// recwriter.h -- batched record output on its own thread.
// emit() used to call fwrite_unlocked() per 32-byte record (and printf() per record for --text).
// RecWriter fills a large buffer in place and hands full buffers to a writer thread that issues
// write(2), so formatting/copying overlaps the reconstruction and there is one syscall per 16 MB.
// The bytes written are exactly what the old code wrote, in the same order.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cerrno>
#include <stdexcept>
#include "trace_format.h"

class RecWriter {
public:
    explicit RecWriter(int fd, size_t buf_bytes = 16u << 20, int nbufs = 4) : fd_(fd), cap_(buf_bytes) {
        for (int i = 0; i < nbufs; i++) free_.push_back(new char[cap_]);
        cur_ = take_free(); th_ = std::thread([this] { loop(); });
    }
    ~RecWriter() { close(); }
    // binary record
    // room for `bytes' (<= the buffer size) written in place by the caller
    inline char* claim(size_t bytes) { if (n_ + bytes > cap_) swap(); char* p = cur_ + n_; n_ += bytes; return p; }
    inline void put(const mtrace_rec& r) { if (n_ + sizeof r > cap_) swap(); memcpy(cur_ + n_, &r, sizeof r); n_ += sizeof r; }
    // the --text line: "%c %lx %d %lx%s\n"
    inline void put_text(char opc, uint64_t addr, int size, uint64_t ip, bool known) {
        if (n_ + 64 > cap_) swap();
        char* p = cur_ + n_; *p++ = opc; *p++ = ' '; p = hex(p, addr); *p++ = ' ';
        p = dec(p, size); *p++ = ' '; p = hex(p, ip); if (!known) { *p++ = ' '; *p++ = '?'; } *p++ = '\n';
        n_ = (size_t)(p - cur_);
    }
    // a preformatted line (sync events)
    inline void put_raw(const char* s, size_t n) { if (n_ + n > cap_) swap(); memcpy(cur_ + n_, s, n); n_ += n; }
    void close() {
        if (closed_) return; closed_ = true;
        if (n_) { push_full(cur_, n_); cur_ = nullptr; n_ = 0; }
        { std::lock_guard<std::mutex> g(mu_); done_ = true; } cv_.notify_all();
        if (th_.joinable()) th_.join();
        if (cur_) delete[] cur_;
        for (char* b : free_) delete[] b; free_.clear();
    }
    bool failed() const { return err_ != 0; }
private:
    static inline char* hex(char* p, uint64_t v) {
        char t[16]; int n = 0; do { t[n++] = "0123456789abcdef"[v & 15]; v >>= 4; } while (v);
        while (n) *p++ = t[--n]; return p; }
    static inline char* dec(char* p, int v) {
        if (v < 0) { *p++ = '-'; v = -v; } char t[12]; int n = 0; do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
        while (n) *p++ = t[--n]; return p; }
    char* take_free() {
        std::unique_lock<std::mutex> g(mu_); cv_.wait(g, [this] { return !free_.empty(); });
        char* b = free_.back(); free_.pop_back(); return b; }
    void push_full(char* b, size_t n) { { std::lock_guard<std::mutex> g(mu_); full_.push_back({b, n}); } cv_.notify_all(); }
    void swap() { push_full(cur_, n_); n_ = 0; cur_ = take_free(); }
    void loop() {
        for (;;) {
            std::pair<char*, size_t> job;
            { std::unique_lock<std::mutex> g(mu_); cv_.wait(g, [this] { return !full_.empty() || done_; });
              if (full_.empty()) return; job = full_.front(); full_.pop_front(); }
            size_t off = 0;
            while (off < job.second) { ssize_t w = ::write(fd_, job.first + off, job.second - off);
                if (w < 0) { if (errno == EINTR) continue; err_ = errno; break; } off += (size_t)w; }
            { std::lock_guard<std::mutex> g(mu_); free_.push_back(job.first); } cv_.notify_all();
        }
    }
    int fd_; size_t cap_; char* cur_ = nullptr; size_t n_ = 0; bool closed_ = false, done_ = false; int err_ = 0;
    std::mutex mu_; std::condition_variable cv_; std::vector<char*> free_; std::deque<std::pair<char*, size_t>> full_;
    std::thread th_;
};
