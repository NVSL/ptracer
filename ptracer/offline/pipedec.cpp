// pipedec.cpp -- PIPELINED PT decoding for a serial reconstruction (see ptdecode.h,
// PtDecoder::start_pipeline).  K forked producers decode PSB-aligned segments of the AUX file and
// stream what PtDecoder::fetch_blk() produced to the consumer through shared-memory rings:
//   BLK  one block, exactly as fetch_blk() left it ready (block fields, the offset/time before it,
//        the live offset/time after it, the pending-boundary state);
//   EV_* one event callback (ptwrite, overflow, enable, disable, resync) with its arguments and the
//        decoder's live offset/time at the moment it fired;
//   EOS  the end of the stream (eos_why/off/status);  SEG_END  the end of a producer's segment.
// Every item carries the decoder counters' increments since the previous item.  An item's KEY is
// the live offset at which it was produced (a block: after it, i.e. the offset its last
// instruction reports).  Segment s = [B[s], B[s+1]) emits exactly the items whose key lies in it;
// its producer starts decoding at the PSB before B[s] (the --jobs --decode-only warm-up rule, which
// makes a fresh decoder agree with a continuous one from B[s] on).  Keys are monotonic, so the
// concatenation over segments is the continuous decoder's item sequence.
#include "ptdecode.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <unistd.h>
#include <signal.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>

namespace {
enum : uint8_t { PI_BLK = 1, PI_PTW, PI_OVF, PI_EN, PI_DIS, PI_RES, PI_EOS, PI_SEGEND, PI_PLAN };
struct PItem {
    uint8_t type, trunc, spec, iclass, size, mode, pend, pkind;
    int32_t isid, st; uint32_t ninsn;
    uint16_t d_resync, d_syncfail, d_ovf, d_ptw, d_walkfail, d_blocks;
    uint64_t ip, end_ip, boff0, btsc0, loff, ltime, a, b;
    int32_t b_ok, lt_ok;
    uint8_t raw[16];
    uint32_t rep;      // a PI_BLK item stands for `rep' (>= 1) identical consecutive items
};
static constexpr size_t PI_CMP = offsetof(PItem, rep);   // bytes two items must share to coalesce
static_assert(sizeof(PItem) <= 128, "PItem");
struct Ring {
    alignas(64) std::atomic<uint64_t> head;    // written by the producer
    alignas(64) std::atomic<uint64_t> tail;    // written by the consumer
    alignas(64) uint64_t cap;
    PItem* items() { return reinterpret_cast<PItem*>(reinterpret_cast<char*>(this) + 192); }
};
static void pipe_nap() { struct timespec ts{0, 20000}; nanosleep(&ts, nullptr); }
// Producer side; the head index is published every 32 items (and at a segment end / flush) and the
// consumer's tail is re-read only when the cached copy says the ring is full: no cache-line
// ping-pong per item.
static thread_local uint64_t tl_head = 0, tl_tail_seen = 0, tl_unpub = 0;
static void ring_flush(Ring* r) { r->head.store(tl_head, std::memory_order_release); tl_unpub = 0; }
static void ring_push(Ring* r, const PItem& it, bool flush = false) {
    if (tl_head - tl_tail_seen >= r->cap) {
        ring_flush(r);
        // full: the consumer is far behind -- back off up to 1 ms (no latency matters here, the ring is large)
        long ns = 20000;
        for (int spin = 0; tl_head - (tl_tail_seen = r->tail.load(std::memory_order_acquire)) >= r->cap; spin++) {
            if (spin < 16) { sched_yield(); continue; }
            struct timespec ts{0, ns}; nanosleep(&ts, nullptr); if (ns < 1000000) ns *= 2; }
    }
    r->items()[tl_head & (r->cap - 1)] = it; tl_head++;
    if (++tl_unpub >= 32 || flush) ring_flush(r);
}
}  // namespace

struct PtDecoder::Pipe {
    int K = 0; size_t ring_bytes = 0; std::vector<Ring*> rings; std::vector<pid_t> pids;
    std::vector<uint64_t> B;             // segment boundaries: B[0] = 0 < B[1] < ... < B[n] = total
    size_t cut = 1;                      // direct mode: the next boundary at which to hand over
    size_t rs = 0;                       // replay: current segment
    size_t rs_c = ~(size_t)0, ri_c = 0;  // pop(): ring index of rs
    struct pt_image_section_cache* own_isc = nullptr;
    std::unordered_map<uint64_t, std::vector<BInsn>> old_walk;   // keeps the direct-mode walk entries alive
    std::vector<uint64_t> c_tail, c_head;   // consumer-side cached indices per ring
    PItem rep_it{}; uint32_t rep_left = 0;  // the copies still to deliver of a coalesced block item
    bool pop(PItem& it, uint64_t& waits) {
        if (rs != rs_c) { rs_c = rs; ri_c = (rs - 1) % (size_t)K; }   // (a division per item was ~6 % of a gemm replay)
        const size_t ri = ri_c; Ring* r = rings[ri];
        uint64_t& t = c_tail[ri]; uint64_t& h = c_head[ri];
        if (t == h) {
            r->tail.store(t, std::memory_order_release);
            for (int spin = 0; (h = r->head.load(std::memory_order_acquire)) == t; spin++) {
                if (spin == 0) waits++;
                if (spin > 256) pipe_nap(); else if (spin > 32) sched_yield();
            }
        }
        const PItem* src = &r->items()[t & (r->cap - 1)];
        __builtin_prefetch((const char*)&r->items()[(t + 16) & (r->cap - 1)]);
        __builtin_prefetch((const char*)&r->items()[(t + 16) & (r->cap - 1)] + 64);
        it = *src; t++;
        if ((t & 63) == 0) r->tail.store(t, std::memory_order_release);
        return true;
    }
    // the item in place (no 128-byte copy).  Valid until the next pop: the tail
    // published here never passes it (t - 1), and the caller is done with it before popping again.
    const PItem* pop_ref(uint64_t& waits) {
        if (rs != rs_c) { rs_c = rs; ri_c = (rs - 1) % (size_t)K; }
        const size_t ri = ri_c; Ring* r = rings[ri];
        uint64_t& t = c_tail[ri]; uint64_t& h = c_head[ri];
        if (t == h) {
            r->tail.store(t, std::memory_order_release);
            for (int spin = 0; (h = r->head.load(std::memory_order_acquire)) == t; spin++) {
                if (spin == 0) waits++;
                if (spin > 256) pipe_nap(); else if (spin > 32) sched_yield();
            }
        }
        const PItem* src = &r->items()[t & (r->cap - 1)];
        __builtin_prefetch((const char*)&r->items()[(t + 16) & (r->cap - 1)]);
        __builtin_prefetch((const char*)&r->items()[(t + 16) & (r->cap - 1)] + 64);
        t++;
        if ((t & 63) == 0) r->tail.store(t - 1, std::memory_order_release);
        return src;
    }
    // the item d places ahead in the current ring, for prefetching only (it may not be written yet:
    // the caller must never act on it, only hint the cache).
    const PItem* peek(size_t d) const { Ring* r = rings[ri_c]; return &r->items()[(c_tail[ri_c] + d) & (r->cap - 1)]; }
    void drain_segment(uint64_t& waits) {   // discard the rest of segment rs
        PItem it; for (;;) { pop(it, waits); if (it.type == PI_SEGEND || it.type == PI_EOS) return; } }
};

bool PtDecoder::start_pipeline(const Sideband& sb, const std::string& auxfile, int K, uint64_t seg_bytes,
                               uint64_t range_begin, uint64_t range_end) {
    if (pipe_ || jit || !blk_ || dec_hash_on || progress_.limit || K < 1 || (skip_ != 0 && skip_ != range_begin)) return false;
    if (plan_ && !range_end) return false;
    const uint64_t fsize = aux_size(auxfile);
    const uint64_t rb = range_begin, re = range_end && range_end < fsize ? range_end : fsize;
    const uint64_t total = re - rb;
    if (!seg_bytes) seg_bytes = std::max<uint64_t>(512ull << 10, total / ((uint64_t)K * 64));
    if (total < 2 * seg_bytes) return false;
    std::vector<uint64_t> psb = psb_offsets(auxfile);
    if (psb.size() < 4) return false;
    Pipe* p = new Pipe(); p->K = K; p->B.push_back(rb);
    size_t j = 0;
    for (uint64_t want = rb + seg_bytes; want < re; want += seg_bytes) {
        while (j < psb.size() && psb[j] < want) j++;
        if (j >= psb.size()) break;
        if (j == 0) continue;                       // a segment needs a PSB before it (warm start)
        if (psb[j] > p->B.back()) p->B.push_back(psb[j]);
    }
    p->B.push_back(fsize);     // the last producer runs to the end of the file (the consumer stops where it stops)
    if (p->B.size() < 3) { delete p; return false; }
    prebuild_shared(auxfile, sb);
    if (!s_img_ || s_sb_ != &sb) { delete p; return false; }
    const size_t nseg = p->B.size() - 1;
    uint64_t cap = 1u << 20;                   // items per ring (128 MB): producers run a segment or more ahead
    if (const char* e = getenv("PTRECON_PIPE_RING")) cap = strtoull(e, nullptr, 0);
    { uint64_t c2 = 1; while (c2 < cap) c2 <<= 1; cap = c2; }   // power of two: index by mask, not division
    p->ring_bytes = 192 + cap * sizeof(PItem);
    for (int k = 0; k < K; k++) {
        void* m = mmap(nullptr, p->ring_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { for (Ring* r : p->rings) munmap(r, p->ring_bytes); delete p; return false; }
        Ring* r = new (m) Ring(); r->head.store(0); r->tail.store(0); r->cap = cap; p->rings.push_back(r);
    }
    fflush(stdout); fflush(stderr);
    const bool nt = no_time;
    for (int k = 0; k < K; k++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork (pipeline)"); break; }
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            Ring* r = p->rings[(size_t)k];
            for (size_t s = 1 + (size_t)k; s < nseg; s += (size_t)K) {
                const uint64_t from = p->B[s], to = s + 1 < nseg ? p->B[s + 1] : 0;
                size_t lo = 0, hi = psb.size();             // the last PSB strictly before `from'
                while (lo < hi) { size_t mid = (lo + hi) / 2; if (psb[mid] < from) lo = mid + 1; else hi = mid; }
                const uint64_t warm = psb[lo - 1];
                try { PtDecoder P(auxfile, sb, warm, 0, nullptr, nullptr, 0); P.no_time = nt;
                      P.produce(r, from, to, s + 1 == nseg); }
                catch (std::exception& e) {
                    fprintf(stderr, "ptrecon pipeline producer, segment %zu: %s\n", s, e.what());
                    PItem it{}; it.type = PI_EOS; it.b = (uint64_t)(const char*)"pipeline producer failed"; ring_push(r, it, true); _exit(1); }
            }
            _exit(0);
        }
        p->pids.push_back(pid);
    }
    if ((int)p->pids.size() != K) { pipe_ = p; pipe_stop(); return false; }
    p->c_tail.assign((size_t)K, 0); p->c_head.assign((size_t)K, 0);
    pipe_ = p; pipe_k = K; pipe_segs = (int)nseg;
    return true;
}

void PtDecoder::pipe_stop() {
    if (!pipe_) return;
    if (pipe_k) fprintf(stderr, "ptrecon: pipeline: %d producers, %d segments, replay from segment %d, %lu items, %lu consumer waits\n",
                        pipe_k, pipe_segs, pipe_from_seg, (unsigned long)n_pipe_items, (unsigned long)n_pipe_waits);
    if (getenv("PTRECON_DEV_REPDBG")) fprintf(stderr, "repdbg consumer: bi<bn %lu pend %lu trunc %lu preinit %lu limit %lu ok %lu\n", (unsigned long)dbg_rr[0], (unsigned long)dbg_rr[1], (unsigned long)dbg_rr[2], (unsigned long)dbg_rr[3], (unsigned long)dbg_rr[4], (unsigned long)dbg_rr[5]);
    for (pid_t pid : pipe_->pids) kill(pid, SIGKILL);
    for (pid_t pid : pipe_->pids) { int st; waitpid(pid, &st, 0); }
    for (Ring* r : pipe_->rings) munmap(r, pipe_->ring_bytes);
    if (rp_) { iscache_ = pipe_->own_isc; }       // the direct decoder's own cache is freed by the destructor
    delete pipe_; pipe_ = nullptr;
}

// Direct mode: an item with live offset `key' is about to be produced.  At or past the next
// boundary, hand over to the producers' stream (returns true: the caller abandons the direct
// decoder and replays).  Not while the E9 pre-init window is open: the producers decode with the
// post-init image, so that segment is skipped (decoded here) and the next boundary is tried.
bool PtDecoder::pipe_cut(uint64_t key) {
    Pipe* p = pipe_;
    if (!p || rp_ || p->cut + 1 >= p->B.size() || key < p->B[p->cut]) return false;
    while (p->cut + 1 < p->B.size() && key >= p->B[p->cut + 1]) p->cut++;   // (a jump over a whole segment)
    if (preinit_ && e9_unswapped_) { p->cut++; return false; }
    // discard the segments decoded here, then switch
    for (p->rs = 1; p->rs < p->cut; p->rs++) p->drain_segment(n_pipe_waits);
    pipe_from_seg = (int)p->cut;
    p->own_isc = iscache_; iscache_ = s_isc_;
    p->old_walk.swap(bwalk_); bwalk_.clear(); rw_ptr_ = nullptr;
    for (auto& f : bfront_) f = BFront();
    rp_ = true; bn_ = bi_ = 0; bpend_err_ = 0; eos_ = false;
    return true;
}

void PtDecoder::pipe_end_plan() {
    if (!pipe_) return;
    const bool was = rp_;
    if (rp_) { bwalk_.swap(pipe_->old_walk); for (auto& f : bfront_) f = BFront(); } rw_ptr_ = nullptr;
    pipe_stop(); rp_ = false;
    if (was) { bn_ = bi_ = 0; bpend_err_ = 0; eos_ = false; synced_ = false; }
}

bool PtDecoder::fetch_blk_replay(PtEvents& ev) {
    Pipe* p = pipe_;
    for (;;) {
        if (bi_ < bn_) return true;
        if (eos_) return false;
        const PItem* itp;
        if (p->rep_left) { p->rep_left--; itp = &p->rep_it; }
        else { itp = p->pop_ref(n_pipe_waits);
               if (itp->type == PI_BLK && itp->rep > 1) { p->rep_it = *itp; p->rep_left = itp->rep - 1; itp = &p->rep_it; } }
        const PItem& it = *itp;
        if (it.type == PI_SEGEND) { p->rs++; continue; }
        n_pipe_items++;
        // staged lookahead over the ring (data only, no effect on the output): the walk-cache front slot
        // of the block 24 items ahead, its vector header 12 ahead, the walk's first/last BInsn 5 ahead and the
        // InsnInfo / fused program they point to 2 ahead -- the dependent misses blk_walk and batch_run pay.
        if (pf_on_ && !jit && !p->rep_left) {
            auto slot = [&](const PItem* q) -> BFront& { const uint64_t k = q->ip | ((uint64_t)(uint32_t)q->isid << 48);
                                                          return bfront_[(size_t)((k * 0x9E3779B97F4A7C15ull) >> 46)]; };
            auto hit = [&](const PItem* q, BFront& f) { return f.key == (q->ip | ((uint64_t)(uint32_t)q->isid << 48)) && f.v; };
            __builtin_prefetch(&slot(p->peek(24)));
            { const PItem* q = p->peek(12); BFront& f = slot(q); if (hit(q, f)) __builtin_prefetch(f.v); }
            { const PItem* q = p->peek(5); BFront& f = slot(q); if (hit(q, f)) { const size_t n = f.v->size();
                  if (n) { const BInsn* d = f.v->data(); __builtin_prefetch(d); __builtin_prefetch(d + n - 1); } } }
            { const PItem* q = p->peek(2); BFront& f = slot(q); if (hit(q, f)) { const size_t n = f.v->size();
                  if (n) { const BInsn* d = f.v->data(); if (d->uptr) __builtin_prefetch(d->uptr); if (d->fptr) __builtin_prefetch(d->fptr); } } }
        }
        rp_loff_ = it.loff - skip_;
        // plan mode: an event is delivered only inside the current plan segment (drain_events' rule)
        const bool deliver = !plan_ || ev_ok();
        if (!deliver && (it.type == PI_PTW || it.type == PI_OVF || it.type == PI_EN || it.type == PI_DIS)) {
            n_resync += it.d_resync; n_sync_fail += it.d_syncfail; n_walk_fail += it.d_walkfail; n_blocks += it.d_blocks;
            if (it.type == PI_OVF) { pending_boundary_ = true; pending_kind_ = 'O'; }
            continue; }
        // a resync outside the plan segment is counted but not delivered (step()'s rule;
        // fetch_blk delivered it, and a node worker took 14 spurious state losses: +181 unknown)
        if (!deliver && it.type == PI_RES) {
            n_resync += it.d_resync; n_sync_fail += it.d_syncfail; n_walk_fail += it.d_walkfail; n_blocks += it.d_blocks;
            pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
            continue; }
        n_resync += it.d_resync; n_sync_fail += it.d_syncfail; n_ovf += it.d_ovf; n_ptw += it.d_ptw;
        n_walk_fail += it.d_walkfail; n_blocks += it.d_blocks;
        rp_loff_ = it.loff - skip_; rp_ltime_ = it.ltime; rp_lt_ok_ = it.lt_ok;
        switch (it.type) {
        case PI_BLK:
            /* every field of pt_block is assigned below: no memset */
            blk_cur_.ip = it.ip; blk_cur_.end_ip = it.end_ip; blk_cur_.isid = it.isid; blk_cur_.ninsn = it.ninsn;
            blk_cur_.mode = (enum pt_exec_mode)it.mode; blk_cur_.iclass = (enum pt_insn_class)it.iclass;
            blk_cur_.truncated = it.trunc; blk_cur_.speculative = it.spec; blk_cur_.size = it.size;
            memcpy(blk_cur_.raw, it.raw, sizeof blk_cur_.raw < sizeof it.raw ? sizeof blk_cur_.raw : sizeof it.raw);
            boff0_ = it.boff0 - skip_; btsc0_ = it.btsc0; btsc0_ok_ = it.b_ok;
            pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
            // JIT blocks -- the time key fetch_blk set before pt_blk_next, and no reuse of the previous walk
            // (blk_walk_jit re-validates its bytes at that key)
            if (jit) { static const bool jit_notime = getenv("PTRECON_JIT_NOTIME") != nullptr;
                       if (!jit_notime && it.b_ok >= 0 && it.btsc0) cur_tsc = it.btsc0; }
            if (rw_ptr_ && it.isid > 0 && it.ip == rw_ip_ && it.isid == rw_isid_ && it.ninsn == rw_n_ && it.end_ip == rw_end_ && it.trunc == rw_trunc_ && it.mode == rw_mode_) bcur_ = rw_ptr_;
            else { bcur_ = blk_walk(blk_cur_); rw_ptr_ = bcur_; rw_ip_ = it.ip; rw_isid_ = it.isid; rw_n_ = it.ninsn; rw_end_ = it.end_ip; rw_trunc_ = it.trunc; rw_mode_ = it.mode; }
            bn_ = it.ninsn; bi_ = 0;
            break;
        case PI_PTW: pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
                     if (ev.on_ptwrite) ev.on_ptwrite(it.a, it.st, it.b); break;
        case PI_OVF: pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
                     if (ev.on_overflow) ev.on_overflow(it.a); break;
        case PI_EN:  pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
                     if (ev.on_enable) ev.on_enable(it.a, true); break;
        case PI_DIS: pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
                     if (ev.on_enable) ev.on_enable(0, false); break;
        case PI_RES: pending_boundary_ = it.pend; pending_kind_ = (char)it.pkind;
                     if (ev.on_resync) ev.on_resync(it.st, it.a); break;
        case PI_EOS: eos_ = true; eos_why = ppipe_ ? "plan pipeline: end of stream" : (const char*)it.b; eos_off = it.a; eos_status = it.st; return false;
        case PI_PLAN: throw std::runtime_error("plan pipeline: positioning item where a block was expected");
        default: throw std::runtime_error("pipeline: bad item");
        }
    }
}

// Producer side: run fetch_blk() over [from, to) (to = 0: to the end) and stream its effects.
void PtDecoder::produce(void* rv, uint64_t from, uint64_t to, bool last) {
    (void)last;
    Ring* R = (Ring*)rv;
    uint64_t c_res = n_resync, c_sf = n_sync_fail, c_ovf = n_ovf, c_ptw = n_ptw, c_wf = n_walk_fail, c_blk = n_blocks;
    bool stop = false;
    auto mk = [&](PItem& it, uint64_t key) -> bool {
        it.d_resync = (uint16_t)(n_resync - c_res); it.d_syncfail = (uint16_t)(n_sync_fail - c_sf); it.d_ovf = (uint16_t)(n_ovf - c_ovf);
        it.d_ptw = (uint16_t)(n_ptw - c_ptw); it.d_walkfail = (uint16_t)(n_walk_fail - c_wf); it.d_blocks = (uint16_t)(n_blocks - c_blk);
        c_res = n_resync; c_sf = n_sync_fail; c_ovf = n_ovf; c_ptw = n_ptw; c_wf = n_walk_fail; c_blk = n_blocks;
        if (to && key >= to) { stop = true; return false; }
        return key >= from;
    };
    // identical consecutive block items (a hot loop's iterations inside one TNT packet:
    // same block, offsets, times, counter increments) travel as one item with a repeat count.
    PItem held{}; bool have_held = false;
    static const bool dbg_on = getenv("PTRECON_DEV_REPDBG") != nullptr; uint64_t dbg_coal = 0, dbg_diff[129] = {0};
    auto flush_held = [&]() { if (have_held) { ring_push(R, held); have_held = false; } };
    auto push_blk = [&](const PItem& it) {
        if (have_held && held.rep < 0xffffffffu && !memcmp(&held, &it, PI_CMP)) { held.rep++; dbg_coal++; return; }
        if (dbg_on && have_held && held.ip == it.ip && held.end_ip == it.end_ip) {
            const unsigned char* x = (const unsigned char*)&held; const unsigned char* y = (const unsigned char*)&it;
            size_t i = 0; while (i < PI_CMP && x[i] == y[i]) i++; dbg_diff[i < PI_CMP ? i : PI_CMP]++; }
        flush_held(); held = it; held.rep = 1; have_held = true; };
    auto evitem = [&](uint8_t type, uint64_t a, uint64_t b, int32_t st) {
        if (stop) return;
        flush_held();
        PItem it{}; it.type = type; it.a = a; it.b = b; it.st = st; it.rep = 1;
        it.loff = skip_ + live_offset(); it.lt_ok = live_time(&it.ltime);
        it.pend = pending_boundary_; it.pkind = (uint8_t)pending_kind_;
        if (mk(it, it.loff)) ring_push(R, it);
    };
    const bool coalesce = !(getenv("PTRECON_PIPE_NOREP") && *getenv("PTRECON_PIPE_NOREP") == '1');
    PtEvents pev;
    pev.on_ptwrite = [&](uint64_t pl, int sz, uint64_t ip) { evitem(PI_PTW, pl, ip, sz); };
    pev.on_overflow = [&](uint64_t tsc) { evitem(PI_OVF, tsc, 0, 0); };
    pev.on_enable = [&](uint64_t ip, bool en) { evitem(en ? PI_EN : PI_DIS, ip, 0, 0); };
    pev.on_resync = [&](int err, uint64_t off) { evitem(PI_RES, off, 0, err); };
    for (;;) {
        const bool ok = fetch_blk(pev);
        if (stop) break;
        if (!ok) {
            flush_held();
            PItem it{}; it.type = PI_EOS; it.a = eos_off; it.st = eos_status; it.b = (uint64_t)eos_why; it.rep = 1;
            it.loff = skip_ + live_offset(); mk(it, 0); ring_push(R, it, true); return;
        }
        PItem it{}; it.type = PI_BLK; it.rep = 1;
        it.ip = blk_cur_.ip; it.end_ip = blk_cur_.end_ip; it.isid = blk_cur_.isid; it.ninsn = bn_;
        it.mode = (uint8_t)blk_cur_.mode; it.iclass = (uint8_t)blk_cur_.iclass; it.trunc = blk_cur_.truncated;
        it.spec = blk_cur_.speculative; it.size = blk_cur_.size;
        memcpy(it.raw, blk_cur_.raw, sizeof blk_cur_.raw < sizeof it.raw ? sizeof blk_cur_.raw : sizeof it.raw);
        it.boff0 = skip_ + boff0_; it.btsc0 = btsc0_; it.b_ok = btsc0_ok_;
        it.loff = skip_ + live_offset(); it.lt_ok = live_time(&it.ltime);
        it.pend = pending_boundary_; it.pkind = (uint8_t)pending_kind_;
        bi_ = bn_; pending_boundary_ = false; pending_kind_ = 'R'; seen_insn_ = true;   // what delivering it does
        if (mk(it, it.loff)) { if (coalesce) push_blk(it); else ring_push(R, it); }
        if (stop) break;
    }
    flush_held();
    if (dbg_on) { fprintf(stderr, "repdbg seg [%lu,%lu) coal %lu diffs:", (unsigned long)from, (unsigned long)to, (unsigned long)dbg_coal);
                  for (int i = 0; i <= (int)PI_CMP; i++) if (dbg_diff[i]) fprintf(stderr, " @%d:%lu", i, (unsigned long)dbg_diff[i]); fprintf(stderr, "\n"); }
    PItem e{}; e.type = PI_SEGEND; e.rep = 1; ring_push(R, e, true);
}

uint64_t PtDecoder::dbg_rr[8];
uint32_t PtDecoder::rp_reps() const {
    const Pipe* p = pipe_;
    if (!p || !rp_ || eos_ || !p->rep_left) return 0;
    if (bi_ < bn_ || !blk_) { dbg_rr[0]++; return 0; }
    // plan mode: batch_avail's whole-block conditions (an identical copy has the same offsets, so the
    // segment tests the first copy passed hold for it too)
    if (plan_ && (!plan_rebatch_ || !positioned_ || seg_first_ || seg_i_ >= plan_->segs.size())) return 0;
    const PItem& it = p->rep_it;
    if (it.pend) { dbg_rr[1]++; return 0; } if (it.trunc) { dbg_rr[2]++; return 0; }
    if (preinit_ && e9_unswapped_) { dbg_rr[3]++; return 0; } if (dec_hash_on || progress_.limit) { dbg_rr[4]++; return 0; }
    dbg_rr[5]++;
    return p->rep_left;
}
void PtDecoder::rp_take_rep() {
    // fetch_blk_replay's PI_BLK case for an identical copy (block fields, offsets, times and the walk
    // are those already in place), then batch_advance(bn_)
    Pipe* p = pipe_; const PItem& it = p->rep_it;
    p->rep_left--; n_pipe_items++;
    n_resync += it.d_resync; n_sync_fail += it.d_syncfail; n_ovf += it.d_ovf; n_ptw += it.d_ptw;
    n_walk_fail += it.d_walkfail; n_blocks += it.d_blocks;
    bi_ = 0; batch_advance(bn_);
}
void PtDecoder::rp_take_reps(uint32_t n) {
    Pipe* p = pipe_; const PItem& it = p->rep_it;
    p->rep_left -= n; n_pipe_items += n;
    n_resync += (uint64_t)n * it.d_resync; n_sync_fail += (uint64_t)n * it.d_syncfail; n_ovf += (uint64_t)n * it.d_ovf; n_ptw += (uint64_t)n * it.d_ptw;
    n_walk_fail += (uint64_t)n * it.d_walkfail; n_blocks += (uint64_t)n * it.d_blocks;
    if (preinit_ && e9_unswapped_) n_e9_preinit_insn += (uint64_t)n * bn_;
    n_insn += (uint64_t)n * bn_; bi_ = bn_; pending_boundary_ = false; pending_kind_ = 'R'; seen_insn_ = true;
}

// ---- plan-level pipeline -----------------------------------------------------------------------------------------
namespace {
struct PPState { Ring* r = nullptr; PItem held{}; bool have = false; uint64_t c_res = 0, c_sf = 0, c_ovf = 0, c_ptw = 0, c_wf = 0, c_blk = 0; };
PPState g_pp;   // producer process only
}
void PtDecoder::pp_flush() { if (g_pp.have) { ring_push(g_pp.r, g_pp.held); g_pp.have = false; } }
static void pp_mk(PtDecoder& d, PItem& it, uint64_t nres, uint64_t nsf, uint64_t novf, uint64_t nptw, uint64_t nwf, uint64_t nblk) {
    it.d_resync = (uint16_t)(nres - g_pp.c_res); it.d_syncfail = (uint16_t)(nsf - g_pp.c_sf); it.d_ovf = (uint16_t)(novf - g_pp.c_ovf);
    it.d_ptw = (uint16_t)(nptw - g_pp.c_ptw); it.d_walkfail = (uint16_t)(nwf - g_pp.c_wf); it.d_blocks = (uint16_t)(nblk - g_pp.c_blk);
    g_pp.c_res = nres; g_pp.c_sf = nsf; g_pp.c_ovf = novf; g_pp.c_ptw = nptw; g_pp.c_wf = nwf; g_pp.c_blk = nblk; (void)d; }
void PtDecoder::pp_emit_blk() {
    PItem it{}; it.type = PI_BLK; it.rep = 1;
    it.ip = blk_cur_.ip; it.end_ip = blk_cur_.end_ip; it.isid = blk_cur_.isid; it.ninsn = bn_;
    it.mode = (uint8_t)blk_cur_.mode; it.iclass = (uint8_t)blk_cur_.iclass; it.trunc = blk_cur_.truncated;
    it.spec = blk_cur_.speculative; it.size = blk_cur_.size;
    memcpy(it.raw, blk_cur_.raw, sizeof blk_cur_.raw < sizeof it.raw ? sizeof blk_cur_.raw : sizeof it.raw);
    it.boff0 = skip_ + boff0_; it.btsc0 = btsc0_; it.b_ok = btsc0_ok_;
    it.loff = skip_ + live_offset(); it.lt_ok = live_time(&it.ltime);
    it.pend = pending_boundary_; it.pkind = (uint8_t)pending_kind_;
    pp_mk(*this, it, n_resync, n_sync_fail, n_ovf, n_ptw, n_walk_fail, n_blocks);
    if (g_pp.have && g_pp.held.rep < 0xffffffffu && !memcmp(&g_pp.held, &it, PI_CMP)) { g_pp.held.rep++; return; }
    pp_flush(); g_pp.held = it; g_pp.have = true;
}
void PtDecoder::pp_emit_eos() {
    pp_flush(); PItem it{}; it.type = PI_EOS; it.a = eos_off; it.st = eos_status; it.rep = 1; it.loff = skip_ + live_offset();
    pp_mk(*this, it, n_resync, n_sync_fail, n_ovf, n_ptw, n_walk_fail, n_blocks); ring_push(g_pp.r, it); }
void PtDecoder::pp_emit_plan(size_t seg, int kind, uint64_t d_reseek) {
    pp_flush(); PItem it{}; it.type = PI_PLAN; it.a = seg; it.st = kind; it.b = d_reseek; it.rep = 1;
    it.loff = skip_ + live_offset(); pp_mk(*this, it, n_resync, n_sync_fail, n_ovf, n_ptw, n_walk_fail, n_blocks); ring_push(g_pp.r, it); }
void PtDecoder::pp_take_plan(size_t seg, int* kind, uint64_t* d_reseek) {
    Pipe* p = pipe_;
    if (pp_g_ + 1 < pp_gb_.size() && seg >= pp_gb_[pp_g_ + 1]) {      // the next group: the rest of this one, then its ring
        if (p->rep_left) p->rep_left = 0;
        if (p->rs >= 1) { for (;;) { const PItem* x = p->pop_ref(n_pipe_waits); if (x->type == PI_SEGEND) break;
                                       n_resync += x->d_resync; n_sync_fail += x->d_syncfail; n_ovf += x->d_ovf; n_ptw += x->d_ptw; n_walk_fail += x->d_walkfail; n_blocks += x->d_blocks; } }
        while (pp_g_ + 1 < pp_gb_.size() && seg >= pp_gb_[pp_g_ + 1]) pp_g_++;
        p->rs = pp_g_ + 1;
    }
    if (p->rep_left) throw std::runtime_error("plan pipeline: positioning inside a repeated block");
    const PItem* x = p->pop_ref(n_pipe_waits);
    if (x->type != PI_PLAN || x->a != seg) throw std::runtime_error("plan pipeline: out of step at segment " + std::to_string(seg) +
        " (item type " + std::to_string(x->type) + " a " + std::to_string(x->a) + " ip " + std::to_string(x->ip) + " loff " + std::to_string(x->loff) +
        "; consumer offset " + std::to_string(offset()) + " bi " + std::to_string(bi_) + "/" + std::to_string(bn_) + " group " + std::to_string(pp_g_) + ")");
    n_resync += x->d_resync; n_sync_fail += x->d_syncfail; n_ovf += x->d_ovf; n_ptw += x->d_ptw; n_walk_fail += x->d_walkfail; n_blocks += x->d_blocks;
    *kind = x->st; *d_reseek = x->b;
    rp_loff_ = x->loff - skip_;
}
bool PtDecoder::start_plan_pipeline(int K) {
    // (a plan decoder sets blk_ when it opens its first file: the same predicate as alloc_decoder())
    { const char* want = getenv("PTRECON_DECODER");
      const bool will_blk = !(want && !strcmp(want, "insn")) && !(getenv("PTRECON_PLAN_BLK") && *getenv("PTRECON_PLAN_BLK") == '0')
                            && !(jit && getenv("PTRECON_JIT_BLK") && *getenv("PTRECON_JIT_BLK") == '0') && !uncached_;
      if (!will_blk) return false; }
    if (!plan_ || pipe_ || rp_ || dec_hash_on || progress_.limit || K < 1 || seg_i_ != 0 || positioned_ || (preinit_ && e9_unswapped_)) {
        if (getenv("PTRECON_PP_DBG")) fprintf(stderr, "plan pipeline not started: plan %d pipe %d rp %d blk %d hash %d lim %d seg %zu pos %d preinit %d\n",
            plan_ != nullptr, pipe_ != nullptr, rp_, blk_, dec_hash_on, progress_.limit != 0, seg_i_, positioned_, preinit_ && e9_unswapped_);
        return false; }
    const auto& sg = plan_->segs; if (sg.size() < 2) return false;
    uint64_t total = 0; for (auto& x : sg) total += x.end > x.begin ? x.end - x.begin : 0;
    uint64_t target = std::max<uint64_t>(256u << 10, total / ((uint64_t)K * 32));
    if (const char* e = getenv("PTRECON_PP_GROUP_KB")) target = strtoull(e, nullptr, 0) << 10;
    std::vector<size_t> gb{0}; uint64_t acc = 0;
    for (size_t i = 0; i < sg.size(); i++) {
        // a group starts where the direct decoder seeks for certain: another file, or (same file) a gap far beyond
        // next_plan()'s 64 KB continue window (the first instruction past the previous segment is within 1 MB of its end)
        const bool forced = i > 0 && (sg[i].file != sg[i - 1].file || sg[i].begin > sg[i - 1].end + (64u << 10) + (1u << 20));
        if (forced && acc >= target) { gb.push_back(i); acc = 0; }
        acc += sg[i].end > sg[i].begin ? sg[i].end - sg[i].begin : 0;
    }
    gb.push_back(sg.size());
    const size_t G = gb.size() - 1; if (G < 2) return false;
    Pipe* p = new Pipe(); p->K = K;
    uint64_t cap = 1u << 18; if (const char* e = getenv("PTRECON_PIPE_RING")) cap = strtoull(e, nullptr, 0);
    { uint64_t c2 = 1; while (c2 < cap) c2 <<= 1; cap = c2; }
    p->ring_bytes = 192 + cap * sizeof(PItem);
    for (int k = 0; k < K; k++) {
        void* m = mmap(nullptr, p->ring_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { for (Ring* r : p->rings) munmap(r, p->ring_bytes); delete p; return false; }
        Ring* r = new (m) Ring(); r->head.store(0); r->tail.store(0); r->cap = cap; p->rings.push_back(r);
    }
    fflush(stdout); fflush(stderr);
    for (int k = 0; k < K; k++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork (plan pipeline)"); break; }
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            g_pp = PPState(); g_pp.r = p->rings[(size_t)k]; tl_head = 0; tl_tail_seen = 0; tl_unpub = 0;
            PtEvents pev;
            auto evitem = [&](uint8_t type, uint64_t a, uint64_t b, int32_t st) {
                pp_flush(); PItem it{}; it.type = type; it.a = a; it.b = b; it.st = st; it.rep = 1;
                it.loff = skip_ + live_offset(); it.lt_ok = live_time(&it.ltime);
                it.pend = pending_boundary_; it.pkind = (uint8_t)pending_kind_;
                pp_mk(*this, it, n_resync, n_sync_fail, n_ovf, n_ptw, n_walk_fail, n_blocks); ring_push(g_pp.r, it); };
            pev.on_ptwrite = [&](uint64_t pl, int sz, uint64_t ip) { evitem(PI_PTW, pl, ip, sz); };
            pev.on_overflow = [&](uint64_t tsc) { evitem(PI_OVF, tsc, 0, 0); };
            pev.on_enable = [&](uint64_t ip, bool en) { evitem(en ? PI_EN : PI_DIS, ip, 0, 0); };
            pev.on_resync = [&](int err, uint64_t off) { evitem(PI_RES, off, 0, err); };
            pev.on_switch_loss = [](uint64_t) {};      // the consumer's own next_plan() fires these
            try {
                pp_ring_ = g_pp.r;
                for (size_t g = (size_t)k; g < G; g += (size_t)K) {
                    seg_i_ = gb[g]; pp_end_ = gb[g + 1]; positioned_ = false; seg_first_ = false; cur_file_ = -1; synced_ = false; eos_ = false;
                    bn_ = bi_ = 0; bpend_err_ = 0;
                    struct pt_insn insn; uint64_t tsc = 0;
                    while (next_plan(insn, &tsc, pev)) skip_nonlast(~0ull);
                    pp_flush(); PItem it{}; it.type = PI_SEGEND; it.rep = 1; ring_push(g_pp.r, it, true);
                }
            } catch (std::exception& e) {
                fprintf(stderr, "ptrecon plan pipeline producer %d: %s\n", k, e.what());
                pp_flush(); PItem it{}; it.type = PI_EOS; it.b = (uint64_t)(const char*)"plan pipeline producer failed"; it.rep = 1; ring_push(g_pp.r, it, true); _exit(1); }
            ring_flush(g_pp.r);
            _exit(0);
        }
        p->pids.push_back(pid);
    }
    if ((int)p->pids.size() != K) { pipe_ = p; pipe_stop(); return false; }
    p->c_tail.assign((size_t)K, 0); p->c_head.assign((size_t)K, 0);
    p->rs = 1; p->cut = ~(size_t)0 >> 1;
    blk_ = true; if (bfront_.empty()) bfront_.resize(1u << 18);   // (the consumer never opens a file: alloc_decoder() never runs)
    pipe_ = p; pipe_k = K; pipe_segs = (int)G; rp_ = true; ppipe_ = true; pp_gb_ = gb; pp_g_ = 0; n_pp_groups = G;
    bn_ = bi_ = 0; bpend_err_ = 0; eos_ = false;
    return true;
}
