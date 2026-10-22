// cjit.cpp -- see cjit.h.
#include "cjit.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <spawn.h>
#include <sys/wait.h>
#include <csignal>
#include <unistd.h>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>

extern char** environ;

namespace cjit {
namespace {
struct Job { std::string name, src; std::atomic<void*>* slot; std::atomic<int>* state; uint64_t key; };
pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER, cv_done = PTHREAD_COND_INITIALIZER;
std::vector<Job>* q = nullptr;
bool running = false; int inflight = 0; uint64_t gen = 0; pid_t cc_pid = 0; char cur_c[300], cur_so[300]; /* the batch in flight (quiesce unlinks it) */ int g_pid = 0; int batch_no = 0;
Stats st;
std::vector<uint64_t> inflight_keys;   // claims of the batch in flight (quiesce releases them)
double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
void at_child() {
    pthread_mutex_init(&mu, nullptr); pthread_cond_init(&cv, nullptr); pthread_cond_init(&cv_done, nullptr);
    q = nullptr;          // the parent's queue object is leaked in the child on purpose (its jobs are resubmitted)
    running = false; inflight = 0; g_pid = (int)getpid(); }
double g_t0 = 0;
// COMPILE BUDGET on a one-cpu affinity mask.  The compiler is a child of this process and inherits its cpu
// set, so with one cpu every second of gcc is a second the replay does not run.  Measured pinned to one P-core
// py_nbody (2 452 functions, ~44 s of batch wall) cjit on 98.9 s / off 85.8-86.1 s; gemm
// (24 functions, 0.86 s) on 84.3 s / off 138.8 s.  So: on one cpu, batches are compiled only while the total
// compile wall stays within max(B0, f x elapsed) (default B0 = 2 s, f = 0.02; PTRECON_CJIT_BUDGET="B0,f",
// "0" = unlimited); a batch over budget is dropped (its programs stay interpreted, state 3).  More cpus: unlimited.
// PER-PROGRAM BENEFIT estimate on a one-cpu mask, replacing the flat 2 s / 2 % budget as the selector.
// A program that reached hot() (4096) runs is compiled only once its run count n satisfies
//   n x ops x S >= K x (Cf + Cop x ops)
// i.e. (Lindy: it will run about as often again) the interpretation time compiling saves exceeds the gcc time the
// replay loses on the shared cpu.  Constants calibrated on py_nbody / gemm 1-core: S = 6 ns per
// fused op-run saved, Cf = 10 ms per function, Cop = 1 ms per op.  PTRECON_CJIT_BENEFIT="K[,Cf_ms,Cop_ms,S_ns]",
// "0" = off (seg-2 rule: every hot program, flat budget).  On by default only on a one-cpu mask (more cpus: unchanged);
// the time budget stays as a safety cap, widened to max(2 s, 0.10 x elapsed) when the benefit rule is on.
double g_bk = -1, g_bcf = 0.010, g_bcop = 0.001, g_bs = 6e-9; int g_ncpu = -1;
static int ncpu() { if (g_ncpu < 0) { cpu_set_t cs; CPU_ZERO(&cs); g_ncpu = sched_getaffinity(0, sizeof cs, &cs) == 0 ? CPU_COUNT(&cs) : 0; } return g_ncpu; }
void benefit_init() {
    if (g_bk >= 0) return;
    const char* v = getenv("PTRECON_CJIT_BENEFIT");
    if (v) { double k = 0, cf = 10, cop = 1, sn = 6; sscanf(v, "%lf,%lf,%lf,%lf", &k, &cf, &cop, &sn); g_bk = k > 0 ? k : 0; g_bcf = cf * 1e-3; g_bcop = cop * 1e-3; g_bs = sn * 1e-9; }
    else g_bk = ncpu() == 1 ? 1.0 : 0;
}
double g_b0 = -1, g_bf = 0;
void budget_init() {
    if (g_b0 >= 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); const int n = sched_getaffinity(0, sizeof cs, &cs) == 0 ? CPU_COUNT(&cs) : 0;
    const char* v = getenv("PTRECON_CJIT_BUDGET");
    if (v) { double b = 0, f = 0; if (sscanf(v, "%lf,%lf", &b, &f) < 1 || b <= 0) { g_b0 = 0; g_bf = 0; } else { g_b0 = b; g_bf = f; } }
    else if (n == 1) { benefit_init(); g_b0 = 2.0; g_bf = g_bk > 0 ? 0.10 : 0.02; }
    else g_b0 = 0;
}
bool benefit_on() { benefit_init(); return g_bk > 0; }
bool over_budget() { budget_init(); if (g_b0 <= 0) return false; const double el = now() - g_t0; return st.cc_s > std::max(g_b0, g_bf * el); }
struct Init { Init() { g_pid = (int)getpid(); g_t0 = now(); pthread_atfork(nullptr, nullptr, at_child); } } init_;

bool compile(std::vector<Job>& jobs) {
    const double t0 = now();
    char base[400]; snprintf(base, sizeof base, "%s/ptrecon-cjit.%d.%d", cache_on() ? getenv("PTRECON_CJIT_CACHE") : getenv("PTRECON_CJIT_DIR") ? getenv("PTRECON_CJIT_DIR") : "/dev/shm", g_pid, batch_no++);
    std::string cfile = std::string(base) + ".c", sofile = std::string(base) + ".so";
    FILE* f = fopen(cfile.c_str(), "w");
    if (!f) return false;
    fputs(prelude(), f);
    for (auto& j : jobs) fputs(j.src.c_str(), f);
    fclose(f);
    const char* cc = getenv("PTRECON_CJIT_CC") ? getenv("PTRECON_CJIT_CC") : "gcc";
    // -O2 for a small batch (a few hot loops: gemm 2 G prefix -2.6 %), -O1 when the queue is
    // long (py: -O2 halves the programs compiled in time, +5 %).  PTRECON_CJIT_OPT forces one level.
    static const char* opt_env = getenv("PTRECON_CJIT_OPT");
    const char* opt = opt_env ? opt_env : (jobs.size() <= 4 ? "-O2" : "-O1");
    const char* argv[] = { cc, opt, "-fPIC", "-shared", "-w", "-fno-strict-aliasing", "-o", sofile.c_str(), cfile.c_str(), nullptr };
    pid_t p; bool ok = false;
    pthread_mutex_lock(&mu);
    const uint64_t my_gen = gen;
    snprintf(cur_c, sizeof cur_c, "%s", cfile.c_str()); snprintf(cur_so, sizeof cur_so, "%s", sofile.c_str());
    int sp = posix_spawnp(&p, cc, nullptr, nullptr, (char* const*)argv, environ);
    if (sp == 0) cc_pid = p;
    pthread_mutex_unlock(&mu);
    if (sp == 0) {
        int status = 0; while (waitpid(p, &status, 0) < 0) {}
        pthread_mutex_lock(&mu); cc_pid = 0; pthread_mutex_unlock(&mu);
        ok = WIFEXITED(status) && WEXITSTATUS(status) == 0; }
    void* h = ok ? dlopen(sofile.c_str(), RTLD_NOW | RTLD_LOCAL) : nullptr;
    bool shared = false;
    if (h && cache_on()) for (auto& j : jobs) if (j.key) {   // publish: f_<key>.so -> this batch's .so (kept)
        char lp[400]; snprintf(lp, sizeof lp, "%s/f_%016lx.so", cache_on() ? getenv("PTRECON_CJIT_CACHE") : "", (unsigned long)j.key);
        if (symlink(sofile.c_str(), lp) == 0 || errno == EEXIST) shared = true; }
    if (!h && cache_on()) for (auto& j : jobs) if (j.key) {   // a failed claim must not leave the others waiting
        char cp[400]; snprintf(cp, sizeof cp, "%s/c_%016lx", getenv("PTRECON_CJIT_CACHE"), (unsigned long)j.key); unlink(cp); }
    if (!getenv("PTRECON_CJIT_KEEP")) { unlink(cfile.c_str()); if (!shared) unlink(sofile.c_str()); }
    pthread_mutex_lock(&mu);
    if (gen == my_gen)    // not abandoned by quiesce(): the FProgs are still alive (publishing under the lock)
        for (auto& j : jobs) {
            void* fn = h ? dlsym(h, j.name.c_str()) : nullptr;
            if (fn) { j.slot->store(fn, std::memory_order_release); j.state->store(2); }
            else j.state->store(3);
        }
    st.batches++; st.funcs += h ? jobs.size() : 0; st.failed += h ? 0 : jobs.size(); st.cc_s += now() - t0;
    pthread_mutex_unlock(&mu);
    if (getenv("PTRECON_CJIT_LOG")) fprintf(stderr, "cjit: batch %d: %zu functions, %s, %.3f s\n", batch_no - 1, jobs.size(), h ? "ok" : "FAILED", now() - t0);
    if (!h && gen == my_gen) { static bool said = false; if (!said) { said = true; fprintf(stderr, "cjit: compiling %zu functions failed (%s); interpreting\n", jobs.size(), cfile.c_str()); } }
    return h != nullptr;
}
void* loop(void*) {
    pthread_mutex_lock(&mu);
    for (;;) {
        while (!q || q->empty()) pthread_cond_wait(&cv, &mu);
        // gather: up to 10 ms more for further hot programs, at most 1024 per batch
        // with the benefit rule (one cpu) every gcc start-up is replay time lost, so gather for up to
        // 250 ms (or 64 programs) instead of 10 ms: fewer, larger batches
        const bool big = benefit_on();
        struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_nsec += (big ? 250 : 10) * 1000000L; while (dl.tv_nsec >= 1000000000) { dl.tv_sec++; dl.tv_nsec -= 1000000000; }
        while (q->size() < (big ? 64u : 256u) && pthread_cond_timedwait(&cv, &mu, &dl) == 0) {}
        std::vector<Job> jobs;
        if (q->size() <= 256) jobs.swap(*q);
        else { jobs.assign(q->begin(), q->begin() + 256); q->erase(q->begin(), q->begin() + 256); }
        if (jobs.empty()) continue;
        if (over_budget()) { for (auto& j : jobs) j.state->store(3); st.dropped += jobs.size(); continue; }   // one-cpu budget
        inflight = 1;
        inflight_keys.clear(); if (cache_on()) for (auto& j : jobs) if (j.key) inflight_keys.push_back(j.key);
        pthread_mutex_unlock(&mu);
        compile(jobs);
        pthread_mutex_lock(&mu);
        inflight = 0; inflight_keys.clear(); pthread_cond_broadcast(&cv_done);
    }
    return nullptr;
}
}  // namespace

static int g_en = -1;
bool enabled() { if (g_en < 0) { const char* v = getenv("PTRECON_CJIT"); g_en = (v && *v == '0') ? 0 : 1; } return g_en == 1; }
void back_to_default() { g_en = -1; }
// opt-in (PTRECON_CJIT_SHARE=1): py_nbody --jobs 40 x2 on 40 cpus, 3 runs each: off 5.05-5.16 s, shared (hot 4096) 5.49-5.67 s,
// hot 16384/65536 5.28-5.31 s -- the compilers take cpus every chunk needs, and a ~2 s chunk gains little from code that arrives late
static bool share_ok() { const char* v = getenv("PTRECON_CJIT_SHARE"); return v && *v == '1'; }
void off_for_chunks(int nchunks) { const char* v = getenv("PTRECON_CJIT"); const char* m = getenv("PTRECON_CJIT_MAXCHUNKS");
    if (!(v && *v == '1') && nchunks > (m ? atoi(m) : 8) && !(share_ok() && cache_on())) g_en = 0; }
bool g_cache = getenv("PTRECON_CJIT_CACHE") && *getenv("PTRECON_CJIT_CACHE");
bool cache_on() { const char* d = getenv("PTRECON_CJIT_CACHE"); return d && *d; }
void* cache_lookup(uint64_t key) {
    char lp[400]; snprintf(lp, sizeof lp, "%s/f_%016lx.so", getenv("PTRECON_CJIT_CACHE"), (unsigned long)key);
    void* h = dlopen(lp, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);   // the same batch .so: already mapped, cheap
    if (!h) { if (access(lp, R_OK)) return nullptr; h = dlopen(lp, RTLD_NOW | RTLD_LOCAL); }
    if (!h) return nullptr;
    char nm[40]; snprintf(nm, sizeof nm, "cj_%016lx", (unsigned long)key);
    return dlsym(h, nm); }
bool cache_claim(uint64_t key) {
    char cp[400]; snprintf(cp, sizeof cp, "%s/c_%016lx", getenv("PTRECON_CJIT_CACHE"), (unsigned long)key);
    int fd = open(cp, O_CREAT | O_EXCL | O_WRONLY, 0600); if (fd < 0) return false; close(fd); return true; }
void cache_create_for_run() {
    if (!share_ok() || !enabled()) return;
    char d[300]; snprintf(d, sizeof d, "%s/ptrecon-cjc.%d", getenv("PTRECON_CJIT_DIR") ? getenv("PTRECON_CJIT_DIR") : "/dev/shm", (int)getpid());
    if (mkdir(d, 0700) == 0) { setenv("PTRECON_CJIT_CACHE", d, 1); g_cache = true; } }
void cache_remove() {
    const char* d = getenv("PTRECON_CJIT_CACHE"); if (!d || !*d) return;
    if (!getenv("PTRECON_CJIT_KEEP")) {
        if (DIR* dp = opendir(d)) { while (struct dirent* e = readdir(dp)) { if (e->d_name[0] == '.') continue; std::string p = std::string(d) + "/" + e->d_name; unlink(p.c_str()); }
                                    closedir(dp); rmdir(d); } }
    unsetenv("PTRECON_CJIT_CACHE"); g_cache = false; }
uint32_t benefit_runs(size_t nops) {   // 0: the benefit rule is off
    benefit_init(); if (g_bk <= 0) return 0;
    const double n = g_bk * (g_bcf + g_bcop * (double)nops) / ((double)(nops ? nops : 1) * g_bs);
    return n >= 4e9 ? 4000000000u : (uint32_t)n; }
uint32_t hot() { static uint32_t h = 0; if (!h) { const char* v = getenv("PTRECON_CJIT_HOT"); h = v ? (uint32_t)strtoul(v, nullptr, 0) : 4096; if (!h) h = 1; } return h; }
int pid() { return g_pid; }
Stats stats() { pthread_mutex_lock(&mu); Stats s = st; pthread_mutex_unlock(&mu); return s; }

void submit(const std::string& name, const std::string& src, std::atomic<void*>* slot, std::atomic<int>* state, uint64_t key) {
    pthread_mutex_lock(&mu);
    if (!q) q = new std::vector<Job>();
    if (!running) { pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&t, &a, loop, nullptr) == 0) running = true; pthread_attr_destroy(&a); }
    q->push_back({name, src, slot, state, key});
    pthread_cond_signal(&cv);
    pthread_mutex_unlock(&mu);
}
void quiesce() {
    pthread_mutex_lock(&mu);
    if (q) { for (auto& j : *q) { j.state->store(3); if (j.key && cache_on()) { char cp[400]; snprintf(cp, sizeof cp, "%s/c_%016lx", getenv("PTRECON_CJIT_CACHE"), (unsigned long)j.key); unlink(cp); } } q->clear(); }
    for (uint64_t k : inflight_keys) { char cp[400]; snprintf(cp, sizeof cp, "%s/c_%016lx", getenv("PTRECON_CJIT_CACHE"), (unsigned long)k); unlink(cp); }
    gen++;                             // the batch in flight (if any) will not publish
    if (cc_pid > 0) { kill(cc_pid, SIGKILL); if (!getenv("PTRECON_CJIT_KEEP")) { unlink(cur_c); unlink(cur_so); } }   // the process may exit before the compile thread cleans up
    pthread_mutex_unlock(&mu);
}

const char* prelude() {
    return R"(#include <stdint.h>
#include <string.h>
#include <emmintrin.h>
typedef struct { uint64_t w[4]; uint32_t kn; uint8_t sz; } Val;
typedef struct { uint64_t addr; _Bool known; int size; int op; } MA;
typedef struct {
  void (*get)(void*, int, int, Val*);
  void (*unop)(void*, uint32_t, const Val*, Val*);
  void (*binop)(void*, uint32_t, const Val*, const Val*, Val*);
  void (*ite_u)(const Val*, const Val*, Val*);
  void (*ccall)(void*, const void*, const Val* const*, Val*);
  void (*sload)(void*, uint64_t, int, Val*);
  void (*sstore)(void*, uint64_t, const Val*, int);
  void (*sforget)(void*, uint64_t, int);
  void (*put)(void*, const void*, const Val*);
  void (*puti)(void*, int, int);
  void (*storeg)(void*, const void*, const Val*, const Val*, const Val*, MA**);
  void (*loadg)(void*, const void*, const Val*, const Val*, const Val*, Val*, MA**);
  void (*cas)(void*, const void*, const Val*, Val*, Val*, MA**);
  void (*llsc)(void*, const void*, const Val*, Val*, MA**);
  void (*dirty)(void*, const void*, const Val*, Val*, MA**);
} H;
static inline uint32_t vmask(int sz) { return sz >= 32 ? 0xffffffffu : ((1u << sz) - 1); }
static inline int vknown(const Val* v) { return (v->kn & vmask(v->sz)) == vmask(v->sz); }
static inline uint64_t vu64(const Val* v) { return v->sz >= 8 ? v->w[0] : (v->w[0] & ((1ull << (8 * v->sz)) - 1)); }
static inline Val vU(int sz) { Val r = {{0,0,0,0},0,(uint8_t)sz}; return r; }
static inline uint32_t knm8(const uint8_t* k) { __m128i x = _mm_loadl_epi64((const __m128i*)k); return (uint32_t)_mm_movemask_epi8(_mm_cmpgt_epi8(x, _mm_setzero_si128())) & 0xffu; }
static inline uint64_t knx8(uint32_t m) { uint64_t x = m & 0xffu; x = (x | (x << 28)) & 0x0000000F0000000Full; x = (x | (x << 14)) & 0x0003000300030003ull; x = (x | (x << 7)) & 0x0101010101010101ull; return x; }
typedef struct { uint64_t rh, ts0, tsl, nu; uint64_t* memo; char* out; uint32_t nnl, tid; int32_t hash, pad; uint64_t* ring; uint64_t rc0; } EmCtx;
typedef struct { uint64_t addr, ip, ts; uint32_t tid; uint8_t op, size; uint16_t flags; } REC;
#define RHM ((1ull << 61) - 1)
#define RHP (0x1f3a5c7e9b2d4861ull % RHM)
static inline uint64_t rmix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
static inline uint64_t rmul(uint64_t a, uint64_t b) { unsigned __int128 z = (unsigned __int128)a * b; uint64_t lo = (uint64_t)(z & RHM), hi = (uint64_t)(z >> 61); uint64_t r = lo + hi; return r >= RHM ? r - RHM : r; }
static inline uint64_t radd(uint64_t a, uint64_t b) { uint64_t r = a + b; return r >= RHM ? r - RHM : r; }
/* exact transcriptions of vexinterp.cpp's Val helpers and binop() cases */
static inline Val vK(uint64_t v, int sz) { Val r = {{0,0,0,0},0,(uint8_t)sz}; r.w[0] = sz >= 8 ? v : (v & ((1ull << (8 * sz)) - 1)); r.kn = vmask(sz); return r; }
static inline uint8_t vbyte(const Val* v, int i) { return (uint8_t)(v->w[i / 8] >> (8 * (i % 8))); }
static inline void vset(Val* r, int i, uint8_t b, int k) { r->w[i / 8] = (r->w[i / 8] & ~((uint64_t)0xff << (8 * (i % 8)))) | ((uint64_t)b << (8 * (i % 8))); if (k) r->kn |= 1u << i; else r->kn &= ~(1u << i); }
static inline Val vwiden(const Val* a, int to, int sign) { Val r = vU(to); if (!vknown(a)) return r; uint64_t v = vu64(a); int from = a->sz;
    if (sign && from < 8 && (v >> (8 * from - 1)) & 1) v |= ~((1ull << (8 * from)) - 1); return vK(v, to); }
static inline Val vnarrow(const Val* a, int to) { Val r = {{0,0,0,0},0,0}; r.sz = (uint8_t)to; r.kn = a->kn & vmask(to); r.w[0] = a->w[0]; r.w[1] = a->w[1]; r.w[2] = a->w[2]; r.w[3] = a->w[3]; if (to < 8) r.w[0] &= (1ull << (8 * to)) - 1; return r; }
static inline int64_t vsext(uint64_t v, int sz) { if (sz >= 8) return (int64_t)v; int sh = 64 - 8 * sz; return ((int64_t)(v << sh)) >> sh; }
#define BOTH (vknown(a) && vknown(c))
static inline Val b_add(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vu64(a) + vu64(c), sz) : vU(sz); }
static inline Val b_sub(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vu64(a) - vu64(c), sz) : vU(sz); }
static inline Val b_mul(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vu64(a) * vu64(c), sz) : vU(sz); }
static inline Val b_and(const Val* a, const Val* c) { int sz = a->sz; Val r = vU(sz); for (int i = 0; i < sz; i++) { int ka = (a->kn >> i) & 1, kc = (c->kn >> i) & 1; uint8_t x = vbyte(a, i) & vbyte(c, i); int k = (ka && kc) || (ka && !vbyte(a, i)) || (kc && !vbyte(c, i)); vset(&r, i, k ? x : 0, k); } return r; }
static inline Val b_or(const Val* a, const Val* c) { int sz = a->sz; Val r = vU(sz); for (int i = 0; i < sz; i++) { int ka = (a->kn >> i) & 1, kc = (c->kn >> i) & 1; uint8_t x = vbyte(a, i) | vbyte(c, i); int k = (ka && kc) || (ka && vbyte(a, i) == 0xff) || (kc && vbyte(c, i) == 0xff); vset(&r, i, k ? x : 0, k); } return r; }
static inline Val b_xor(const Val* a, const Val* c) { int sz = a->sz; Val r = vU(sz); for (int i = 0; i < sz; i++) { int k = ((a->kn >> i) & 1) && ((c->kn >> i) & 1); vset(&r, i, k ? (vbyte(a, i) ^ vbyte(c, i)) : 0, k); } return r; }
static inline Val b_shl(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vu64(c) >= 64 ? 0 : vu64(a) << (vu64(c) & 63), sz) : vU(sz); }
static inline Val b_shr(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vu64(c) >= 64 ? 0 : vu64(a) >> (vu64(c) & 63), sz) : vU(sz); }
static inline Val b_sar(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK((uint64_t)(vsext(vu64(a), sz) >> (vu64(c) > 63 ? 63 : vu64(c))), sz) : vU(sz); }
static inline Val b_eq(const Val* a, const Val* c) { return BOTH ? vK(vu64(a) == vu64(c), 1) : vU(1); }
static inline Val b_ne(const Val* a, const Val* c) { return BOTH ? vK(vu64(a) != vu64(c), 1) : vU(1); }
static inline Val b_lts(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vsext(vu64(a), sz) < vsext(vu64(c), sz), 1) : vU(1); }
static inline Val b_les(const Val* a, const Val* c) { int sz = a->sz; return BOTH ? vK(vsext(vu64(a), sz) <= vsext(vu64(c), sz), 1) : vU(1); }
static inline Val b_ltu(const Val* a, const Val* c) { return BOTH ? vK(vu64(a) < vu64(c), 1) : vU(1); }
static inline Val b_leu(const Val* a, const Val* c) { return BOTH ? vK(vu64(a) <= vu64(c), 1) : vU(1); }
#undef BOTH
#define EMIT(A, SZ, OP) do { const int k_ = vknown(A); wp->addr = k_ ? vu64(A) : 0; wp->known = k_; wp->size = (SZ); wp->op = (OP); ++wp; } while (0)
)";
}
}  // namespace cjit
