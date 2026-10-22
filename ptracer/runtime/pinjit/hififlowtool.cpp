/*
 * hififlowtool.cpp -- the PTracer HiFi-mode Pintool (Pin JIT instrumentation), compiled into
 * hififlow.cpp for the HiFi offline-reconstruction experiment (run/05_hifi_recon.sh).
 *
 * WHAT THIS IS.  The user's definition (): HiFi is Pin's JIT instrumentation --
 * simple, highest fidelity, no dangerous tricks -- and that is what the paper's HiFi is
 * (design.tex: "HiFi mode relies on Intel Pin's Just-in-Time (JIT) option that recompiles
 * the binaries ... The logging stores the value to a circular buffer in memory that is
 * periodically flushed to disks").  Everything else in this tree (E9Patch static rewriting,
 * naked trampolines) is the FAST substrate.
 *
 * WHAT IT LOGS.  Exactly the critical values a PTracer v2 spec names, at the addresses it
 * names, in the program order docs/SPEC_FORMAT.md section 1 mandates.  The spec is
 * pre-flattened by runtime/pinjit/mkplan.py into a PLAN (see that file for the grammar),
 * because Pin's CRT has no JSON parser.
 *
 * HOW IT LOGS.  Through Pin's buffering API: PIN_DefineTraceBuffer with a record size of
 * EIGHT BYTES and one INS_InsertFillBuffer per logged value.  A fill buffer is filled by
 * inlined code in the code cache, not by a callout, so the common case costs one store and
 * a pointer bump.  Because a record IS one little-endian 64-bit value, the buffer's raw
 * bytes are already the body of the v2 critical-value file:
 *
 *     cv.<pid>.<tid>.bin  =  struct { char magic[4]="PTCV"; uint32 version=2;
 *                                     uint32 tid; uint32 sync; }
 *                            followed by raw little-endian uint64 values in program order.
 *
 * so offline/ptrecon consumes it with no change (SPEC_FORMAT section 3, "Buffer option").
 *
 * LIMIT, stated here so it is not discovered downstream: `sync` is 0.  The sync marker is a
 * PTWRITE of the exact running value count. This tool does not implement it yet.
 * Pin 3.20 DOES expose PIN_GetBufferPointer(CONTEXT*, BUFFER_ID), so the earlier claim
 * that an analysis routine cannot read the fill position was incorrect. A sparse-marker
 * design must validate callback ordering, completed buffers and thread state; it need not
 * be assumed to require a callout at every value. Code-cache PT reconstruction is ALSO
 * missing: the compatible CV format alone is not an end-to-end trace. See pin_jit.md.
 *
 * Knobs:
 *   -plan  FILE     the PLAN from mkplan.py (required)
 *   -cvdir DIR      where cv.<pid>.<tid>.bin go; "/dev/null" discards (timing runs)
 *   -pages N        trace-buffer size in 4 KB pages per thread (default 8192 = 32 MB,
 *                   the same ring size runtime/rt/ptlogrt.c uses)
 *   -stats 1        print per-thread value counts to stderr at exit
 *   -memread load   m: items: inlined exact-width load where provably safe (default) |
 *                   safecopy (PIN_SafeCopy callout everywhere, the old path; see
 *                  
 */
#include "pin.H"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

/* ------------------------------------------------------------------ knobs */
KNOB<std::string> KnobPlan(KNOB_MODE_WRITEONCE, "pintool", "plan", "",
                           "PLAN file from runtime/pinjit/mkplan.py");
KNOB<std::string> KnobCvDir(KNOB_MODE_WRITEONCE, "pintool", "cvdir", ".",
                            "directory for cv.<pid>.<tid>.bin ('/dev/null' discards)");
KNOB<UINT32> KnobPages(KNOB_MODE_WRITEONCE, "pintool", "pages", "8192",
                       "trace-buffer pages per thread (4 KB each)");
KNOB<BOOL> KnobStats(KNOB_MODE_WRITEONCE, "pintool", "stats", "0", "print stats at exit");
KNOB<BOOL> KnobRepRcx(KNOB_MODE_WRITEONCE, "pintool", "reprcx", "1",
                      "log %rcx before every REP string instruction whose site does not already log it");
/* Synchronization events: RTN instrumentation of the pthread / semaphore /
 * process functions; one record { w0, w1, tsc } per event (w0/w1 = the Fast format of runtime/rt/ptsync.S)
 * into sync.<pid>.<tid>.bin -- a SEPARATE stream, so the cv value stream is unchanged. */
KNOB<BOOL> KnobSyncEv(KNOB_MODE_WRITEONCE, "pintool", "syncev", "0", "log synchronization events");
KNOB<std::string> KnobSyncDir(KNOB_MODE_WRITEONCE, "pintool", "syncdir", "",
                              "directory of sync.<pid>.<tid>.bin (default: -cvdir)");
/* The paper's HiFi-PTWRITE arm (evaluation.tex): "Enabling PTWRITE instructions achieves
 * better performance in Fast mode ... But it often causes worse performance in HiFi mode,
 * because Intel Pin's JIT cannot inline PTWRITEs."  -sink ptwrite reproduces exactly that:
 * every logged value becomes an ANALYSIS CALL that executes one `ptwrite %rdi', instead of
 * the inlined INS_InsertFillBuffer store.  Pin cannot inline an analysis routine that
 * contains PTWRITE, so this is a callout per value by construction -- which is the paper's
 * stated reason for the arm being slower, not an artefact of this implementation.
 * Default "buffer" leaves every other row byte-for-byte unchanged. */
KNOB<std::string> KnobSink(KNOB_MODE_WRITEONCE, "pintool", "sink", "buffer",
                           "value sink: buffer (inlined fill buffer) | ptwrite (callout per value)");
/* How an m:<bytes> item reads the instruction's memory operand.
 * "load" (default): an INLINED plain load of exactly <bytes> at IARG_MEMORYREAD_EA, used only when the
 * instruction itself unconditionally reads at least <bytes> there (standard memop, no REP, not
 * predicated), so the load can fault only where the application instruction would fault next.
 * Every other m: site keeps the PIN_SafeCopy callout.  "safecopy": the pre- behaviour, a
 * PIN_SafeCopy callout at every m: site.  On Sapphire Rapids a PIN_SafeCopy callout from
 * instrumented code left the rest of the run paying ASSISTS.SSE_AVX_MIX on every SSE instruction
 * (syr2k_large 2.4 s -> 112 s from 12 start-up calls in ld.so); the logged value is identical. */
KNOB<std::string> KnobMemRead(KNOB_MODE_WRITEONCE, "pintool", "memread", "load",
                              "m: item read: load (inlined, safe cases) | safecopy (PIN_SafeCopy callout)");

/* ------------------------------------------------------------------- plan */
enum ItemKind { IT_REG, IT_XMM_LO, IT_XMM_HI, IT_MEM, IT_FSBASE, IT_SHMEM };   /* IT_SHMEM: -logshared value */

struct Item {
    ItemKind kind;
    REG reg;        /* IT_REG / IT_XMM_* */
    UINT32 size;    /* IT_MEM */
};

struct Site {
    std::vector<Item> before;
    std::vector<Item> after;
};

/* plan as read: image path -> (link-time address -> Site) */
static std::map<std::string, std::map<ADDRINT, Site> > g_plan;
/* resolved at image load: runtime address -> Site* */
static std::map<ADDRINT, const Site *> g_sites;
static UINT64 g_resolved = 0, g_images = 0;

static REG reg_by_name(const std::string &n)
{
    static const char *gp[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                 "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
    // Pin 4's POSIX headers also define global REG_R* ucontext enumerators.
    static const REG gr[16] = {LEVEL_BASE::REG_RAX, LEVEL_BASE::REG_RCX,
        LEVEL_BASE::REG_RDX, LEVEL_BASE::REG_RBX, LEVEL_BASE::REG_RSP,
        LEVEL_BASE::REG_RBP, LEVEL_BASE::REG_RSI, LEVEL_BASE::REG_RDI,
        LEVEL_BASE::REG_R8, LEVEL_BASE::REG_R9, LEVEL_BASE::REG_R10,
        LEVEL_BASE::REG_R11, LEVEL_BASE::REG_R12, LEVEL_BASE::REG_R13,
        LEVEL_BASE::REG_R14, LEVEL_BASE::REG_R15};
    for (int i = 0; i < 16; i++)
        if (n == gp[i])
            return gr[i];
    if (n.size() > 3 && n.compare(0, 3, "xmm") == 0) {
        int k = atoi(n.c_str() + 3);
        if (k >= 0 && k < 16)
            return (REG)(REG_XMM0 + k);
    }
    return REG_INVALID();
}

static bool parse_item(const std::string &s, Item *out)
{
    if (s == "f") { out->kind = IT_FSBASE; out->reg = REG_INVALID(); out->size = 8; return true; }
    if (s.compare(0, 2, "r:") == 0) {
        out->kind = IT_REG; out->size = 8; out->reg = reg_by_name(s.substr(2));
        return out->reg != REG_INVALID();
    }
    if (s.compare(0, 2, "m:") == 0) {
        out->kind = IT_MEM; out->reg = REG_INVALID();
        out->size = (UINT32)atoi(s.c_str() + 2);
        return out->size >= 1 && out->size <= 8;
    }
    if (s.compare(0, 2, "x:") == 0) {
        size_t c = s.rfind(':');
        if (c == std::string::npos || c < 2) return false;
        out->reg = reg_by_name(s.substr(2, c - 2));
        out->kind = (s.compare(c + 1, std::string::npos, "hi") == 0) ? IT_XMM_HI : IT_XMM_LO;
        out->size = 8;
        return out->reg != REG_INVALID();
    }
    return false;
}

static void load_plan(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "[hifitool] cannot open plan %s\n", path.c_str());
        PIN_ExitProcess(2);
    }
    char line[65536];
    std::string img;
    UINT64 nsite = 0, nval = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'I' && line[1] == ' ') {
            char *p = line + 2, *e = p + strlen(p);
            while (e > p && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ')) --e;
            img.assign(p, e - p);
        } else if (line[0] == 'S' && line[1] == ' ') {
            char *save = NULL;
            strtok_r(line, " \t\n", &save);                 /* "S" */
            char *a = strtok_r(NULL, " \t\n", &save);
            char *w = strtok_r(NULL, " \t\n", &save);
            char *n = strtok_r(NULL, " \t\n", &save);
            if (!a || !w || !n) continue;
            ADDRINT addr = (ADDRINT)strtoull(a, NULL, 16);
            int cnt = atoi(n);
            Site &st = g_plan[img][addr];
            std::vector<Item> &v = (w[0] == 'A') ? st.after : st.before;
            for (int i = 0; i < cnt; i++) {
                char *t = strtok_r(NULL, " \t\n", &save);
                Item it;
                if (!t || !parse_item(std::string(t), &it)) {
                    fprintf(stderr, "[hifitool] bad plan item near %s\n", a);
                    PIN_ExitProcess(2);
                }
                v.push_back(it);
                nval++;
            }
            nsite++;
        }
    }
    fclose(f);
    fprintf(stderr, "[hifitool] plan: %lu images, %lu addresses, %lu values/execution\n",
            (unsigned long)g_plan.size(), (unsigned long)nsite, (unsigned long)nval);
}

/* --------------------------------------------------------- per-thread sink */
struct CvHdr { char magic[4]; UINT32 version; UINT32 tid; UINT32 sync; };

struct TLS {
    int fd;
    UINT64 nvalues;
    OS_THREAD_ID ostid;
    int sfd;                                    /* sync.<pid>.<tid>.bin (-syncev 1) */
    std::vector<UINT64> sbuf;                   /* pending {w0, w1, tsc} triples */
    struct SFrame { ADDRINT sp, ret, a0, a1, a2, aux; };
    std::vector<SFrame> sframes;                /* entry state of the instrumented calls in flight */
};
static TLS_KEY g_tls;
static std::vector<TLS *> g_all;
static PIN_LOCK g_lock;
static BUFFER_ID g_buf;
static bool g_discard = false;

/* ------------------------------------------------------------- sync events */
enum { SY_CREATE = 1, SY_START, SY_EXIT, SY_JOIN, SY_FORK, SY_FORK_CHILD, SY_VFORK, SY_EXEC, SY_WAIT, SY_LOCK,
       SY_TRYLOCK, SY_UNLOCK, SY_TIMEDLOCK, SY_CWAIT_BEGIN, SY_CWAIT_END, SY_CSIGNAL, SY_CBROADCAST, SY_RDLOCK,
       SY_WRLOCK, SY_TRYRDLOCK, SY_TRYWRLOCK, SY_RWUNLOCK, SY_SEMWAIT, SY_SEMTRYWAIT, SY_SEMTIMEDWAIT, SY_SEMPOST,
       SY_SPAWN };
/* how: B = before (obj = a0, or 0 with OBJ0), A = after; obj: a0 | a0p64 | a0p32 | ret | waitid; res: rc | neg */
enum SyHow { SB, SA, SCOND, SFORK };
enum SyObj { O_A0, O_ZERO, O_A0P64, O_A0P32, O_RET, O_WAITID };
enum SyRes { R_RC, R_NEG };
struct SyFn { const char *name; int op; SyHow how; SyObj obj; SyRes res; };
static const SyFn g_syfns[] = {
    {"pthread_create", SY_CREATE, SA, O_A0P64, R_RC}, {"pthread_join", SY_JOIN, SA, O_A0, R_RC},
    {"pthread_tryjoin_np", SY_JOIN, SA, O_A0, R_RC}, {"pthread_timedjoin_np", SY_JOIN, SA, O_A0, R_RC},
    {"fork", SY_FORK, SFORK, O_RET, R_NEG}, {"vfork", SY_VFORK, SB, O_ZERO, R_RC},
    {"posix_spawn", SY_SPAWN, SA, O_A0P32, R_RC}, {"posix_spawnp", SY_SPAWN, SA, O_A0P32, R_RC},
    {"execve", SY_EXEC, SB, O_ZERO, R_RC}, {"execv", SY_EXEC, SB, O_ZERO, R_RC}, {"execvp", SY_EXEC, SB, O_ZERO, R_RC},
    {"execvpe", SY_EXEC, SB, O_ZERO, R_RC}, {"fexecve", SY_EXEC, SB, O_ZERO, R_RC},
    {"wait", SY_WAIT, SA, O_RET, R_NEG}, {"waitpid", SY_WAIT, SA, O_RET, R_NEG}, {"wait4", SY_WAIT, SA, O_RET, R_NEG},
    {"waitid", SY_WAIT, SA, O_WAITID, R_NEG},
    {"pthread_mutex_lock", SY_LOCK, SA, O_A0, R_RC}, {"pthread_mutex_trylock", SY_TRYLOCK, SA, O_A0, R_RC},
    {"pthread_mutex_timedlock", SY_TIMEDLOCK, SA, O_A0, R_RC}, {"pthread_mutex_clocklock", SY_TIMEDLOCK, SA, O_A0, R_RC},
    {"pthread_mutex_unlock", SY_UNLOCK, SB, O_A0, R_RC},
    {"pthread_cond_wait", SY_CWAIT_END, SCOND, O_A0, R_RC}, {"pthread_cond_timedwait", SY_CWAIT_END, SCOND, O_A0, R_RC},
    {"pthread_cond_clockwait", SY_CWAIT_END, SCOND, O_A0, R_RC},
    {"pthread_cond_signal", SY_CSIGNAL, SB, O_A0, R_RC}, {"pthread_cond_broadcast", SY_CBROADCAST, SB, O_A0, R_RC},
    {"pthread_rwlock_rdlock", SY_RDLOCK, SA, O_A0, R_RC}, {"pthread_rwlock_wrlock", SY_WRLOCK, SA, O_A0, R_RC},
    {"pthread_rwlock_tryrdlock", SY_TRYRDLOCK, SA, O_A0, R_RC}, {"pthread_rwlock_trywrlock", SY_TRYWRLOCK, SA, O_A0, R_RC},
    {"pthread_rwlock_timedrdlock", SY_RDLOCK, SA, O_A0, R_RC}, {"pthread_rwlock_timedwrlock", SY_WRLOCK, SA, O_A0, R_RC},
    {"pthread_rwlock_unlock", SY_RWUNLOCK, SB, O_A0, R_RC},
    {"sem_wait", SY_SEMWAIT, SA, O_A0, R_RC}, {"sem_trywait", SY_SEMTRYWAIT, SA, O_A0, R_RC},
    {"sem_timedwait", SY_SEMTIMEDWAIT, SA, O_A0, R_RC}, {"sem_clockwait", SY_SEMTIMEDWAIT, SA, O_A0, R_RC},
    {"sem_post", SY_SEMPOST, SB, O_A0, R_RC},
};
static bool g_syncev = false;
/* calls made from libc / ld.so themselves (ld.so's rtld lock is pthread_mutex_lock, dl_fini, ...) are not
 * logged: Fast interposes only calls that cross into libc through the PLT, and HiFi matches that */
static ADDRINT g_sy_ex[4][2]; static int g_sy_nex = 0;
static inline bool sy_internal(ADDRINT ret)
{
    for (int i = 0; i < g_sy_nex; i++) if (ret >= g_sy_ex[i][0] && ret < g_sy_ex[i][1]) return true;
    return false;
}
static std::string g_syncdir;
static UINT64 g_sync_n = 0;

static void sync_open(struct TLS *t);
static void sync_flush(struct TLS *t)
{
    if (t->sfd >= 0 && !t->sbuf.empty()) {
        size_t want = t->sbuf.size() * 8, off = 0;
        const char *p = (const char *)t->sbuf.data();
        while (off < want) { ssize_t w = write(t->sfd, p + off, want - off); if (w <= 0) break; off += (size_t)w; }
    }
    t->sbuf.clear();
}
static inline UINT64 sy_rdtsc() { UINT32 lo, hi; __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi)); return ((UINT64)hi << 32) | lo; }
static void sync_put(struct TLS *t, int op, UINT64 res, ADDRINT ret, UINT64 obj)
{
    if (!t) return;
    if (res > 255) res = 255;
    t->sbuf.push_back((1ULL << 63) | ((UINT64)op << 56) | (res << 48) | ((UINT64)ret & 0xffffffffffffULL));
    t->sbuf.push_back(obj);
    t->sbuf.push_back(sy_rdtsc());
    __atomic_add_fetch(&g_sync_n, 1, __ATOMIC_RELAXED);
    if (t->sbuf.size() >= 3 * 4096) sync_flush(t);
}
static VOID SyBefore(THREADID tid, ADDRINT fi, ADDRINT sp, ADDRINT a0, ADDRINT a1, ADDRINT a2)
{
    struct TLS *t = (struct TLS *)PIN_GetThreadData(g_tls, tid);
    if (!t) return;
    const SyFn &f = g_syfns[fi];
    ADDRINT ret = 0;
    PIN_SafeCopy(&ret, (const VOID *)sp, sizeof ret);
    if (sy_internal(ret)) return;
    if (f.how == SB) { sync_put(t, f.op, 0, ret, f.obj == O_ZERO ? 0 : a0); return; }
    if (f.how == SCOND) { sync_put(t, SY_UNLOCK, 0, ret, a1); sync_put(t, SY_CWAIT_BEGIN, 0, ret, a0); }
    TLS::SFrame fr; fr.sp = sp; fr.ret = ret; fr.a0 = a0; fr.a1 = a1; fr.a2 = a2; fr.aux = (ADDRINT)PIN_GetPid();
    t->sframes.push_back(fr);
}
static VOID SyAfter(THREADID tid, ADDRINT fi, ADDRINT sp, ADDRINT rv)
{
    struct TLS *t = (struct TLS *)PIN_GetThreadData(g_tls, tid);
    if (!t) return;
    /* the frame of THIS call: entered with the same %rsp (IPOINT_AFTER sits on the `ret');
     * frames of calls whose IPOINT_AFTER never ran (a longjmp, a tail call) are dropped */
    while (!t->sframes.empty() && t->sframes.back().sp < sp) t->sframes.pop_back();
    if (t->sframes.empty() || t->sframes.back().sp != sp) return;
    TLS::SFrame fr = t->sframes.back(); t->sframes.pop_back();
    const SyFn &f = g_syfns[fi];
    const int r32 = (int)(UINT32)rv;
    UINT64 obj = 0;
    switch (f.obj) {
    case O_A0: obj = fr.a0; break;
    case O_ZERO: break;
    case O_RET: obj = (UINT64)(INT64)r32; break;
    case O_A0P64: if (r32 == 0 && fr.a0) PIN_SafeCopy(&obj, (const VOID *)fr.a0, 8); break;
    case O_A0P32: if (r32 == 0 && fr.a0) { INT32 v = 0; PIN_SafeCopy(&v, (const VOID *)fr.a0, 4); obj = (UINT64)(INT64)v; } break;
    case O_WAITID: if (r32 == 0 && fr.a2) { INT32 v = 0; PIN_SafeCopy(&v, (const VOID *)(fr.a2 + 16), 4); obj = (UINT64)(INT64)v; } break;
    }
    const UINT64 res = f.res == R_NEG ? (r32 < 0 ? 255 : 0) : (UINT64)(UINT32)r32;
    if (f.how == SFORK && r32 == 0) { sync_put(t, SY_FORK_CHILD, 0, fr.ret, fr.aux); return; }
    sync_put(t, f.op, res, fr.ret, obj);
    if (f.how == SCOND) sync_put(t, SY_LOCK, 0, fr.ret, fr.a1);
}
static void sync_image(IMG img)
{
    if (!g_syncev) return;
    {   const std::string n = IMG_Name(img); const size_t sl = n.rfind('/');
        const std::string b = sl == std::string::npos ? n : n.substr(sl + 1);
        if ((b.compare(0, 7, "libc.so") == 0 || b.compare(0, 8, "ld-linux") == 0 || IMG_IsInterpreter(img)) && g_sy_nex < 4) {
            g_sy_ex[g_sy_nex][0] = IMG_LowAddress(img); g_sy_ex[g_sy_nex][1] = IMG_HighAddress(img) + 1; g_sy_nex++; } }
    /* RTN_FindByName (which resolves glibc's aliases), plus every OTHER routine of that name: glibc keeps
     * old symbol versions as separate functions (pthread_cond_wait@GLIBC_2.2.5 vs @@GLIBC_2.3.2) */
    std::vector<std::pair<RTN, size_t> > todo; std::set<ADDRINT> seen;
    const size_t NF = sizeof g_syfns / sizeof g_syfns[0];
    for (size_t i = 0; i < NF; i++) {
        RTN r = RTN_FindByName(img, g_syfns[i].name);
        if (RTN_Valid(r) && seen.insert(RTN_Address(r)).second) todo.push_back(std::make_pair(r, i));
    }
    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
    for (RTN r = SEC_RtnHead(sec); RTN_Valid(r); r = RTN_Next(r)) {
        std::string nm = RTN_Name(r);
        size_t at = nm.find('@'); if (at != std::string::npos) nm.resize(at);
        for (size_t i = 0; i < NF; i++)
            if (nm == g_syfns[i].name && seen.insert(RTN_Address(r)).second) { todo.push_back(std::make_pair(r, i)); break; }
    }
    for (size_t k = 0; k < todo.size(); k++) {
        RTN rtn = todo[k].first; const size_t i = todo[k].second;
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)SyBefore, IARG_THREAD_ID, IARG_ADDRINT, (ADDRINT)i,
                       IARG_REG_VALUE, REG_STACK_PTR, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 1, IARG_FUNCARG_ENTRYPOINT_VALUE, 2, IARG_END);
        if (g_syfns[i].how != SB)
            RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)SyAfter, IARG_THREAD_ID, IARG_ADDRINT, (ADDRINT)i,
                           IARG_REG_VALUE, REG_STACK_PTR, IARG_FUNCRET_EXITPOINT_VALUE, IARG_END);
        RTN_Close(rtn);
    }
}
static void sync_open(struct TLS *t)
{
    t->sfd = -1;
    if (!g_syncev) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/sync.%d.%d.bin", g_syncdir.c_str(), (int)PIN_GetPid(), (int)t->ostid);
    t->sfd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CvHdr h; memcpy(h.magic, "PTSY", 4); h.version = 1; h.tid = (UINT32)t->ostid; h.sync = 0;
    if (t->sfd >= 0 && write(t->sfd, &h, sizeof(h)) != (ssize_t)sizeof(h))
        fprintf(stderr, "[hifitool] short sync header write\n");
}
static VOID SyForkChild(THREADID tid, const CONTEXT *, VOID *)
{   /* the child's only thread: its own file (the parent's buffered records stay with the parent) */
    struct TLS *t = (struct TLS *)PIN_GetThreadData(g_tls, tid);
    if (!t) return;
    t->sbuf.clear(); t->ostid = PIN_GetTid();
    if (t->sfd >= 0) close(t->sfd);
    sync_open(t);
}

static VOID ThreadStart(THREADID tid, CONTEXT *ctxt, INT32, VOID *)
{
    TLS *t = new TLS;
    t->nvalues = 0;
    t->ostid = PIN_GetTid();
    char path[1024];
    if (g_discard)
        snprintf(path, sizeof(path), "/dev/null");
    else
        snprintf(path, sizeof(path), "%s/cv.%d.%d.bin", KnobCvDir.Value().c_str(),
                 (int)PIN_GetPid(), (int)t->ostid);
    t->fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (t->fd >= 0 && !g_discard) {
        CvHdr h;
        memcpy(h.magic, "PTCV", 4);
        h.version = 2;
        h.tid = (UINT32)t->ostid;
        h.sync = 0;                     /* see the header comment: no sync markers */
        if (write(t->fd, &h, sizeof(h)) != (ssize_t)sizeof(h))
            fprintf(stderr, "[hifitool] short header write\n");
    }
    sync_open(t);
    if (g_syncev && tid != 0)       /* START(pthread_self() = the thread's %fs base) */
        sync_put(t, SY_START, 0, 0, PIN_GetContextReg(ctxt, REG_SEG_FS_BASE));
    PIN_SetThreadData(g_tls, t, tid);
    PIN_GetLock(&g_lock, tid + 1);
    g_all.push_back(t);
    PIN_ReleaseLock(&g_lock);
}

static VOID ThreadFini(THREADID tid, const CONTEXT *ctxt, INT32, VOID *)
{
    TLS *t = (TLS *)PIN_GetThreadData(g_tls, tid);
    if (t && g_syncev) {
        if (tid != 0) sync_put(t, SY_EXIT, 0, 0, PIN_GetContextReg(ctxt, REG_SEG_FS_BASE));
        sync_flush(t);
        if (t->sfd >= 0) { close(t->sfd); t->sfd = -1; }
    }
    if (t && t->fd >= 0) {
        close(t->fd);
        t->fd = -1;
    }
}

/* Pin calls this when a thread's trace buffer is full and once when it exits. */
static VOID *BufferFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf,
                        UINT64 n, VOID *)
{
    TLS *t = (TLS *)PIN_GetThreadData(g_tls, tid);
    if (t) {
        t->nvalues += n;
        if (t->fd >= 0 && n) {
            size_t want = (size_t)n * 8, off = 0;
            const char *p = (const char *)buf;
            while (off < want) {
                ssize_t w = write(t->fd, p + off, want - off);
                if (w <= 0) break;
                off += (size_t)w;
            }
        }
    }
    return buf;
}

/* ------------------------------------------------- analysis for the odd cases */
/* A memop logs the VALUE at the instruction's effective address; a fill buffer cannot
 * dereference, so the value is produced into a tool scratch register first and the fill
 * buffer copies that register.  Both are inserted at IPOINT_BEFORE, and Pin runs analysis
 * inserted at the same point in insertion order, so program order is preserved. */
static REG g_scr;

static ADDRINT PIN_FAST_ANALYSIS_CALL ReadMem(ADDRINT ea, UINT32 sz)
{
    UINT64 v = 0;
    PIN_SafeCopy(&v, (const VOID *)ea, sz);
    return (ADDRINT)v;
}

/* Inlinable exact-width loads (straight-line, no call): Pin inlines them into the code cache, so
 * no bridge and no PIN_SafeCopy runs.  Zero-extension matches ReadMem's zero-initialised copy of
 * the low <sz> little-endian bytes. */
static ADDRINT PIN_FAST_ANALYSIS_CALL Load8(ADDRINT ea) { return *(const volatile UINT64 *)ea; }
static ADDRINT PIN_FAST_ANALYSIS_CALL Load4(ADDRINT ea) { return *(const volatile UINT32 *)ea; }
static ADDRINT PIN_FAST_ANALYSIS_CALL Load2(ADDRINT ea) { return *(const volatile UINT16 *)ea; }
static ADDRINT PIN_FAST_ANALYSIS_CALL Load1(ADDRINT ea) { return *(const volatile UINT8 *)ea; }
static bool g_memload = true;
static UINT64 g_mem_inline = 0, g_mem_safecopy = 0, g_mem_faults = 0;

/* A fault in inlined analysis code is a Pin INTERNAL exception, which by default kills the process
 * ("Tool (or Pin) caused signal 11") instead of reaching the application's own handler.  The inlined
 * load faults only where the instrumented instruction is about to fault on the same bytes (a JVM
 * implicit null check, a probing SIGSEGV handler, ...).  This handler gives such a fault exactly the
 * PIN_SafeCopy semantics: the logged value is 0, execution resumes after the load, and the
 * application instruction then raises the real, application-visible fault.  It only accepts a
 * faulting MOV/MOVZX whose destination is a GPR and whose source is memory, i.e. the Load* body;
 * any other internal fault is left to Pin (EHR_CONTINUE_SEARCH). */
extern "C" {
#include "xed-interface.h"
}
static EXCEPT_HANDLING_RESULT LoadFault(THREADID, EXCEPTION_INFO *ei, PHYSICAL_CONTEXT *pc, VOID *)
{
    if (PIN_GetExceptionClass(PIN_GetExceptionCode(ei)) != EXCEPTCLASS_ACCESS_FAULT)
        return EHR_CONTINUE_SEARCH;
    const ADDRINT ip = PIN_GetPhysicalContextReg(pc, REG_INST_PTR);
    UINT8 bytes[15];
    size_t got = PIN_SafeCopy(bytes, (const VOID *)ip, sizeof bytes);
    if (got == 0)
        return EHR_CONTINUE_SEARCH;
    xed_decoded_inst_t x;
    xed_decoded_inst_zero(&x);
    xed_decoded_inst_set_mode(&x, XED_MACHINE_MODE_LONG_64, XED_ADDRESS_WIDTH_64b);
    if (xed_decode(&x, bytes, (unsigned)got) != XED_ERROR_NONE)
        return EHR_CONTINUE_SEARCH;
    const xed_iclass_enum_t ic = xed_decoded_inst_get_iclass(&x);
    if ((ic != XED_ICLASS_MOV && ic != XED_ICLASS_MOVZX) ||
        xed_decoded_inst_number_of_memory_operands(&x) != 1 ||
        !xed_decoded_inst_mem_read(&x, 0) || xed_decoded_inst_mem_written(&x, 0))
        return EHR_CONTINUE_SEARCH;
    const REG dst = INS_XedExactMapToPinReg(xed_decoded_inst_get_reg(&x, XED_OPERAND_REG0));
    if (!REG_is_gr(REG_FullRegName(dst)))
        return EHR_CONTINUE_SEARCH;
    PIN_SetPhysicalContextReg(pc, REG_FullRegName(dst), 0);
    PIN_SetPhysicalContextReg(pc, REG_INST_PTR, ip + xed_decoded_inst_get_length(&x));
    __atomic_add_fetch(&g_mem_faults, 1, __ATOMIC_RELAXED);
    return EHR_HANDLED;
}

/* The inlined load is used only where it cannot fault unless the instrumented instruction
 * itself is about to fault on the same bytes. */
static AFUNPTR inline_loader(INS ins, UINT32 sz)
{
    if (!g_memload || !INS_IsMemoryRead(ins) || !INS_IsStandardMemop(ins) ||
        INS_HasRealRep(ins) || INS_IsPredicated(ins))
        return NULL;
    /* IARG_MEMORYREAD_EA is the first memory operand that is read */
    UINT32 n = INS_MemoryOperandCount(ins), i = 0;
    while (i < n && !INS_MemoryOperandIsRead(ins, i)) i++;
    if (i == n || INS_MemoryOperandSize(ins, i) < sz)
        return NULL;
    switch (sz) {
    case 8: return (AFUNPTR)Load8;
    case 4: return (AFUNPTR)Load4;
    case 2: return (AFUNPTR)Load2;
    case 1: return (AFUNPTR)Load1;
    default: return NULL;
    }
}

/* IARG_REG_CONST_REFERENCE is a raw register-byte buffer in newer Pin kits.
 * These copies also preserve the old PIN_REGISTER qword layout on x86-64. */
static ADDRINT PIN_FAST_ANALYSIS_CALL XmmLo(const VOID *r)
{
    UINT64 value; memcpy(&value, r, sizeof value); return (ADDRINT)value;
}
static ADDRINT PIN_FAST_ANALYSIS_CALL XmmHi(const VOID *r)
{
    UINT64 value; memcpy(&value, static_cast<const UINT8 *>(r) + 8, sizeof value);
    return (ADDRINT)value;
}

/* ---------------------------------------------------- the PTWRITE sink (-sink ptwrite)
 * `ptwrite %rdi' = F3 REX.W 0F AE /4 with rm = rdi.  Written as bytes so the tool builds
 * with any assembler in the kit's tool chain.  When IA32_RTIT_CTL.PTWEn is clear the
 * instruction retires without emitting a packet, so a -sink ptwrite run outside a PT
 * capture is still correct, just unrecorded. */
static bool g_ptw = false;

#define PTW_EMIT(v) __asm__ __volatile__(".byte 0xf3,0x48,0x0f,0xae,0xe7" \
                                         : : "D" ((UINT64)(v)) : )

static VOID PIN_FAST_ANALYSIS_CALL PtwReg(ADDRINT v) { PTW_EMIT(v); }

static VOID PIN_FAST_ANALYSIS_CALL PtwMem(ADDRINT ea, UINT32 sz)
{
    UINT64 v = 0;
    PIN_SafeCopy(&v, (const VOID *)ea, sz);
    PTW_EMIT(v);
}

static VOID PIN_FAST_ANALYSIS_CALL PtwVal(ADDRINT v) { PTW_EMIT(v); }

static VOID PIN_FAST_ANALYSIS_CALL PtwXmmLo(const VOID *r)
{
    UINT64 v; memcpy(&v, r, sizeof v); PTW_EMIT(v);
}

static VOID PIN_FAST_ANALYSIS_CALL PtwXmmHi(const VOID *r)
{
    UINT64 v; memcpy(&v, static_cast<const UINT8 *>(r) + 8, sizeof v); PTW_EMIT(v);
}

/* ------------------------------------------------------------ instrumentation */
static UINT64 g_inserted = 0, g_skipped_after = 0;

/* Pin runs a REP instruction's IPOINT_BEFORE and IPOINT_AFTER analysis once per ITERATION, but
 * the reconstructor consumes a site's values once per EXECUTED INSTRUCTION.  Measured on Pin 4.4
 * (tests/rep_site/repprobe.cpp): count n >= 1 runs BEFORE n times (FIRST_REP_ITERATION true on the
 * first, EXECUTING true) and AFTER n times; count 0 runs BEFORE once with FIRST_REP_ITERATION false
 * and EXECUTING false, and AFTER once.  So on a REP site every value is guarded (If/Then):
 *   BEFORE: logged iff FIRST_REP_ITERATION || !EXECUTING   (first iteration, or the count-0 call)
 *   AFTER:  logged iff this is the last iteration: rcx == 0, or REPE/REPNE stopped on ZF
 * i.e. exactly once per execution, count 0 included; the reconstructor applies a site's values
 * once per instruction instance, count 0 included. */
static ADDRINT PIN_FAST_ANALYSIS_CALL RepFirst(BOOL first, BOOL exe) { return first | !exe; }
static ADDRINT PIN_FAST_ANALYSIS_CALL RepLast(ADDRINT rcx, ADDRINT fl, UINT32 kind)
{
    return rcx == 0 || (kind == 1 && !(fl & 0x40)) || (kind == 2 && (fl & 0x40));
}
static UINT32 rep_kind(INS ins)   /* 0 plain REP, 1 REPE (cmps/scas), 2 REPNE */
{
    if (INS_RepnePrefix(ins)) return 2;
    /* Pin may report the base iclass (CMPSB) or the REPE_ one: match the mnemonic */
    const std::string m = INS_Mnemonic(ins);
    return (m.find("CMPS") != std::string::npos || m.find("SCAS") != std::string::npos) ? 1 : 0;
}
/* The guard is evaluated ONCE per (REP instruction, IPOINT) into a tool register and every guarded
 * value tests that register: evaluating RepLast(rflags) in front of each value lost ZF-terminated
 * REPE executions on the second value of a site (Pin 4.4, tests/rep_site BA plan). */
static REG g_rep = REG_INVALID();
static ADDRINT PIN_FAST_ANALYSIS_CALL RepFlag(ADDRINT f) { return f; }
static void rep_eval(INS ins, IPOINT ip)
{
    if (ip == IPOINT_BEFORE)
        INS_InsertCall(ins, ip, (AFUNPTR)RepFirst, IARG_FAST_ANALYSIS_CALL, IARG_FIRST_REP_ITERATION, IARG_EXECUTING,
                       IARG_RETURN_REGS, g_rep, IARG_END);
    else
        INS_InsertCall(ins, ip, (AFUNPTR)RepLast, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, LEVEL_BASE::REG_RCX,
                       IARG_REG_VALUE, LEVEL_BASE::REG_RFLAGS, IARG_UINT32, rep_kind(ins), IARG_RETURN_REGS, g_rep, IARG_END);
}
static void rep_guard(INS ins, IPOINT ip)
{
    INS_InsertIfCall(ins, ip, (AFUNPTR)RepFlag, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, g_rep, IARG_END);
}
/* INS_InsertCall / INS_InsertFillBuffer, guarded on a REP site */
#define XCALL(...) do { if (rep) { rep_guard(ins, ip); INS_InsertThenCall(ins, ip, __VA_ARGS__); } else INS_InsertCall(ins, ip, __VA_ARGS__); } while (0)
#define XFILL(...) do { if (rep) { rep_guard(ins, ip); INS_InsertFillBufferThen(ins, ip, g_buf, __VA_ARGS__); } else INS_InsertFillBuffer(ins, ip, g_buf, __VA_ARGS__); } while (0)
static UINT64 g_rep_sites = 0;

static void emit_items(INS ins, IPOINT ip, const std::vector<Item> &items)
{
    const bool rep = INS_HasRealRep(ins);
    if (rep) { g_rep_sites++; rep_eval(ins, ip); }
    for (size_t i = 0; i < items.size(); i++) {
        const Item &it = items[i];
        if (g_ptw) {
            /* one callout per value; program order is insertion order at the same IPOINT */
            switch (it.kind) {
            case IT_REG:
                XCALL((AFUNPTR)PtwReg, IARG_FAST_ANALYSIS_CALL,
                               IARG_REG_VALUE, it.reg, IARG_END);
                break;
            case IT_FSBASE:
                XCALL((AFUNPTR)PtwReg, IARG_FAST_ANALYSIS_CALL,
                               IARG_REG_VALUE, REG_SEG_FS_BASE, IARG_END);
                break;
            case IT_MEM:
            case IT_SHMEM:
                if (AFUNPTR ld = inline_loader(ins, it.size)) {
                    XCALL(ld, IARG_FAST_ANALYSIS_CALL, IARG_MEMORYREAD_EA,
                                   IARG_RETURN_REGS, g_scr, IARG_END);
                    XCALL((AFUNPTR)PtwVal, IARG_FAST_ANALYSIS_CALL,
                                   IARG_REG_VALUE, g_scr, IARG_END);
                    g_mem_inline++;
                } else {
                    XCALL((AFUNPTR)PtwMem, IARG_FAST_ANALYSIS_CALL,
                                   IARG_MEMORYREAD_EA, IARG_UINT32, it.size, IARG_END);
                    g_mem_safecopy++;
                }
                break;
            case IT_XMM_LO:
            case IT_XMM_HI:
                INS_InsertCall(ins, ip,
                               (AFUNPTR)(it.kind == IT_XMM_LO ? PtwXmmLo : PtwXmmHi),
                               IARG_FAST_ANALYSIS_CALL, IARG_REG_CONST_REFERENCE, it.reg,
                               IARG_END);
                break;
            }
            g_inserted++;
            continue;
        }
        switch (it.kind) {
        case IT_REG:
            XFILL(IARG_REG_VALUE, it.reg, 0, IARG_END);
            break;
        case IT_FSBASE:
            XFILL(IARG_REG_VALUE, REG_SEG_FS_BASE, 0, IARG_END);
            break;
        case IT_MEM:
        case IT_SHMEM:
            if (AFUNPTR ld = inline_loader(ins, it.size)) {
                XCALL(ld, IARG_FAST_ANALYSIS_CALL, IARG_MEMORYREAD_EA,
                               IARG_RETURN_REGS, g_scr, IARG_END);
                g_mem_inline++;
            } else {
                XCALL((AFUNPTR)ReadMem, IARG_FAST_ANALYSIS_CALL,
                               IARG_MEMORYREAD_EA, IARG_UINT32, it.size,
                               IARG_RETURN_REGS, g_scr, IARG_END);
                g_mem_safecopy++;
            }
            XFILL(IARG_REG_VALUE, g_scr, 0, IARG_END);
            break;
        case IT_XMM_LO:
        case IT_XMM_HI:
            XCALL((AFUNPTR)(it.kind == IT_XMM_LO ? XmmLo : XmmHi),
                           IARG_FAST_ANALYSIS_CALL, IARG_REG_CONST_REFERENCE, it.reg,
                           IARG_RETURN_REGS, g_scr, IARG_END);
            XFILL(IARG_REG_VALUE, g_scr, 0, IARG_END);
            break;
        }
        g_inserted++;
    }
}

#include "jitplan_flow.h"

/* The site for INS, or NULL.  with -reprcx 1 every REP string instruction also logs %rcx
 * (IPOINT_BEFORE, first iteration only, after the plan's own items) unless the plan already logs it
 * there.  The static plan derives a REP count only for REPs inside a CFG the analysis recovered; a
 * REP reached through an entry it never saw (libc's IFUNC-selected memset/memmove *_erms tails) has
 * no count root, so its count arrived as an unlogged caller value and the replay lost the whole
 * string operation.  Applied identically by hifitool, hififlow's code stream and -cvtag, so the
 * reconstruction consumes the extra value from the per-block item list with no plan change. */
static UINT64 g_rep_rcx = 0;
/* (-logshared): a load the plan has no site for is recomputed offline from the replay's shadow, which
 * holds only what THIS thread stored (own stack, plus logged memop values).  A load from memory another
 * thread or the kernel wrote (a worker-built heap header, a libevent base the main thread set up, vvar
 * clock data) is therefore unknown, and so is every address derived from it.  -logshared 1 logs the loaded
 * value of each such load whose value FEEDS AN ADDRESS, as an IT_SHMEM item in the ordinary cv stream.
 * The selection is static and per instruction, so hifitool, hififlow's code stream and -cvtag agree by
 * construction (the reconstruction reads the item from the block record):
 *   - one explicit memory read of 1/2/4/8 bytes that writes a GPR; not a REP/string op, push/pop,
 *     call/ret, or an %rsp-based operand (statically the thread's own stack); the plan has no site there;
 *   - forward taint from the written GPRs to the end of the basic block (any instruction that reads a
 *     tainted GPR taints the GPRs it writes; one that writes without reading untaints them): selected
 *     if a tainted GPR is the base/index of a later memory operand (REP string pointers included) or the
 *     REP count.  -lscons 1 also selects when the taint is still live at the block end or at a call.
 * The own-stack test is a RUNTIME property: with -logshared 1 the value is logged unconditionally (the
 * cv cursor must not depend on an address the replay may not know) and the replay uses it only when the
 * address is outside the thread's own stack mapping (tls.bin words 2-3).  -logshared 2 = count only:
 * every memory read, every selected one, and the selected ones whose address is outside the own stack
 * (inline compare against the per-thread bounds taken at thread start), per thread at exit. */
KNOB<UINT32> KnobLogShared(KNOB_MODE_WRITEONCE, "pintool", "logshared", "0",
                           "0 off | 1 log non-site address-feeding loads | 2 count only");
KNOB<UINT32> KnobLsCons(KNOB_MODE_WRITEONCE, "pintool", "lscons", "0", "1 = also select taint live at block end / call; 2 = also a store of a tainted GPR (spill)");
static UINT64 g_ls_static = 0;
static void ls_addregs(INS j, std::set<REG> *used)
{
    for (UINT32 o = 0; o < INS_OperandCount(j); o++) if (INS_OperandIsMemory(j, o)) {
        REG b = INS_OperandMemoryBaseReg(j, o), x = INS_OperandMemoryIndexReg(j, o);
        if (REG_valid(b)) used->insert(REG_FullRegName(b));
        if (REG_valid(x)) used->insert(REG_FullRegName(x));
    }
    if (INS_HasRealRep(j)) used->insert(LEVEL_BASE::REG_RCX);
}
static bool ls_candidate(INS ins, const Site *s, UINT32 *size)
{
    if (!INS_IsMemoryRead(ins) || INS_HasRealRep(ins) || INS_IsCall(ins) || INS_IsRet(ins) || INS_IsStringop(ins)) return false;
    const xed_iclass_enum_t op = (xed_iclass_enum_t)INS_Opcode(ins);
    if (op == XED_ICLASS_POP || op == XED_ICLASS_PUSH || op == XED_ICLASS_LEAVE) return false;
    if (INS_MemoryOperandCount(ins) != 1 || !INS_MemoryOperandIsRead(ins, 0) || INS_MemoryOperandIsWritten(ins, 0)) return false;
    const UINT32 sz = INS_MemoryOperandSize(ins, 0);
    if (sz != 1 && sz != 2 && sz != 4 && sz != 8) return false;
    if (REG_FullRegName(INS_MemoryBaseReg(ins)) == LEVEL_BASE::REG_RSP) return false;
    if (s) return false;          /* the plan already has a site here: its items are what the analysis needs */
    std::set<REG> t;
    for (UINT32 i = 0; i < INS_MaxNumWRegs(ins); i++) { REG r = REG_FullRegName(INS_RegW(ins, i)); if (REG_is_gr64(r)) t.insert(r); }
    if (t.empty()) return false;
    *size = sz;
    for (INS j = INS_Next(ins); INS_Valid(j); j = INS_Next(j)) {
        std::set<REG> u; ls_addregs(j, &u);
        for (REG r : u) if (t.count(r)) return true;
        if (INS_IsCall(j)) return KnobLsCons.Value() != 0;
        bool rd = false;
        for (UINT32 i = 0; i < INS_MaxNumRRegs(j) && !rd; i++) if (t.count(REG_FullRegName(INS_RegR(j, i)))) rd = true;
        if (rd && KnobLsCons.Value() >= 2 && INS_IsMemoryWrite(j)) return true;
        for (UINT32 i = 0; i < INS_MaxNumWRegs(j); i++) { REG r = REG_FullRegName(INS_RegW(j, i)); if (!REG_is_gr64(r)) continue; if (rd) t.insert(r); else t.erase(r); }
        if (t.empty()) return false;
    }
    return KnobLsCons.Value() != 0 && !t.empty();
}
/* (-logvec 1): the -logshared selection covers GPR loads only.  The rayon main thread's residual (91 % of
 * its unknowns, one memmove rep movsb in par_extend) is rooted in a 16-byte VECTOR load of a struct another
 * thread built: LinkedList::pop_front moves a worker-written node's Vec {ptr,len} with movups xmm0,[rdi+8],
 * spills xmm0 to the stack, frees the node, and later reloads ptr/len into GPRs from the own stack.  The plan
 * has no site (single-thread, that value would be in the shadow).  -logvec 1 logs the loaded register (two
 * IT_XMM_LO/HI after items, the plan's own item kinds) at each 16-byte xmm load that has no plan site, is not
 * %rsp-based, and whose register, in the same block before it is redefined, is stored to an %rsp-based slot
 * or moved into a GPR.  The value is the register's true content, so it is sound unconditionally (registers
 * are thread-private); the replay needs no change. */
KNOB<UINT32> KnobLogVec(KNOB_MODE_WRITEONCE, "pintool", "logvec", "0", "1 = log 16-byte xmm loads (no site, not %rsp-based) spilled to the stack / moved to a GPR in the block");
static UINT64 g_lv_static = 0;
static bool lv_candidate(INS ins, const Site *s, REG *xr)
{
    if (s || !INS_IsMemoryRead(ins) || INS_IsMemoryWrite(ins) || INS_HasRealRep(ins) || INS_IsStringop(ins)) return false;
    if (INS_MemoryOperandCount(ins) != 1 || INS_MemoryOperandSize(ins, 0) != 16) return false;
    if (REG_FullRegName(INS_MemoryBaseReg(ins)) == LEVEL_BASE::REG_RSP) return false;
    if (INS_MaxNumWRegs(ins) != 1 || !REG_is_xmm(INS_RegW(ins, 0))) return false;
    const REG r = INS_RegW(ins, 0), fr = REG_FullRegName(r);
    for (INS j = INS_Next(ins); INS_Valid(j); j = INS_Next(j)) {
        bool rd = false, wr = false, gw = false;
        for (UINT32 i = 0; i < INS_MaxNumRRegs(j); i++) if (REG_FullRegName(INS_RegR(j, i)) == fr) rd = true;
        for (UINT32 i = 0; i < INS_MaxNumWRegs(j); i++) { REG w = INS_RegW(j, i); if (REG_FullRegName(w) == fr) wr = true; if (REG_is_gr64(REG_FullRegName(w))) gw = true; }
        if (rd && INS_IsMemoryWrite(j) && REG_FullRegName(INS_MemoryBaseReg(j)) == LEVEL_BASE::REG_RSP) { *xr = r; return true; }
        if (rd && gw) { *xr = r; return true; }
        if (wr || INS_IsCall(j)) return false;
    }
    return false;
}
static const Site *site_for_base(INS ins, Site *tmp);
static const Site *site_for_ls(INS ins, Site *tmp);
static const Site *site_for(INS ins, Site *tmp)
{
    const Site *s = site_for_ls(ins, tmp);
    REG xr;
    if (KnobLogVec.Value() != 1 || !lv_candidate(ins, s, &xr)) return s;
    if (s != tmp) { tmp->before.clear(); tmp->after.clear(); }
    Item lo; lo.kind = IT_XMM_LO; lo.reg = xr; lo.size = 8; Item hi = lo; hi.kind = IT_XMM_HI;
    tmp->after.push_back(lo); tmp->after.push_back(hi);
    g_lv_static++;
    return tmp;
}
static const Site *site_for_ls(INS ins, Site *tmp)
{
    const Site *s = site_for_base(ins, tmp);
    UINT32 sz;
    if (KnobLogShared.Value() != 1 || !ls_candidate(ins, s, &sz)) return s;
    if (s != tmp) { if (s) *tmp = *s; else { tmp->before.clear(); tmp->after.clear(); } }
    Item m; m.kind = IT_SHMEM; m.reg = REG_INVALID(); m.size = sz;
    tmp->before.push_back(m);     /* after the plan's own items, like -reprcx */
    return tmp;
}
static const Site *site_for_base(INS ins, Site *tmp)
{
    std::map<ADDRINT, const Site *>::const_iterator it = g_sites.find(INS_Address(ins));
    const Site *s = it == g_sites.end() ? NULL : it->second;
    if (!s && KnobJitBridge.Value() && jit_site(INS_Address(ins), tmp)) s = tmp;
    if (!KnobRepRcx.Value() || !INS_HasRealRep(ins)) return s;
    if (s) for (size_t i = 0; i < s->before.size(); i++)
        if (s->before[i].kind == IT_REG && s->before[i].reg == LEVEL_BASE::REG_RCX) return s;
    if (s != tmp) { if (s) *tmp = *s; else { tmp->before.clear(); tmp->after.clear(); } }
    Item r; r.kind = IT_REG; r.reg = LEVEL_BASE::REG_RCX; r.size = 8;
    tmp->before.push_back(r);
    return tmp;
}

struct LsCnt { UINT64 lo, hi, all, cand, off, os, pad[2]; };
static LsCnt g_lsc[1024];
static VOID PIN_FAST_ANALYSIS_CALL LsAll(THREADID t) { if (t < 1024) g_lsc[t].all++; }
static VOID PIN_FAST_ANALYSIS_CALL LsCand(THREADID t, ADDRINT ea)
{ if (t < 1024) { LsCnt &c = g_lsc[t]; c.cand++; c.all++; if (ea < c.lo || ea >= c.hi) c.off++; } }
static VOID Instruction(INS ins, VOID *)
{
    Site dynamic;
    if (KnobLogShared.Value() == 2 && INS_IsMemoryRead(ins)) {
        UINT32 sz; if (ls_candidate(ins, site_for(ins, &dynamic), &sz)) { g_ls_static++;
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)LsCand, IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_MEMORYREAD_EA, IARG_END); }
        else INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)LsAll, IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_END);
    }
    const Site *s = site_for(ins, &dynamic);
    if (!s) return;
    if (s == &dynamic && INS_HasRealRep(ins) && KnobRepRcx.Value()) g_rep_rcx++;
    if (!s->before.empty())
        emit_items(ins, IPOINT_BEFORE, s->before);
    if (!s->after.empty()) {
        if (INS_HasFallThrough(ins))
            emit_items(ins, IPOINT_AFTER, s->after);
        else
            g_skipped_after += s->after.size();
    }
}

static VOID ImageLoad(IMG img, VOID *)
{
    jit_image(img);
    sync_image(img);
    const std::string name = IMG_Name(img);
    std::map<std::string, std::map<ADDRINT, Site> >::iterator p = g_plan.find(name);
    if (p == g_plan.end()) {
        /* fall back on the basename: LD_LIBRARY_PATH shadow trees and /proc paths differ */
        size_t sl = name.rfind('/');
        const std::string base = (sl == std::string::npos) ? name : name.substr(sl + 1);
        for (p = g_plan.begin(); p != g_plan.end(); ++p) {
            size_t s2 = p->first.rfind('/');
            if ((s2 == std::string::npos ? p->first : p->first.substr(s2 + 1)) == base)
                break;
        }
        if (p == g_plan.end())
            return;
    }
    const ADDRINT off = IMG_LoadOffset(img);
    for (std::map<ADDRINT, Site>::const_iterator s = p->second.begin();
         s != p->second.end(); ++s) {
        g_sites[s->first + off] = &s->second;
        g_resolved++;
    }
    g_images++;
    fprintf(stderr, "[hifitool] image %s base+%lx: %lu addresses\n", name.c_str(),
            (unsigned long)off, (unsigned long)p->second.size());
    PIN_RemoveInstrumentation();      /* traces cached before this image loaded */
}

static VOID Fini(INT32, VOID *)
{
    if (!KnobStats.Value())
        return;
    UINT64 tot = 0;
    for (size_t i = 0; i < g_all.size(); i++)
        tot += g_all[i]->nvalues;
    fprintf(stderr, "[hifitool] threads=%lu values=%lu bytes=%lu addresses=%lu "
                    "inserted=%lu skipped_after=%lu\n",
            (unsigned long)g_all.size(), (unsigned long)tot, (unsigned long)(tot * 8),
            (unsigned long)g_resolved, (unsigned long)g_inserted,
            (unsigned long)g_skipped_after);
    fprintf(stderr, "[hifitool] reprcx=%d rep_sites_with_added_rcx=%lu\n", (int)KnobRepRcx.Value(), (unsigned long)g_rep_rcx);
    fprintf(stderr, "[hifitool] logshared=%u lscons=%d static_selected_instrumentations=%lu\n", KnobLogShared.Value(), (int)KnobLsCons.Value(), (unsigned long)g_ls_static);
    if (KnobLogVec.Value()) fprintf(stderr, "[hifitool] logvec=%u static_selections=%lu (hififlow: each instruction is selected twice, code record + instrumentation)\n", KnobLogVec.Value(), (unsigned long)g_lv_static);
    if (KnobLogShared.Value() == 2) for (int t = 0; t < 1024; t++) if (g_lsc[t].all)
        fprintf(stderr, "[hifitool] LSCOUNT tid=%lu stack=[%#lx,%#lx) loads=%lu selected=%lu selected_offstack=%lu share_off=%.4f%%\n", (unsigned long)g_lsc[t].os,
                (unsigned long)g_lsc[t].lo, (unsigned long)g_lsc[t].hi, (unsigned long)g_lsc[t].all, (unsigned long)g_lsc[t].cand, (unsigned long)g_lsc[t].off, 100.0 * g_lsc[t].off / g_lsc[t].all);
    if (g_syncev) fprintf(stderr, "[hifitool] sync_events=%lu\n", (unsigned long)g_sync_n);
    fprintf(stderr, "[hifitool] memread=%s mem_inline=%lu mem_safecopy=%lu mem_faults=%lu\n",
            g_memload ? "load" : "safecopy", (unsigned long)g_mem_inline,
            (unsigned long)g_mem_safecopy, (unsigned long)g_mem_faults);
    fprintf(stderr, "[hifitool] rep_site_instrumentations=%lu (values logged on the first REP iteration only)\n",
            (unsigned long)g_rep_sites);
    fprintf(stderr, "[hifitool] jit_added=%llu jit_moved=%llu jit_removed=%llu jit_requested=%llu\n",
            (unsigned long long)g_jitadded, (unsigned long long)g_jitmoved,
            (unsigned long long)g_jitremoved, (unsigned long long)g_jitrequested);
    if (KnobJitMiss.Value())
        fprintf(stderr, "[hifitool] jit_late_objects=%llu jit_late_sites=%llu "
                        "jit_dyn_traces=%llu jit_blocks_unplanned=%llu jit_blocks_planned=%llu\n",
                (unsigned long long)g_jit_late_objects, (unsigned long long)g_jit_late_sites,
                (unsigned long long)g_jit_dyn_traces, (unsigned long long)g_jitmiss_blocks,
                (unsigned long long)g_jithit_blocks);
    jit_miss_report(stderr);
}

int main(int argc, char *argv[])
{
    if (PIN_Init(argc, argv)) {
        fprintf(stderr, "usage: pin -t hifitool.so -plan PLAN [-cvdir DIR] -- <prog>\n");
        return 1;
    }
    if (KnobPlan.Value().empty() && !KnobJitBridge.Value()) {
        fprintf(stderr, "[hifitool] -plan or -jitbridge is required\n");
        return 1;
    }
    if (KnobPlan.Value().empty())
        fprintf(stderr, "[hifitool] no ELF PLAN: only runtime-published generated-code "
                        "sites will be instrumented\n");
    PIN_InitSymbols();
    if (!KnobPlan.Value().empty()) load_plan(KnobPlan.Value());
    if (KnobSink.Value() != "buffer" && KnobSink.Value() != "ptwrite") {
        fprintf(stderr, "[hifitool] -sink must be buffer or ptwrite\n");
        return 1;
    }
    g_ptw = (KnobSink.Value() == "ptwrite");
    if (KnobMemRead.Value() != "load" && KnobMemRead.Value() != "safecopy") {
        fprintf(stderr, "[hifitool] -memread must be load or safecopy\n");
        return 1;
    }
    g_memload = (KnobMemRead.Value() == "load");
    if (g_memload)
        PIN_AddInternalExceptionHandler(LoadFault, 0);
    /* in PTWRITE mode the values travel in the Intel PT packet stream, so there is no
     * cv.<pid>.<tid>.bin to write and no fill buffer to drain */
    g_discard = g_ptw || (KnobCvDir.Value() == "/dev/null");
    PIN_InitLock(&g_lock);
    PIN_InitLock(&g_jitlock);
    g_tls = PIN_CreateThreadDataKey(NULL);
    g_scr = PIN_ClaimToolRegister();
    g_rep = PIN_ClaimToolRegister();
    if (!REG_valid(g_rep)) {
        fprintf(stderr, "[hifitool] no REP-guard register available\n");
        return 1;
    }
    if (!REG_valid(g_scr)) {
        fprintf(stderr, "[hifitool] no scratch register available\n");
        return 1;
    }
    g_buf = PIN_DefineTraceBuffer(8, KnobPages.Value(), BufferFull, 0);
    if (g_buf == BUFFER_ID_INVALID) {
        fprintf(stderr, "[hifitool] PIN_DefineTraceBuffer failed\n");
        return 1;
    }
    g_syncev = KnobSyncEv.Value();
    g_syncdir = KnobSyncDir.Value().empty() ? KnobCvDir.Value() : KnobSyncDir.Value();
    if (g_syncev) PIN_AddForkFunction(FPOINT_AFTER_IN_CHILD, SyForkChild, 0);
    PIN_AddThreadStartFunction(ThreadStart, 0);
    PIN_AddThreadFiniFunction(ThreadFini, 0);
    IMG_AddInstrumentFunction(ImageLoad, 0);
    if (KnobJitMiss.Value())
        TRACE_AddInstrumentFunction(JitTrace, 0);
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;
}
