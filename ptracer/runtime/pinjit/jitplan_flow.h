// Included after the canonical Site/Item definitions. Opt-in dynamic plans use
// exactly the same emit_items/trace buffer as static ELF plans.
#include "jitbridge.h"
#include <algorithm>
#include <cstddef>
#include <limits>
// ON BY DEFAULT ().  HiFi is ONE definition for every suite: the same Pintool,
// the same fill-buffer emitter, ELF PLAN sites plus runtime-published generated-code sites.
// On a target that generates no code and loads no publisher, this costs one symbol lookup
// per image at load time and nothing at all afterwards.
KNOB<BOOL> KnobJitBridge(KNOB_MODE_WRITEONCE, "pintool", "jitbridge", "1",
                       "accept runtime-published JIT critical-site plans");
// Diagnostic only, OFF in every timing row: 1 = count code objects Pin had already
// translated when their plan arrived (they executed uninstrumented); 2 = also count basic
// -block EXECUTIONS in generated code, split by whether a plan was published at the time
// that block was translated.  2 adds one increment per generated-code block: never time it.
KNOB<UINT32> KnobJitMiss(KNOB_MODE_WRITEONCE, "pintool", "jitmiss", "0",
                       "0 off | 1 object-level first-execution misses | 2 + execution-weighted");
struct JitObject { ADDRINT length; std::map<ADDRINT, Site> sites; };
static std::map<ADDRINT, JitObject> g_jitobjects;
static PIN_LOCK g_jitlock;
static UINT64 g_jitadded = 0, g_jitmoved = 0, g_jitremoved = 0, g_jitrequested = 0;
/* ---- first-execution miss accounting (KnobJitMiss), see the knob comment ---------------
 * Pin translates a trace at its FIRST execution.  So "Pin had already translated an address
 * inside this code object when the object's plan arrived" is exactly "this object executed
 * before our instrumentation existed for it".  The publication then invalidates the range,
 * so the misses are the executions BEFORE it, which `g_jitmiss_blocks' counts directly. */
static std::map<ADDRINT, ADDRINT> g_imgrange;      /* image low -> high, loaded images */
static std::map<ADDRINT, ADDRINT> g_translated;    /* generated-code trace low -> high */
static UINT64 g_jit_late_objects = 0, g_jit_late_sites = 0;
/* The published objects' bounding range, WIDENED ONLY, read without the lock.  This is what
 * makes "on by default" free: on a target that publishes nothing the range is empty and the
 * per-instruction hook is two compares and no lock.  A torn or stale read can only make the
 * check conservative or make one translation miss a just-published object -- and that
 * publication invalidates the range right after, forcing the translation to be redone. */
static volatile ADDRINT g_jitlo = ~(ADDRINT)0, g_jithi = 0;

static void jit_widen(ADDRINT base, ADDRINT length)
{
    if (base < g_jitlo) g_jitlo = base;
    if (base + length > g_jithi) g_jithi = base + length;
}
static UINT64 g_jitmiss_blocks = 0, g_jithit_blocks = 0;   /* generated-code BBL executions */
static std::map<ADDRINT, UINT64> g_miss_by_addr;           /* -jitmiss 3: unplanned, by trace */
static UINT64 g_jit_dyn_traces = 0;

static bool in_image(ADDRINT a)
{
    std::map<ADDRINT, ADDRINT>::const_iterator i = g_imgrange.upper_bound(a);
    if (i == g_imgrange.begin()) return false;
    --i;
    return a <= i->second;
}

static VOID PIN_FAST_ANALYSIS_CALL CountMiss(UINT64 *cell) { (*cell)++; }

static void jit_fail(const char *why)
{
    fprintf(stderr, "[hifitool] JIT bridge refused: %s\n", why);
    PIN_ExitProcess(71);
}

static bool jit_range(ADDRINT base, ADDRINT size)
{
    return base && size && size <= (1u << 27) &&
           base <= std::numeric_limits<ADDRINT>::max() - size;
}

// Caller holds only our data lock. Never call Pin invalidation while holding it:
// an instrumentation callback may be waiting on us while owning Pin's VM lock.
static void jit_erase_overlaps(ADDRINT base, ADDRINT length,
                              std::vector<std::pair<ADDRINT, ADDRINT> > *invalidated)
{
    for (auto it = g_jitobjects.begin(); it != g_jitobjects.end();) {
        if (it->first < base + length && base < it->first + it->second.length) {
            invalidated->push_back(std::make_pair(it->first, it->second.length));
            // Pin 3.20's STLport map::erase returns void; erase a copy and step first.
            std::map<ADDRINT, JitObject>::iterator dead = it++;
            g_jitobjects.erase(dead);
        } else ++it;
    }
}

static VOID JitPublish(ADDRINT address, THREADID tid)
{
    PtPinRequest req;
    if (PIN_SafeCopy(&req, reinterpret_cast<const void *>(address), sizeof req) != sizeof req ||
        req.version != 1 || req.acknowledged || !jit_range(req.base, req.length))
        jit_fail("bad request/range");
    JitObject object; object.length = req.length;
    if (req.operation == PT_PIN_ADD) {
        if (req.count > 65536) jit_fail("too many sites");
        std::vector<PtjSite> sites(req.count);
        const size_t bytes = sites.size() * sizeof(PtjSite);
        if (bytes && PIN_SafeCopy(&sites[0], reinterpret_cast<const void *>(req.sites), bytes) != bytes)
            jit_fail("unreadable sites");
        std::stable_sort(sites.begin(), sites.end(), [](const PtjSite &a, const PtjSite &b) {
            return a.id < b.id;
        });
        for (const auto &s : sites) {
            if (s.off >= req.length || s.when > PTJ_WHEN_AFTER || s.kind > PTJ_KIND_MEMOP ||
                s.kf || s.nregs > PTJ_MAXREG) jit_fail("unsupported site/keyframe");
            Site &target = object.sites[s.off];
            auto &items = (s.when == PTJ_WHEN_AFTER || s.kind == PTJ_KIND_LOAD) ? target.after : target.before;
            if (s.kind == PTJ_KIND_MEMOP) {
                if (s.when != PTJ_WHEN_BEFORE || s.size < 1 || s.size > 8) jit_fail("bad memory site");
                items.push_back(Item{IT_MEM, REG_INVALID(), s.size});
            } else {
                if (!s.nregs) jit_fail("empty register site");
                for (unsigned k = 0; k < s.nregs; k++) {
                    const unsigned r = s.regs[k];
                    if (r >= 32) jit_fail("unsupported register");
                    static const char *gp[] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                               "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
                    if (r < 16) items.push_back(Item{IT_REG, reg_by_name(gp[r]), 8});
                    else {
                        items.push_back(Item{IT_XMM_LO, (REG)(REG_XMM0 + r - 16), 8});
                        items.push_back(Item{IT_XMM_HI, (REG)(REG_XMM0 + r - 16), 8});
                    }
                }
            }
        }
    } else if (req.operation != PT_PIN_MOVE && req.operation != PT_PIN_REMOVE) {
        jit_fail("unknown operation");
    }
    if (req.operation == PT_PIN_MOVE && !jit_range(req.destination, req.length))
        jit_fail("bad destination");
    std::vector<std::pair<ADDRINT, ADDRINT> > invalidated;
    invalidated.push_back(std::make_pair(req.base, req.length));
    PIN_GetLock(&g_jitlock, tid + 1);
    if (req.operation == PT_PIN_ADD) {
        if (KnobJitMiss.Value()) {
            /* any part of this object already translated == it already executed */
            std::map<ADDRINT, ADDRINT>::const_iterator t = g_translated.lower_bound(req.base);
            bool late = (t != g_translated.end() && t->first < req.base + req.length);
            if (!late && t != g_translated.begin()) { --t; late = (req.base <= t->second); }
            if (late) { g_jit_late_objects++; g_jit_late_sites += req.count; }
        }
        jit_erase_overlaps(req.base, req.length, &invalidated);
        g_jitobjects[req.base] = object;
        jit_widen(req.base, req.length);
        g_jitadded++; g_jitrequested += req.count;
        { static FILE *rl = getenv("PTJ_JITRANGE_LOG") ? fopen(getenv("PTJ_JITRANGE_LOG"), "a") : 0;   /* diag: published code ranges */
          if (rl) { fprintf(rl, "%llx %llx %llu\n", (unsigned long long)req.base, (unsigned long long)(req.base + req.length), (unsigned long long)req.count); fflush(rl); } }
    } else if (req.operation == PT_PIN_MOVE) {
        auto old = g_jitobjects.find(req.base);
        if (old != g_jitobjects.end() && old->second.length == req.length) {
            object = old->second;
            g_jitobjects.erase(old);
            jit_erase_overlaps(req.destination, req.length, &invalidated);
            g_jitobjects[req.destination] = object;
            jit_widen(req.destination, req.length);
            g_jitmoved++;
        }
        // Unanalyzed/gated code can also move: it has no sites to republish.
    } else {
        jit_erase_overlaps(req.base, req.length, &invalidated); g_jitremoved++;
    }
    PIN_ReleaseLock(&g_jitlock);
    for (const auto &range : invalidated)
        PIN_RemoveInstrumentationInRange(range.first, range.first + range.second - 1);
    if (req.operation == PT_PIN_MOVE)
        PIN_RemoveInstrumentationInRange(req.destination, req.destination + req.length - 1);
    const UINT64 ack = 1;
    if (PIN_SafeCopy(reinterpret_cast<void *>(address + offsetof(PtPinRequest, acknowledged)),
                     &ack, sizeof ack) != sizeof ack) jit_fail("unwritable acknowledgement");
}

static bool jit_site(ADDRINT address, Site *out)
{
    if (address < g_jitlo || address >= g_jithi) return false;   /* lock-free common case */
    bool found = false;
    PIN_GetLock(&g_jitlock, 1);
    auto it = g_jitobjects.upper_bound(address);
    if (it != g_jitobjects.begin()) {
        --it;
        auto s = it->second.sites.find(address - it->first);
        if (s != it->second.sites.end()) { *out = s->second; found = true; }
    }
    PIN_ReleaseLock(&g_jitlock);
    return found;
}

/* Records generated-code translations and, at -jitmiss 2, counts their executions split by
 * whether a plan existed when the block was translated.  Registered only when -jitmiss > 0. */
static VOID JitTrace(TRACE trace, VOID *)
{
    const ADDRINT lo = TRACE_Address(trace);
    if (in_image(lo)) return;
    const ADDRINT hi = lo + TRACE_Size(trace) - 1;
    PIN_GetLock(&g_jitlock, 1);
    g_translated[lo] = hi > lo ? hi : lo;
    g_jit_dyn_traces++;
    bool known = false;
    std::map<ADDRINT, JitObject>::const_iterator o = g_jitobjects.upper_bound(lo);
    if (o != g_jitobjects.begin()) { --o; known = (lo < o->first + o->second.length); }
    PIN_ReleaseLock(&g_jitlock);
    if (KnobJitMiss.Value() < 2) return;
    UINT64 *cell = known ? &g_jithit_blocks : &g_jitmiss_blocks;
    /* -jitmiss 3: ALSO attribute the unplanned executions to the address they happen at, so
     * "a third of the generated code ran uninstrumented" can be turned into "which code".
     * One extra counter per unplanned trace; strictly a diagnostic, never a timing row. */
    UINT64 *where = 0;
    if (!known && KnobJitMiss.Value() >= 3) {
        PIN_GetLock(&g_jitlock, 1);
        where = &g_miss_by_addr[lo];
        PIN_ReleaseLock(&g_jitlock);
    }
    for (BBL b = TRACE_BblHead(trace); BBL_Valid(b); b = BBL_Next(b)) {
        BBL_InsertCall(b, IPOINT_BEFORE, (AFUNPTR)CountMiss, IARG_FAST_ANALYSIS_CALL,
                       IARG_PTR, cell, IARG_END);
        if (where)
            BBL_InsertCall(b, IPOINT_BEFORE, (AFUNPTR)CountMiss, IARG_FAST_ANALYSIS_CALL,
                           IARG_PTR, where, IARG_END);
    }
}

/* Top unplanned sites, printed at fini under -jitmiss 3. */
static void jit_miss_report(FILE *f)
{
    if (KnobJitMiss.Value() < 3 || g_miss_by_addr.empty()) return;
    std::vector<std::pair<UINT64, ADDRINT> > v;
    for (std::map<ADDRINT, UINT64>::const_iterator i = g_miss_by_addr.begin();
         i != g_miss_by_addr.end(); ++i)
        if (i->second) v.push_back(std::make_pair(i->second, i->first));
    std::sort(v.begin(), v.end());
    fprintf(f, "[hifitool] unplanned-execution attribution: %llu distinct traces\n",
            (unsigned long long)v.size());
    size_t shown = v.size() < 20000 ? v.size() : 20000;
    for (size_t k = 0; k < shown; k++) {
        const std::pair<UINT64, ADDRINT> &e = v[v.size() - 1 - k];
        fprintf(f, "[hifitool] miss_at %#lx %llu\n",
                (unsigned long)e.second, (unsigned long long)e.first);
    }
}

static void jit_image(IMG image)
{
    if (!KnobJitBridge.Value()) return;
    if (KnobJitMiss.Value()) {
        PIN_GetLock(&g_jitlock, 1);
        g_imgrange[IMG_LowAddress(image)] = IMG_HighAddress(image);
        PIN_ReleaseLock(&g_jitlock);
    }
    RTN marker = RTN_FindByName(image, "ptj_pin_publish");
    if (!RTN_Valid(marker)) return;
    RTN_Open(marker);
    RTN_InsertCall(marker, IPOINT_BEFORE, (AFUNPTR)JitPublish,
                   IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_THREAD_ID, IARG_END);
    RTN_Close(marker);
    fprintf(stderr, "[hifitool] JIT marker in %s\n", IMG_Name(image).c_str());
}
