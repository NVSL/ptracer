/*
 * hififlow.cpp -- the production HiFi
 * Pintool (hififlowtool.cpp, compiled in UNCHANGED below) plus a software control-flow log that makes
 * a Pin-JIT HiFi run reconstructible without mapping Pin's code cache.
 *
 * WHY.  Under Pin JIT the program runs from Pin's code cache, so Intel PT records code-cache IPs
 * (and Pin VM execution) that ptrecon cannot map back to the original program.  Instead of
 * reverse-engineering the cache, this tool records the ORIGINAL control flow through public Pin
 * APIs only:
 *
 *   code.<pid>.bin       one record per distinct (address, bytes) basic block Pin translated, with
 *                        the ORIGINAL bytes Pin saw at translation time and XED-derived metadata
 *                        per instruction.  Written at instrumentation time only (no runtime cost).
 *                        Because the bytes are captured when Pin compiles the block, a JIT code
 *                        object that is later replaced gets a NEW id: the log is time-versioned by
 *                        construction.
 *   flow.<pid>.<tid>.bin 4-byte block id per EXECUTED basic block, written by one inlined
 *                        INS_InsertFillBuffer at the block head (a store + pointer bump, no call).
 *   cv.<pid>.<tid>.bin   exactly hifitool's critical-value stream (unchanged code path).
 *   gt.<pid>.<tid>.bin   (-gt 1, validation only) Pin-side same-process ground truth: 16-byte
 *                        {effective address, ip|size|write} per executed memory operand.
 *   img.<pid>.txt        image name, low, high, load offset, for resolving PLAN addresses.
 *
 * The reconstructor (ptracer/offline/hifirecon.cpp) walks flow ids -> original instructions ->
 * VEX interpretation, consuming cv values at the plan's sites in program order.
 *
 * Knobs (in addition to hifitool's): -flow 0|1 (default 1), -gt 0|1, -fdir DIR (default = -cvdir;
 * "/dev/null" discards flow and gt too).
 */
#include "pin.H"
static VOID hf_start_program();
#define PIN_StartProgram hf_start_program
#include "hififlowtool.cpp"
#undef PIN_StartProgram

KNOB<BOOL> KnobFlow(KNOB_MODE_WRITEONCE, "pintool", "flow", "1", "log executed basic-block ids");
KNOB<BOOL> KnobGt(KNOB_MODE_WRITEONCE, "pintool", "gt", "0", "log Pin-side ground truth (validation)");
KNOB<std::string> KnobFdir(KNOB_MODE_WRITEONCE, "pintool", "fdir", "", "dir for flow/gt/code (default -cvdir)");
KNOB<BOOL> KnobCvTag(KNOB_MODE_WRITEONCE, "pintool", "cvtag", "0",
                     "debug: parallel stream with the site ip of every logged value (cvtag.<pid>.<tid>.bin)");
KNOB<BOOL> KnobRle(KNOB_MODE_WRITEONCE, "pintool", "rle", "1",
                   "self-loop blocks: count taken back edges in a tool register, one record at loop exit");
KNOB<BOOL> KnobElide(KNOB_MODE_WRITEONCE, "pintool", "elide", "1",
                     "self-loops whose exit test reads only registers: log nothing per iteration, log those "
                     "registers at the loop exit; the reconstructor replays until they match (needs -rle 1)");
KNOB<UINT32> KnobStamp(KNOB_MODE_WRITEONCE, "pintool", "stamp", "0",
                       "PT time alignment: 0 none; 1 PTWRITE (tid, flow position) at every flow-buffer flush; "
                       "2 also PTWRITE (block id) at every self-loop exit");
KNOB<UINT32> KnobFpages(KNOB_MODE_WRITEONCE, "pintool", "fpages", "0",
                        "flow-buffer pages per thread (0 = -pages); with -stamp >= 1 this sets the stamp period");
KNOB<BOOL> KnobStampLog(KNOB_MODE_WRITEONCE, "pintool", "stamplog", "0",
                        "validation: stamp.<pid>.<tid>.bin = {payload, rdtsc} of every flush stamp");
KNOB<UINT32> KnobDbg(KNOB_MODE_WRITEONCE, "pintool", "dbg", "0",
                     "COST DIAGNOSTIC ONLY (breaks the flow log): 1 no instrumentation on self-loop blocks, "
                     "2 no flow instrumentation at all (buffers/register still set up), 3 = 2 + no block_id()");
KNOB<UINT32> KnobRepMode(KNOB_MODE_WRITEONCE, "pintool", "repmode", "1",
                         "block marker on a REP-headed block: 0 plain IPOINT_BEFORE, 1 first-iteration only");

static BUFFER_ID g_fbuf = BUFFER_ID_INVALID, g_gbuf = BUFFER_ID_INVALID, g_tbuf = BUFFER_ID_INVALID;
static TLS_KEY g_ftls;
static BUFFER_ID g_cbuf = BUFFER_ID_INVALID;   /* loop counts for RLE records, 8 bytes each */
static REG g_cnt;                              /* per-thread tool register: taken back edges so far */
static UINT64 g_nloops = 0, g_nelide = 0;
static std::string g_fdir;
static bool g_fdiscard = false;
static int g_codefd = -1, g_imgfd = -1;
static PIN_LOCK g_flock;
static UINT32 g_nextid = 1;                       /* 0 is never a valid id */
static std::map<std::pair<ADDRINT, std::string>, UINT32> g_ids;
static UINT64 g_nbbl_instr = 0;

struct FTLS { int ffd, gfd, tfd, cfd, sfd, xfd; UINT64 nflow, ngt, nloop; };

/* Time alignment with Intel PT (s.9 of the note).  A PTWRITE executed by the tool lands in the
 * thread's own PT stream between the timing packets (TSC/MTC) around it, so each stamp ties a flow
 * position to PT time.  Payload tags: bits 63..56. */
/* PTW_EMIT (`ptwrite %rdi`) comes from hififlowtool.cpp. */
static const UINT64 STAMP_FLUSH = 0xa5ull << 56, STAMP_LOOP = 0xa6ull << 56;
static VOID PIN_FAST_ANALYSIS_CALL StampLoop(ADDRINT v) { PTW_EMIT(v); }
static std::vector<FTLS *> g_fall;

#pragma pack(push, 1)
struct CodeHdr { UINT32 magic, id; UINT64 addr; UINT32 nins, nbytes, nitems; };
struct InsMeta { UINT8 len, flags, base, index, scale, seg, ngt, pred; UINT16 gpw, msize; UINT8 nbefore, nafter, pad0, pad1; INT64 disp; };
#pragma pack(pop)
enum { F_REP = 1, F_FALL = 2, F_CTRL = 4, F_MEMR = 8, F_SYSCALL = 16, F_CONDBR = 32, F_MEMW = 64, F_HASMEM = 128 };

static const REG kGp[16] = {LEVEL_BASE::REG_RAX, LEVEL_BASE::REG_RCX, LEVEL_BASE::REG_RDX,
    LEVEL_BASE::REG_RBX, LEVEL_BASE::REG_RSP, LEVEL_BASE::REG_RBP, LEVEL_BASE::REG_RSI,
    LEVEL_BASE::REG_RDI, LEVEL_BASE::REG_R8, LEVEL_BASE::REG_R9, LEVEL_BASE::REG_R10,
    LEVEL_BASE::REG_R11, LEVEL_BASE::REG_R12, LEVEL_BASE::REG_R13, LEVEL_BASE::REG_R14,
    LEVEL_BASE::REG_R15};

static UINT8 gp_index(REG r)
{
    if (!REG_valid(r)) return 255;
    if (r == LEVEL_BASE::REG_RIP || r == LEVEL_BASE::REG_EIP) return 16;
    REG f = REG_FullRegName(r);
    for (int i = 0; i < 16; i++) if (f == kGp[i]) return (UINT8)i;
    return 254;                                   /* something else (not a GP register) */
}

static void write_all(int fd, const void *p, size_t n)
{
    const char *c = (const char *)p;
    while (n) { ssize_t w = write(fd, c, n); if (w <= 0) return; c += w; n -= (size_t)w; }
}

/* The critical-value items hifitool instruments at this instruction RIGHT NOW (same lookup,
 * same fall-through rule as hifitool's Instruction()).  Recording them per block version makes the
 * reconstruction independent of WHEN a plan became active: code Pin translated before an image's
 * plan was resolved (e.g. libc IFUNC resolvers run by ld.so during relocation, before Pin's image
 * callback) carries no items, exactly as it ran. */
static void site_items(INS ins, std::string *ser, UINT8 *nb, UINT8 *na)
{
    *nb = *na = 0;
    Site dynamic;
    const Site *s = site_for(ins, &dynamic);
    if (!s) return;
    const std::vector<Item> *lists[2] = {&s->before, &s->after};
    for (int w = 0; w < 2; w++) {
        if (w == 1 && !INS_HasFallThrough(ins)) break;
        for (size_t i = 0; i < lists[w]->size(); i++) {
            const Item &x = (*lists[w])[i];
            UINT8 rec[4] = {(UINT8)x.kind, 255, (UINT8)x.size, (UINT8)w};
            if (x.kind == IT_REG) rec[1] = gp_index(x.reg);
            else if (x.kind == IT_XMM_LO || x.kind == IT_XMM_HI) rec[1] = (UINT8)(x.reg - REG_XMM0);
            ser->append((const char *)rec, 4);
            if (w == 0) (*nb)++; else (*na)++;
        }
    }
}

/* lmode: 0 plain, 1 counted self-loop, 2 elided self-loop (regs[0..nr-1] logged at exit).  An
 * elided loop that carries CV items is demoted to counted (*lmode is updated): the positional CV
 * cursor must never depend on the replay converging.  Recorded in the TAIL's InsMeta:
 * pad0 = lmode | nr << 2 | regs[0] << 4, pad1 = regs[1] | regs[2] << 4. */
static UINT32 block_id(BBL bbl, UINT8 *lmode, const UINT8 *regs, int nr)
{
    const ADDRINT a = BBL_Address(bbl);
    const USIZE sz = BBL_Size(bbl);
    std::string bytes(sz, '\0');
    size_t got = PIN_SafeCopy(&bytes[0], (const VOID *)a, sz);
    if (got != sz) bytes.resize(got);
    std::vector<InsMeta> meta;
    std::string items;
    for (INS ins = BBL_InsHead(bbl); INS_Valid(ins); ins = INS_Next(ins)) {
        InsMeta m; memset(&m, 0, sizeof m);
        m.len = (UINT8)INS_Size(ins);
        if (INS_HasRealRep(ins)) m.flags |= F_REP;
        if (INS_HasFallThrough(ins)) m.flags |= F_FALL;
        if (INS_IsControlFlow(ins)) m.flags |= F_CTRL;
        if (INS_IsBranch(ins) && INS_HasFallThrough(ins)) m.flags |= F_CONDBR;
        if (INS_IsSyscall(ins)) m.flags |= F_SYSCALL;
        if (INS_IsMemoryRead(ins)) m.flags |= F_MEMR;
        if (INS_IsMemoryWrite(ins)) m.flags |= F_MEMW;
        if (INS_MemoryOperandCount(ins) > 0) {
            m.flags |= F_HASMEM;
            m.msize = (UINT16)INS_MemoryOperandSize(ins, 0);
            m.base = gp_index(INS_MemoryBaseReg(ins));
            m.index = gp_index(INS_MemoryIndexReg(ins));
            m.scale = (UINT8)INS_MemoryScale(ins);
            m.disp = (INT64)INS_MemoryDisplacement(ins);
            REG s = INS_SegmentRegPrefix(ins);
            m.seg = s == LEVEL_BASE::REG_SEG_FS ? 1 : s == LEVEL_BASE::REG_SEG_GS ? 2 : 0;
        }
        if (!INS_IsPrefetch(ins)) m.ngt = (UINT8)INS_MemoryOperandCount(ins);
        if (INS_IsPredicated(ins)) m.pred = 1;
        for (UINT32 i = 0; i < INS_MaxNumWRegs(ins); i++) {
            UINT8 g = gp_index(INS_RegW(ins, i));
            if (g < 16) m.gpw |= (UINT16)(1u << g);
        }
        site_items(ins, &items, &m.nbefore, &m.nafter);
        meta.push_back(m);
    }
    if (*lmode == 2 && !items.empty()) *lmode = 1;
    if (!meta.empty() && *lmode) {
        InsMeta &t = meta.back();
        t.pad0 = (UINT8)(*lmode | (*lmode == 2 ? nr << 2 | regs[0] << 4 : 0));
        t.pad1 = (UINT8)(*lmode == 2 ? (nr > 1 ? regs[1] : 0) | (nr > 2 ? regs[2] << 4 : 0) : 0);
    }
    std::string key = bytes;
    key.append(items);
    key.append((const char *)&meta.back().pad0, 2);
    std::pair<ADDRINT, std::string> k2(a, key);
    std::map<std::pair<ADDRINT, std::string>, UINT32>::iterator it = g_ids.find(k2);
    if (it != g_ids.end()) return it->second;
    UINT32 id = g_nextid++;
    g_ids[k2] = id;
    if (g_codefd >= 0) {
        CodeHdr h; h.magic = 0x324c4242; /* "BBL2" */ h.id = id; h.addr = a;
        h.nins = (UINT32)meta.size(); h.nbytes = (UINT32)bytes.size(); h.nitems = (UINT32)(items.size() / 4);
        write_all(g_codefd, &h, sizeof h);
        write_all(g_codefd, &meta[0], meta.size() * sizeof(InsMeta));
        write_all(g_codefd, bytes.data(), bytes.size());
        write_all(g_codefd, items.data(), items.size());
    }
    return id;
}

/* Counted-loop elision s.8).  For a self-loop block whose
 * conditional back edge is decided by a register-only flag setter S, return the GP registers R that
 * S reads (<= 3) and 0 otherwise.  Exactness argument: the branch outcome of an iteration is a
 * function f of the values of R at the END of that iteration (S writes no register, or writes one
 * whose input is recoverable from its output, or the jcc reads only result flags; no instruction
 * between S and the jcc writes R).  At the exit iteration n the logged values V = R_n make f
 * not-taken, at every k < n f was taken, so R_k != V for all k < n: the FIRST iteration whose R
 * equals V is the exit.  The reconstructor replays until that match; nothing is guessed. */
static bool jcc_reads_cf(xed_iclass_enum_t j)
{
    return j == XED_ICLASS_JB || j == XED_ICLASS_JNB || j == XED_ICLASS_JBE || j == XED_ICLASS_JNBE;
}
static bool jcc_result_only(xed_iclass_enum_t j)   /* ZF / SF / PF only */
{
    return j == XED_ICLASS_JZ || j == XED_ICLASS_JNZ || j == XED_ICLASS_JS || j == XED_ICLASS_JNS ||
           j == XED_ICLASS_JP || j == XED_ICLASS_JNP;
}
static bool writes_flags(INS ins)
{
    for (UINT32 i = 0; i < INS_MaxNumWRegs(ins); i++) if (REG_is_flags_any_size_type(INS_RegW(ins, i))) return true;
    return false;
}
KNOB<BOOL> KnobRspLoop(KNOB_MODE_WRITEONCE, "pintool", "rsploop", "1", "1 = never elide a loop that writes %rsp (default, keeps %rsp exact); 0 = allow eliding it");
static int elide_regs(BBL bbl, UINT8 regs[3])
{
    INS tail = BBL_InsTail(bbl);
    const xed_iclass_enum_t j = (xed_iclass_enum_t)INS_Opcode(tail);
    const bool is_jcc = jcc_result_only(j) || jcc_reads_cf(j) || j == XED_ICLASS_JL || j == XED_ICLASS_JNL ||
                        j == XED_ICLASS_JLE || j == XED_ICLASS_JNLE || j == XED_ICLASS_JO || j == XED_ICLASS_JNO;
    if (!is_jcc) return 0;
    INS setter = INS_Invalid();
    UINT32 wafter = 0;
    for (INS i = BBL_InsHead(bbl); INS_Valid(i) && i != tail; i = INS_Next(i)) {
        if (INS_HasRealRep(i) || INS_IsSyscall(i) || INS_IsControlFlow(i)) return 0;
        /* A loop that moves %rsp (e.g. a stack-zeroing loop) is counted, never elided: if its exit
         * register is unknown at entry, an elided replay cannot decide the exit and would leave %rsp
         * unknown for the rest of the activation.  Counting keeps %rsp exact at the price of one
         * inlined add per iteration. */
        if (KnobRspLoop.Value()) for (UINT32 r = 0; r < INS_MaxNumWRegs(i); r++) if (REG_FullRegName(INS_RegW(i, r)) == REG_STACK_PTR) return 0;
        if (writes_flags(i)) { setter = i; wafter = 0; continue; }
        for (UINT32 r = 0; r < INS_MaxNumWRegs(i); r++) { UINT8 g = gp_index(INS_RegW(i, r)); if (g < 16) wafter |= 1u << g; }
    }
    if (!INS_Valid(setter) || INS_MemoryOperandCount(setter) != 0) return 0;
    const xed_iclass_enum_t c = (xed_iclass_enum_t)INS_Opcode(setter);
    bool ok = false;
    if (c == XED_ICLASS_CMP || c == XED_ICLASS_TEST) ok = true;
    else if (c == XED_ICLASS_AND || c == XED_ICLASS_OR || c == XED_ICLASS_XOR) ok = true;   /* CF=OF=0 */
    else if (c == XED_ICLASS_INC || c == XED_ICLASS_DEC) ok = !jcc_reads_cf(j);             /* CF untouched */
    else if (c == XED_ICLASS_ADD || c == XED_ICLASS_SUB) {
        if (INS_OperandCount(setter) < 2 || !INS_OperandIsReg(setter, 0)) return 0;
        const bool same = INS_OperandIsReg(setter, 1) &&
                          REG_FullRegName(INS_OperandReg(setter, 0)) == REG_FullRegName(INS_OperandReg(setter, 1));
        ok = !same || jcc_result_only(j);                 /* input recoverable from output, or result-only */
    }
    if (!ok) return 0;
    /* R must not depend on memory.  The replay knows loaded values only through the CV/shadow,
     * so a loop whose exit register is (transitively, incl. loop-carried and vector) fed by a load in
     * the body -- strlen/memchr scans, digit loops, pointer chasing -- cannot be decided.  Such a loop
     * is counted (-rle) instead of elided, which is exact.  Monotone taint over full registers, iterated
     * to a fixpoint across the back edge. */
    {
        std::vector<bool> taint(REG_LAST + 1, false);
        for (bool chg = true; chg; ) {
            chg = false;
            for (INS i = BBL_InsHead(bbl); INS_Valid(i) && i != tail; i = INS_Next(i)) {
                bool t = INS_IsMemoryRead(i);
                /* the flags carry taint too (one carrier, REG_RFLAGS): a sequence that feeds a loaded
                 * value into the exit register through CF cannot be decided if the loop is elided. */
                for (UINT32 r = 0; r < INS_MaxNumRRegs(i) && !t; r++) {
                    REG f = REG_FullRegName(INS_RegR(i, r));
                    if (REG_valid(f) && REG_is_flags_any_size_type(f)) f = REG_RFLAGS;
                    if (REG_valid(f) && taint[f]) t = true;
                }
                if (!t) continue;
                for (UINT32 r = 0; r < INS_MaxNumWRegs(i); r++) {
                    REG f = REG_FullRegName(INS_RegW(i, r));
                    if (REG_valid(f) && REG_is_flags_any_size_type(f)) f = REG_RFLAGS;
                    if (REG_valid(f) && !taint[f]) { taint[f] = true; chg = true; }
                }
            }
        }
        for (UINT32 r = 0; r < INS_MaxNumRRegs(setter); r++) {
            REG f = REG_FullRegName(INS_RegR(setter, r));
            if (REG_valid(f) && REG_is_flags_any_size_type(f)) f = REG_RFLAGS;
            if (REG_valid(f) && taint[f]) return 0;
        }
    }
    int n = 0;
    for (UINT32 r = 0; r < INS_MaxNumRRegs(setter); r++) {
        UINT8 g = gp_index(INS_RegR(setter, r));
        if (g == 254 || g == 16) { if (REG_is_flags_any_size_type(INS_RegR(setter, r))) continue; return 0; }
        if (g >= 16) continue;
        bool dup = false; for (int q = 0; q < n; q++) if (regs[q] == g) dup = true;
        if (dup) continue;
        if (n == 3 || (wafter & (1u << g))) return 0;
        regs[n++] = g;
    }
    return n;
}

/* (): manual flow/loop buffers.  Pin's trace-buffer API (INS_InsertFillBuffer) checks
 * the buffer space once per TRACE ENTRY for every fill in the trace; a self-loop whose back edge
 * re-enters its own trace head (the usual case: the loop body is the trace's first block and the
 * exit path's records are later blocks of the same trace) therefore pays that check on EVERY
 * iteration even though the loop itself logs nothing (gemm: 1.3 G iterations, flow 2.5x HiFi).
 * Here each record is an inlined store through a tool register plus an inlined end test at the
 * record itself, so an iteration that logs nothing costs nothing.  The buffer END is aligned to
 * MB_SPAN and the base is not, so (p & (MB_SPAN-1)) == 0 means exactly "full".  -mbuf 0 = old path. */
/* -gtstart S (validation only).  The gt log of a JIT-scale run is bounded by -gtcap, so a window that
 * starts at VM start-up only ever validates the start-up code.  With -gtstart S > 0 every gt fill is guarded by
 * a per-thread flag that opens at the first flow-buffer flush of that thread at or after S seconds (monotonic,
 * from the tool's main).  A flush is an exact flow position: the flag flips between two flow records, so the
 * thread's gt stream starts at a known flow index.  That index is written to pgtwin.<pid>.<tid>.bin (header,
 * then {flow index K = flow records flushed so far, ns since t0}); the reconstructor replays the whole flow log
 * (it needs the state) but compares gt only from flow record K-1 or K (whichever owns the first gt record).
 * Flow, loop and cv streams are unchanged: they are logged from the start.  Requires -mbuf 1. */
KNOB<std::string> KnobGtStart(KNOB_MODE_WRITEONCE, "pintool", "gtstart", "0",
                              "validation: open the gt window S seconds after start, per thread at its next flow flush; 0 = from the start");
static double g_gtstart = 0; static UINT64 g_t0ns = 0;
static const UINT32 GT_MAXT = 8192; static volatile UINT32 g_gopen[GT_MAXT];
static UINT64 now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (UINT64)ts.tv_sec * 1000000000ull + ts.tv_nsec; }
static ADDRINT PIN_FAST_ANALYSIS_CALL GtOpen(THREADID tid) { return (UINT32)tid < GT_MAXT && g_gopen[tid]; }
static ADDRINT PIN_FAST_ANALYSIS_CALL GtOpenExec(THREADID tid, BOOL exe) { return exe && (UINT32)tid < GT_MAXT && g_gopen[tid]; }
KNOB<BOOL> KnobMbuf(KNOB_MODE_WRITEONCE, "pintool", "mbuf", "1", "manual flow/loop buffers (0 = Pin trace buffers)");
static const ADDRINT MB_SPAN = (ADDRINT)1 << 20, MB_CAP = ((ADDRINT)1 << 20) - 64;
static REG g_fptr, g_cptr;
static VOID *FlowFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf, UINT64 n, VOID *);
static VOID *LoopFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf, UINT64 n, VOID *);
static ADDRINT PIN_FAST_ANALYSIS_CALL Put4(ADDRINT p, UINT32 v) { *(UINT32 *)p = v; return p + 4; }
static ADDRINT PIN_FAST_ANALYSIS_CALL Put8(ADDRINT p, ADDRINT v) { *(UINT64 *)p = v; return p + 8; }
static ADDRINT PIN_FAST_ANALYSIS_CALL AtEnd(ADDRINT p) { return (p & (MB_SPAN - 1)) == 0; }
static ADDRINT FlowFlushM(THREADID tid, ADDRINT p)
{
    const ADDRINT b = p - MB_CAP;
    FlowFull(BUFFER_ID_INVALID, tid, NULL, (VOID *)b, MB_CAP / 4, NULL);
    return b;
}
static ADDRINT LoopFlushM(THREADID tid, ADDRINT p)
{
    const ADDRINT b = p - MB_CAP;
    LoopFull(BUFFER_ID_INVALID, tid, NULL, (VOID *)b, MB_CAP / 8, NULL);
    return b;
}
static ADDRINT mb_alloc()
{
    const ADDRINT raw = (ADDRINT)malloc(2 * MB_SPAN);
    return ((raw + MB_CAP + MB_SPAN - 1) & ~(MB_SPAN - 1)) - MB_CAP;   /* base; base + MB_CAP is aligned */
}
static ADDRINT mb_base(ADDRINT p) { return ((p - 1) & ~(MB_SPAN - 1)) + MB_SPAN - MB_CAP; }   /* p in (base, base+CAP] */
static ADDRINT PIN_FAST_ANALYSIS_CALL IsFirstRep(BOOL first, BOOL exe);
static ADDRINT PIN_FAST_ANALYSIS_CALL IsExec(BOOL exe) { return exe; }
static void put_flow(INS ins, IPOINT ip, UINT32 v, bool firstrep)
{
    if (!KnobMbuf.Value()) {
        if (firstrep) {
            INS_InsertIfCall(ins, ip, (AFUNPTR)IsFirstRep, IARG_FAST_ANALYSIS_CALL, IARG_FIRST_REP_ITERATION, IARG_EXECUTING, IARG_END);
            INS_InsertFillBufferThen(ins, ip, g_fbuf, IARG_UINT32, v, 0, IARG_END);
        } else INS_InsertFillBuffer(ins, ip, g_fbuf, IARG_UINT32, v, 0, IARG_END);
        return;
    }
    if (firstrep) {
        INS_InsertIfCall(ins, ip, (AFUNPTR)IsFirstRep, IARG_FAST_ANALYSIS_CALL, IARG_FIRST_REP_ITERATION, IARG_EXECUTING, IARG_END);
        INS_InsertThenCall(ins, ip, (AFUNPTR)Put4, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, g_fptr, IARG_UINT32, v,
                           IARG_RETURN_REGS, g_fptr, IARG_END);
    } else
        INS_InsertCall(ins, ip, (AFUNPTR)Put4, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, g_fptr, IARG_UINT32, v,
                       IARG_RETURN_REGS, g_fptr, IARG_END);
    INS_InsertIfCall(ins, ip, (AFUNPTR)AtEnd, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, g_fptr, IARG_END);
    INS_InsertThenCall(ins, ip, (AFUNPTR)FlowFlushM, IARG_THREAD_ID, IARG_REG_VALUE, g_fptr, IARG_RETURN_REGS, g_fptr, IARG_END);
}
static void put_loop(INS ins, IPOINT ip, REG r)
{
    if (!KnobMbuf.Value()) { INS_InsertFillBuffer(ins, ip, g_cbuf, IARG_REG_VALUE, r, 0, IARG_END); return; }
    INS_InsertCall(ins, ip, (AFUNPTR)Put8, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, g_cptr, IARG_REG_VALUE, r,
                   IARG_RETURN_REGS, g_cptr, IARG_END);
    INS_InsertIfCall(ins, ip, (AFUNPTR)AtEnd, IARG_FAST_ANALYSIS_CALL, IARG_REG_VALUE, g_cptr, IARG_END);
    INS_InsertThenCall(ins, ip, (AFUNPTR)LoopFlushM, IARG_THREAD_ID, IARG_REG_VALUE, g_cptr, IARG_RETURN_REGS, g_cptr, IARG_END);
}

static ADDRINT PIN_FAST_ANALYSIS_CALL CntInc(ADDRINT c) { return c + 1; }
static ADDRINT PIN_FAST_ANALYSIS_CALL CntZero() { return 0; }
/* count-0 REP runs IPOINT_BEFORE once with FIRST_REP_ITERATION false and EXECUTING false; it is
 * still one execution of the block, so it gets its marker too */
static ADDRINT PIN_FAST_ANALYSIS_CALL IsFirstRep(BOOL first, BOOL exe) { return first | !exe; }

static VOID FlowTrace(TRACE trace, VOID *)
{
    for (BBL bbl = TRACE_BblHead(trace); BBL_Valid(bbl); bbl = BBL_Next(bbl)) {
        INS head = BBL_InsHead(bbl);
        if (KnobFlow.Value() && KnobDbg.Value() >= 2) {
            if (KnobDbg.Value() == 2) { UINT8 lm = 0; block_id(bbl, &lm, NULL, 0); }
        } else if (KnobFlow.Value()) {
            INS tail = BBL_InsTail(bbl);
            const bool selfloop = KnobRle.Value() && !INS_HasRealRep(head) && INS_IsDirectBranch(tail) &&
                                  INS_HasFallThrough(tail) &&
                                  INS_DirectControlFlowTargetAddress(tail) == BBL_Address(bbl);
            UINT8 regs[3] = {0, 0, 0};
            const int nr = selfloop && KnobElide.Value() ? elide_regs(bbl, regs) : 0;
            UINT8 lmode = selfloop ? (nr ? 2 : 1) : 0;
            UINT32 id = block_id(bbl, &lmode, regs, nr);
            if (KnobDbg.Value() == 1 && lmode) lmode = 3;   /* diagnostic: leave the loop uninstrumented */
            if (lmode && KnobStamp.Value() >= 2)
                INS_InsertCall(tail, IPOINT_AFTER, (AFUNPTR)StampLoop, IARG_FAST_ANALYSIS_CALL,
                               IARG_ADDRINT, (ADDRINT)(STAMP_LOOP | id), IARG_END);
            if (lmode == 3) {
            } else if (lmode == 2) {
                /* Nothing per iteration.  Loop exit (fall-through): one flow record id|2^31, then the
                 * exit-test registers into the loop stream. */
                put_flow(tail, IPOINT_AFTER, id | 0x80000000u, false);
                for (int q = 0; q < nr; q++) put_loop(tail, IPOINT_AFTER, kGp[regs[q]]);
                g_nelide++;
            } else if (selfloop) {
                /* Iterations 2..n are counted, not logged: taken back edge -> ++cnt (inlined);
                 * fall-through (loop exit) -> one flow record id|2^31 plus the count, then cnt = 0.
                 * No block-head record: every execution of this block ends in one of the two edges. */
                INS_InsertCall(tail, IPOINT_TAKEN_BRANCH, (AFUNPTR)CntInc, IARG_FAST_ANALYSIS_CALL,
                               IARG_REG_VALUE, g_cnt, IARG_RETURN_REGS, g_cnt, IARG_END);
                put_flow(tail, IPOINT_AFTER, id | 0x80000000u, false);
                put_loop(tail, IPOINT_AFTER, g_cnt);
                INS_InsertCall(tail, IPOINT_AFTER, (AFUNPTR)CntZero, IARG_FAST_ANALYSIS_CALL,
                               IARG_RETURN_REGS, g_cnt, IARG_END);
                g_nloops++;
            } else if (INS_HasRealRep(head) && KnobRepMode.Value() == 1) {
                put_flow(head, IPOINT_BEFORE, id, true);
            } else {
                put_flow(head, IPOINT_BEFORE, id, false);
            }
            g_nbbl_instr++;
        }
        if (KnobGt.Value()) {
            for (INS ins = head; INS_Valid(ins); ins = INS_Next(ins)) {
                if (INS_IsPrefetch(ins)) continue;
                UINT32 n = INS_MemoryOperandCount(ins);
                for (UINT32 i = 0; i < n; i++) {
                    UINT64 sz = INS_MemoryOperandSize(ins, i);
                    if (sz > 0x7fff) sz = 0x7fff;
                    UINT64 meta = (UINT64)INS_Address(ins) | (sz << 48) |
                                  ((UINT64)(INS_MemoryOperandIsWritten(ins, i) ? 1 : 0) << 63);
                    if (g_gtstart > 0) {   /* windowed gt; IARG_EXECUTING carries the REP count-0 and cmov guards */
                        if (INS_HasRealRep(ins) || INS_IsPredicated(ins))
                            INS_InsertIfCall(ins, IPOINT_BEFORE, (AFUNPTR)GtOpenExec, IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_EXECUTING, IARG_END);
                        else INS_InsertIfCall(ins, IPOINT_BEFORE, (AFUNPTR)GtOpen, IARG_FAST_ANALYSIS_CALL, IARG_THREAD_ID, IARG_END);
                        INS_InsertFillBufferThen(ins, IPOINT_BEFORE, g_gbuf, IARG_MEMORYOP_EA, i, 0, IARG_ADDRINT, (ADDRINT)meta, 8, IARG_END);
                    } else
                    if (INS_HasRealRep(ins)) {   /* Pin 4.4 fills a predicated buffer on the count-0 call of a
                                                     REP (EXECUTING false), which accesses no memory: guard it */
                        INS_InsertIfCall(ins, IPOINT_BEFORE, (AFUNPTR)IsExec, IARG_FAST_ANALYSIS_CALL, IARG_EXECUTING, IARG_END);
                        INS_InsertFillBufferThen(ins, IPOINT_BEFORE, g_gbuf, IARG_MEMORYOP_EA, i, 0,
                                                 IARG_ADDRINT, (ADDRINT)meta, 8, IARG_END);
                    } else
                    INS_InsertFillBufferPredicated(ins, IPOINT_BEFORE, g_gbuf,
                                                   IARG_MEMORYOP_EA, i, 0,
                                                   IARG_ADDRINT, (ADDRINT)meta, 8, IARG_END);
                }
            }
        }
    }
}

static int open_stream(const char *kind, OS_THREAD_ID tid, UINT32 magic)
{
    if (g_fdiscard) return -1;
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.%d.%d.bin", g_fdir.c_str(), kind, (int)PIN_GetPid(), (int)tid);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) { UINT32 h[4] = {magic, 1, (UINT32)tid, 0}; write_all(fd, h, sizeof h); }
    return fd;
}

/* the mapping that holds sp, from /proc/self/maps.  The main thread's "[stack]" grows down on
 * demand, so its lower bound is taken as hi - 8 MiB (the default RLIMIT_STACK); 0/0 if not found. */
static void stack_range(UINT64 sp, UINT64 *lo, UINT64 *hi)
{
    *lo = *hi = 0;
    int fd = open("/proc/self/maps", O_RDONLY); if (fd < 0) return;
    static char buf[1 << 20]; size_t n = 0; ssize_t r;
    while (n < sizeof buf - 1 && (r = read(fd, buf + n, sizeof buf - 1 - n)) > 0) n += (size_t)r;
    close(fd); buf[n] = 0;
    for (char *l = buf; l && *l; ) {
        char *e = strchr(l, '\n'); if (e) *e = 0;
        char *q; UINT64 a = strtoull(l, &q, 16), b = (*q == '-') ? strtoull(q + 1, &q, 16) : 0;
        if (a <= sp && sp < b) { *lo = a; *hi = b; if (strstr(l, "[stack]") && b > (8ull << 20)) *lo = b - (8ull << 20); break; }
        l = e ? e + 1 : NULL;
    }
}

static VOID FThreadStart(THREADID tid, CONTEXT *ctxt, INT32, VOID *)
{
    FTLS *t = new FTLS; t->nflow = t->ngt = 0; t->xfd = -1;
    OS_THREAD_ID os = PIN_GetTid();
    t->ffd = KnobFlow.Value() ? open_stream("flow", os, 0x4c465450 /* PTFL */) : -1;
    t->gfd = KnobGt.Value() ? open_stream("pgt", os, 0x47505450 /* PTPG */) : -1;
    t->tfd = KnobCvTag.Value() ? open_stream("cvtag", os, 0x47545650 /* PVTG */) : -1;
    t->cfd = (KnobFlow.Value() && KnobRle.Value()) ? open_stream("loop", os, 0x504f4c50 /* PLOP */) : -1;
    t->nloop = 0;
    t->sfd = KnobStampLog.Value() ? open_stream("stamp", os, 0x504d5453 /* STMP */) : -1;
    /* tls.<pid>.<tid>.bin: the thread's %fs base as the kernel set it (CLONE_SETTLS) when the thread starts.
     * A new thread never runs arch_prctl, so without this the reconstruction starts every non-main thread
     * with fs_base unknown and loses every %fs-relative access (TLS, tcache, errno) until something logs it. */
    if (KnobFlow.Value()) { int xfd = open_stream("tls", os, 0x4c545450 /* PTTL */);
        if (xfd >= 0) { UINT64 v[4] = {PIN_GetContextReg(ctxt, REG_SEG_FS_BASE), PIN_GetContextReg(ctxt, REG_STACK_PTR), 0, 0};
            PIN_GetLock(&g_flock, tid + 1); stack_range(v[1], &v[2], &v[3]); PIN_ReleaseLock(&g_flock); if (tid < 1024) { g_lsc[tid].lo = v[2]; g_lsc[tid].hi = v[3]; g_lsc[tid].os = os; }   /* the thread's own stack mapping [lo, hi) */
            write_all(xfd, v, sizeof v); close(xfd); } }
    PIN_SetThreadData(g_ftls, t, tid);
    if (KnobFlow.Value() && KnobMbuf.Value()) {
        PIN_SetContextReg(ctxt, g_fptr, mb_alloc());
        if (KnobRle.Value()) PIN_SetContextReg(ctxt, g_cptr, mb_alloc());
    }
    PIN_GetLock(&g_flock, tid + 1); g_fall.push_back(t); PIN_ReleaseLock(&g_flock);
}

static VOID FThreadFini(THREADID tid, const CONTEXT *ctxt, INT32, VOID *)
{
    FTLS *t = (FTLS *)PIN_GetThreadData(g_ftls, tid);
    if (!t) return;
    if (KnobFlow.Value() && KnobMbuf.Value()) {   /* drain the partial manual buffers */
        ADDRINT p = PIN_GetContextReg(ctxt, g_fptr), b = mb_base(p);
        FlowFull(BUFFER_ID_INVALID, tid, NULL, (VOID *)b, (p - b) / 4, NULL);
        if (KnobRle.Value()) { p = PIN_GetContextReg(ctxt, g_cptr); b = mb_base(p); LoopFull(BUFFER_ID_INVALID, tid, NULL, (VOID *)b, (p - b) / 8, NULL); }
    }
    if (t->ffd >= 0) { close(t->ffd); t->ffd = -1; }
    if (t->gfd >= 0) { close(t->gfd); t->gfd = -1; }
    if (t->tfd >= 0) { close(t->tfd); t->tfd = -1; }
    if (t->cfd >= 0) { close(t->cfd); t->cfd = -1; }
    if (t->sfd >= 0) { close(t->sfd); t->sfd = -1; }
    if (t->xfd >= 0) { close(t->xfd); t->xfd = -1; }
}

static VOID *FlowFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf, UINT64 n, VOID *)
{
    FTLS *t = (FTLS *)PIN_GetThreadData(g_ftls, tid);
    if (t) { t->nflow += n; if (t->ffd >= 0 && n) write_all(t->ffd, buf, (size_t)n * 4); }
    if (t && g_gtstart > 0 && (UINT32)tid < GT_MAXT && !g_gopen[tid]) {
        const UINT64 dt = now_ns() - g_t0ns;
        if (dt >= (UINT64)(g_gtstart * 1e9)) {
            int fd = open_stream("pgtwin", PIN_GetTid(), 0x4e575450 /* PTWN */);
            if (fd >= 0) { UINT64 r[2] = {t->nflow, dt}; write_all(fd, r, sizeof r); close(fd); }
            g_gopen[tid] = 1;   /* after the marker: no gt record precedes it */
        }
    }
    if (t && KnobStamp.Value()) {
        const UINT64 v = STAMP_FLUSH | ((UINT64)(PIN_GetTid() & 0xffff) << 40) | (t->nflow & 0xffffffffffull);
        PTW_EMIT(v);
        if (t->sfd >= 0) { UINT64 r[2] = {v, __builtin_ia32_rdtsc()}; write_all(t->sfd, r, sizeof r); }
    }
    return buf;
}

KNOB<UINT32> KnobGtCap(KNOB_MODE_WRITEONCE, "pintool", "gtcap", "0",
                       "validation: end the application cleanly (PIN_ExitApplication) once all threads have "
                       "written this many MB of gt; bounds JIT-scale captures by volume instead of a kill; 0 = never");
static volatile UINT64 g_gttot = 0; static volatile UINT32 g_gtstop = 0;
static VOID *GtFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf, UINT64 n, VOID *)
{
    FTLS *t = (FTLS *)PIN_GetThreadData(g_ftls, tid);
    /* The chunk that crosses the cap is NOT written here: PIN_ExitApplication does not return, and Pin flushes this
     * thread's buffer again at thread fini, which then writes it (once).  Losing it instead would only shorten gt. */
    if (KnobGtCap.Value() && n && __sync_add_and_fetch(&g_gttot, n * 16) > ((UINT64)KnobGtCap.Value() << 20)
        && __sync_bool_compare_and_swap(&g_gtstop, 0, 1)) {
        fprintf(stderr, "[hififlow] gtcap %u MB reached: PIN_ExitApplication\n", KnobGtCap.Value());
        PIN_ExitApplication(0);
    }
    if (t) { t->ngt += n; if (t->gfd >= 0 && n) write_all(t->gfd, buf, (size_t)n * 16); }
    return buf;
}

static VOID *LoopFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf, UINT64 n, VOID *)
{
    FTLS *t = (FTLS *)PIN_GetThreadData(g_ftls, tid);
    if (t) { t->nloop += n; if (t->cfd >= 0 && n) write_all(t->cfd, buf, (size_t)n * 8); }
    return buf;
}

static VOID *TagFull(BUFFER_ID, THREADID tid, const CONTEXT *, VOID *buf, UINT64 n, VOID *)
{
    FTLS *t = (FTLS *)PIN_GetThreadData(g_ftls, tid);
    if (t && t->tfd >= 0 && n) write_all(t->tfd, buf, (size_t)n * 8);
    return buf;
}

/* -cvtag 1: mirror hifitool's Instruction() exactly (same sites, same IPOINTs, same
 * fall-through rule), logging the site ip instead of the value. */
static VOID TagIns(INS ins, VOID *)
{
    Site dynamic;
    const Site *s = site_for(ins, &dynamic);
    if (!s) return;
    /* a REP site logs its values on the first iteration only (hifitool emit_items); so does its tag */
    const bool rep = INS_HasRealRep(ins);
    if (rep && !s->before.empty()) rep_eval(ins, IPOINT_BEFORE);
    for (size_t i = 0; i < s->before.size(); i++) {
        if (rep) {
            rep_guard(ins, IPOINT_BEFORE);
            INS_InsertFillBufferThen(ins, IPOINT_BEFORE, g_tbuf, IARG_ADDRINT, INS_Address(ins), 0, IARG_END);
        } else INS_InsertFillBuffer(ins, IPOINT_BEFORE, g_tbuf, IARG_ADDRINT, INS_Address(ins), 0, IARG_END);
    }
    if (rep && !s->after.empty() && INS_HasFallThrough(ins)) rep_eval(ins, IPOINT_AFTER);
    if (!s->after.empty() && INS_HasFallThrough(ins))
        for (size_t i = 0; i < s->after.size(); i++) {
            if (rep) {
                rep_guard(ins, IPOINT_AFTER);
                INS_InsertFillBufferThen(ins, IPOINT_AFTER, g_tbuf, IARG_ADDRINT, INS_Address(ins) | (1ull << 63), 0, IARG_END);
            } else INS_InsertFillBuffer(ins, IPOINT_AFTER, g_tbuf, IARG_ADDRINT, INS_Address(ins) | (1ull << 63), 0, IARG_END);
        }
}

static VOID FImageLoad(IMG img, VOID *)
{
    if (g_imgfd < 0) return;
    char line[2048];
    int n = snprintf(line, sizeof line, "%s %lx %lx %lx\n", IMG_Name(img).c_str(),
                     (unsigned long)IMG_LowAddress(img), (unsigned long)IMG_HighAddress(img),
                     (unsigned long)IMG_LoadOffset(img));
    if (n > 0) write_all(g_imgfd, line, (size_t)n);
}

/* Signals / faults: Pin delivers asynchronous signals only at trace boundaries, but a synchronous
 * fault can end a block early.  Record where (flow position, interrupted ip) so a reconstructor can
 * truncate that block instead of running past the fault.  Counted here; see the note. */
static UINT64 g_ctxchg = 0;
/* ctx.<pid>.<tid>.bin -- one record per context change: flow position (entries before it), reason,
 * the new rip, the 16 GPRs and fs base of the NEW context, and the old %rsp.  A signal handler starts on
 * a kernel-built frame below the interrupted %rsp and rt_sigreturn reloads the whole register file from
 * it; the replay cannot derive either, so it takes them from here (memcached's SIGINT handler gave 3
 * wrong addresses in ). */
static VOID CtxChange(THREADID tid, CONTEXT_CHANGE_REASON reason, const CONTEXT *from, CONTEXT *to, INT32 info, VOID *)
{
    __sync_fetch_and_add(&g_ctxchg, 1);
    FTLS *t = (FTLS *)PIN_GetThreadData(g_ftls, tid);
    if (!t || !to || !from || !KnobFlow.Value() || !KnobMbuf.Value()) return;
    static const REG gr[16] = {LEVEL_BASE::REG_RAX, LEVEL_BASE::REG_RCX, LEVEL_BASE::REG_RDX, LEVEL_BASE::REG_RBX, LEVEL_BASE::REG_RSP, LEVEL_BASE::REG_RBP, LEVEL_BASE::REG_RSI, LEVEL_BASE::REG_RDI,
                               LEVEL_BASE::REG_R8, LEVEL_BASE::REG_R9, LEVEL_BASE::REG_R10, LEVEL_BASE::REG_R11, LEVEL_BASE::REG_R12, LEVEL_BASE::REG_R13, LEVEL_BASE::REG_R14, LEVEL_BASE::REG_R15};
    ADDRINT p = PIN_GetContextReg(from, g_fptr);
    UINT64 r[22];
    r[0] = t->nflow + (p - mb_base(p)) / 4; r[1] = (UINT64)reason | ((UINT64)(UINT32)info << 32);
    r[2] = PIN_GetContextReg(to, REG_INST_PTR);
    for (int i = 0; i < 16; i++) r[3 + i] = PIN_GetContextReg(to, gr[i]);
    r[19] = PIN_GetContextReg(to, REG_SEG_FS_BASE); r[20] = PIN_GetContextReg(from, REG_STACK_PTR); r[21] = 0;
    if (t->xfd < 0) t->xfd = open_stream("ctx", PIN_GetTid(), 0x58544350 /* PCTX */);
    if (t->xfd >= 0) write_all(t->xfd, r, sizeof r);
}

static VOID FFini(INT32, VOID *)
{
    UINT64 f = 0, g = 0;
    for (size_t i = 0; i < g_fall.size(); i++) { f += g_fall[i]->nflow; g += g_fall[i]->ngt; }
    fprintf(stderr, "[hififlow] blocks=%lu distinct_block_versions=%lu block_instrumentations=%lu "
                    "flow_records=%lu gt_records=%lu context_changes=%lu selfloop_blocks=%lu elided_blocks=%lu\n",
            (unsigned long)g_ids.size(), (unsigned long)(g_nextid - 1), (unsigned long)g_nbbl_instr,
            (unsigned long)f, (unsigned long)g, (unsigned long)g_ctxchg, (unsigned long)g_nloops, (unsigned long)g_nelide);
    if (g_codefd >= 0) close(g_codefd);
    if (g_imgfd >= 0) close(g_imgfd);
}

static VOID hf_start_program()
{
    g_fdir = KnobFdir.Value().empty() ? KnobCvDir.Value() : KnobFdir.Value();
    g_fdiscard = (g_fdir == "/dev/null");
    PIN_InitLock(&g_flock);
    g_ftls = PIN_CreateThreadDataKey(NULL);
    if (!g_fdiscard) {
        char path[1024];
        snprintf(path, sizeof path, "%s/code.%d.bin", g_fdir.c_str(), (int)PIN_GetPid());
        g_codefd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        snprintf(path, sizeof path, "%s/img.%d.txt", g_fdir.c_str(), (int)PIN_GetPid());
        g_imgfd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    }
    if (KnobFlow.Value() && KnobMbuf.Value()) {
        g_fptr = PIN_ClaimToolRegister(); g_cptr = PIN_ClaimToolRegister();
        if (!REG_valid(g_fptr) || !REG_valid(g_cptr)) { fprintf(stderr, "[hififlow] mbuf setup failed\n"); PIN_ExitProcess(1); }
    }
    if (KnobFlow.Value()) {
        g_fbuf = PIN_DefineTraceBuffer(4, KnobFpages.Value() ? KnobFpages.Value() : KnobPages.Value(), FlowFull, 0);
        if (KnobRle.Value()) {
            g_cbuf = PIN_DefineTraceBuffer(8, KnobPages.Value(), LoopFull, 0);
            g_cnt = PIN_ClaimToolRegister();
            if (g_cbuf == BUFFER_ID_INVALID || !REG_valid(g_cnt)) { fprintf(stderr, "[hififlow] rle setup failed\n"); PIN_ExitProcess(1); }
        }
        if (g_fbuf == BUFFER_ID_INVALID) { fprintf(stderr, "[hififlow] flow buffer failed\n"); PIN_ExitProcess(1); }
    }
    g_t0ns = now_ns(); g_gtstart = atof(KnobGtStart.Value().c_str());
    if (g_gtstart > 0 && (!KnobGt.Value() || !KnobMbuf.Value() || !KnobFlow.Value())) { fprintf(stderr, "[hififlow] -gtstart needs -gt 1 -flow 1 -mbuf 1\n"); PIN_ExitProcess(1); }
    if (KnobGt.Value()) {
        g_gbuf = PIN_DefineTraceBuffer(16, KnobPages.Value(), GtFull, 0);
        if (g_gbuf == BUFFER_ID_INVALID) { fprintf(stderr, "[hififlow] gt buffer failed\n"); PIN_ExitProcess(1); }
    }
    if (KnobCvTag.Value()) {
        g_tbuf = PIN_DefineTraceBuffer(8, KnobPages.Value(), TagFull, 0);
        INS_AddInstrumentFunction(TagIns, 0);
    }
    PIN_AddThreadStartFunction(FThreadStart, 0);
    PIN_AddThreadFiniFunction(FThreadFini, 0);
    IMG_AddInstrumentFunction(FImageLoad, 0);
    TRACE_AddInstrumentFunction(FlowTrace, 0);
    PIN_AddContextChangeFunction(CtxChange, 0);
    PIN_AddFiniFunction(FFini, 0);
    PIN_StartProgram();
}
