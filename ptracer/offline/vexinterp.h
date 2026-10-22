// vexinterp.h -- interpreter for the per-instruction VEX blocks produced by vexlift.h with a
// known/unknown lattice. This is PTracer's "trace reconstructor": it executes the program
// skeleton, takes critical values from the log, and emits every memory access it can compute.
#pragma once
#include "vexlift.h"
#include "shadowmap.h"
#include <array>
#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct MemAccess { uint64_t addr; bool known; int size; int op; /* 0 load 1 store 2 rmw */ };

class VexInterp {
public:
    VexInterp();
    // guest state (VEX AMD64 layout, 1024 bytes)
    uint8_t g[1024]; uint8_t gk[1024];
    void set_reg(int off, int size, uint64_t v);       // define a register (known)
    void set_reg_unknown(int off, int size);
    Val get_reg(int off, int size) const;
    void all_unknown();                                 // e.g. after a PT overflow
    // Guest state that is an ABI invariant rather than something the value log carries: the
    // direction flag (see vexinterp.cpp).  Applied at construction and after all_unknown().
    void init_invariants();
    // shadow memory (stack slots + logged memop values)
    void shadow_store(uint64_t addr, const Val& v, int size);
    Val shadow_load(uint64_t addr, int size) const;
    void shadow_clear() { if (fake_.on) { fake_.cleared = true; fake_.fab = false; } shadow_.clear(); }
    size_t shadow_chunks() const { return shadow_.size(); }
    // Forget `size' bytes at `addr' (an un-liftable instruction stored there: the shadow no longer
    // knows what is in that memory, but everything else stays valid).
    void shadow_forget(uint64_t addr, int size) { if (__builtin_expect(fake_.on, 0)) fake_touch(addr, size); if (!shadow_.size()) return; for (int i = 0; i < size;) { uint64_t key = (addr + i) >> 3; int o = (addr + i) & 7; int n = 8 - o; if (n > size - i) n = size - i;
            ShadowChunk* c = shadow_.find(key); if (c) { for (int j = 0; j < n; j++) c->kn &= (uint8_t)~(1 << (o + j)); if (!c->kn) shadow_.erase(key); } i += n; } }
    void memop_define(uint64_t addr, uint64_t v, int size) { Val x = Val::K(v, size); shadow_store_force(addr, x, size); }
    // ---- chunk seeds (recon_seed.cpp) ------------------------------------------
    // dump_state/load_state: the whole machine state (guest state, known masks, shadow memory).
    // Fake mode (pass 1 of a seeded --jobs run): every register lane starts as a TAGGED fake value
    // (fake64(lane)), every stack chunk read or written for the first time is fabricated as a tagged
    // fake (fake64(128 + i), key fake_.keys[i]), and removals of chunks the run never had are noted
    // (fake_.pages), so the parent can express the chunk's final state in terms of its seed.
    void dump_state(std::vector<uint8_t>& out) const;
    bool load_state(const uint8_t*& p, const uint8_t* e);
    struct Fake { bool on = false, fab = false, cleared = false, pages_ovf = false; uint64_t win_lo = 0, win_hi = 0; uint32_t next = 128;
                  std::vector<uint64_t> keys; uint64_t last_key = ~0ull; uint32_t heap = 0; int var = 0;
                  // chunk keys removed/overwritten that this run did not hold: one 512-bit map per 4 KB page
                  std::unordered_map<uint64_t, std::array<uint64_t, 8>> touched; uint64_t tp_page = ~0ull; uint64_t* tp_bits = nullptr;
                  bool is_touched(uint64_t k) { auto it = touched.find(k >> 9); return it != touched.end() && ((it->second[(k >> 6) & 7] >> (k & 63)) & 1); }
                  void touch(uint64_t k) { if ((k >> 9) != tp_page) { tp_page = k >> 9; tp_bits = touched[tp_page].data(); } tp_bits[(k >> 6) & 7] |= 1ull << (k & 63); } };
    Fake fake_;
    static uint64_t fake64(uint32_t s) { return ((uint64_t)(0x5e000000u + (s << 8)) << 32) | (uint64_t)(0x40000000u + (s << 12)); }   // both halves < 2^31 (also in run 2): sign extensions agree
    // run 2 of pass 1: both halves shifted by a per-seed pseudo-random multiple of 64 in [2^20, 2^29) (no carry between halves)
    static uint32_t fake_dh(uint32_t s, int h) { uint64_t x = (uint64_t)s * 2 + 1 + (uint64_t)h + 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; x ^= x >> 31;
        return (uint32_t)((1u << 20) + (x % ((1u << 29) - (1u << 20)))) & ~63u; }
    static uint64_t fake_delta(uint32_t s) { return ((uint64_t)fake_dh(s, 1) << 32) | fake_dh(s, 0); }
    static uint64_t fakev(uint32_t s, int var) { return var ? fake64(s) + fake_delta(s) : fake64(s); }
    void fake_begin(int var);
    void fake_touch(uint64_t addr, int size);   // fabricate (window) or note a removal page (elsewhere)
    void fake_fab(uint64_t addr, int size);
    void fake_fab_heap(uint64_t addr, int size);
    // a fake value for the `rel'-th logged value since the chunk's start (pass 1 does not know the cursor)
    uint64_t fake_cv(uint64_t rel) { if (next_sym_full()) return 0; fake_.keys.push_back((1ull << 63) | rel); return fakev(fake_.next++, fake_.var); }
    bool next_sym_full() const { return fake_.next >= 65536; }
    // execute one instruction block; `emit` receives each memory access; returns false if the block was not ok
    bool exec(const IRBlockC& b, const std::function<void(const MemAccess&)>& emit);
    // Same, appending the accesses to `sink' (nullptr: discard them).  Uses the compiled skeleton
    // (compile()) unless PTRECON_INTERP=tree or the block's shape is not handled.
    bool exec(const IRBlockC& b, std::vector<MemAccess>* sink);
    static bool tree_mode();                            // PTRECON_INTERP=tree: always the tree walker
    uint64_t n_prog = 0, n_prog_fallback = 0, n_stmt_dead = 0, n_expr_folded = 0;   // compile stats
    // ---- fused programs (a run of instructions compiled into one op list) -------------
    // Each instruction's compiled skeleton is concatenated with its slots/constants renumbered
    // into one slot space (constants preloaded, never written), an OP_MARK per instruction records
    // where its memory accesses start in the sink, and a backward pass over the WHOLE run removes
    // guest-state writes that a later instruction of the run overwrites before anything reads
    // them (the flags thunk of nearly every ALU instruction), then the computations only they fed.
    // Guest state, shadow memory and the access sequence after the run equal running the
    // instructions one by one; nothing outside reads state inside the run.
    struct FProg { std::vector<COp> ops; std::vector<int32_t> args; std::vector<Val> slots; int ninsn = 0; uint64_t gen = 0; int ops_in = 0; int nemit = 0;
        // per-instruction emission facts filled by the caller (Recon::build_fused_at)
        // per EXECUTED instruction (a run may also cover pure-instrumentation trampoline instructions,
        // which execute nothing): its record ip, position in the run, InsnInfo and bytes
        struct IEm { uint64_t ip; uint64_t iph; uint8_t nmem; uint8_t rmw; uint32_t pos; const void* xi; const uint8_t* raw; int len; };
        std::vector<IEm> iem; uint64_t ip_lo = 0, ip_hi = 0;
        uint32_t span = 0, nskip = 0;           // decoder instructions the run covers; pure-instrumentation ones among them
        void* sb = nullptr;                    // the superblock this (whole-block) program heads (Recon::SBlk)
        // pseudo-instructions in the run -- buffer-sink value definitions (flat_ent_ index each,
        // in order; they also count in nskip), lift failures (iem entries without accesses), patch jumps;
        // last_real: the last iem entry that is not a lift failure (-1: none; rip_/ts_ stay)
        std::vector<uint32_t> ents; uint32_t nlf = 0, npj = 0; int last_real = -1; int64_t kf_idx = -1;   // kf_idx: the run ends at a keyframe branch (pending_kf_)
        bool last_tramp = false, first_tramp = false, has_orig = false, need_noovf = false; uint64_t last_orig_ip = 0;
        bool avail_cut = false;   // the run stopped only because the batch had no more instructions then (a longer batch rebuilds it)
        // the access sequence when it is static (no guarded/atomic/dirty op): per access its
        // instruction, op and size; the caller adds its record constants and a hash memo
        struct AEnt { uint32_t insn, pos = 0; uint8_t op, size; uint8_t c_fl = 0xff; uint64_t iph = 0, cst = 0, c_ts = 0, c_mix = 0; };
        std::vector<AEnt> acc; bool acc_static = false; std::vector<uint64_t> memo;   // memo: (ts, flags, mix) per access for the compiled emission
        // the program compiled to native code (cjit.cpp), set by the compile thread
        std::atomic<void*> jit{nullptr}; std::atomic<int> jstate{0}; uint32_t nexec = 0; int jpid = 0; uint64_t cjh = 0; uint32_t nrun = 0; };   // cjh: shared-cache key, 0 = not computed
    // ps[i].kind != 0 replaces blks[i] by a pseudo-instruction (blks[i] unused): 1 = an
    // executed instruction VEX could not lift (OP_UNK per GP/vector register in gp/vec, own MARK),
    // 2 = a buffer-sink value definition (OP_CVDEF of the run's k-th logged value at guest offset off; no MARK)
    struct FPseudo { uint8_t kind = 0; int32_t off = 0; uint32_t k = 0, gp = 0, vec = 0; };
    bool build_fused(const IRBlockC* const* blks, int n, FProg& F, const FPseudo* ps = nullptr);
    const uint64_t* cvp_ = nullptr;          // OP_CVDEF's values (set by the caller before a run that has any)
    static size_t cvp_offset();
    // writes the run's accesses to a buffer it owns (*acc) and instruction i's first access index to marks[i] (marks[ninsn] = total)
    void exec_fused(FProg& F, const MemAccess** acc, uint32_t* marks);
    std::vector<MemAccess> fbuf_;
    // ---- compiled fused programs ------------------------------------------------------------
    // cj_source() writes C source for F (one function `name'; prelude: cjit::prelude()) whose effect
    // on guest state, shadow memory, the access buffer and the marks is exactly exec_fused's: the
    // same handlers with the operands resolved, constants inlined and slots as locals; everything
    // but the inline fast paths calls the helpers below (the interpreter's own code).  false: F is
    // not handled (a slot read before it is written).
    bool cj_source(const FProg& F, const std::string& name, std::string& out) const;
    // Emission context of a compiled program with a static access plan (FProg::acc): the program
    // also produces the run's records (mtrace_rec, 32 bytes each, into `out' if non-null) and the
    // rec_hash steps, exactly as Recon::emit_fused_fast does.  Layout shared with cjit::prelude().
    // memo: null, or room for 3 words per access (address, time, flags) for the caller's rec_hash steps
    struct EmCtx { uint64_t rh, ts0, tsl, nu; uint64_t* memo; char* out; uint32_t nnl, tid; int32_t hash, pad; uint64_t* ring; uint64_t rc0; };   // ring: Recon's recent-ip ring (32 entries), rc0 its count
    bool has_jit(const FProg& F) const { return F.jit.load(std::memory_order_acquire) != nullptr; }
    // run F's compiled code with emission (precondition: has_jit(F), F.acc_static)
    void exec_fused_em(FProg& F, EmCtx* e);
    static const void* cj_helpers();
    uint64_t n_cj_runs = 0, n_cj_submitted = 0, n_cj_refused = 0, n_cj_deferred = 0, n_cj_cache_hits = 0, n_cj_waits = 0; bool cj_on_ = false; std::string cj_src_;
    void cj_submit(FProg& F);
    uint64_t n_fused = 0, n_fused_ops_in = 0, n_fused_ops_out = 0;
    // statistics
    uint64_t n_unknown_ops = 0, n_ccall_unknown = 0, n_ccall_known = 0;
    uint64_t stack_lo = 0, stack_hi = 0;   // shadow-store acceptance window (0 = around rsp)
    // register helpers (VEX offsets)
    static int reg_off(const std::string& name, int* size);
private:
    friend struct CJAccess;
    std::vector<Val> t_;
    const std::function<void(const MemAccess&)>* emit_ = nullptr;
    std::vector<MemAccess>* sink_ = nullptr; bool use_sink_ = false; bool tree_ = false;
    void emit_acc(const Val& addr, int size, int op) { MemAccess m{addr.known() ? addr.u64() : 0, addr.known(), size, op};
        if (use_sink_) { if (sink_) sink_->push_back(m); } else if (emit_) (*emit_)(m); }
    bool exec_tree(const IRBlockC& b);
    void compile(const IRBlockC& b);
    void run_prog(const CProg& p);
    Val ccall_args(const IRNodeC& n, Val* argbuf, size_t nargs);
    ShadowMap shadow_;                                    // 8 bytes of memory + known mask, keyed by addr>>3
    void shadow_store_force(uint64_t addr, const Val& v, int size);
    Val eval(const IRBlockC& b, int ni);
    Val unop(IROp op, const Val& a);
    Val binop(IROp op, const Val& a, const Val& c);
    Val ccall(const IRBlockC& b, const IRNodeC& n);
    bool in_stack_window(uint64_t addr) const;
};
