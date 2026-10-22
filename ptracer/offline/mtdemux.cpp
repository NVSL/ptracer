// mtdemux.cpp -- ptrecon --mt: multi-threaded reconstruction of a per-CPU Intel PT capture.
//
// Design.  A `pt_capture2 --cpu A --cpu B ...' capture gives one AUX
// stream per core plus, per core, the kernel's context-switch records (PERF_RECORD_SWITCH_CPU_WIDE
// with pid/tid/time).  Because the events exclude ring 0, a core's stream is a sequence of USER
// REGIONS [TIP.PGE, TIP.PGD], each executed by exactly one thread, and a switch always happens in
// the gap between two regions.  So:
//
//   1. SEGMENT   every core's stream once, decode-only, recording a boundary at every TIP.PGE, at
//                every PT overflow and after every decoder resync, with the decoder's offset and
//                its TSC there (Recon::run_segment, one forked child per core);
//   2. ATTRIBUTE each region to the thread of the LAST switch-in on that core whose time is <= the
//                region's start time (the record time is perf's clock, converted back to TSC by
//                pt_capture2 with the mmap page's own time_mult/shift/zero); regions of other
//                processes that shared the core are dropped;
//   3. PLAN      per thread: its regions from all cores, in time order (a migrating thread simply
//                continues on another core's file); a region that begins at an overflow/resync
//                is LOSSY -- the switch-in itself may be inside the gap, so the thread's state is
//                dropped there exactly as after an overflow, and the unknown records that follow
//                until its next logged value are counted apart (`unknown_after_switch');
//   4. RECONSTRUCT each thread with an ordinary Recon on a plan-mode PtDecoder (ptdecode.h):
//                per-thread machine state, per-thread cv file (or the v3 mux file demuxed by tid),
//                per-thread ground truth, per-thread fs_base from the sideband's `threads'; one
//                forked child per thread, --mt-jobs at a time;
//   5. MERGE     the per-thread mtraces by TSC into one file whose records carry the real tid.
//
// The single-thread reconstructor is not changed: a plan-mode decoder hands it exactly the
// instructions and PTW events of its thread, in that thread's program order, and nothing else.
#include <sys/resource.h>
#include <thread>
#include "recon.h"
#include <memory>
#include <ctime>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include <glob.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct BndRec { uint64_t off, tsc, ip; uint32_t kind, pad; };     // on-disk boundary record (32 B); pad = instructions
                                                                 // decoded since the previous record (clamped; a cost estimate)
struct Region {
    int cpu = 0; uint64_t begin = 0, end = 0, tsc = 0, ip = 0, insns = 0; char kind = 'E';
    uint32_t pid = 0, tid = 0; bool lossy = false, slack_used = false, inherited = false, conflict = false;
    bool realigned = false;   // start time fell in a switch-out..switch-in gap; moved forward
    int64_t gap = 0;      // region start TSC - switch-in TSC (attribution margin)
};

std::string with_suffix(const std::string& base, const std::string& suf) { return base + suf; }

// All switch records of a core, sorted by time.  `ins' gets the switch-ins of REAL tasks: a
// switch-in of the idle task (pid 0) never owns a user region, so it is kept only in `all' --
// where it marks, like a switch-out, that the previous task left the core.
void load_switches(const std::string& path, std::vector<SwRec>& ins, std::vector<SwRec>& all) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return;
    SwRec r;
    while (fread(&r, sizeof r, 1, f) == 1) if (r.tsc) { all.push_back(r); if ((r.flags & SW_IN) && r.pid != 0) ins.push_back(r); }
    fclose(f);
    auto by_tsc = [](const SwRec& a, const SwRec& b) { return a.tsc < b.tsc; };
    std::stable_sort(ins.begin(), ins.end(), by_tsc); std::stable_sort(all.begin(), all.end(), by_tsc);
}
std::vector<BndRec> load_bnd(const std::string& path) {
    std::vector<BndRec> v;
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return v;
    BndRec r;
    while (fread(&r, sizeof r, 1, f) == 1) v.push_back(r);
    fclose(f);
    return v;
}
uint64_t file_size(const std::string& p) { struct stat st; return stat(p.c_str(), &st) ? 0 : (uint64_t)st.st_size; }

std::vector<std::string> glob_files(const std::string& pattern) {
    std::vector<std::string> out; glob_t g; memset(&g, 0, sizeof g);
    if (glob(pattern.c_str(), 0, nullptr, &g) == 0) for (size_t i = 0; i < g.gl_pathc; i++) out.push_back(g.gl_pathv[i]);
    globfree(&g);
    return out;
}
// cv file header: "PTCV" v2 (tid in the header), "PTCW" v3 (multiplexed), else legacy.
struct CvInfo { std::string path; uint32_t magic = 0, tid = 0; };
CvInfo cv_info(const std::string& path) {
    CvInfo c; c.path = path; unsigned char h[16];
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return c;
    if (fread(h, 1, 16, f) == 16) { c.magic = h[0] | (h[1] << 8) | (h[2] << 16) | ((uint32_t)h[3] << 24); c.tid = h[8] | (h[9] << 8) | (h[10] << 16) | ((uint32_t)h[11] << 24); }
    fclose(f); return c;
}

}  // namespace

// ---- 1. the segmentation pass (runs in a forked child, one per core) -------------------------
int Recon::run_segment(const std::string& aux, const std::string& out) {
    if (PtDecoder::aux_size(aux) == 0) {      // a core the process never ran on: no regions
        FILE* f = fopen(out.c_str(), "wb"); if (!f) { perror("segment out"); return 1; } fclose(f);
        fprintf(stderr, "segment %s: empty AUX file (the process never ran on this core)\n", aux.c_str());
        return 0;
    }
    // One core's stream carries the start-up window (the core the process was exec'd on)
    // and the others do not, and which one it is is not known before decoding.  So the pass is
    // tried in the PRE-INIT phase (unpatched images, e9phase.h) and re-run WITHOUT it if the
    // decoder reports that the phase explained nothing -- it lost sync before any image's loader
    // ran, i.e. this core's stream begins after the images were already patched.  The re-run
    // costs the few thousand instructions decoded before that first sync loss.
    for (int preinit = 1; preinit >= 0; preinit--) {
    PtDecoder dec(aux, sb_, 0, 0, have_jit_ ? &jit_ : nullptr, nullptr, preinit);
    dec.no_time = false;
    FILE* f = fopen(out.c_str(), "wb"); if (!f) { perror("segment out"); return 1; }
    setvbuf(f, nullptr, _IOFBF, 1 << 20);
    uint64_t n = 0;
    PtEvents ev;
    ev.on_boundary = [&](char k, uint64_t off, uint64_t tsc, uint64_t ip) { BndRec r{off, tsc, ip, (uint32_t)(unsigned char)k, 0}; fwrite(&r, sizeof r, 1, f); n++; };
    ev.on_ptwrite = [](uint64_t, int, uint64_t) {};
    ev.on_overflow = [](uint64_t) {};
    ev.on_resync = [](int, uint64_t) {};
    ev.on_enable = [](uint64_t, bool) {};
    struct pt_insn insn; uint64_t tsc = 0;
    bool bail = false;
    // the rest of a block but its last instruction has the offset of the instruction
    // just delivered and produces no boundary: counted without delivery (skip_nonlast, as --decode-only)
    const bool skip = !(getenv("PTRECON_SEG_NOSKIP") && *getenv("PTRECON_SEG_NOSKIP") == '1');
    while (dec.next(insn, &tsc, ev)) {
        if (preinit && dec.e9_preinit_useless()) { bail = true; break; }
        if (skip) dec.skip_nonlast(~0ull);
    }
    fclose(f);
    if (bail || (preinit && dec.e9_preinit_useless())) {
        fprintf(stderr, "segment %s: this core's stream does not cover the start-up window "
                        "(sync lost at %lu with nothing patched) -- decoding it again without the "
                        "pre-init phase\n", aux.c_str(), (unsigned long)dec.e9_abort_off);
        continue;
    }
    const uint64_t fsz = PtDecoder::aux_size(aux), stopped = dec.offset();
    fprintf(stderr, "segment %s: %lu instructions, %lu boundaries, %lu overflows, %lu resyncs, %lu sync failures, %lu psb+ headers repaired, %lu e9 swaps, ended at %lu of %lu bytes%s%s\n",
            aux.c_str(), (unsigned long)dec.n_insn, (unsigned long)n, (unsigned long)dec.n_ovf,
            (unsigned long)dec.n_resync, (unsigned long)dec.n_sync_fail, (unsigned long)dec.n_psb_repaired, (unsigned long)dec.n_e9_swap,
            (unsigned long)stopped, (unsigned long)fsz,
            dec.eos_why[0] ? " -- " : "", dec.eos_why);
    return 0;
    }
    return 0;
}

int Recon::segment_part(const Sideband& sb_, JitTable* jitp, const std::string& aux, const std::string& out, uint64_t from, uint64_t to,
                        const std::vector<uint64_t>& cuts, const std::vector<uint64_t>& psb) {
    if (from == 0) {
        for (int preinit = 1; preinit >= 0; preinit--) {
            PtDecoder dec(aux, sb_, 0, 0, jitp, nullptr, preinit);
            dec.no_time = false;
            { static const bool walk = getenv("PTRECON_SEG_WALK") != nullptr; static const bool noskip = getenv("PTRECON_SEG_NOSKIP") && *getenv("PTRECON_SEG_NOSKIP") == '1';
              dec.lazy_walk = jitp && !preinit && !walk && !noskip; }
            FILE* f = fopen(out.c_str(), "wb"); if (!f) { perror("segment out"); return 1; }
            setvbuf(f, nullptr, _IOFBF, 1 << 20);
            size_t c = 0; bool stopped = false; uint64_t n = 0, n_last = 0;
            auto lim = [&]() { return c < cuts.size() ? cuts[c] : ~0ull; };
            auto cross = [&](uint64_t off) {      // at/after the current cut: stop, or move to the next one
                while (!stopped && off >= lim()) { if (!dec.e9_pending()) { stopped = true; break; } c++; } };
            PtEvents ev; struct timespec ta{}; clock_gettime(CLOCK_MONOTONIC, &ta);
            ev.on_boundary = [&](char k, uint64_t off, uint64_t tsc, uint64_t ip) {
                cross(off); if (stopped) return;
                BndRec r{off, tsc, ip, (uint32_t)(unsigned char)k, (uint32_t)std::min<uint64_t>(dec.n_insn - n_last, 0xffffffffu)}; n_last = dec.n_insn; fwrite(&r, sizeof r, 1, f); n++; };
            ev.on_ptwrite = [](uint64_t, int, uint64_t) {}; ev.on_overflow = [](uint64_t) {};
            ev.on_resync = [](int, uint64_t) {}; ev.on_enable = [](uint64_t, bool) {};
            struct pt_insn insn; uint64_t tsc = 0; bool bail = false;
            const bool skip = !(getenv("PTRECON_SEG_NOSKIP") && *getenv("PTRECON_SEG_NOSKIP") == '1');
            while (!stopped && dec.next(insn, &tsc, ev)) {
                if (preinit && dec.e9_preinit_useless()) { bail = true; break; }
                if (dec.offset() >= lim()) cross(dec.offset());
                if (skip && !stopped) dec.skip_nonlast(~0ull);   // (same offset: same cut test)
            }
            fclose(f);
            if (bail || (preinit && dec.e9_preinit_useless())) continue;
            FILE* q = fopen((out + ".q").c_str(), "w"); if (q) { fprintf(q, "%zu\n", stopped ? c : cuts.size()); fclose(q); }
            if (getenv("PTRECON_SEG_TIMES")) { struct timespec tb{}; clock_gettime(CLOCK_MONOTONIC, &tb);
                fprintf(stderr, "segtime %s part 0 preinit %d: %.2f s\n", aux.c_str(), preinit, (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) / 1e9); }
            fprintf(stderr, "segment %s part 0: %lu instructions, %lu boundaries, stopped at cut %zu of %zu\n", aux.c_str(),
                    (unsigned long)dec.n_insn, (unsigned long)n, stopped ? c : cuts.size(), cuts.size());
            return 0;
        }
        return 0;
    }
    size_t lo = 0, hi = psb.size();                       // the last PSB strictly before `from'
    while (lo < hi) { size_t mid = (lo + hi) / 2; if (psb[mid] < from) lo = mid + 1; else hi = mid; }
    const uint64_t warm = lo ? psb[lo - 1] : 0;
    PtDecoder dec(aux, sb_, warm, 0, jitp, nullptr, 0);
    dec.no_time = false;
    { static const bool walk = getenv("PTRECON_SEG_WALK") != nullptr; static const bool noskip = getenv("PTRECON_SEG_NOSKIP") && *getenv("PTRECON_SEG_NOSKIP") == '1';
      dec.lazy_walk = jitp && !walk && !noskip; }   // JIT captures
    FILE* f = fopen(out.c_str(), "wb"); if (!f) { perror("segment out"); return 1; }
    setvbuf(f, nullptr, _IOFBF, 1 << 20);
    bool stopped = false; uint64_t n_last = 0;
    PtEvents ev;
    // 'S' (first instruction of the decode) is an artefact of starting mid-stream: the continuous
    // decoder reports no boundary there.  Everything else is keyed by offset like the --jobs partition.
    ev.on_boundary = [&](char k, uint64_t off, uint64_t tsc, uint64_t ip) {
        if (to && off >= to) { stopped = true; return; }
        if (k == 'S' || off < from) return;
        BndRec r{off, tsc, ip, (uint32_t)(unsigned char)k, (uint32_t)std::min<uint64_t>(dec.n_insn - n_last, 0xffffffffu)}; n_last = dec.n_insn; fwrite(&r, sizeof r, 1, f); };
    ev.on_ptwrite = [](uint64_t, int, uint64_t) {}; ev.on_overflow = [](uint64_t) {};
    ev.on_resync = [](int, uint64_t) {}; ev.on_enable = [](uint64_t, bool) {};
    struct pt_insn insn; uint64_t tsc = 0;
    const bool skip = !(getenv("PTRECON_SEG_NOSKIP") && *getenv("PTRECON_SEG_NOSKIP") == '1');
    struct timespec ta{}; clock_gettime(CLOCK_MONOTONIC, &ta);
    while (!stopped && dec.next(insn, &tsc, ev)) { if (to && dec.offset() >= to) break; if (skip) dec.skip_nonlast(~0ull); }
    fclose(f);
    if (getenv("PTRECON_SEG_TIMES")) { struct timespec tb{}; clock_gettime(CLOCK_MONOTONIC, &tb);   // diagnostic
        fprintf(stderr, "segtime %s from %lu to %lu: %.2f s, %lu insns, %lu resyncs\n", aux.c_str(), (unsigned long)from, (unsigned long)to,
                (tb.tv_sec - ta.tv_sec) + (tb.tv_nsec - ta.tv_nsec) / 1e9, (unsigned long)dec.n_insn, (unsigned long)dec.n_resync); }
    return 0;
}

// ---- the driver ------------------------------------------------------------------------------
int Recon::run_threads(const ReconOptions& o) {
    // phase timeline on stderr ("ptrecon --mt: t=<s> <phase>")
    struct timespec t00{}; clock_gettime(CLOCK_MONOTONIC, &t00);
    auto ph = [&](const char* what) { struct timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
        fprintf(stderr, "ptrecon --mt: t=%.2f %s\n", (t.tv_sec - t00.tv_sec) + (t.tv_nsec - t00.tv_nsec) / 1e9, what); };
    Sideband sb = Sideband::load(o.sideband);
    if (!sb.per_cpu()) {
        fprintf(stderr, "ptrecon --mt: %s is not a per-CPU capture (sideband version %d, no `cpus' list) -- "
                        "capture with pt_capture2 --cpu A [--cpu B ...]\n", o.sideband.c_str(), sb.sb_version);
        return 2;
    }
    const uint32_t pid = (uint32_t)sb.pid;
    const std::string base = o.out.empty() ? (o.summary.empty() ? std::string("ptrecon.mt") : o.summary) : o.out;
    const int ncpu = (int)sb.cpus.size();
    std::vector<std::string> aux_files; for (auto& c : sb.cpus) aux_files.push_back(c.aux);
    for (auto& c : sb.cpus) if (c.data_lost) fprintf(stderr, "warning: cpu %d lost %lu perf data records -- some switch records are missing\n", c.cpu, (unsigned long)c.data_lost);

    ph("setup done; segmentation");
    // the shared per-process Recon (see step 4) is built BEFORE the segmentation pass, whose
    // children use it too (they each parsed every site map: ~0.3 s of CPU per child, 96 children at n96).
    // for a JIT capture split into parts, the children get a sideband + code table of their own
    // (Recon::segment_part needs nothing else) and the shared Recon is built by the parent WHILE they run
    // (node_acorn: 12 s of site-map setup was serial before segmentation).  PTRECON_SEG_SERIALSETUP=1 = old order.
    const bool noshare = getenv("PTRECON_MT_NOSHARE") != nullptr;
    std::unique_ptr<Recon> shared;
    double t_shared = 0;
    auto build_shared = [&]() {
        if (noshare || shared) return;
        ReconOptions po = o; po.mt = false; po.jobs = 1; po.tid = 0; po.plan = nullptr;
        po.aux.clear(); po.skip_bytes = po.end_bytes = po.emit_from = 0;
        po.cvfile.clear(); po.cvfiles.clear();
        po.gt_in.clear(); po.gt_compare = false; po.gt_out.clear();
        po.out.clear(); po.summary.clear(); po.site_stats.clear(); po.cv_audit.clear();
        po.resync_log.clear(); po.text = false;
        // --decode-only never consults a site map or spec (the thread runs only the decoder loop): a JIT
        // capture's decode-only run skips their setup (node: 12 s of site-map JSON + tables, the decode row's critical
        // path).  Per-thread decoder output (instructions, resyncs, decode window) is identical; PTRECON_DECODE_FULLSETUP=1 = old.
        if (o.decode_only && !o.jitdumps.empty() && !getenv("PTRECON_DECODE_FULLSETUP")) { po.specs.clear(); po.sitemaps.clear(); po.orig_images.clear(); }
        struct timespec a{}, b{}; clock_gettime(CLOCK_MONOTONIC, &a);
        try { shared.reset(new Recon(po)); }
        catch (std::exception& e) { fprintf(stderr, "ptrecon --mt: shared setup failed (%s); "
                                            "falling back to per-thread construction\n", e.what()); shared.reset(); }
        clock_gettime(CLOCK_MONOTONIC, &b);
        t_shared = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
        if (shared) fprintf(stderr, "ptrecon --mt: shared per-process setup built once in %.2f s "
                                    "(was rebuilt in every thread child)\n", t_shared);
    };
    int parts = std::max(1, o.mt_jobs / std::max(1, ncpu));
    // JIT captures over-decompose 3x (PTRECON_SEG_OVER; equal-byte parts hold 0.3-1.4 G instructions on java_scrabble: the
    // densest part was the pass's critical path; the region dump is identical at 24 and 96 parts per core)
    if (!o.jitdumps.empty()) { int ov = 3; if (const char* e = getenv("PTRECON_SEG_OVER")) ov = std::max(1, atoi(e));
                               parts = std::max(1, ov * o.mt_jobs / std::max(1, ncpu)); }
    if (const char* e = getenv("PTRECON_SEG_PARTS")) parts = std::max(1, atoi(e));
    // JIT captures are split too (a part's decoder takes its time key from the PSB+TSC it starts at);
    // PTRECON_JIT_SEG1=1 = one pass per core, as before
    if (!o.jitdumps.empty() && getenv("PTRECON_JIT_SEG1")) parts = 1;
    const bool overlap = !o.jitdumps.empty() && !noshare && parts > 1 && !getenv("PTRECON_SEG_SERIALSETUP")
                         && !getenv("PTRECON_MT_REUSE_BND") && !getenv("PTRECON_SEG_OWNRECON");
    JitTable segjit; JitTable* sjit = nullptr;
    if (overlap) { Recon::build_jit_table(o.jitdumps, sb, segjit, true); if (!segjit.empty()) sjit = &segjit; ph("segmentation code table built"); }
    else build_shared();
    // ---- 1. segment every core's stream (parallel children) ---------------------------------
    // one decoder built here first, so the merged-trampoline memfds (PtDecoder::
    // merge_trampolines, a per-process cache keyed by the mappings) exist before the fork and every
    // segmentation child inherits them instead of copying ~2.6 GB of sparse span itself.
    // JIT captures too, with the shared Recon's code table (the merge skips the mappings it serves)
    JitTable* wjit = overlap ? sjit : (!o.jitdumps.empty() && shared && shared->have_jit_) ? &shared->jit_ : nullptr;
    if ((o.jitdumps.empty() || wjit) && ncpu > 0 && !getenv("PTRECON_SEG_NOWARM")) {
        try { for (int i = 0; i < ncpu; i++) if (file_size(sb.cpus[i].aux)) { PtDecoder w(sb.cpus[i].aux, sb, 0, 0, wjit, nullptr, 0); break; } }
        catch (std::exception&) {} }
    std::vector<std::string> bnd_files(ncpu);
    // a core's pass is split into PSB-aligned parts decoded in parallel (PTRECON_SEG_PARTS,
    // default max(1, mt_jobs / cores)), concatenated in order; part 0 keeps the pre-init logic.
    // diagnostic: PTRECON_MT_REUSE_BND=1 reuses the boundary files a --keep-thread-files run left (same base)
    bool reuse_bnd = getenv("PTRECON_MT_REUSE_BND") != nullptr;
    if (reuse_bnd) for (int i = 0; i < ncpu; i++) { bnd_files[i] = with_suffix(base, ".cpu" + std::to_string(sb.cpus[i].cpu) + ".bnd");
        struct stat stb; if (stat(bnd_files[i].c_str(), &stb)) reuse_bnd = false; }
    if (reuse_bnd) fprintf(stderr, "ptrecon --mt: PTRECON_MT_REUSE_BND: reusing the boundary files of an earlier run\n");
    else if (parts > 1) {
      struct Part { int cpu; size_t p; uint64_t from, to; std::string out; };
      std::vector<std::vector<uint64_t>> cuts(ncpu), psbs(ncpu); std::vector<Part> todo;
      { std::vector<std::thread> pw;   // the per-core PSB scans (java: 4 x 2 GB, 3.5 s serial) in parallel; memoized
        for (int i = 0; i < ncpu; i++) if (file_size(sb.cpus[i].aux)) pw.emplace_back([&sb, i]() { PtDecoder::prewarm_file(sb.cpus[i].aux); });
        for (auto& t : pw) t.join(); }
      for (int i = 0; i < ncpu; i++) {
          bnd_files[i] = with_suffix(base, ".cpu" + std::to_string(sb.cpus[i].cpu) + ".bnd");
          const uint64_t fsz = file_size(sb.cpus[i].aux);
          if (fsz) psbs[i] = PtDecoder::psb_offsets(sb.cpus[i].aux);    // (memoized: computed in parallel just above)
          for (int k = 1; k < parts && psbs[i].size() > 2; k++) {
              const uint64_t want = fsz * (uint64_t)k / (uint64_t)parts;
              auto it = std::lower_bound(psbs[i].begin(), psbs[i].end(), want);
              if (it == psbs[i].end() || it == psbs[i].begin()) continue;
              if (cuts[i].empty() || *it > cuts[i].back()) cuts[i].push_back(*it);
          }
          todo.push_back({i, 0, 0, 0, bnd_files[i] + ".p0"});
          for (size_t k = 0; k < cuts[i].size(); k++)
              todo.push_back({i, k + 1, cuts[i][k], k + 1 < cuts[i].size() ? cuts[i][k + 1] : 0, bnd_files[i] + ".p" + std::to_string(k + 1)});
      }
      std::vector<pid_t> kids;
      for (auto& t : todo) {
          if (!file_size(sb.cpus[t.cpu].aux)) { if (t.p == 0) { FILE* f = fopen(t.out.c_str(), "wb"); if (f) fclose(f); FILE* q = fopen((t.out + ".q").c_str(), "w"); if (q) { fprintf(q, "0\n"); fclose(q); } } continue; }
          pid_t p = fork(); if (p < 0) { perror("fork"); return 2; }
          if (p == 0) {
              ReconOptions co = o; co.mt = false; co.plan = nullptr; co.aux = sb.cpus[t.cpu].aux; co.out.clear(); co.summary.clear(); co.site_stats.clear(); co.gt_in.clear(); co.gt_compare = false;
              int rc = 1;
              // 3 parts per worker are runnable at once; the parent building the shared Recon must not get a
              // quarter of a cpu (node: its setup IS the critical path) -> the parts run at a lower priority
              if (overlap && !getenv("PTRECON_SEG_NONICE")) { if (setpriority(PRIO_PROCESS, 0, 10)) {} }
              try { if (overlap) rc = Recon::segment_part(sb, sjit, sb.cpus[t.cpu].aux, t.out, t.from, t.to, cuts[t.cpu], psbs[t.cpu]);
                    else if (shared && !getenv("PTRECON_SEG_OWNRECON")) rc = shared->run_segment_part(sb.cpus[t.cpu].aux, t.out, t.from, t.to, cuts[t.cpu], psbs[t.cpu]); else { Recon r(co); rc = r.run_segment_part(sb.cpus[t.cpu].aux, t.out, t.from, t.to, cuts[t.cpu], psbs[t.cpu]); } } catch (std::exception& e) { fprintf(stderr, "ptrecon --mt segment cpu %d part %zu: %s\n", sb.cpus[t.cpu].cpu, t.p, e.what()); }
              _exit(rc);
          }
          kids.push_back(p);
      }
      if (overlap) { ph("segmentation parts forked; building the shared setup"); build_shared(); ph("shared setup done"); }
      int bad = 0; for (pid_t p : kids) { int st = 0; waitpid(p, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad++; }
      if (bad) { fprintf(stderr, "ptrecon --mt: %d segmentation parts failed\n", bad); return 1; }
      for (int i = 0; i < ncpu; i++) {      // concatenate: part 0 up to its stop cut q, then parts q..
          size_t q = cuts[i].size(); { FILE* f = fopen((bnd_files[i] + ".p0.q").c_str(), "r"); if (f) { if (fscanf(f, "%zu", &q) != 1) q = cuts[i].size(); fclose(f); } remove((bnd_files[i] + ".p0.q").c_str()); }
          FILE* out = fopen(bnd_files[i].c_str(), "wb"); if (!out) { perror("bnd"); return 1; }
          std::vector<char> buf(1 << 20);
          for (size_t k = 0; k <= cuts[i].size(); k++) {
              const std::string pf = bnd_files[i] + ".p" + std::to_string(k);
              if (k == 0 || k > q) { FILE* in = fopen(pf.c_str(), "rb"); if (in) { size_t n; while ((n = fread(buf.data(), 1, buf.size(), in)) > 0) fwrite(buf.data(), 1, n, out); fclose(in); } }
              remove(pf.c_str());
          }
          fclose(out);
      }
    } else
    { std::vector<pid_t> kids;
      for (int i = 0; i < ncpu; i++) {
          bnd_files[i] = with_suffix(base, ".cpu" + std::to_string(sb.cpus[i].cpu) + ".bnd");
          pid_t p = fork(); if (p < 0) { perror("fork"); return 2; }
          if (p == 0) {
              ReconOptions co = o; co.mt = false; co.plan = nullptr; co.aux = sb.cpus[i].aux; co.out.clear(); co.summary.clear(); co.site_stats.clear(); co.gt_in.clear(); co.gt_compare = false;
              int rc = 1;
              try { if (shared && !getenv("PTRECON_SEG_OWNRECON")) rc = shared->run_segment(sb.cpus[i].aux, bnd_files[i]); else { Recon r(co); rc = r.run_segment(sb.cpus[i].aux, bnd_files[i]); } } catch (std::exception& e) { fprintf(stderr, "ptrecon --mt segment cpu %d: %s\n", sb.cpus[i].cpu, e.what()); }
              _exit(rc);
          }
          kids.push_back(p);
      }
      int bad = 0; for (pid_t p : kids) { int st = 0; waitpid(p, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad++; }
      if (bad) { fprintf(stderr, "ptrecon --mt: %d of %d segmentation passes failed\n", bad, ncpu); return 1; } }

    ph("segmentation done");
    // ---- 2. attribute the regions -------------------------------------------------------------
    std::vector<Region> regions;
    struct CpuStat { uint64_t regions = 0, target = 0, other = 0, unattributed = 0, inherited = 0, slack = 0, lossy = 0, conflict = 0, realigned = 0, bytes_target = 0, bytes_other = 0, bytes_unattr = 0, sw = 0; int64_t min_gap = INT64_MAX; };
    std::vector<CpuStat> cs(ncpu);
    std::set<uint32_t> other_pids;
    for (int i = 0; i < ncpu; i++) {
        std::vector<SwRec> sw, swall; load_switches(sb.cpus[i].switch_file, sw, swall);
        std::vector<BndRec> bn = load_bnd(bnd_files[i]);
        const uint64_t fsz = file_size(sb.cpus[i].aux);
        cs[i].sw = sw.size();
        if (sw.empty()) fprintf(stderr, "warning: cpu %d has no switch-in records (%s) -- its regions cannot be attributed\n", sb.cpus[i].cpu, sb.cpus[i].switch_file.c_str());
        std::vector<Region> rs;
        for (size_t k = 0; k < bn.size(); k++) {
            Region r; r.cpu = i; r.begin = bn[k].off; r.end = k + 1 < bn.size() ? bn[k + 1].off : fsz; r.tsc = bn[k].tsc; r.ip = bn[k].ip; r.kind = (char)bn[k].kind;
            r.insns = k + 1 < bn.size() ? bn[k + 1].pad : 0;
            r.lossy = (r.kind == 'O' || r.kind == 'R');
            if (r.end < r.begin) r.end = r.begin;
            if (r.tsc == 0) {                                 // no time yet: inherit the previous region's thread
                if (!rs.empty()) { r.pid = rs.back().pid; r.tid = rs.back().tid; r.tsc = rs.back().tsc; r.inherited = true; }
            } else if (!sw.empty()) {
                // last switch-in with tsc <= T + slack
                const uint64_t T = r.tsc + o.sw_slack_tsc;
                size_t lo = 0, hi = sw.size();
                while (lo < hi) { size_t mid = (lo + hi) / 2; if (sw[mid].tsc <= T) lo = mid + 1; else hi = mid; }
                // Consistency: by the records, was that task still on the core at T?  The last
                // record at or before T must be its switch-in, not a later switch-out / idle
                // switch-in.
                bool left = false;
                if (lo) { size_t alo = 0, ahi = swall.size();
                          while (alo < ahi) { size_t mid = (alo + ahi) / 2; if (swall[mid].tsc <= T) alo = mid + 1; else ahi = mid; }
                          if (alo) { const SwRec& last = swall[alo - 1]; const SwRec& s = sw[lo - 1];
                                     left = !((last.flags & SW_IN) && last.pid == s.pid && last.tid == s.tid) && last.tsc > s.tsc; } }
                // `left' means the region's start time falls in the GAP between a
                // switch-OUT and the next switch-in: by the records no task of the process was on
                // the core then, yet the region is user code of the process, so the time and the
                // records disagree.  Which of the two is in doubt?  Only the time: libipt's TSC at
                // a TIP.PGE is a LOWER BOUND (it is the last TSC/MTC/CYC the decoder has seen, and
                // the wall clock has moved on since), so the region's true start is at or after
                // `r.tsc' -- never before it.  The thread that switched OUT before `r.tsc' is
                // therefore EXCLUDED, and the only candidate left is the NEXT switch-in.  Taking
                // the last switch-in instead handed a departed thread a stretch of the thread that
                // replaced it: on memcached mc3 exactly ONE region did this (cpu2 [2007167,2007405),
                // tsc ...894608, 1267 ticks after worker 2086503's switch-out and 44 ticks before
                // 2086498's switch-in), and it cost 2086503 54 345 unknown + 16 007 wrong + 61 323
                // gt-only -- the foreign stretch also consumed 90 cv values the thread never logged,
                // so its positional value cursor stayed 90 ahead until the next sync marker.
                // The correction is bounded by construction (the next switch-in is 19-134 tsc ahead
                // on that capture, i.e. clock noise) and it can only ever move a region FORWARD.
                if (left && lo < sw.size()) {
                    const SwRec& s = sw[lo];
                    r.pid = s.pid; r.tid = s.tid; r.gap = (int64_t)r.tsc - (int64_t)s.tsc; r.realigned = true; r.slack_used = true;
                } else if (lo) {
                    const SwRec& s = sw[lo - 1]; r.pid = s.pid; r.tid = s.tid; r.gap = (int64_t)r.tsc - (int64_t)s.tsc; r.slack_used = s.tsc > r.tsc;
                    if (r.gap < cs[i].min_gap) cs[i].min_gap = r.gap;
                    if (left) r.conflict = true;   // no switch-in after it: the best guess stands
                }
            }
            // A "conflict" is a region whose start time disagrees with the switch records.  When
            // the region continues the SAME thread as the region immediately before it (and is
            // contiguous with it in the file) the attribution is not in doubt -- so it is
            // not counted as an attribution conflict.
            if (r.conflict && !rs.empty() && rs.back().pid == r.pid && rs.back().tid == r.tid && rs.back().end == r.begin)
                r.conflict = false;
            rs.push_back(r);
        }
        // Merging adjacent regions of one thread HERE, per CPU, would be wrong on a per-CPU
        // capture: two regions of thread T that are adjacent in cpu K's AUX
        // FILE are adjacent in K's trace, not in T's EXECUTION -- between them T may have run for
        // a long time on another core, whose regions live in another file.  The merged region
        // keeps the FIRST one's timestamp, so the plan (sorted by tsc) would replay a later
        // stretch of the thread at an earlier position: on memcached -t 4 worker 2070279 executed
        // markers with payloads 159757, 163853, 167950, 172046 ... but the plan decoded them
        // 159757, 172046, 176143, 163853 -- a cpu2 block that had been merged across a 0.7 ms gap
        // during which the thread ran on cpu3.  Out-of-order replay makes the positional cv cursor
        // run ahead (+6304) and then behind (-9792) and the registers it anchors stale: 39 966 of
        // that thread's 1 554 356 compared records were wrong, with 77 157 gt-only and 3 756
        // recon-only.  The merge therefore happens in step 3, on the thread's OWN time-sorted region
        // list, where "adjacent" means adjacent IN THE THREAD'S EXECUTION.
        std::vector<Region>& merged = rs;
        if (false) for (auto& r : rs) {
            // NEVER merge a LOSSY boundary ('O' overflow / 'R' decoder resync) into the
            // region before it, even when the thread is the same.  Merging hid the loss: the
            // plan then held one non-lossy segment, `next_plan' fired no switch-loss/overflow
            // hook at the gap, and the reconstruction carried its registers and its POSITIONAL
            // cv cursor straight across instructions it never saw -- so the values logged inside
            // the gap were never consumed (the cursor ran behind until the next sync marker
            // pulled it straight) and every address derived from a register that the gap had
            // changed was reported as a confident WRONG address instead of an unknown one.
            // Measured on memcached -t 4 (cpu2, 9 decoder resyncs): worker 4060747 lost the
            // seven-instruction prologue of `event_base_loop' -- its ONE anchor of the
            // `event_base *' -- to a resync inside its own region, and reported 51 813 wrong
            // addresses (3.5 % of the thread) from that stale pointer.
            (void)r;
        }
        for (auto& r : merged) {
            cs[i].regions++;
            const uint64_t bytes = r.end - r.begin;
            if (r.tid == 0) { cs[i].unattributed++; cs[i].bytes_unattr += bytes; }
            else if (r.pid != pid) { cs[i].other++; cs[i].bytes_other += bytes; other_pids.insert(r.pid); }
            else { cs[i].target++; cs[i].bytes_target += bytes; if (r.lossy) cs[i].lossy++; if (r.slack_used) cs[i].slack++; if (r.inherited) cs[i].inherited++; if (r.conflict) cs[i].conflict++; if (r.realigned) cs[i].realigned++; }
        }
        regions.insert(regions.end(), merged.begin(), merged.end());
        if (!o.keep_thread_files) remove(bnd_files[i].c_str());
    }
    if (!o.sw_dump.empty()) {
        FILE* f = fopen(o.sw_dump.c_str(), "w");
        if (f) { fprintf(f, "# cpu begin end kind tsc pid tid lossy slack_used inherited conflict realigned gap_tsc ip\n");
                 for (auto& r : regions) fprintf(f, "%d %lu %lu %c %lu %u %u %d %d %d %d %d %ld %#lx\n", sb.cpus[r.cpu].cpu, (unsigned long)r.begin, (unsigned long)r.end, r.kind, (unsigned long)r.tsc, r.pid, r.tid, (int)r.lossy, (int)r.slack_used, (int)r.inherited, (int)r.conflict, (int)r.realigned, (long)r.gap, (unsigned long)r.ip);
                 fclose(f); }
        if (getenv("PTRECON_MT_SEGONLY")) { fprintf(stderr, "ptrecon --mt: PTRECON_MT_SEGONLY: stopping after the region dump\n"); return 0; }
    }
    for (int i = 0; i < ncpu; i++)
        fprintf(stderr, "cpu %d: %lu switch-ins, %lu regions: %lu of pid %u (%.1f MB, %lu lossy, %lu needed slack, %lu inherited, %lu conflicts, %lu gap-realigned), %lu of other pids (%.1f MB), %lu unattributed (%.1f MB); min start-after-switch gap %ld tsc\n",
                sb.cpus[i].cpu, (unsigned long)cs[i].sw, (unsigned long)cs[i].regions, (unsigned long)cs[i].target, pid, cs[i].bytes_target / 1e6,
                (unsigned long)cs[i].lossy, (unsigned long)cs[i].slack, (unsigned long)cs[i].inherited, (unsigned long)cs[i].conflict, (unsigned long)cs[i].realigned,
                (unsigned long)cs[i].other, cs[i].bytes_other / 1e6, (unsigned long)cs[i].unattributed, cs[i].bytes_unattr / 1e6,
                cs[i].min_gap == INT64_MAX ? 0L : (long)cs[i].min_gap);

    // ---- 3. per-thread plans -----------------------------------------------------------------
    std::map<uint32_t, ThreadPlan> plans; std::map<uint32_t, uint64_t> insn_of;   // segmentation's instruction estimate
    for (auto& r : regions) {
        if (r.pid != pid || r.tid == 0 || r.end <= r.begin) continue;
        insn_of[r.tid] += r.insns;
        ThreadPlan& p = plans[r.tid]; p.tid = r.tid; p.files = aux_files;
        AuxSeg s; s.file = r.cpu; s.begin = r.begin; s.end = r.end; s.tsc = r.tsc; s.lossy = r.lossy;
        p.segs.push_back(s);
    }
    for (auto& kv : plans) {
        std::vector<AuxSeg>& v = kv.second.segs;
        std::stable_sort(v.begin(), v.end(), [](const AuxSeg& a, const AuxSeg& b) { return a.tsc < b.tsc; });
        // Merge only what is adjacent in THIS THREAD's execution -- consecutive in its own
        // time-sorted plan, in the same file, contiguous in that file -- and never across a lossy
        // ('O'/'R') boundary.  That is the syscall PGD/PGE pair and the preemption that
        // came straight back, and it cannot reorder the thread's own execution.
        std::vector<AuxSeg> m;
        for (auto& s2 : v) {
            if (!m.empty() && !s2.lossy && m.back().file == s2.file && m.back().end == s2.begin) { m.back().end = s2.end; continue; }
            m.push_back(s2);
        }
        v.swap(m);
    }
    std::map<uint32_t, uint64_t> fs_of; for (auto& t : sb.threads) fs_of[t.tid] = t.fs_base;
    if (o.tid) { auto it = plans.find(o.tid); if (it == plans.end()) { fprintf(stderr, "ptrecon --mt: no regions for --tid %u\n", o.tid); return 1; }
                 ThreadPlan one = it->second; plans.clear(); plans[o.tid] = one; }
    fprintf(stderr, "ptrecon --mt: pid %u, %zu threads with regions (%zu in the sideband's thread list)%s\n", pid, plans.size(), sb.threads.size(),
            other_pids.empty() ? "" : (", " + std::to_string(other_pids.size()) + " other pids shared the cores").c_str());
    for (auto& kv : plans) {
        uint64_t bytes = 0, lossy = 0; std::set<int> cpus; for (auto& s : kv.second.segs) { bytes += s.end - s.begin; lossy += s.lossy; cpus.insert(s.file); }
        fprintf(stderr, "  tid %u: %zu segments (%lu lossy) on %zu core(s), %.1f MB, fs_base %#lx%s\n", kv.first, kv.second.segs.size(), (unsigned long)lossy, cpus.size(), bytes / 1e6,
                (unsigned long)(fs_of.count(kv.first) ? fs_of[kv.first] : 0), fs_of.count(kv.first) ? "" : "  [not in the sideband's thread list]");
    }
    if (plans.empty()) { fprintf(stderr, "ptrecon --mt: nothing to reconstruct\n"); return 1; }

    // ---- per-thread inputs: cv files, ground truth ---------------------------------------------
    std::vector<CvInfo> cvs;
    { std::vector<std::string> paths = o.cvfiles;
      if (!o.cv_dir.empty()) for (auto& p : glob_files(o.cv_dir + "/cv.*.bin")) paths.push_back(p);
      std::sort(paths.begin(), paths.end()); paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
      for (auto& p : paths) cvs.push_back(cv_info(p)); }
    auto cv_for = [&](uint32_t tid) -> std::string {
        for (auto& c : cvs) if (c.magic == 0x56435450u /* PTCV */ && c.tid == tid) return c.path;
        for (auto& c : cvs) if (c.magic == 0x57435450u /* PTCW */) return c.path;
        return "";
    };
    std::string gt_dir = o.gt_dir;
    if (gt_dir.empty() && !o.gt_in.empty()) { struct stat st; if (!stat(o.gt_in.c_str(), &st) && S_ISDIR(st.st_mode)) gt_dir = o.gt_in; }
    auto gt_for = [&](uint32_t tid) -> std::string {
        if (gt_dir.empty()) return "";
        std::string p = gt_dir + "/gt." + std::to_string(pid) + "." + std::to_string(tid) + ".bin";
        struct stat st; return stat(p.c_str(), &st) ? std::string() : p;
    };

    ph("plans done");
    // ---- 4. reconstruct every thread (forked children, mt_jobs at a time) --------------------
    std::vector<uint32_t> tids; for (auto& kv : plans) tids.push_back(kv.first);
    if (!o.mt_tids.empty()) {                       // --mt-tid: reconstruct only the named threads
        std::vector<uint32_t> keep;
        for (uint32_t t : o.mt_tids) if (plans.count(t)) keep.push_back(t); else fprintf(stderr, "ptrecon --mt: --mt-tid %u has no regions in this capture\n", t);
        tids.swap(keep);
        fprintf(stderr, "ptrecon --mt: --mt-tid selected %zu of %zu thread(s)\n", tids.size(), plans.size());
        if (tids.empty()) return 1;
    }
    int jobs = o.mt_jobs > 0 ? o.mt_jobs : (o.jobs > 1 ? o.jobs : std::min<int>((int)tids.size(), 8));
    // --mt-split N -- every thread runs at once and a thread's stream is split into chunks
    // (Recon::run_parallel on its plan), N workers shared in proportion to the thread's bytes.
    std::map<uint32_t, int> split_of;
    if (o.mt_split > 1) {
        uint64_t sum = 0; std::map<uint32_t, uint64_t> by;
        for (uint32_t t : tids) { uint64_t b = Recon::plan_total(plans[t]); by[t] = b; sum += b; plans[t].index(); }
        // JIT captures: allocate by estimated WORK = segmentation's instruction count + a per-segment cost
        // (each plan segment is a seek + sync; ~3000 instruction-equivalents, measured on java_scrabble's lossy threads).
        // Bytes were a poor proxy there: 700 MB threads of 0.4 G instructions got 7 workers each while 2 G-instruction
        // threads of 110 MB got none.  OPT-IN (PTRECON_MT_SPLIT_WORK=1): faster on Java (-7 %), but
        // splitting the well-anchored 2 G-instruction threads costs chunk-start state: unknown 1.839 G -> 1.887 G.
        // (the decode-only run splits exactly like the full run, so full - decode = execution)
        // PTRECON_MT_SPLIT_WORK=2 (opt-in): by work, but a thread the byte rule leaves unsplit stays unsplit
        // (newly split well-anchored threads cost unknowns); its workers go to the other threads by work.
        const char* sw = getenv("PTRECON_MT_SPLIT_WORK");
        std::set<uint32_t> pin1;
        if (!o.jitdumps.empty() && sw && *sw == '2') {
            std::map<uint32_t, int> sb; for (uint32_t t : tids) sb[t] = 1;
            for (int left = o.mt_split - (int)tids.size(); left > 0; left--) {
                uint32_t best = 0; double bv = -1;
                for (uint32_t t : tids) { const double v = (double)by[t] / sb[t]; if (v > bv) { bv = v; best = t; } }
                if (bv <= 0) break; sb[best]++; }
            for (uint32_t t : tids) if (sb[t] == 1) pin1.insert(t);
        }
        if (!o.jitdumps.empty() && sw && (*sw == '1' || *sw == '2')) {
            uint64_t tot = 0; for (uint32_t t : tids) tot += insn_of[t];
            if (tot) { sum = 0; for (uint32_t t : tids) { const uint64_t c = insn_of[t] + 3000ull * plans[t].segs.size(); by[t] = c; sum += c; } }
        }
        // greedy min-max -- every thread starts with one chunk worker and each further worker goes to the
        // thread with the most bytes per worker (proportional rounding left 50-70 s JIT worker threads unsplit
        // next to a main thread split 8 ways).  PTRECON_MT_SPLIT_PROP=1 = the proportional rule.
        if (getenv("PTRECON_MT_SPLIT_PROP"))
            for (uint32_t t : tids) { int j = sum ? (int)((double)o.mt_split * (double)by[t] / (double)sum + 0.5) : 1; split_of[t] = j < 1 ? 1 : j; }
        else {
            for (uint32_t t : tids) split_of[t] = 1;
            for (int left = o.mt_split - (int)tids.size(); left > 0; left--) {
                uint32_t best = 0; double bv = -1;
                for (uint32_t t : tids) { if (pin1.count(t)) continue; const double v = (double)by[t] / split_of[t]; if (v > bv) { bv = v; best = t; } }
                if (bv <= 0) break;
                split_of[best]++;
            }
        }
        jobs = (int)tids.size();
        fprintf(stderr, "ptrecon --mt-split %d:", o.mt_split);
        for (uint32_t t : tids) if (split_of[t] > 1) fprintf(stderr, " tid %u x%d", t, split_of[t]);
        fprintf(stderr, "\n");
    }
    if (jobs < 1) jobs = 1;
    auto tpath = [&](uint32_t tid, const char* suf) { return base + ".tid" + std::to_string(tid) + suf; };
    // every plan-mode decoder (each thread, each --mt-split chunk) opened the per-core files with a full
    // PSB+ repair scan, and split planning re-scanned their PSBs: fill both memos here, once, before the forks
    // (JIT captures; PTRECON_MT_NOPREWARM=1 = old)
    if (!o.jitdumps.empty() && !getenv("PTRECON_MT_NOPREWARM")) {
        std::vector<std::thread> pw; for (auto& c : sb.cpus) if (file_size(c.aux)) pw.emplace_back([&c]() { PtDecoder::prewarm_file(c.aux); });
        for (auto& t : pw) t.join(); ph("per-core files prewarmed"); }
    std::map<pid_t, uint32_t> running; size_t next = 0; int bad = 0;
    std::vector<uint32_t> failed_tids;
    // ---- ONE Recon for the whole process ---------------------------------------------------
    // Everything Recon's constructor builds -- both site-map JSONs, the rebased site tables, the
    // interval map over every mapping, the trampoline map, the original-code index -- is a
    // property of the PROCESS, not of the thread, so a thread child need not rebuild it.
    // Build it ONCE here, with no thread named (no plan, no cv file, no gt ring), and let each
    // child `retarget_thread()' it; fork shares the tables physically, copy-on-write.
    // PTRECON_MT_NOSHARE=1 is the ablation: construct per child, as before.
    while (next < tids.size() || !running.empty()) {
        while (next < tids.size() && (int)running.size() < jobs) {
            uint32_t tid = tids[next++];
            pid_t p = fork(); if (p < 0) { perror("fork"); return 2; }
            if (p == 0) {
                ReconOptions co = o; co.mt = false; co.jobs = 1; co.tid = tid; co.plan = &plans[tid]; co.aux.clear();
                co.skip_bytes = co.end_bytes = co.emit_from = 0;
                // TLS base: the kernel's value for this thread (sideband `threads'); a worker thread
                // has it from its first instruction (CLONE_SETTLS), so it is seeded from the start;
                // the main thread keeps the single-thread rule (0 until the loader's arch_prctl).
                auto fit = fs_of.find(tid);
                if (fit != fs_of.end() && fit->second) co.fs_base = fit->second;
                co.seed_fs = (tid != pid) && !o.no_seed_fs;
                co.cvfile = cv_for(tid); co.cvfiles.clear();
                co.gt_in = gt_for(tid); co.gt_compare = !co.gt_in.empty();
                co.gt_out = (co.gt_compare && !o.gt_out.empty()) ? tpath(tid, ".gt.mtrace") : std::string();
                co.out = o.out.empty() ? std::string() : tpath(tid, ".mtrace");
                co.summary = tpath(tid, ".json");
                co.site_stats = o.site_stats.empty() ? std::string() : tpath(tid, ".sites.csv");
                co.cv_audit = o.cv_audit.empty() ? std::string() : tpath(tid, ".cvaudit");
                co.resync_log = o.resync_log.empty() ? std::string() : tpath(tid, ".resync");
                co.text = false;
                int rc = 1;
                struct timespec a{}, b{}, c{}; clock_gettime(CLOCK_MONOTONIC, &a);
                try {
                    if (split_of.count(tid) && split_of[tid] > 1 && co.gt_in.empty()) {
                        co.jobs = split_of[tid]; clock_gettime(CLOCK_MONOTONIC, &b);
                        // the main thread borrows too (its own setup was 14 s of node's critical path); its records are
                        // identical either way.  PTRECON_MT_SPLIT_OWNMAIN=1 = its own setup.
                        { extern Recon* g_par_prebuilt; g_par_prebuilt = (tid != pid || !getenv("PTRECON_MT_SPLIT_OWNMAIN")) ? shared.get() : nullptr; }
                        rc = Recon::run_parallel(co); }
                    else if (shared) { shared->retarget_thread(co); clock_gettime(CLOCK_MONOTONIC, &b); rc = shared->run(); }
                    else { Recon r(co); clock_gettime(CLOCK_MONOTONIC, &b); rc = r.run(); }
                } catch (std::exception& e) { fprintf(stderr, "ptrecon --mt tid %u: %s\n", tid, e.what()); rc = 1; }
                clock_gettime(CLOCK_MONOTONIC, &c);
                fprintf(stderr, "ptrecon --mt tid %u: setup %.2f s, run %.2f s\n", tid,
                        (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9,
                        (c.tv_sec - b.tv_sec) + (c.tv_nsec - b.tv_nsec) / 1e9);
                { void vx_opstat_dump(); vx_opstat_dump(); }
                _exit(rc);
            }
            running[p] = tid;
        }
        int st = 0; pid_t p = waitpid(-1, &st, 0);
        if (p < 0) break;
        auto it = running.find(p); if (it == running.end()) continue;
        if (!WIFEXITED(st) || WEXITSTATUS(st)) { bad++; failed_tids.push_back(it->second); fprintf(stderr, "ptrecon --mt: thread %u failed (status %d)\n", it->second, st); }
        running.erase(it);
    }
    if (bad) {
        // A failed thread must NOT throw away the other N-1: the run is a COMPLETENESS
        // measurement, and a missing aggregate is worse than an aggregate that says how
        // many threads it is missing.  The failures are dropped from the merge, recorded
        // in the summary as `threads_failed'/`threads_failed_tids', printed here, and the
        // exit status is still non-zero -- but only after the summary has been written.
        fprintf(stderr, "ptrecon --mt: %d of %zu threads failed; they are EXCLUDED from the "
                "totals below and listed as threads_failed_tids in the summary\n", bad, tids.size());
        std::vector<uint32_t> keep;
        for (uint32_t t : tids)
            if (std::find(failed_tids.begin(), failed_tids.end(), t) == failed_tids.end()) keep.push_back(t);
        tids.swap(keep);
    }

    ph("threads done");
    // ---- 5. merge the per-thread traces by time --------------------------------------------
    uint64_t out_records = 0, no_ts = 0;
    if (!o.out.empty()) {
        struct Rd { FILE* f; mtrace_rec cur; bool has; uint32_t tid; std::vector<mtrace_rec> buf; size_t pos; };
        std::vector<Rd> rds;
        for (uint32_t tid : tids) {
            std::string cp = tpath(tid, ".mtrace"); FILE* f = fopen(cp.c_str(), "rb");
            if (!f) { fprintf(stderr, "ptrecon --mt: missing %s\n", cp.c_str()); continue; }
            fseek(f, sizeof(mtrace_hdr), SEEK_SET);
            Rd r; r.f = f; r.has = false; r.tid = tid; r.pos = 0; r.buf.resize(1 << 14); r.buf.clear();
            rds.push_back(std::move(r));
        }
        auto refill = [&](Rd& r) -> bool {
            if (r.pos < r.buf.size()) { r.cur = r.buf[r.pos++]; r.has = true; return true; }
            r.buf.resize(1 << 14); size_t n = fread(r.buf.data(), sizeof(mtrace_rec), r.buf.size(), r.f); r.buf.resize(n); r.pos = 0;
            if (!n) { r.has = false; return false; }
            r.cur = r.buf[r.pos++]; r.has = true; return true;
        };
        for (auto& r : rds) refill(r);
        FILE* out = fopen(o.out.c_str(), "wb"); if (!out) { perror("out"); return 1; }
        setvbuf(out, nullptr, _IOFBF, 1 << 20);
        mtrace_hdr h{}; h.magic = MTRACE_MAGIC; h.version = MTRACE_VERSION; h.tid_count = (uint32_t)tids.size(); h.flags = MTH_TIME_MERGED | (o.output_stride > 1 ? MTH_SAMPLED_OUTPUT : 0);
        fwrite(&h, sizeof h, 1, out);
        // k-way merge on (ts, tid); a record with ts == 0 (no timing) is emitted in file order
        // relative to its own thread and ahead of any timed record, so per-thread order is kept.
        using Key = std::pair<uint64_t, size_t>;
        std::priority_queue<Key, std::vector<Key>, std::greater<Key>> pq;
        for (size_t i = 0; i < rds.size(); i++) if (rds[i].has) pq.push({rds[i].cur.ts, i});
        std::set<uint32_t> tids_seen;
        while (!pq.empty()) {
            Key k = pq.top(); pq.pop(); Rd& r = rds[k.second];
            if (r.cur.ts == 0) no_ts++;
            tids_seen.insert(r.cur.tid);
            fwrite(&r.cur, sizeof r.cur, 1, out); out_records++;
            if (refill(r)) pq.push({r.cur.ts, k.second});
        }
        // tid_count = the distinct tids IN THE FILE (a thread that ran no instrumented code --
        // the buffer sink's own drain thread -- has a plan but no record).
        h.tid_count = (uint32_t)tids_seen.size();
        fseek(out, 0, SEEK_SET); fwrite(&h, sizeof h, 1, out);
        fclose(out);
        for (auto& r : rds) { fclose(r.f); if (!o.keep_thread_files) remove(tpath(r.tid, ".mtrace").c_str()); }
    }

    // ---- summary: totals + per thread + per cpu ------------------------------------------------
    static const char* SUM[] = {"records", "unknown_addr", "instructions", "pt_overflows", "resyncs", "sync_failures", "psb_repaired", "output_records",
                                "ptw_packets", "ptw_used", "ptw_unconsumed", "lift_failures", "lift_fail_state_kept",
                                "rep_unknown_count", "rmw_records", "instr_skipped", "cv_values", "cv_used", "cv_missing",
                                "memory_omission_events", "lift_failed_memory_instructions", "rep_collapsed_unknown_count", "lifted_memory_shortfall",
                                "sync_markers", "sync_realigned", "sync_events", "sync_events_bad", "sync_events_obj_unknown", "sync_events_noret", "sync_events_derived", "sync_events_derive_agree", "sync_events_derive_disagree", "sync_events_derive_fail", "unanchored_records", "delta_skipped", "delta_unknown",
                                "keyframes_taken", "keyframe_reanchors_after_resync", "segments", "segments_lossy",
                                "segments_unsyncable", "skipped_insn", "reseeks", "switch_losses", "unknown_after_switch", "records_after_switch",
                                "tramp_mismatch", "tramp_unknown_origin", "fb_logsite", "patch_jumps",
                                "e9_swaps", "e9_swaps_triggered", "e9_preinit_insn", nullptr};
    static const char* GT[] = {"records_compared", "identical", "unknown", "wrong", "excluded", "gt_only", "recon_only", "gt_tail", "gt_records", "gt_physical_records", "gt_rep_ranges",
                               "records_after_gt_end", "wrong_in_overflow", "wrong_same_page", "wrong_unanchored", "unknown_in_overflow", "unknown_unanchored",
                               "wrong_after_switch", "resync_after_streak", "gt_skipped_after_streak", "resync_after_loss", "gt_skipped_after_loss", "gt_rewound_after_loss", "resync_after_loss_failed", nullptr};
    std::vector<uint64_t> tot; for (int i = 0; SUM[i]; i++) tot.push_back(0);
    std::vector<uint64_t> gtot; for (int i = 0; GT[i]; i++) gtot.push_back(0);
    bool any_gt = false; double wall_max = 0; std::string thrjs = "[";
    for (uint32_t tid : tids) {
        std::string sp = tpath(tid, ".json"); Json j;
        try { j = json_load(sp); } catch (std::exception&) { fprintf(stderr, "ptrecon --mt: cannot read %s\n", sp.c_str()); continue; }
        for (int i = 0; SUM[i]; i++) tot[i] += j[SUM[i]].u64();
        if (j["gt"].t == Json::OBJ) { any_gt = true; for (int i = 0; GT[i]; i++) gtot[i] += j["gt"][GT[i]].u64(); }
        double w = j["wall_s"].dbl(); if (w > wall_max) wall_max = w;
        uint64_t nseg = 0, nlossy = 0, bytes = 0; for (auto& s : plans[tid].segs) { nseg++; nlossy += s.lossy; bytes += s.end - s.begin; }
        thrjs += std::string(thrjs.size() > 1 ? "," : "") + "{\"tid\":" + std::to_string(tid) + ",\"main\":" + (tid == pid ? "true" : "false") +
                 ",\"fs_base\":" + std::to_string(fs_of.count(tid) ? fs_of[tid] : 0) +
                 ",\"plan_segments\":" + std::to_string(nseg) + ",\"plan_segments_lossy\":" + std::to_string(nlossy) + ",\"aux_bytes\":" + std::to_string(bytes) +
                 ",\"cv\":" + json_quote(cv_for(tid)) + ",\"gt_in\":" + json_quote(gt_for(tid));
        for (int i = 0; SUM[i]; i++) thrjs += std::string(",\"") + SUM[i] + "\":" + std::to_string(j[SUM[i]].u64());
        if (j.has("rec_hash")) thrjs += ",\"rec_hash\":" + std::to_string(j["rec_hash"].u64());
        thrjs += ",\"decode_window\":" + json_dump(j["decode_window"]) +
                 ",\"memory_omission_examples\":" + json_dump(j["memory_omission_examples"]) +
                 ",\"gt\":" + json_dump(j["gt"]) + ",\"images\":" + json_dump(j["images"]) + ",\"wall_s\":" + std::to_string(w) + "}";
        if (!o.keep_thread_files) remove(sp.c_str());
    }
    thrjs += "]";
    std::string failjs = "[";
    for (size_t i = 0; i < failed_tids.size(); i++) failjs += std::string(i ? "," : "") + std::to_string(failed_tids[i]);
    failjs += "]";
    std::string cpujs = "[";
    for (int i = 0; i < ncpu; i++)
        cpujs += std::string(i ? "," : "") + "{\"cpu\":" + std::to_string(sb.cpus[i].cpu) + ",\"aux\":" + json_quote(sb.cpus[i].aux) + ",\"aux_bytes\":" + std::to_string(sb.cpus[i].aux_bytes) +
                 ",\"switch_ins\":" + std::to_string(cs[i].sw) + ",\"regions\":" + std::to_string(cs[i].regions) + ",\"regions_target\":" + std::to_string(cs[i].target) +
                 ",\"regions_other_pid\":" + std::to_string(cs[i].other) + ",\"regions_unattributed\":" + std::to_string(cs[i].unattributed) +
                 ",\"regions_lossy\":" + std::to_string(cs[i].lossy) + ",\"regions_slack_used\":" + std::to_string(cs[i].slack) + ",\"regions_inherited\":" + std::to_string(cs[i].inherited) + ",\"regions_conflict\":" + std::to_string(cs[i].conflict) + ",\"regions_gap_realigned\":" + std::to_string(cs[i].realigned) +
                 ",\"bytes_target\":" + std::to_string(cs[i].bytes_target) + ",\"bytes_other_pid\":" + std::to_string(cs[i].bytes_other) + ",\"bytes_unattributed\":" + std::to_string(cs[i].bytes_unattr) +
                 ",\"min_gap_tsc\":" + std::to_string(cs[i].min_gap == INT64_MAX ? 0 : cs[i].min_gap) + ",\"aux_lost_bytes\":" + std::to_string(sb.cpus[i].lost) + ",\"data_lost\":" + std::to_string(sb.cpus[i].data_lost) + "}";
    cpujs += "]";
    std::string js = "{";
    for (int i = 0; SUM[i]; i++) js += std::string(i ? "," : "") + "\"" + SUM[i] + "\":" + std::to_string(tot[i]);
    js += ",\"output_stride\":" + std::to_string(o.output_stride);
    js += ",\"mt\":true,\"pid\":" + std::to_string(pid) + ",\"threads_reconstructed\":" + std::to_string(tids.size()) + ",\"threads_in_sideband\":" + std::to_string(sb.threads.size()) +
          ",\"other_pids\":" + std::to_string(other_pids.size()) + ",\"sw_slack_tsc\":" + std::to_string(o.sw_slack_tsc) + ",\"mt_jobs\":" + std::to_string(jobs) +
          ",\"merged_records\":" + std::to_string(out_records) + ",\"merged_no_ts\":" + std::to_string(no_ts) +
          ",\"threads_failed\":" + std::to_string(bad) + ",\"threads_failed_tids\":" + failjs;
    if (any_gt) { js += ",\"gt\":{"; for (int i = 0; GT[i]; i++) js += std::string(i ? "," : "") + "\"" + GT[i] + "\":" + std::to_string(gtot[i]);
                  const uint64_t cmp = gtot[1] + gtot[2] + gtot[3]; char b[64];
                  snprintf(b, sizeof b, "%.6f", cmp ? 100.0 * (double)gtot[3] / (double)cmp : 0.0); js += std::string(",\"wrong_pct\":") + b;
                  snprintf(b, sizeof b, "%.6f", cmp ? 100.0 * (double)gtot[1] / (double)cmp : 0.0); js += std::string(",\"identical_pct\":") + b;
                  snprintf(b, sizeof b, "%.6f", cmp ? 100.0 * (double)gtot[2] / (double)cmp : 0.0); js += std::string(",\"unknown_pct\":") + b + "}"; }
    else js += ",\"gt\":null";
    js += ",\"threads\":" + thrjs + ",\"cpus\":" + cpujs + ",\"wall_s\":" + std::to_string(wall_max) + "}\n";
    if (!o.summary.empty()) { FILE* f = fopen(o.summary.c_str(), "w"); if (f) { fputs(js.c_str(), f); fclose(f); } }
    fprintf(stderr, "%s", js.c_str());
    if (!o.out.empty()) fprintf(stderr, "ptrecon --mt: %lu records from %zu threads merged by time -> %s\n", (unsigned long)out_records, tids.size(), o.out.c_str());
    ph("end");
    return bad ? 1 : 0;
}
