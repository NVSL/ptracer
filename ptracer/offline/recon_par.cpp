// recon_par.cpp -- ptrecon --jobs N: chunked, parallel reconstruction.
//
// WHERE A CHUNK MAY START.  Three things have to hold at a chunk boundary, and
// they are three different mechanisms:
//   1. the hardware decoder can only be entered at a PSB packet;
//   2. the machine state is all-unknown at a fresh chunk, so a KEYFRAME (a guarded site that logs
//      its registers every K-th execution) has to re-define the registers;
//   3. with the buffer sink the value stream is POSITIONAL, so a SYNC MARKER (`ptwrite <running
//      count>') has to re-align the cursor.
// Checking only 1 is not enough: a capture with neither 2 nor 3 split anyway (gemm MEDIUM, one
// sync marker, no keyframes) comes out 84 % unknown at 20 jobs.  So a probe per boundary
// finds the first point at or after each ideal split where 2 and 3 have both happened with no
// state loss after them (Recon::run_anchor_scan), the chunk decodes from the PSB before that
// point, and a capture that cannot supply such points is reconstructed SERIALLY with the reason
// printed (Recon::can_split).  Chunk 0 starts at byte 0, not at the first PSB: `skip_bytes != 0'
// turns off the pre-init image view, which would cost 99 338 records on whole-program CPython.
//
// Each chunk is reconstructed by a forked child with its own PtDecoder and its own VexInterp,
// which decodes [warm, bound[k+1]) but emits only from bound[k]: the warm-up prefix rebuilds the
// registers, the shadow memory and the cursor that a serial run would have had there.  The chunk
// `.mtrace' files are concatenated in chunk order, which reproduces program order because the
// chunks partition the stream.
//
// Why processes and not threads: Recon holds a VexLifter (libVEX has global state and is not
// thread safe) and large per-chunk caches; fork() also gives copy-on-write sharing of everything
// the parent had already read.  The mmap()ed AUX file is shared physically between all children.
#include "recon.h"
#include "cjit.h"
#include <map>
#include <memory>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <functional>
#include <sys/wait.h>
#include <ctime>
#include <cerrno>
#include <atomic>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static std::string chunk_path(const std::string& base, int i, const char* suffix) {
    char buf[64]; snprintf(buf, sizeof buf, ".chunk%03d%s", i, suffix);
    return base + buf;
}

static double par_now() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
Recon* g_par_prebuilt = nullptr;

int Recon::run_parallel(const ReconOptions& o) {
    const double t_par0 = par_now(); double t_setup = 0, t_probe = 0, t_chunks = 0;
    std::vector<double> c_t0, c_t1;   // per-chunk fork / reap time (round 2), relative to t_par0
    // the shared compile cache this run created (removed by the creating process only)
    struct CacheGuard { bool own = false; pid_t pid = getpid(); ~CacheGuard() { if (own && getpid() == pid) cjit::cache_remove(); } } cache_guard;
    bool& own_cache = cache_guard.own;
    // a thread plan (--mt-split) is split in its VIRTUAL stream (ThreadPlan::vbase); every
    // segment start and every PSB inside a segment is an entry point there.
    const uint64_t total = o.plan ? plan_total(*o.plan) : PtDecoder::aux_size(o.aux);
    std::vector<uint64_t> psb;
    if (!o.plan) psb = PtDecoder::psb_offsets(o.aux);
    else { ThreadPlan ip = *o.plan; ip.index(); std::map<int, std::vector<uint64_t>> fp;
        for (size_t i = 0; i < ip.segs.size(); i++) {
            const AuxSeg& sg = ip.segs[i]; if (sg.end <= sg.begin) continue;
            psb.push_back(ip.vbase[i]);
            auto it = fp.find(sg.file); if (it == fp.end()) it = fp.emplace(sg.file, PtDecoder::psb_offsets(ip.files[(size_t)sg.file])).first;
            // binary search (the old scan over every PSB of the file per segment was O(segments x PSBs):
            // java_scrabble 987293 = 338 k segments x 680 k PSBs, the split thread's parent spent minutes here)
            for (auto pi = std::upper_bound(it->second.begin(), it->second.end(), sg.begin); pi != it->second.end() && *pi < sg.end; ++pi)
                psb.push_back(ip.vbase[i] + (*pi - sg.begin));
        }
        std::sort(psb.begin(), psb.end()); psb.erase(std::unique(psb.begin(), psb.end()), psb.end()); }
    // `par' is how many children run at once; `jobs' below is the number of CHUNKS.  Over-
    // decomposition (chunks_per_job > 1) lets a pool of `par' workers absorb chunks of unequal
    // cost; it is cheap now that a chunk's warm-up is one anchor window, not a whole chunk.
    int par = o.jobs < 1 ? 1 : o.jobs;
    // auto = over-decompose up to 4 chunks per worker while a chunk keeps >= 8 MB of AUX
    // (measured, whole-program CPython nbody at 8 workers: 1 -> 71.4 s, 4 -> 62.6 s, 16 -> 65.1 s;
    // the start-up half of the trace replays ~2x slower per instruction, so equal byte ranges
    // are unequal work, and a chunk costs ~0.2-0.3 s of fixed setup + warm-up).
    int cpj = o.chunks_per_job > 0 ? o.chunks_per_job
            : (int)std::max<uint64_t>(1, std::min<uint64_t>(4, total / ((uint64_t)par * (8ull << 20))));
    int jobs = par * cpj; if (jobs > (int)psb.size()) jobs = (int)psb.size();
    if (jobs < 1) jobs = 1;
    if (par > jobs) par = jobs;
    // A fork pool: at most `par' children at once; `child(k)' runs in the child and returns its
    // exit code, `done(k, ok)' runs in the parent in COMPLETION order.  Returns the failure count.
    // a fork of this (large) process costs ~18 ms of page-table copying, and the old
    // pool forked every chunk from the parent one after another: at 96 workers x 384 chunks the
    // parent's fork loop WAS the critical path (6.6 s of chunks for a 1.9 s longest chunk).  Now
    // `par' WORKER processes are spawned through a fork TREE (depth log2 par), and each worker claims
    // chunk indices from a shared counter (dynamic load balancing / work stealing) and forks the
    // chunk from ITSELF -- a pristine copy, so every chunk still starts from the shared setup -- so
    // the forks run in parallel.  Completion is reported through shared memory; `done' runs in the
    // parent as chunks finish.  PTRECON_FLATPOOL=1 restores the old pool.
    auto pool = [&](int n, const std::function<int(int)>& child, const std::function<void(int, bool)>& done,
                    std::vector<double>* ts0 = nullptr, std::vector<double>* ts1 = nullptr) -> int {
        if (ts0) ts0->assign(n, 0); if (ts1) ts1->assign(n, 0);
        if (getenv("PTRECON_FLATPOOL") || n < 2 || par < 2) {
        int next = 0, running = 0, bad = 0; std::map<pid_t, int> who;
        while (next < n || running) {
            while (next < n && running < par) {
                pid_t p = fork();
                if (p < 0) { perror("fork"); _exit(2); }
                if (p == 0) { int rc = 1; try { rc = child(next); } catch (std::exception& e) { fprintf(stderr, "ptrecon child %d: %s\n", next, e.what()); rc = 1; } _exit(rc); }
                if (ts0) (*ts0)[next] = par_now() - t_par0;
                who[p] = next++; running++;
            }
            int st = 0; pid_t p = waitpid(-1, &st, 0);
            if (p < 0) { perror("waitpid"); break; }
            auto it = who.find(p); if (it == who.end()) continue;
            int k = it->second; who.erase(it); running--;
            bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 0; if (!ok) bad++;
            if (ts1) (*ts1)[k] = par_now() - t_par0;
            if (done) done(k, ok);
        }
        return bad;
        }
        struct Shm { std::atomic<int> next; std::atomic<int> nfin; };
        const size_t bytes = sizeof(Shm) + (size_t)n * (sizeof(std::atomic<int>) + 2 * sizeof(double));
        void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { perror("mmap pool"); return n; }
        Shm* sh = new (m) Shm(); sh->next.store(0); sh->nfin.store(0);
        std::atomic<int>* st = reinterpret_cast<std::atomic<int>*>((char*)m + sizeof(Shm));
        for (int k = 0; k < n; k++) new (&st[k]) std::atomic<int>(0);
        double* T0 = reinterpret_cast<double*>((char*)m + sizeof(Shm) + (size_t)n * sizeof(std::atomic<int>));
        double* T1 = T0 + n;
        const int W = std::min(par, n);
        prctl(PR_SET_CHILD_SUBREAPER, 1);
        fflush(stdout); fflush(stderr);
        auto worker = [&]() {
            for (;;) {
                const int k = sh->next.fetch_add(1);
                if (k >= n) break;
                T0[k] = par_now() - t_par0;
                pid_t p = fork();
                if (p < 0) { st[k].store(2); sh->nfin.fetch_add(1); continue; }
                if (p == 0) { int rc = 1; try { rc = child(k); } catch (std::exception& e) { fprintf(stderr, "ptrecon child %d: %s\n", k, e.what()); rc = 1; }
                              fflush(stdout); fflush(stderr); _exit(rc); }
                int ws = 0; while (waitpid(p, &ws, 0) < 0 && errno == EINTR) {}
                T1[k] = par_now() - t_par0;
                st[k].store(WIFEXITED(ws) && WEXITSTATUS(ws) == 0 ? 1 : 2); sh->nfin.fetch_add(1);
            }
            _exit(0);
        };
        // fork tree: a process responsible for workers [lo, hi) forks one for [mid, hi) and keeps [lo, mid)
        pid_t top = fork();
        if (top < 0) { perror("fork"); return n; }
        if (top == 0) {
            int lo = 0, hi = W;
            while (hi - lo > 1) { const int mid = (lo + hi) / 2; pid_t q = fork();
                                  if (q < 0) { hi = mid; continue; }
                                  if (q == 0) { lo = mid; } else hi = mid; }
            worker();
        }
        int bad = 0, reported = 0; std::vector<char> rep(n, 0);
        while (reported < n) {
            if (sh->nfin.load() > reported) {
                for (int k = 0; k < n; k++) if (!rep[k] && st[k].load()) {
                    rep[k] = 1; reported++; const bool ok = st[k].load() == 1; if (!ok) bad++;
                    if (ts0) (*ts0)[k] = T0[k]; if (ts1) (*ts1)[k] = T1[k];
                    if (done) done(k, ok); }
            } else {
                struct timespec ts{0, 1000000}; nanosleep(&ts, nullptr);
                // every worker gone without finishing everything (crashed)?  reap and give up on the rest
                int ws; pid_t r = waitpid(-1, &ws, WNOHANG);
                if (r < 0 && errno == ECHILD && sh->nfin.load() == reported) {
                    for (int k = 0; k < n; k++) if (!rep[k]) { rep[k] = 1; reported++; bad++; if (done) done(k, false); }
                }
            }
        }
        while (waitpid(-1, nullptr, 0) > 0 || errno == EINTR) {}   // reap the workers (subreaper)
        prctl(PR_SET_CHILD_SUBREAPER, 0);
        munmap(m, bytes);
        return bad;
    };
    const std::string base = o.out.empty() ? (o.summary.empty() ? std::string("ptrecon.par") : o.summary) : o.out;

    // ---- ONE setup for all the workers ---------------------------------------------------
    // A Recon per chunk child would re-parse both site-map JSONs, re-sort the interval map over
    // every executable mapping and re-read the cv value stream -- per worker, and then hold them
    // in per-worker memory.  All of that describes the PROCESS, not the chunk, so it is
    // built once here and inherited through fork(), physically shared copy-on-write.  The parent
    // never runs it, so each child forks from a pristine Recon and only has to say which chunk it
    // is (retarget()) -- and the boundary probes below use the same object.
    std::unique_ptr<Recon> shared;
    // --mt-split hands in the process-wide Recon its thread child inherited (g_par_prebuilt): it is
    // retargeted to the thread instead of rebuilt (12.6 s of JSON per split thread on node_acorn), and released
    // (not freed) on return.
    bool par_borrowed = false;
    struct Release { std::unique_ptr<Recon>& p; bool& b; ~Release() { if (b) p.release(); } } release_{shared, par_borrowed};
    if (o.gt_in.empty()) {
        ReconOptions so = o;
        so.jobs = 1; so.out.clear(); so.summary.clear(); so.text = false;
        so.skip_bytes = 0; so.end_bytes = 0; so.emit_from = 0; so.delta_seed_in.clear();
        try { if (g_par_prebuilt && !getenv("PTRECON_MT_SPLIT_OWNSETUP")) { shared.reset(g_par_prebuilt); par_borrowed = true; shared->retarget_thread(so); }   // --mt-split
              else shared.reset(new Recon(so));
              if (getenv("PTRECON_TSETUP")) fprintf(stderr, "tsetup ctor done %.3f\n", par_now() - t_par0);
              shared->preload_values(); if (getenv("PTRECON_TSETUP")) fprintf(stderr, "tsetup preload_values %.3f\n", par_now() - t_par0);
              if (!o.decode_only && !getenv("PTRECON_NOORIGPRE")) shared->preload_orig_images();
              shared->prebuild_decoder(); if (getenv("PTRECON_TSETUP")) fprintf(stderr, "tsetup prebuild_decoder %.3f\n", par_now() - t_par0); }
        catch (std::exception& e) { fprintf(stderr, "ptrecon: shared setup failed (%s)\n", e.what()); shared.reset(); }
    }
    t_setup = par_now() - t_par0;

    // ---- refusing, loudly, instead of producing garbage --------------------------------
    // Splitting at the PSB nearest each even division of the file, whatever the capture
    // contains, does not work.  On a build with no keyframes and no sync markers there is NOTHING at a
    // mid-trace PSB to re-anchor a chunk on, and the chunks reconstruct an all-unknown machine:
    // 84 % of gemm MEDIUM comes out with unknown addresses at 20 jobs.  A capture that cannot be
    // split is therefore reconstructed serially, with the reason
    // on stderr, which is the answer the caller actually wanted.
    auto run_serial = [&](const std::string& why) -> int {
        fprintf(stderr, "ptrecon --jobs %d: NOT splitting this capture -- %s.\n"
                        "ptrecon: falling back to SERIAL reconstruction (the output is the serial output).\n",
                o.jobs, why.c_str());
        cjit::back_to_default();
        ReconOptions so = o;
        so.jobs = 1; so.skip_bytes = 0; so.end_bytes = 0; so.emit_from = 0;
        so.seed_fs = o.plan ? o.seed_fs : false; so.delta_seed_in.clear(); so.anchor_scan_out.clear(); so.delta_scan_out.clear();   // --mt-split fallback keeps the thread %fs seeding
        if (shared) { shared->retarget(so); return shared->run(); }
        try { Recon r(so); return r.run(); }
        catch (std::exception& e) { fprintf(stderr, "ptrecon: %s\n", e.what()); return 1; }
    };
    if (const char* rr = getenv("PTRECON_SEED_RESOLVE")) {   // dev: P1 P2 ENTRY OUT
        char a[512], b[512], c[512], d[512]; if (sscanf(rr, "%511s %511s %511s %511s", a, b, c, d) != 4) return 2;
        std::string why; bool ok = Recon::seed_resolve(a, c, d, &why, shared.get(), b); fprintf(stderr, "seed_resolve: %d %s\n", ok, why.c_str()); return ok ? 0 : 1; }
    if (const char* rr = getenv("PTRECON_SEED_SAME")) {
        char a[512], b[512]; if (sscanf(rr, "%511s %511s", a, b) != 2) return 2;
        std::string why; bool ok = Recon::seed_same(a, b, &why); fprintf(stderr, "seed_same: %d %s\n", ok, why.c_str()); return ok ? 0 : 1; }
    if (psb.empty()) return run_serial("there is no PSB packet in " + o.aux + ", so the stream can "
                                       "only be entered at its beginning");
    if (!shared)     return run_serial("the shared setup failed");
    if (jobs < 2)    return run_serial("fewer than two chunks are available");
    // `--decode-only' interprets nothing and emits no records, so it has no replay state to
    // re-anchor: any exact partition of the stream at PSB packets is sound for what it counts.
    // Everything else needs the anchors below.
    const bool need_anchor = !o.decode_only;
    // SEEDED chunks (recon_seed.cpp) need no anchor.  PTRECON_SEEDS=1 always, 0 never,
    // default: when the capture cannot be split at anchors or anchors are too rare (< half the boundaries).
    // (thread plans, --mt-split: off unless PTRECON_SEEDS_PLAN=1 -- chunks there are not yet exact)
    // OPT-IN: PTRECON_SEEDS=1 always seeded, 2 = seeded when the capture cannot be split at anchors or fewer than
    // half the boundaries are anchored; default 0 (anchors / serial fallback, as before).  At 96 chunks seeded is a win on
    // gemm and 2mm but not yet on atax/correlation/doitgen (mispredictions -> serial re-runs) or fdtd/lu (slow fake runs).
    const int seed_mode = (o.plan && !getenv("PTRECON_SEEDS_PLAN")) ? 0 : getenv("PTRECON_SEEDS") ? atoi(getenv("PTRECON_SEEDS")) : 0;
    bool seeds = need_anchor && !shared->have_jit_ && seed_mode == 1;
    if (need_anchor && !seeds) { std::string why; if (!shared->can_split(why)) { if (seed_mode == 2 && !shared->have_jit_) seeds = true; else return run_serial(why); } }
    const bool probe = need_anchor && !seeds;

    // ---- round 0: SOUND chunk boundaries ----------------------------------------------------
    // A chunk can only be replayed independently if, where it starts emitting, the decoder has
    // synchronised (a PSB), the registers have been re-defined (a KEYFRAME fired and its site
    // logged) and -- with the buffer sink -- the positional value cursor has been re-aligned (a
    // SYNC MARKER).  None of those is at a fixed distance from an even division of the file, so
    // one cheap probe per boundary decodes forward from the PSB at or before the ideal split
    // point and reports the first offset at which all of it holds.  The probes are independent of
    // the machine state (both events are instruction-stream facts), run concurrently, and only
    // decode as far as the next anchor.
    const uint64_t ideal = total / (uint64_t)jobs;
    // (a probe searches half of the LOCAL chunk size past its split point: an anchor further than that
    //  does not pay for itself)
    auto psb_ge = [&](uint64_t x) -> uint64_t {      // the first PSB at or after x (0 if none)
        size_t lo = 0, hi = psb.size();
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (psb[mid] < x) lo = mid + 1; else hi = mid; }
        return lo < psb.size() ? psb[lo] : 0;
    };
    auto psb_le = [&](uint64_t x) -> uint64_t {      // the last PSB at or before x
        size_t lo = 0, hi = psb.size();
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (psb[mid] <= x) lo = mid + 1; else hi = mid; }
        return lo ? psb[lo - 1] : psb.front();
    };
    std::vector<uint64_t> tgt, tsz;     // split points and the local chunk size there
    // the TAIL (last quarter of the stream) is cut into half-size chunks when there are
    // several chunks per worker, so the last chunks a worker claims are short and the workers finish
    // together (PTRECON_TAIL_SPLIT=0: equal sizes, as before).
    const bool tail_split = jobs >= 2 * par && !(getenv("PTRECON_TAIL_SPLIT") && *getenv("PTRECON_TAIL_SPLIT") == '0');
    if (!tail_split) for (int k = 1; k < jobs; k++) { tgt.push_back(total * (uint64_t)k / (uint64_t)jobs); tsz.push_back(ideal); }
    else { const uint64_t head = total - total / 4;
           for (uint64_t x = ideal; x < total; ) { tgt.push_back(x); const uint64_t sz = x < head ? ideal : ideal / 2; tsz.push_back(sz); x += sz; }
           while (!tgt.empty() && tgt.back() + tsz.back() / 4 >= total) { tgt.pop_back(); tsz.pop_back(); } }
    if (probe) {
        int pbad = pool((int)tgt.size(), [&](int k) -> int {
                ReconOptions co = o; co.jobs = 1; co.out.clear(); co.summary.clear(); co.site_stats.clear();
                co.skip_bytes = psb_le(tgt[k]); co.emit_from = 0; co.delta_seed_in.clear();
                static const double span = getenv("PTRECON_DEV_PROBE_SPAN") ? atof(getenv("PTRECON_DEV_PROBE_SPAN")) : 0.5;   // dev
                const uint64_t sk = (uint64_t)(tsz[k] * span) ? (uint64_t)(tsz[k] * span) : 1;
                co.end_bytes = tgt[k] + sk < total ? tgt[k] + sk : total;
                co.anchor_scan_out = chunk_path(base, k, ".anchor");
                shared->retarget(co); return shared->run_anchor_scan();
            }, nullptr);
        if (pbad) fprintf(stderr, "ptrecon: %d of %zu boundary probes failed\n", pbad, tgt.size());
    }
    // Chunk 0 starts at byte 0, NOT at the first PSB: `skip_bytes != 0' turns off the pre-init
    // view of the E9Patch images and empties the log-on-change table, so a chunk 0 that
    // began at the first PSB would reconstruct the loader window against the PATCHED bytes and lose
    // 99 338 records of whole-program CPython -- at every job count, because chunk 0 is the same
    // chunk every time.  libipt synchronises forward to
    // that PSB by itself.
    std::vector<uint64_t> bound; bound.push_back(0);
    std::vector<uint64_t> bfrom; bfrom.push_back(0);   // the PSB each boundary's probe started at
    long long lag_sum = 0; int lag_n = 0, missed = 0, too_close = 0;
    // too few anchors (lu, gemm, ...) -> seeded chunks at PSBs instead
    if (probe && seed_mode == 2 && !shared->have_jit_) {
        int nok = 0;
        for (size_t k = 0; k < tgt.size(); k++) { const std::string ap = chunk_path(base, (int)k, ".anchor"); FILE* f = fopen(ap.c_str(), "r"); int ok = 0;
            if (f) { if (fscanf(f, "%d", &ok) != 1) ok = 0; fclose(f); } nok += ok ? 1 : 0; }
        if (2 * nok < (int)tgt.size()) { seeds = true; for (size_t k = 0; k < tgt.size(); k++) remove(chunk_path(base, (int)k, ".anchor").c_str()); }
    }
    const bool at_psb = !need_anchor || seeds;
    for (size_t k = 0; k < tgt.size(); k++) {
        int ok = 0; unsigned long anchor = 0, from = 0, kfo = 0, syo = 0, nkf = 0, nin = 0;
        if (at_psb) { anchor = (unsigned long)psb_ge(tgt[k]); ok = anchor != 0 && anchor > psb.front(); }
        else {
            const std::string ap = chunk_path(base, (int)k, ".anchor");
            FILE* f = fopen(ap.c_str(), "r");
            if (f) { if (fscanf(f, "%d %lu %lu %lu %lu %lu %lu", &ok, &anchor, &from, &kfo, &syo, &nkf, &nin) != 7) ok = 0;
                     fclose(f); }
            remove(ap.c_str());
        }
        if (!ok) { missed++; continue; }
        // Two boundaries closer than a quarter of an ideal chunk are one boundary: keep the first.
        if ((uint64_t)anchor <= bound.back() + tsz[k] / 4) { too_close++; continue; }
        lag_sum += (long long)anchor - (long long)tgt[k]; lag_n++;
        bound.push_back((uint64_t)anchor); bfrom.push_back(!at_psb ? (uint64_t)from : (uint64_t)anchor);
    }
    bound.push_back(total);
    t_probe = par_now() - t_par0 - t_setup;
    jobs = (int)bound.size() - 1;
    if (jobs < 2)
        return run_serial("no sound chunk boundary was found: the capture re-anchors (keyframe + "
                          "sync marker) too rarely to place even one chunk boundary");
    if (o.exact_warm)
        fprintf(stderr, "ptrecon --jobs: --exact-warm: every chunk replays the trace from byte 0 before its own\n"
                        "ptrecon:   range, which is bit-identical to serial by construction and costs most of the\n"
                        "ptrecon:   parallel speed-up\n");
    fprintf(stderr, "ptrecon --jobs: %zu PSB packets in %.1f MB; %d of %d chunk boundaries %s"
                    " (mean %+.1f KB from the ideal split; %d without an anchor, %d merged) -> %d chunks\n",
            psb.size(), total / 1048576.0, lag_n, (int)tgt.size(),
            seeds ? "at a PSB (seeded chunks)" : need_anchor ? "anchored" : "at a PSB (--decode-only: no replay state to anchor)",
            lag_n ? (double)lag_sum / lag_n / 1024.0 : 0.0, missed, too_close, jobs);
    if (jobs > 8 && !o.decode_only && !getenv("PTRECON_CJIT_CACHE")) { cjit::cache_create_for_run(); own_cache = cjit::cache_on(); }
    cjit::off_for_chunks(jobs);   // many short chunks: no compiler per chunk process (cjit.h) -- unless they share one cache

    // Where each chunk warms up from (the same rule the round-2 children use below).  The warm
    // start must be a PSB -- it is where the decoder is entered -- and must lie at or before the
    // boundary, so that the keyframe and the sync marker the boundary was chosen for are inside
    // the warm window.  A warm start that would reach the first PSB is pulled back to byte 0
    // instead, so that such a chunk sees the same pre-init image the serial run does.
    auto warm_start = [&](int k) -> uint64_t {
        if (k <= 0) return 0;
        if (o.exact_warm) return 0;       // --exact-warm: replay the whole prefix
        // --decode-only: nothing is interpreted, so there is no state to warm -- decoding the
        // previous chunk's tail again was pure duplicated work.  One PSB period is still decoded
        // first: libipt reports the instructions of the packet that ENDS at the boundary PSB with
        // offset() == bound[k], which the predecessor drops and only a decoder that was
        // already running before the PSB hands out.
        if (at_psb) { uint64_t w = psb_le(bound[k] - 1); return w <= psb.front() ? 0 : w; }
        if (!o.warm_legacy) {
            // The ANCHOR window: the probe for this boundary started decoding at
            // the PSB `bfrom[k]' and found the keyframe and the sync marker AFTER it, so warming
            // from there re-establishes exactly what the boundary was placed for.  `--warm-kb'
            // adds margin (never past the previous chunk's start).
            uint64_t want = bfrom[k];
            if (o.warm_min) { uint64_t w2 = want > o.warm_min ? want - o.warm_min : 0;
                              if (w2 < bound[k - 1]) w2 = std::min(bound[k - 1], want); want = w2; }
            if (want <= psb.front()) return 0;
            return psb_le(want);
        }
        uint64_t want = bound[k] > o.warm_bytes ? bound[k] - o.warm_bytes : 0;
        if (want < bound[k - 1]) want = bound[k - 1];
        if (want <= psb.front()) return 0;
        size_t lo = 0, hi = psb.size();
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (psb[mid] < want) lo = mid + 1; else hi = mid; }
        if (lo < psb.size() && psb[lo] < bound[k]) return psb[lo];
        return psb_le(bound[k] - 1);
    };

    // ---- round 1: the log-on-change scan --------------------------------------------------
    // Only for a build that actually has log-on-change values, and only when asked: it costs one
    // extra PT decode of the file (spread over the same N processes).  Chunk k scans its OWN range
    // and reports the last value logged at every site, plus a snapshot at chunk k+1's warm start,
    // which is the point chunk k+1 has to start its table from.
    bool scan = o.delta_scan && shared->has_delta();
    if (scan && o.plan) return run_serial("a thread plan with log-on-change values cannot be delta-scanned (not implemented)");
    if (o.delta_scan && !scan)
        fprintf(stderr, "ptrecon --delta-scan: no log-on-change values in these site maps -- skipped\n");
    if (scan) {
        std::vector<pid_t> sk;
        for (int k = 0; k < jobs; k++) {
            pid_t p = fork();
            if (p < 0) { perror("fork"); return 2; }
            if (p == 0) {
                ReconOptions co = o; co.jobs = 1; co.out.clear(); co.summary.clear(); co.site_stats.clear();
                co.skip_bytes = bound[k]; co.end_bytes = bound[k + 1]; co.emit_from = 0;
                co.scan_snap_at = (k + 1 < jobs) ? warm_start(k + 1) : 0;
                co.delta_scan_out = chunk_path(base, k, ".dscan");
                int rc = 1;
                try { Recon r(co); rc = r.run_delta_scan(); } catch (std::exception& e) { fprintf(stderr, "ptrecon dscan %d: %s\n", k, e.what()); rc = 1; }
                if (rc) fprintf(stderr, "ptrecon dscan %d: rc=%d\n", k, rc);
                _exit(rc);
            }
            sk.push_back(p);
        }
        int sbad = 0;
        for (size_t k = 0; k < sk.size(); k++) { int st = 0; waitpid(sk[k], &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) sbad++; }
        if (sbad) { fprintf(stderr, "ptrecon: %d of %d delta-scan chunks failed\n", sbad, jobs); return 1; }
        // Fold the per-chunk reports, in chunk order, into the exact table at each warm start.
        // A part's rule per entry: logged after this part's last state loss -> that value; else, if
        // the part had a state loss at all -> unknown; else -> whatever came in.
        size_t N = 0;
        {   std::string p0 = chunk_path(base, 0, ".dscan"); FILE* f = fopen(p0.c_str(), "rb");
            if (!f) { fprintf(stderr, "ptrecon: missing %s\n", p0.c_str()); return 1; }
            uint64_t hdr[4]; if (fread(hdr, 8, 4, f) != 4) { fclose(f); return 1; } N = (size_t)hdr[1]; fclose(f); }
        std::vector<uint64_t> run_val(N, 0); std::vector<uint8_t> run_ok(N, 1);   // the table at bound[0]
        auto load = [&](int k, std::vector<uint64_t>& sv, std::vector<uint64_t>& ss, uint64_t& sl,
                              std::vector<uint64_t>& fv, std::vector<uint64_t>& fs, uint64_t& fl) -> bool {
            std::string p = chunk_path(base, k, ".dscan"); FILE* f = fopen(p.c_str(), "rb"); if (!f) return false;
            uint64_t hdr[4]; if (fread(hdr, 8, 4, f) != 4 || hdr[1] != N) { fclose(f); return false; }
            sl = hdr[2]; fl = hdr[3];
            sv.resize(N); ss.resize(N); fv.resize(N); fs.resize(N);
            bool ok = fread(sv.data(), 8, N, f) == N && fread(ss.data(), 8, N, f) == N
                   && fread(fv.data(), 8, N, f) == N && fread(fs.data(), 8, N, f) == N;
            fclose(f); return ok;
        };
        auto apply = [&](std::vector<uint64_t>& tv, std::vector<uint8_t>& tok,
                         const std::vector<uint64_t>& v, const std::vector<uint64_t>& sq, uint64_t ls) {
            for (size_t i = 0; i < N; i++) {
                if (sq[i] && sq[i] > ls) { tv[i] = v[i]; tok[i] = 1; }
                else if (ls) tok[i] = 0;
            }
        };
        for (int k = 1; k < jobs; k++) {
            std::vector<uint64_t> sv, ss, fv, fs; uint64_t sl = 0, fl = 0;
            if (!load(k - 1, sv, ss, sl, fv, fs, fl)) { fprintf(stderr, "ptrecon: bad delta scan for chunk %d\n", k - 1); return 1; }
            std::vector<uint64_t> seed_val = run_val; std::vector<uint8_t> seed_ok = run_ok;
            apply(seed_val, seed_ok, sv, ss, sl);                 // ... up to chunk k's warm start
            std::string sp = chunk_path(base, k, ".dseed");
            FILE* f = fopen(sp.c_str(), "wb"); if (!f) { perror("dseed"); return 1; }
            uint64_t n64 = N; fwrite(&n64, 8, 1, f);
            fwrite(seed_val.data(), 8, N, f); fwrite(seed_ok.data(), 1, N, f); fclose(f);
            apply(run_val, run_ok, fv, fs, fl);                   // ... to the end of chunk k-1
        }
        for (int k = 0; k < jobs; k++) remove(chunk_path(base, k, ".dscan").c_str());
        size_t known = 0; for (size_t i = 0; i < N; i++) known += run_ok[i] ? 1 : 0;
        fprintf(stderr, "ptrecon --delta-scan: %zu logged values, %zu known at the last boundary\n", N, known);
    }

    // ---- round 2: the chunks, `par' at a time; the parent appends each finished chunk's trace to
    // the output as soon as every chunk before it is done, so the concatenation overlaps the tail
    // of the reconstruction instead of following it.
    uint64_t out_records = 0;
    FILE* out = nullptr; std::vector<char> buf;
    if (!o.out.empty()) {
        out = fopen(o.out.c_str(), "wb"); if (!out) { perror("out"); return 1; }
        mtrace_hdr h{}; h.magic = MTRACE_MAGIC; h.version = MTRACE_VERSION; h.tid_count = 1;
        fwrite(&h, sizeof h, 1, out); buf.resize(4 << 20);
    }
    std::vector<char> fin(jobs, 0); int merged = 0; bool merge_bad = false; std::string seed_stats;
    auto merge_ready = [&]() {
        while (merged < jobs && fin[merged]) {
            if (out && !merge_bad) {
                std::string cp = chunk_path(base, merged, ".mtrace");
                FILE* in = fopen(cp.c_str(), "rb");
                if (!in) { fprintf(stderr, "ptrecon: missing %s\n", cp.c_str()); merge_bad = true; }
                else {
                    // kernel-side copy (copy_file_range), no user-space bounce buffer;
                    // falls back to read/write where the kernel/filesystem refuses it.
                    bool done = false;
                    {   struct stat st; fflush(out);
                        if (fstat(fileno(in), &st) == 0 && st.st_size >= (off_t)sizeof(mtrace_hdr)) {
                            loff_t ioff = sizeof(mtrace_hdr); size_t left = (size_t)st.st_size - sizeof(mtrace_hdr), copied = 0;
                            while (left) { ssize_t c = copy_file_range(fileno(in), &ioff, fileno(out), nullptr, left, 0);
                                           if (c <= 0) break; left -= (size_t)c; copied += (size_t)c; }
                            if (!left) { done = true; out_records += copied / sizeof(mtrace_rec); }
                            else if (copied) { fprintf(stderr, "ptrecon: copy_file_range stopped mid-chunk\n"); merge_bad = true; done = true; }
                        } }
                    if (!done) { fseek(in, sizeof(mtrace_hdr), SEEK_SET); size_t n;
                       while ((n = fread(buf.data(), 1, buf.size(), in)) > 0) { fwrite(buf.data(), 1, n, out); out_records += n / sizeof(mtrace_rec); } }
                    fclose(in); remove(cp.c_str()); }
            }
            merged++;
        }
    };
    int pipe_per_chunk = 0;
    if (jobs < par && !o.decode_only) {
        pipe_per_chunk = std::min(8, (par - jobs) / jobs);
        if (const char* e = getenv("PTRECON_CHUNK_PIPE")) pipe_per_chunk = atoi(e);
        if (pipe_per_chunk < 2) pipe_per_chunk = 0;
        if (pipe_per_chunk) fprintf(stderr, "ptrecon --jobs: %d chunks for %d workers -> %d pipeline producers per chunk\n", jobs, par, pipe_per_chunk);
    }
    int bad = 0;
    if (seeds) {
        // ---- seeded chunks (recon_seed.cpp) -------------------------------------------
        auto sp = [&](int k, const char* x) { return chunk_path(base, k, x); };
        auto chunk_opts = [&](int k) { ReconOptions co = o; co.jobs = 1;
            const uint64_t warm = warm_start(k);
            co.skip_bytes = warm; co.end_bytes = bound[k + 1]; co.emit_from = warm < bound[k] ? bound[k] : 0;
            co.seed_fs = warm > 0 && !o.no_seed_fs;
            co.out = o.out.empty() ? std::string() : sp(k, ".mtrace"); co.summary = sp(k, ".json");
            co.site_stats = o.site_stats.empty() ? std::string() : sp(k, ".sites.csv"); co.text = false;
            co.seed_out = k + 1 < jobs ? sp(k, ".send") : std::string();
            return co; };
        // mid points: a PSB 1/8 of a chunk before each boundary.  The fake runs of chunk k-1 also export their state
        // there, so a chunk whose prediction has uncertain bytes can warm up from a prediction made 1/8 chunk earlier
        // (the uncertain bytes are usually re-derived by then: a counter reloaded from its stack slot)
        std::vector<uint64_t> mid(jobs + 1, 0);
        if (!getenv("PTRECON_SEED_NOMID")) for (int k = 2; k < jobs; k++) { const uint64_t m = psb_ge(bound[k] - (bound[k] - bound[k - 1]) / 8);
            if (m > bound[k - 1] && m < bound[k] && psb_le(m - 1) > bound[k - 1]) mid[k] = m; }
        auto run_co = [&](const ReconOptions& co) -> int { if (shared) { shared->retarget(co); return shared->run(); } Recon r(co); return r.run(); };
        // ONE pool: [chunk 0 for real] [two fake runs of every chunk 1..n-2] [chunks 1..n-1 for real].  The parent
        // extends the chain of predictions as fake runs finish; a real chunk's child waits for its prediction file
        // (every fake task is claimed before any real one, so the wait always ends).
        const double ts1 = par_now();
        const int nf = 2 * std::max(0, jobs - 2), N = 1 + nf + 2 * (jobs - 1);   // + one double-run task per chunk (no-op unless its prediction has uncertain bytes)
        std::vector<char> pred(jobs, 0), okv(jobs, 0), okb(jobs, 0), fd0(jobs, 0), fd1(jobs, 0); pred[0] = 1;
        int npred = 0, nextk = 1; bool broken = false, ok0 = false; double t_chain_s = 0, t_fake_end = 0;
        auto mark_broken = [&](int from) { broken = true; for (int j = from; j < jobs; j++) { FILE* f = fopen(sp(j, ".nopred").c_str(), "w"); if (f) fclose(f); } };
        auto advance = [&]() {
            while (!broken && nextk + 1 < jobs && pred[nextk] && fd0[nextk] && fd1[nextk]) {
                const double t0 = par_now(); std::string why;
                if (fd0[nextk] == 1 && fd1[nextk] == 1 && mid[nextk + 1]) { std::string w2;   // (before .pred: a waiting chunk sees both)
                    if (!Recon::seed_resolve(sp(nextk, ".m1"), sp(nextk, ".pred"), sp(nextk + 1, ".mpred"), &w2, shared.get(), sp(nextk, ".m1b"))) remove(sp(nextk + 1, ".mpred").c_str()); }
                if (fd0[nextk] == 1 && fd1[nextk] == 1 && Recon::seed_resolve(sp(nextk, ".p1"), sp(nextk, ".pred"), sp(nextk + 1, ".pred"), &why, shared.get(), sp(nextk, ".p1b"))) { pred[nextk + 1] = 1; npred++; }
                else { fprintf(stderr, "ptrecon --jobs seeds: no prediction for chunk %d (%s)\n", nextk + 1, why.c_str()); mark_broken(nextk + 1); }
                if (!getenv("PTRECON_SEED_KEEP")) for (const char* x : {".p1", ".p1json", ".p1b", ".p1bjson", ".m1", ".m1b"}) remove(sp(nextk, x).c_str());
                t_chain_s += par_now() - t0; nextk++;
            } };
        bad = pool(N, [&](int i) -> int {
                if (i == 0) return run_co(chunk_opts(0));
                if (i <= nf) { const int k = 1 + (i - 1) / 2; const bool isB = (i - 1) % 2;
                    ReconOptions co = chunk_opts(k);
                    if (!co.emit_from) return 4;     // a seeded chunk needs a warm-up before its range
                    co.seed_fake = isB ? 2 : 1; co.out.clear(); co.summary = sp(k, isB ? ".p1bjson" : ".p1json"); co.site_stats.clear(); co.rec_hash = false; co.seed_out = sp(k, isB ? ".p1b" : ".p1");
                    if (mid[k + 1]) { co.seed_mid_at = mid[k + 1]; co.seed_mid_out = sp(k, isB ? ".m1b" : ".m1"); }
                    return run_co(co); }
                const bool dbl = i - nf >= jobs; const int k = dbl ? i - nf - (jobs - 1) : i - nf; ReconOptions co = chunk_opts(k);
                const std::string pf = sp(k, ".pred"), nf2 = sp(k, ".nopred");
                for (;;) { if (access(pf.c_str(), F_OK) == 0) break; if (access(nf2.c_str(), F_OK) == 0) return 5;
                           struct timespec ts{0, 2000000}; nanosleep(&ts, nullptr); }
                const bool unc = Recon::seed_has_unc(pf);
                const std::string mp = sp(k, ".mpred");
                if (unc && mid[k] && access(mp.c_str(), F_OK) == 0) {   // warm up from the mid prediction; verified exactly at the boundary
                    if (dbl) return 0;
                    co.skip_bytes = psb_le(mid[k] - 1); co.seed_import_at = mid[k]; co.seed_in = mp; co.seed_entry_out = sp(k, ".entry");
                    { FILE* f = fopen(sp(k, ".usedmid").c_str(), "w"); if (f) fclose(f); }
                    return run_co(co); }
                if (unc) co.rec_hash = true;      // the two runs are compared by their record streams
                if (dbl) { if (!unc) return 0;
                    co.seed_unc_fake = true; co.out.clear(); co.site_stats.clear(); co.summary = sp(k, ".bjson");
                    if (!co.seed_out.empty()) co.seed_out = sp(k, ".bsend"); }
                else co.seed_entry_out = sp(k, ".entry");
                co.seed_in = pf;
                return run_co(co);
            }, [&](int i, bool ok) {
                if (i == 0) { ok0 = ok; if (ok && jobs > 1 && rename(sp(0, ".send").c_str(), sp(1, ".pred").c_str()) == 0) { pred[1] = 1; npred++; } else mark_broken(1); }
                else if (i <= nf) { const int k = 1 + (i - 1) / 2; ((i - 1) % 2 ? fd1 : fd0)[k] = ok ? 1 : 2; t_fake_end = par_now(); }
                else if (i - nf < jobs) { okv[i - nf] = ok; return; }
                else { okb[i - nf - (jobs - 1)] = ok; return; }
                advance(); }, &c_t0, &c_t1);
        (void)bad; bad = 0;
        const double ts2 = t_fake_end ? t_fake_end : par_now(), ts3 = ts2 + t_chain_s, ts4 = par_now();
        std::vector<char> ran(jobs, 0); ran[0] = ok0;
        const int b1 = ok0 ? 0 : 1;
        // verify the chain; a chunk whose prediction has uncertain bytes must also have run the same with them = fakes
        // (same records, same end state but in bytes that stay uncertain); a mispredicted chunk is re-run from the true state
        int nmiss = 0, nrerun = 0, ndbl = 0, nmid = 0; bool give_up = false;
        auto sum3 = [&](const std::string& p, std::string& out) -> bool { try { Json j = json_load(p); out = std::to_string(j["records"].u64()) + "/" + std::to_string(j["unknown_addr"].u64()) + "/" + std::to_string(j["rec_hash"].u64()); return j.has("rec_hash"); } catch (std::exception&) { return false; } };
        auto double_ok = [&](int k, std::string* why) -> bool {   // run a (.json/.send) against run b (.bjson/.bsend)
            std::string a, b; if (!sum3(sp(k, ".json"), a) || !sum3(sp(k, ".bjson"), b) || a != b) { *why = "double run: records differ (" + a + " vs " + b + ")"; return false; }
            if (k + 1 < jobs && !Recon::seed_mark_uncertain(sp(k, ".send"), sp(k, ".bsend"), why)) return false;
            return true; };
        auto rerun = [&](int k, const std::string& from, bool dbl_run) -> int {
            pid_t p = fork();
            if (p == 0) { int rc = 1; try { ReconOptions co = chunk_opts(k); co.seed_in = from; co.seed_entry_out = sp(k, ".entry");
                              if (dbl_run) co.rec_hash = true; rc = run_co(co); } catch (std::exception& e) { fprintf(stderr, "ptrecon seed rerun %d: %s\n", k, e.what()); } _exit(rc); }
            pid_t q = -1;
            if (dbl_run) { q = fork();
                if (q == 0) { int rc = 1; try { ReconOptions co = chunk_opts(k); co.seed_in = from; co.seed_unc_fake = true; co.rec_hash = true; co.out.clear(); co.site_stats.clear();
                                  co.summary = sp(k, ".bjson"); if (!co.seed_out.empty()) co.seed_out = sp(k, ".bsend"); rc = run_co(co); } catch (std::exception&) {} _exit(rc); } }
            int st = 0, bad2 = 0; waitpid(p, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad2++;
            if (q > 0) { waitpid(q, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad2++; }
            return bad2; };
        for (int k = 1; k < jobs && !give_up; k++) {
            const std::string truth = k == 1 ? sp(1, ".pred") : sp(k - 1, ".send");   // chunk 0's end state became pred[1]
            const bool usedmid = access(sp(k, ".usedmid").c_str(), F_OK) == 0;
            std::string why; bool good = pred[k] && okv[k] && Recon::seed_same(sp(k, ".entry"), truth, &why, usedmid ? std::string() : sp(k, ".pred"));
            if (usedmid) nmid++;
            else if (good && Recon::seed_has_unc(sp(k, ".pred"))) { ndbl++; good = okb[k] && double_ok(k, &why); }
            if (!good) {
                nmiss++;
                if (getenv("PTRECON_SEED_DEBUG")) fprintf(stderr, "ptrecon --jobs seeds: chunk %d mispredicted (%s)\n", k, pred[k] ? why.c_str() : "no prediction");
                nrerun++;
                const bool dbl_run = Recon::seed_has_unc(truth);
                if (rerun(k, truth, dbl_run)) { fprintf(stderr, "ptrecon: seeded chunk %d failed\n", k); bad++; break; }
                if (dbl_run) { std::string w2; if (!double_ok(k, &w2)) { fprintf(stderr, "ptrecon --jobs seeds: chunk %d depends on state its predecessor left uncertain (%s): serial\n", k, w2.c_str()); give_up = true; } }
            }
        }
        if (!getenv("PTRECON_SEED_KEEP")) for (int k = 0; k < jobs; k++) for (const char* x : {".pred", ".entry", ".send", ".nopred", ".bjson", ".bsend", ".mpred", ".usedmid"}) remove(sp(k, x).c_str());
        if (give_up) { for (int k = 0; k < jobs; k++) { remove(sp(k, ".json").c_str()); remove(sp(k, ".mtrace").c_str()); }
                       if (out) { fclose(out); out = nullptr; }
                       return run_serial("a seeded chunk depends on state its predecessor left uncertain"); }
        if (b1 && !ran[0]) bad++;
        fprintf(stderr, "ptrecon --jobs seeds: %d double runs, %d warm-ups from a mid prediction\n", ndbl, nmid);
        fprintf(stderr, "ptrecon --jobs seeds: %d chunks, %d predicted, %d mispredicted (re-run serially); round1 %.2f s, chain %.2f s, round2 %.2f s, verify+reruns %.2f s\n",
                jobs, npred, nmiss, ts2 - ts1, ts3 - ts2, ts4 - ts3, par_now() - ts4);
        seed_stats = "{\"chunks\":" + std::to_string(jobs) + ",\"double\":" + std::to_string(ndbl) + ",\"midwarm\":" + std::to_string(nmid) + ",\"predicted\":" + std::to_string(npred) + ",\"mispredicted\":" + std::to_string(nmiss) +
                     ",\"t_round1\":" + std::to_string(ts2 - ts1) + ",\"t_chain\":" + std::to_string(ts3 - ts2) + ",\"t_round2\":" + std::to_string(ts4 - ts3) +
                     ",\"t_verify\":" + std::to_string(par_now() - ts4) + "}";
        if (!bad) { for (int k = 0; k < jobs; k++) fin[k] = 1; merge_ready(); }
    } else
    bad = pool(jobs, [&](int k) -> int {
            ReconOptions co = o;
            co.jobs = 1;
            // ---- chunk warm-up -------------------------------------------------------------
            // Chunk k decodes [warm, bound[k+1]) but only EMITS from bound[k]: the prefix
            // [warm, bound[k]) is the tail of chunk k-1, decoded with every record and every
            // counter suppressed, purely so that the registers, the shadow memory, the
            // log-on-change table and the buffer-sink cursor are warm when chunk k's own range
            // begins.  The warm start is snapped to a PSB (nothing else can be synchronised on)
            // and never crosses the previous chunk's own start, so the extra work is bounded by
            // one chunk.  Chunk 0 has nothing to warm up from.
            uint64_t warm = warm_start(k);
            if (scan && k > 0) co.delta_seed_in = chunk_path(base, k, ".dseed");
            co.skip_bytes = warm; co.end_bytes = bound[k + 1];
            co.emit_from = warm < bound[k] ? bound[k] : 0;
            // Process invariants the chunk cannot re-derive: the image load bases already come
            // from the sideband, the TLS base does not (the analyzer anchors `fs_base' once, at
            // `main', so no mid-stream chunk ever sees it logged).  It is a process constant for a
            // single-threaded target, so seed it -- but only in a chunk that does not start at the
            // beginning of the trace, where %fs is still 0 until the loader sets it.
            co.seed_fs = warm > 0 && !o.no_seed_fs;
            co.out = o.out.empty() ? std::string() : chunk_path(base, k, ".mtrace");
            co.summary = chunk_path(base, k, ".json");
            co.site_stats = o.site_stats.empty() ? std::string() : chunk_path(base, k, ".sites.csv");
            co.text = false;
            // fewer chunks than workers (a capture that re-anchors rarely, e.g. lu: one
            // boundary -> chunk 0 = 3/4 of the stream): the spare cpus become pipeline producers.
            if (pipe_per_chunk > 0) co.pipe_k = pipe_per_chunk;
            if (shared) { shared->retarget(co); return shared->run(); }
            Recon r(co); return r.run();
        }, [&](int k, bool ok) { if (ok) { fin[k] = 1; merge_ready(); } }, &c_t0, &c_t1);
    t_chunks = par_now() - t_par0 - t_setup - t_probe;
    if (scan) for (int k = 1; k < jobs; k++) remove(chunk_path(base, k, ".dseed").c_str());
    if (out) fclose(out);
    if (bad || merge_bad) { fprintf(stderr, "ptrecon: %d of %d chunks failed\n", bad, jobs); return 1; }

    // ---- merge the per-chunk summaries ----------------------------------------------
    static const char* SUM[] = {"records", "unknown_addr", "instructions", "pt_overflows", "resyncs", "sync_failures", "psb_repaired",
                                "ptw_packets", "ptw_used", "ptw_unconsumed", "lift_failures",
                                "lift_fail_state_kept", "rep_unknown_count", "rmw_records",
                                "memory_omission_events", "lift_failed_memory_instructions", "rep_collapsed_unknown_count", "lifted_memory_shortfall",
                                "instr_skipped", "cv_used", "cv_missing", "sync_markers",
                                "sync_realigned", "sync_events", "sync_events_bad", "sync_events_obj_unknown", "sync_events_noret", "sync_events_derived", "sync_events_derive_agree", "sync_events_derive_disagree", "sync_events_derive_fail", "unknown_ops", "ccall_known", "ccall_unknown",
                                "tramp_mismatch", "tramp_unknown_origin", "fb_logsite", "unanchored_records",
                                "delta_skipped", "delta_unknown", "delta_reanchored", "patch_jumps",
                                "keyframes_taken", "keyframe_reanchors_after_resync",
                                "keyframe_site_defs",
                                "warm_instructions", nullptr};
    std::vector<uint64_t> tot; for (int i = 0; SUM[i]; i++) tot.push_back(0);
    // --jitdump: the code table is read-only, so each chunk child loads it and reports its own
    // read accounting; these are sums, and the coverage is recomputed from them.
    static const char* JIT[] = {"reads_object", "reads_trampoline", "reads_slab", "reads_anon_dump",
                                "reads_unmapped", "reads_ambiguous", "reads_time_keyed",
                                "reads_composed", "time_behind", "jit_instructions", "jit_records",
                                "jit_unknown_addr", "jit_lift_failures", nullptr};
    std::vector<uint64_t> jtot; for (int i = 0; JIT[i]; i++) jtot.push_back(0);
    bool have_jit = false; uint64_t jit_versions = 0, jit_moved = 0, jit_removed = 0, jit_behind_max = 0;
    uint64_t delta_values = 0;                 // the same in every chunk, not a sum
    uint64_t kf_values = 0, kf_resync_values = 0;      // ditto
    // Per-image records / unknown addresses, merged by path (the serial run's `images').
    struct ImgAgg { uint64_t lo = ~0ull, hi = 0, records = 0, unknown = 0; std::string code; };
    std::map<std::string, ImgAgg> imgs; std::vector<std::string> img_order;
    double wall_max = 0; std::string chunkjs = "[";
    uint64_t rh = 0; bool have_rh = false;
    for (int k = 0; k < jobs; k++) {
        std::string sp = chunk_path(base, k, ".json");
        Json j;
        try { j = json_load(sp); } catch (std::exception&) { fprintf(stderr, "ptrecon: cannot read %s\n", sp.c_str()); continue; }
        for (int i = 0; SUM[i]; i++) tot[i] += j[SUM[i]].u64();
        if (j.has("rec_hash") && o.rec_hash) { have_rh = true; rh = rh_add(rh_mul(rh, rh_pow(j["records"].u64())), j["rec_hash"].u64()); }
        if (j["delta_values"].u64() > delta_values) delta_values = j["delta_values"].u64();
        if (j["keyframe_values"].u64() > kf_values) kf_values = j["keyframe_values"].u64();
        if (j["keyframe_resync_values"].u64() > kf_resync_values) kf_resync_values = j["keyframe_resync_values"].u64();
        for (auto& im : j["images"].arr) {
            const std::string path = im["path"].str();
            auto it = imgs.find(path);
            if (it == imgs.end()) { imgs[path] = ImgAgg(); img_order.push_back(path); it = imgs.find(path); }
            ImgAgg& a = it->second;
            uint64_t lo = im["start"].u64(), hi = im["end"].u64();
            if (lo < a.lo) a.lo = lo; if (hi > a.hi) a.hi = hi;
            a.records += im["records"].u64(); a.unknown += im["unknown_addr"].u64();
            if (a.code.empty()) a.code = im["code"].str();
        }
        if (j["jit"].t == Json::OBJ) { have_jit = true;
            for (int i = 0; JIT[i]; i++) jtot[i] += j["jit"][JIT[i]].u64();
            if (j["jit"]["versions"].u64() > jit_versions) jit_versions = j["jit"]["versions"].u64();
            if (j["jit"]["moved"].u64() > jit_moved) jit_moved = j["jit"]["moved"].u64();
            if (j["jit"]["removed"].u64() > jit_removed) jit_removed = j["jit"]["removed"].u64();
            if (j["jit"]["time_behind_max_tsc"].u64() > jit_behind_max) jit_behind_max = j["jit"]["time_behind_max_tsc"].u64(); }
        double w = j["wall_s"].dbl(); if (w > wall_max) wall_max = w;
        chunkjs += std::string(k ? "," : "") + "{\"chunk\":" + std::to_string(k) +
                   ",\"aux_from\":" + std::to_string(bound[k]) + ",\"aux_to\":" + std::to_string(bound[k + 1]) +
                   ",\"records\":" + std::to_string(j["records"].u64()) +
                   ",\"unknown_addr\":" + std::to_string(j["unknown_addr"].u64()) +
                   (j.has("rec_hash") ? ",\"rec_hash\":" + std::to_string(j["rec_hash"].u64()) : std::string()) +
                   ",\"instructions\":" + std::to_string(j["instructions"].u64()) +
                   ",\"unanchored_records\":" + std::to_string(j["unanchored_records"].u64()) +
                   ",\"aux_warm_from\":" + std::to_string(j["aux_skip_bytes"].u64()) +
                   ",\"warm_instructions\":" + std::to_string(j["warm_instructions"].u64()) +
                   ",\"decode_window\":" + json_dump(j["decode_window"]) +
                   ",\"memory_omission_examples\":" + json_dump(j["memory_omission_examples"]) +
                   ",\"wall_s\":" + std::to_string(w) +
                   ",\"t0\":" + std::to_string(k < (int)c_t0.size() ? c_t0[k] : 0) + ",\"t1\":" + std::to_string(k < (int)c_t1.size() ? c_t1[k] : 0) + "}";
        remove(sp.c_str());
    }
    chunkjs += "]";

    // ---- merge the per-chunk --site-stats CSVs --------------------------------------
    // One row per logged value; `count', `repeats' and `skipped' are sums over the
    // chunks, everything else is identical in all of them.  (A chunk boundary loses one
    // repeat comparison per value, which is noise against the counts this feeds.)
    if (!o.site_stats.empty()) {
        std::vector<std::string> order; std::map<std::string, std::array<uint64_t, 3>> num;
        std::map<std::string, std::string> head;
        std::string hdr;
        for (int k = 0; k < jobs; k++) {
            std::string cp = chunk_path(base, k, ".sites.csv");
            FILE* f = fopen(cp.c_str(), "r"); if (!f) continue;
            char line[4096]; bool first = true;
            while (fgets(line, sizeof line, f)) {
                std::string L(line); while (!L.empty() && (L.back() == '\n' || L.back() == '\r')) L.pop_back();
                if (first) { first = false; if (hdr.empty()) hdr = L; continue; }
                // image,site,orig_addr,tramp_addr,kind,arg,payload_bits,count,repeats,skipped
                std::vector<std::string> c; size_t p0 = 0;
                for (size_t i = 0; i <= L.size(); i++)
                    if (i == L.size() || L[i] == ',') { c.push_back(L.substr(p0, i - p0)); p0 = i + 1; }
                if (c.size() < 10) continue;
                std::string key = c[0] + "," + c[1] + "," + c[5];
                auto it = num.find(key);
                if (it == num.end()) { order.push_back(key); num[key] = {0, 0, 0};
                    head[key] = c[0] + "," + c[1] + "," + c[2] + "," + c[3] + "," + c[4] + "," + c[5] + "," + c[6];
                    it = num.find(key); }
                it->second[0] += strtoull(c[7].c_str(), nullptr, 10);
                it->second[1] += strtoull(c[8].c_str(), nullptr, 10);
                it->second[2] += strtoull(c[9].c_str(), nullptr, 10);
            }
            fclose(f); remove(cp.c_str());
        }
        FILE* f = fopen(o.site_stats.c_str(), "w");
        if (f) {
            fprintf(f, "%s\n", hdr.empty() ? "image,site,orig_addr,tramp_addr,kind,arg,payload_bits,count,repeats,skipped" : hdr.c_str());
            for (auto& k : order) fprintf(f, "%s,%lu,%lu,%lu\n", head[k].c_str(),
                (unsigned long)num[k][0], (unsigned long)num[k][1], (unsigned long)num[k][2]);
            fclose(f);
            fprintf(stderr, "site stats: %s (%zu logged values, merged from %d chunks)\n",
                    o.site_stats.c_str(), order.size(), jobs);
        }
    }
    std::string js = "{";
    for (int i = 0; SUM[i]; i++) js += std::string(i ? "," : "") + "\"" + SUM[i] + "\":" + std::to_string(tot[i]);
    {   // per-image breakdown, biggest first
        std::vector<std::pair<std::string, ImgAgg>> v(imgs.begin(), imgs.end());
        std::sort(v.begin(), v.end(), [](const std::pair<std::string, ImgAgg>& a,
                                         const std::pair<std::string, ImgAgg>& b) {
            return a.second.records > b.second.records; });
        js += ",\"images\":["; bool first = true;
        for (auto& kv : v) {
            if (!kv.second.records) continue;
            js += std::string(first ? "" : ",") + "{\"path\":" + json_quote(kv.first) +
                  ",\"start\":" + std::to_string(kv.second.lo) + ",\"end\":" + std::to_string(kv.second.hi) +
                  ",\"code\":" + json_quote(kv.second.code) +
                  ",\"records\":" + std::to_string(kv.second.records) +
                  ",\"unknown_addr\":" + std::to_string(kv.second.unknown) + "}";
            first = false;
        }
        js += "]";
    }
    if (have_jit) {
        uint64_t served = jtot[0] + jtot[1] + jtot[2] + jtot[3] + jtot[4];
        js += ",\"jit\":{\"versions\":" + std::to_string(jit_versions) + ",\"moved\":" + std::to_string(jit_moved) +
              ",\"removed\":" + std::to_string(jit_removed);
        for (int i = 0; JIT[i]; i++) js += std::string(",\"") + JIT[i] + "\":" + std::to_string(jtot[i]);
        js += ",\"time_behind_max_tsc\":" + std::to_string(jit_behind_max) +
              ",\"code_byte_coverage\":" + std::to_string(served ? (double)(jtot[0] + jtot[1] + jtot[2]) / (double)served : 0.0) + "}";
    } else js += ",\"jit\":null";
    js += ",\"delta_values\":" + std::to_string(delta_values);
    js += ",\"keyframe_values\":" + std::to_string(kf_values) + ",\"keyframe_resync_values\":" + std::to_string(kf_resync_values);
    if (have_rh) js += ",\"rec_hash\":" + std::to_string(rh);
    js += ",\"warm_bytes\":" + std::to_string(o.warm_legacy ? o.warm_bytes : o.warm_min);
    js += ",\"warm_rule\":" + std::string(seeds ? "\"seeded\"" : o.exact_warm ? "\"exact\"" : !need_anchor ? "\"none\"" : o.warm_legacy ? "\"legacy\"" : "\"anchor\"");
    if (!seed_stats.empty()) js += ",\"seeds\":" + seed_stats;
    js += ",\"par\":" + std::to_string(par);
    js += ",\"t_setup\":" + std::to_string(t_setup) + ",\"t_probe\":" + std::to_string(t_probe) +
          ",\"t_chunks\":" + std::to_string(t_chunks) + ",\"t_total\":" + std::to_string(par_now() - t_par0);
    js += ",\"jobs\":" + std::to_string(jobs) + ",\"wall_s\":" + std::to_string(wall_max) +
          ",\"insn_per_s\":" + std::to_string(wall_max > 0 ? tot[2] / wall_max : 0) +
          ",\"chunks\":" + chunkjs + "}\n";
    if (!o.summary.empty()) { FILE* f = fopen(o.summary.c_str(), "w"); if (f) { fputs(js.c_str(), f); fclose(f); } }
    fprintf(stderr, "%s", js.c_str());
    if (!o.out.empty()) fprintf(stderr, "ptrecon --jobs %d: %lu records concatenated -> %s\n", jobs, (unsigned long)out_records, o.out.c_str());
    return 0;
}
