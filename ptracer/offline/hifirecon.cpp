#include <tuple>
#include <algorithm>
#include <unordered_set>
#include <dirent.h>
// hifirecon.cpp -- HiFi (Pin JIT) offline reconstructor.
//
// Inputs, all written by runtime/pinjit/hififlow.so in ONE process run:
//   code.<pid>.bin          original bytes + XED metadata of every basic-block version Pin translated
//   img.<pid>.txt           image name / low / high / load offset
//   flow.<pid>.<tid>.bin    executed block ids (the ORIGINAL control flow, in order)
//   cv.<pid>.<tid>.bin      hifitool's critical values (unchanged format, PTCV v2, sync 0)
//   pgt.<pid>.<tid>.bin     optional: Pin-side same-process ground truth, used ONLY for comparison
// and the PLAN the run used (runtime/pinjit/mkplan.py output).
//
// Reconstruction reads code/img/flow/cv/plan only.  The ground truth is consumed strictly after
// each instruction has been interpreted and never feeds the machine state.
//
//   hifirecon --dir D --pid P --tid T --plan PLAN [--gt] [--json OUT] [--verbose N]
#include "vexinterp.h"
#include <libvex_guest_offsets.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <algorithm>
#include <vector>
static std::map<uint64_t,uint64_t> g_unk_ip;  // diag: unknown gt records by ip (bit63 = skipped by an unresolved elided loop)
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#pragma pack(push, 1)
struct CodeHdr { uint32_t magic, id; uint64_t addr; uint32_t nins, nbytes, nitems; };
struct InsMeta { uint8_t len, flags, base, index, scale, seg, ngt, pred; uint16_t gpw, msize; uint8_t nbefore, nafter, pad0, pad1; int64_t disp; };
struct GtRec { uint64_t ea, meta; };
#pragma pack(pop)
enum { F_REP = 1, F_FALL = 2, F_CTRL = 4, F_MEMR = 8, F_SYSCALL = 16, F_CONDBR = 32, F_MEMW = 64, F_HASMEM = 128 };

enum ItemKind { IT_REG, IT_XMM_LO, IT_XMM_HI, IT_MEM, IT_FSBASE, IT_SHMEM };
struct Item { ItemKind kind; int off; int size; };
struct PSite { std::vector<Item> before, after; };

static const int GPOFF[16] = {OFFSET_amd64_RAX, OFFSET_amd64_RCX, OFFSET_amd64_RDX, OFFSET_amd64_RBX,
    OFFSET_amd64_RSP, OFFSET_amd64_RBP, OFFSET_amd64_RSI, OFFSET_amd64_RDI, OFFSET_amd64_R8,
    OFFSET_amd64_R9, OFFSET_amd64_R10, OFFSET_amd64_R11, OFFSET_amd64_R12, OFFSET_amd64_R13,
    OFFSET_amd64_R14, OFFSET_amd64_R15};

struct Ins { uint64_t ip; InsMeta m; const uint8_t* bytes; std::unique_ptr<IRBlockC> blk; const PSite* site = nullptr; PSite own; };
struct Block { uint64_t addr = 0, end = 0; std::vector<Ins> ins; std::vector<uint8_t> bytes; bool lifted = false;
               int lmode = 0, nr = 0; uint8_t rr[3] = {0, 0, 0}; uint32_t wmask = 0; };

struct Img { std::string name; uint64_t lo, hi, off; uint64_t gt = 0, ident = 0, unk = 0, wrong = 0, recs = 0, recunk = 0; };

static const uint8_t* map_file(const std::string& p, size_t* len, bool must = true) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) { if (must) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); } *len = 0; return nullptr; }
    struct stat st; fstat(fd, &st); *len = st.st_size;
    if (!*len) { close(fd); return nullptr; }
    void* m = mmap(nullptr, *len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) { fprintf(stderr, "mmap %s failed\n", p.c_str()); exit(2); }
    return (const uint8_t*)m;
}
static std::string base_name(const std::string& s) { size_t k = s.rfind('/'); return k == std::string::npos ? s : s.substr(k + 1); }

int main(int argc, char** argv) {
    std::string dir, plan, json; int pid = 0, tid = 0; bool use_gt = false; bool abi_restore = true; int verbose = 0; uint64_t max_report = 20; int mt_shadow = 1; int hint_ea = 0; int elide_cap = 31; int ua_rule = 0; int stop_trunc = 1; uint64_t max_insn_flow = 1ull << 28;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "--dir") dir = nx(); else if (a == "--pid") pid = atoi(nx().c_str()); else if (a == "--tid") tid = atoi(nx().c_str());
        else if (a == "--plan") plan = nx(); else if (a == "--gt") use_gt = true; else if (a == "--no-abi-restore") abi_restore = false; else if (a == "--json") json = nx();
        else if (a == "--mt-shadow") mt_shadow = atoi(nx().c_str()); else if (a == "--verbose") verbose = atoi(nx().c_str()); else if (a == "--max-report") max_report = strtoull(nx().c_str(), 0, 10);
        // address-only memory operands VEX lifts without an access (clflush/clflushopt/clwb; Pin reports them
        // as a 64-byte read): take the address from the operand and the pre-execution registers
        else if (a == "--hint-ea") hint_ea = atoi(nx().c_str());
        // log2 of the iterations an elided loop may replay before it is declared unresolved (its remaining gt
        // records become unknown).  A spin loop that waits on another thread's store never meets its exit value in the
        // shadow; at 2^31 one such loop cost ~13 min on the Java main thread.
        else if (a == "--elide-cap") elide_cap = atoi(nx().c_str());
        // watchdog -- most instructions ONE flow record (a block, an RLE loop, an elided loop) may replay.
        // Past it the record's remaining iterations are declared unresolved (their gt records unknown, never
        // guessed) and a WATCHDOG line names the record, so a recon can never hang silently.
        else if (a == "--max-insn-flow") max_insn_flow = strtoull(nx().c_str(), 0, 0);
        // unknown-address stores while %rsp is KNOWN (the old rule only cleared when %rsp was unknown).
        // 1 = sound rule: forget every shadowed word outside the thread's own stack, and the own-stack shadow
        // too whenever the store could alias it (its base register points into the own stack, or is unknown,
        // or the address has no base register / is segment-relative / implicit).  0 = old behaviour.
        else if (a == "--ua-rule") ua_rule = atoi(nx().c_str());
        else if (a == "--stop-trunc") stop_trunc = atoi(nx().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (dir.empty() || !pid) { fprintf(stderr, "usage: hifirecon --dir D --pid P [--tid T] --plan PLAN [--gt] [--json OUT]\n"); return 2; }
    if (!tid) tid = pid;
    const std::string sfx = "." + std::to_string(pid) + "." + std::to_string(tid) + ".bin";

    // ---- images
    std::vector<Img> imgs;
    { FILE* f = fopen((dir + "/img." + std::to_string(pid) + ".txt").c_str(), "r");
      if (!f) { fprintf(stderr, "no img file\n"); return 2; }
      char name[4096]; unsigned long lo, hi, off;
      while (fscanf(f, "%4095s %lx %lx %lx", name, &lo, &hi, &off) == 4) imgs.push_back({name, lo, hi, off});
      fclose(f); }
    auto img_of = [&](uint64_t ip) -> Img* { for (auto& m : imgs) if (ip >= m.lo && ip <= m.hi) return &m; return nullptr; };
    Img other{"(no image)", 0, 0, 0};

    // ---- plan -> runtime-address sites
    std::unordered_map<uint64_t, PSite> sites;
    uint64_t plan_imgs_resolved = 0, plan_imgs_unresolved = 0;
    if (!plan.empty()) { FILE* f = fopen(plan.c_str(), "r"); if (!f) { fprintf(stderr, "no plan\n"); return 2; }
      static char line[1 << 16]; const Img* cur = nullptr; bool cur_valid = false;
      while (fgets(line, sizeof line, f)) {
          if (line[0] == 'I' && line[1] == ' ') {
              std::string n = line + 2; while (!n.empty() && (n.back() == '\n' || n.back() == ' ')) n.pop_back();
              cur = nullptr;
              for (auto& m : imgs) if (m.name == n) { cur = &m; break; }
              if (!cur) for (auto& m : imgs) if (base_name(m.name) == base_name(n)) { cur = &m; break; }
              cur_valid = cur != nullptr; if (cur) plan_imgs_resolved++; else { plan_imgs_unresolved++; fprintf(stderr, "plan image not loaded: %s\n", n.c_str()); }
          } else if (line[0] == 'S' && line[1] == ' ' && cur_valid) {
              char* save = nullptr; strtok_r(line, " \t\n", &save);
              char* a = strtok_r(nullptr, " \t\n", &save); char* w = strtok_r(nullptr, " \t\n", &save); char* n = strtok_r(nullptr, " \t\n", &save);
              uint64_t addr = strtoull(a, nullptr, 16) + cur->off; int cnt = atoi(n);
              PSite& s = sites[addr]; std::vector<Item>& v = w[0] == 'A' ? s.after : s.before;
              for (int k = 0; k < cnt; k++) {
                  std::string t = strtok_r(nullptr, " \t\n", &save); Item it{IT_REG, -1, 8};
                  if (t == "f") { it.kind = IT_FSBASE; it.off = OFFSET_amd64_FS_CONST; }
                  else if (t.compare(0, 2, "r:") == 0) { it.kind = IT_REG; int sz; it.off = VexInterp::reg_off(t.substr(2), &sz); }
                  else if (t.compare(0, 2, "m:") == 0) { it.kind = IT_MEM; it.size = atoi(t.c_str() + 2); }
                  else if (t.compare(0, 2, "x:") == 0) { size_t c = t.rfind(':'); int sz; it.off = VexInterp::reg_off(t.substr(2, c - 2), &sz);
                      it.kind = t.compare(c + 1, std::string::npos, "hi") == 0 ? IT_XMM_HI : IT_XMM_LO; if (it.kind == IT_XMM_HI) it.off += 8; }
                  else { fprintf(stderr, "bad plan item %s\n", t.c_str()); return 2; }
                  if (it.kind != IT_MEM && it.off < 0) { fprintf(stderr, "bad plan reg %s\n", t.c_str()); return 2; }
                  v.push_back(it);
              }
          }
      }
      fclose(f); }

    // ---- code versions
    std::unordered_map<uint32_t, Block> blocks;
    { size_t len; const uint8_t* p = map_file(dir + "/code." + std::to_string(pid) + ".bin", &len); size_t o = 0;
      while (o + sizeof(CodeHdr) <= len) {
          CodeHdr h; memcpy(&h, p + o, sizeof h); o += sizeof h;
          if (h.magic != 0x324c4242) { fprintf(stderr, "bad code record at %zu\n", o); return 2; }
          Block& b = blocks[h.id]; b.addr = h.addr; b.ins.resize(h.nins);
          const InsMeta* mm = (const InsMeta*)(p + o); o += h.nins * sizeof(InsMeta);
          b.bytes.assign(p + o, p + o + h.nbytes); b.bytes.resize(h.nbytes + 32, 0); o += h.nbytes;
          const uint8_t* itp = p + o; o += 4ull * h.nitems;
          uint64_t ip = h.addr;
          for (uint32_t k = 0; k < h.nins; k++) {
              Ins& in = b.ins[k]; in.ip = ip; memcpy(&in.m, &mm[k], sizeof(InsMeta)); in.bytes = nullptr; ip += mm[k].len;
              for (int q = 0; q < in.m.nbefore + in.m.nafter; q++, itp += 4) {
                  Item it{(ItemKind)itp[0], -1, itp[2]};
                  if (it.kind == IT_REG) it.off = itp[1] < 16 ? GPOFF[itp[1]] : -1;
                  else if (it.kind == IT_XMM_LO || it.kind == IT_XMM_HI) { int sz; it.off = VexInterp::reg_off("xmm" + std::to_string(itp[1]), &sz); if (it.kind == IT_XMM_HI) it.off += 8; }
                  else if (it.kind == IT_FSBASE) it.off = OFFSET_amd64_FS_CONST;
                  if (it.kind != IT_MEM && it.kind != IT_SHMEM && it.off < 0) { fprintf(stderr, "bad item\n"); return 2; }
                  (itp[3] ? in.own.after : in.own.before).push_back(it);
              }
              if (in.m.nbefore + in.m.nafter) in.site = &in.own;
              b.wmask |= in.m.gpw;
          }
          b.end = ip;
          if (h.nins) { const InsMeta& t = mm[h.nins - 1]; b.lmode = t.pad0 & 3;
              if (b.lmode == 2) { b.nr = (t.pad0 >> 2) & 3; b.rr[0] = t.pad0 >> 4; b.rr[1] = t.pad1 & 15; b.rr[2] = t.pad1 >> 4; } }
      } }

    size_t flen, cvlen, gtlen = 0;
    const uint8_t* fmap = map_file(dir + "/flow" + sfx, &flen);
    const uint8_t* cvmap = map_file(dir + "/cv" + sfx, &cvlen, false);
    const uint8_t* gtmap = use_gt ? map_file(dir + "/pgt" + sfx, &gtlen) : nullptr;
    const uint32_t* flow = (const uint32_t*)(fmap + 16); const uint64_t nflow = (flen - 16) / 4;
    const uint64_t* cv = cvmap ? (const uint64_t*)(cvmap + 16) : nullptr; const uint64_t ncv = cvmap ? (cvlen - 16) / 8 : 0;
    size_t taglen = 0; const uint8_t* tagmap = map_file(dir + "/cvtag" + sfx, &taglen, false);
    const uint64_t* tag = tagmap ? (const uint64_t*)(tagmap + 16) : nullptr; const uint64_t ntag = tagmap ? (taglen - 16) / 8 : 0;
    uint64_t tag_mismatch = 0; uint64_t cur_fi = 0; bool cur_after = false;
    const GtRec* gt = gtmap ? (const GtRec*)(gtmap + 16) : nullptr; const uint64_t ngt = gtmap ? (gtlen - 16) / 16 : 0;
    // windowed gt (hififlow -gtstart).  pgtwin holds {K, ns}: the thread's gt starts at its flow flush that
    // completed K flow records.  Everything is replayed; gt is compared from flow record K-1 (if it owns the first
    // gt record: the record written at the flush) else from K.  No pgtwin file = gt from the start (as before).
    const GtRec* const gt_all = gt; uint64_t win_k = 0, win_ns = 0; bool win_on = true;
    if (gt) { size_t wl = 0; const uint8_t* wm = map_file(dir + "/pgtwin" + sfx, &wl, false);
        if (wm && wl >= 32) { win_k = ((const uint64_t*)(wm + 16))[0]; win_ns = ((const uint64_t*)(wm + 16))[1]; win_on = false; gt = nullptr; }
        else if (ngt) fprintf(stderr, "hifirecon: no pgtwin: gt compared from the start\n"); }
    uint64_t win_fi = 0;

    VexLifter lifter; VexInterp vi;
    // tls.<pid>.<tid>.bin (hififlow): the %fs base the kernel gave this thread at creation.  A non-main
    // thread never runs arch_prctl, so without it every %fs-relative access of the thread starts unknown.
    uint64_t fs_seeded = 0, own_lo = 0, own_hi = 0, memop_dropped = 0;
    { size_t tl; const uint8_t* tp = map_file(dir + "/tls" + sfx, &tl, false);
      if (tp && tl >= 24) { uint64_t fs; memcpy(&fs, tp + 16, 8); if (fs) { vi.set_reg(OFFSET_amd64_FS_CONST, 8, fs); fs_seeded = fs; } }
      // (--mt-shadow): in a multi-threaded process another thread may write any non-stack memory (and
      // another thread's stack) between this thread's logged load and its next read of the same bytes, so
      // the shadow serves only this thread's OWN stack mapping [lo, hi) (hififlow tls.bin words 2-3):
      // stores outside it are forgotten and logged memop values outside it are not kept.  1 = only when
      // the process has more than one flow stream (thread), 2 = always, 0 = off (single-thread rule).
      if (tp && tl >= 48 && mt_shadow) { uint64_t r[2]; memcpy(r, tp + 32, 16);
        int nthr = 0; { std::string pre = "flow." + std::to_string(pid) + "."; DIR* dd = opendir(dir.c_str());
          if (dd) { for (struct dirent* de; (de = readdir(dd)); ) if (!strncmp(de->d_name, pre.c_str(), pre.size())) nthr++; closedir(dd); } }
        if (r[0] < r[1] && (mt_shadow == 2 || nthr > 1)) { own_lo = r[0]; own_hi = r[1]; vi.stack_lo = r[0]; vi.stack_hi = r[1]; }
        fprintf(stderr, "hifirecon: threads=%d own stack [%#lx,%#lx) mt_shadow=%d %s\n", nthr, (unsigned long)r[0], (unsigned long)r[1], mt_shadow, own_hi ? "APPLIED" : "off"); } }
    uint64_t shm_n = 0, shm_ea_unknown = 0, shm_own = 0, shm_diff = 0, shm_same = 0, shm_used = 0, shm_pa = 0; int shm_ps = 0;
    uint64_t cvpos = 0, gtpos = 0, n_insn = 0, n_rec = 0, n_rec_unknown = 0, n_lift_fail = 0, n_lift_fail_mem = 0, n_hint_ea = 0;
    uint64_t g_ident = 0, g_unk = 0, g_wrong = 0, g_recon_only = 0, g_gt_ip_mismatch = 0, cv_exhausted = 0, memop_ea_unknown = 0;
    uint64_t after_on_branch = 0, missing_block = 0, reported = 0, shadow_clear_ua = 0, ua_store_stack_gt = 0; bool ua_store = false;
    // --ua-rule: logged memop values the shadow holds OUTSIDE the own stack (only possible without --mt-shadow:
    // under it, IT_MEM outside the own stack is dropped and IT_SHMEM is forgotten after its instruction).
    std::unordered_set<uint64_t> ua_heapdefs; uint64_t ua_clear_all = 0, ua_clear_outside = 0, ua_forgot = 0;
    // Call stack of the callee-saved registers (SysV: rbx rbp r12-r15) as they were at each call.
    struct CallRec { uint64_t ret, sp; Val r[6]; };
    static const int CSR[6] = {3, 5, 12, 13, 14, 15};
    std::vector<CallRec> calls; uint64_t abi_restored = 0, abi_dropped = 0;
    std::vector<MemAccess> acc; acc.reserve(64);
    // diag: ring of 'load-unknown events' (a load at a KNOWN address left a written GPR unknown)
    struct LUE { uint64_t ip, ea, gtpos, insn; };
    // diag (DIAG_TRAIL=prefix, DIAG_TRAIL_IP=hex): ring of the last 2^20 replayed instructions (ip, gt span,
    // GPRs unknown after, bytes); dumped at the first 4 executions of DIAG_TRAIL_IP whose %rsi is unknown.
    struct TR { uint64_t ip, g0; uint32_t ng; uint16_t unk; uint8_t len, pad; uint8_t by[16]; };
    static std::vector<TR> trail; static uint64_t trn = 0; static const char* trp = getenv("DIAG_TRAIL");
    static uint64_t trip = getenv("DIAG_TRAIL_IP") ? strtoull(getenv("DIAG_TRAIL_IP"), 0, 16) : 0; static int trd = 0;
    if (trp) trail.resize(1u << 20); LUE lue[16]; unsigned lue_n = 0; std::map<uint64_t,int> repunk_seen;
    auto emitf = std::function<void(const MemAccess&)>([&](const MemAccess& m) { acc.push_back(m); });

    auto ea_of = [&](const Ins& in, bool* known) -> uint64_t {
        const InsMeta& m = in.m; *known = true; uint64_t a = (uint64_t)m.disp;
        if (m.base == 16) a += in.ip + m.len;
        else if (m.base < 16) { Val v = vi.get_reg(GPOFF[m.base], 8); if (!v.known()) *known = false; a += v.u64(); }
        else if (m.base != 255) *known = false;
        if (m.index < 16) { Val v = vi.get_reg(GPOFF[m.index], 8); if (!v.known()) *known = false; a += v.u64() * m.scale; }
        else if (m.index != 255) *known = false;
        if (m.seg == 1) { Val v = vi.get_reg(OFFSET_amd64_FS_CONST, 8); if (!v.known()) *known = false; a += v.u64(); }
        else if (m.seg == 2) *known = false;
        return a;
    };
    auto next_cv = [&](bool* ok) -> uint64_t { if (cvpos >= ncv) { *ok = false; cv_exhausted++; return 0; } *ok = true; return cv[cvpos++]; };
    auto apply = [&](const Ins& in, const std::vector<Item>& items) {
        for (const Item& it : items) {
            if (tag && cvpos < ntag) {   // debug stream: the site ip Pin logged this value at
                uint64_t want = in.ip | (cur_after ? (1ull << 63) : 0);
                if (tag[cvpos] != want && tag_mismatch++ < 5)
                    fprintf(stderr, "CVTAG mismatch at cv %lu: recon ip %#lx%s, pin logged %#lx%s (flow %lu)\n", (unsigned long)cvpos,
                            (unsigned long)in.ip, cur_after ? " after" : "", (unsigned long)(tag[cvpos] & ~(1ull << 63)), (tag[cvpos] >> 63) ? " after" : "", (unsigned long)cur_fi);
            }
            bool ok; uint64_t v = next_cv(&ok); if (!ok) return;
            // -logshared: the value a load read from memory this thread's shadow cannot vouch for.  Used
            // for that one load when the address is outside the own stack (inside it the shadow is
            // authoritative), then forgotten again under --mt-shadow so no later unlogged load reuses a value
            // another thread may have changed.  A shadow value that disagrees is counted (shm_diff).
            if (it.kind == IT_SHMEM) { bool k; uint64_t a = ea_of(in, &k); shm_n++;
                if (!k) { shm_ea_unknown++; continue; }
                Val spv = vi.get_reg(OFFSET_amd64_RSP, 8);   // the single-thread shadow window (VexInterp::in_stack_window)
                const bool own = own_hi ? (a >= own_lo && a + it.size <= own_hi) : (spv.known() && a + 4096 >= spv.u64() && a < spv.u64() + (16u << 20));
                if (own) { shm_own++; continue; }
                Val old = vi.shadow_load(a, it.size); uint64_t msk = it.size >= 8 ? ~0ull : ((1ull << (8 * it.size)) - 1);
                if (old.known()) { if ((old.u64() & msk) != (v & msk)) shm_diff++; else shm_same++; }
                vi.memop_define(a, v & msk, it.size); shm_used++; if (own_hi) { shm_pa = a; shm_ps = it.size; }
                continue; }
            if (it.kind == IT_MEM) { bool k; uint64_t a = ea_of(in, &k); if (k && (!own_hi || (a >= own_lo && a + it.size <= own_hi))) { vi.memop_define(a, v, it.size);
                    if (ua_rule && !own_hi) { Val spv = vi.get_reg(OFFSET_amd64_RSP, 8); if (!(spv.known() && a + 4096 >= spv.u64() && a < spv.u64() + (16u << 20)))
                        for (uint64_t q = a >> 3; q <= (a + it.size - 1) >> 3; q++) ua_heapdefs.insert(q); } } else if (k) memop_dropped++; else memop_ea_unknown++; }
            else vi.set_reg(it.off, 8, v);
        }
    };

    // Self-loop run-length records (hififlow -rle 1): flow entry id|2^31 = that block executed
    // (count + 1) times in a row, count taken from loop.<pid>.<tid>.bin in order.
    size_t lplen = 0; const uint8_t* lpmap = map_file(dir + "/loop" + sfx, &lplen, false);
    const uint64_t* lpc = lpmap ? (const uint64_t*)(lpmap + 16) : nullptr; const uint64_t nlp = lpmap ? (lplen - 16) / 8 : 0;
    uint64_t lppos = 0, loop_exhausted = 0, rle_iters = 0;
    uint64_t proj_hits = 0, wd_hits = 0, wd_insn = 0, fixpt_hits = 0, heavy_n = 0, next_prog = 1ull << 30; uint64_t elided_loops = 0, elided_iters = 0, elide_unresolved = 0, elide_unresolved_clear = 0, elide_unresolved_gt = 0, elide_inv_mismatch = 0;
    auto addr_of = [&](uint64_t f) -> uint64_t { auto it = blocks.find(flow[f] & 0x7fffffffu); return it == blocks.end() ? 0 : it->second.addr; };
    // context changes (hififlow ctx.<pid>.<tid>.bin, 22 words each): at flow position pos the thread
    // continues in a context the replay cannot derive -- a signal handler on a kernel-built frame, or the
    // register file rt_sigreturn reloads.  Take the new GPRs and fs base from the record; vector registers
    // and flag inputs become unknown; the frame the kernel wrote below the old %rsp is forgotten.
    size_t ctxlen = 0; const uint8_t* ctxmap = map_file(dir + "/ctx" + sfx, &ctxlen, false);
    const uint64_t* ctxr = ctxmap ? (const uint64_t*)(ctxmap + 16) : nullptr; const uint64_t nctx = ctxmap ? (ctxlen - 16) / 176 : 0;
    uint64_t ctxpos = 0, ctx_applied = 0;
    uint64_t trunc_flow = 0, trunc_insn_at = 0, dflag_abi = 0, dflag_lost = 0;   // --stop-trunc: flow entries not replayed (incomplete tail)
    for (uint64_t fi = 0; fi < nflow; fi++) {
        if (!win_on && win_k && fi + 1 >= win_k) {
            bool now = fi >= win_k;
            if (!now && ngt) { auto wb = blocks.find(flow[fi] & 0x7fffffffu); uint64_t ip0 = gt_all[0].meta & 0xffffffffffffull;
                now = wb != blocks.end() && ip0 >= wb->second.addr && ip0 < wb->second.end; }
            if (now) { win_on = true; gt = gt_all; win_fi = fi;
                fprintf(stderr, "hifirecon: gt window opens at flow %lu (K=%lu, t=%.3f s), insn %lu\n", (unsigned long)fi, (unsigned long)win_k, win_ns / 1e9, (unsigned long)n_insn); }
        }
        while (ctxpos < nctx && ctxr[ctxpos * 22] <= fi) {
            const uint64_t* r = ctxr + ctxpos * 22; const uint32_t why = (uint32_t)r[1];
            if (why == 1 || why == 2) {   // CONTEXT_CHANGE_REASON_SIGNAL / _SIGRETURN
                if (why == 1) { if (r[3 + 4] < r[20] && r[20] - r[3 + 4] < (1u << 20)) vi.shadow_forget(r[3 + 4], (int)(r[20] - r[3 + 4])); else vi.shadow_clear(); }
                for (int q = 0; q < 16; q++) vi.set_reg(GPOFF[q], 8, r[3 + q]);
                vi.set_reg(OFFSET_amd64_FS_CONST, 8, r[19]);
                for (int q = 0; q < 16 * 4; q++) vi.set_reg_unknown(OFFSET_amd64_YMM0 + 8 * q, 8);
                vi.set_reg_unknown(OFFSET_amd64_CC_DEP1, 8); vi.set_reg_unknown(OFFSET_amd64_CC_DEP2, 8); vi.set_reg_unknown(OFFSET_amd64_CC_NDEP, 8);
            } else vi.all_unknown();
            if (verbose) fprintf(stderr, "CTX flow=%lu reason=%u rip=%#lx rsp=%#lx old_rsp=%#lx\n", (unsigned long)fi, why, (unsigned long)r[2], (unsigned long)r[3 + 4], (unsigned long)r[20]);
            ctxpos++; ctx_applied++;
        }
        auto bit = blocks.find(flow[fi] & 0x7fffffffu);
        if (bit == blocks.end()) { missing_block++; vi.all_unknown(); continue; }
        Block& b = bit->second;
        // the capture ends inside the run (-gtcap's PIN_ExitApplication, a killed process): the flow
        // stream can hold blocks whose critical values (cv) or loop counts never reached their streams.
        // Replaying such a block with values missing produces wrong addresses (1305 in the last 17 k
        // instructions).  Stop BEFORE the first block whose cv items (upper bound: every before/after item,
        // times the RLE count) or loop record is not fully present; everything from there on, in the flow
        // and in the gt log, is reported as TRUNCATED, never as unknown or wrong.
        if (stop_trunc) {
            // per iteration: every before item, and every after item except on a block-ending conditional
            // branch, whose after items run only when it falls through (at most once for a self-loop)
            uint64_t need = 0, once = 0;
            for (size_t k = 0; k < b.ins.size(); k++) { const Ins& in = b.ins[k]; if (!in.site) continue;
                need += in.site->before.size();
                if ((in.m.flags & F_CONDBR) && k + 1 == b.ins.size()) once += in.site->after.size(); else need += in.site->after.size(); }
            bool short_loop = false; uint64_t rr = 1;
            if (flow[fi] & 0x80000000u) { if (b.lmode == 2) short_loop = lppos + b.nr > nlp; else if (lppos < nlp) rr = lpc[lppos] + 1; else short_loop = true; }
            if (short_loop || cvpos + need * rr + once > ncv) { trunc_flow = nflow - fi; trunc_insn_at = n_insn; break; }
        }
        uint64_t reps = 1;
        // Elided self-loop (hififlow -elide 1, lmode 2): the loop stream holds the exit-test registers
        // R at the exit; replay until R matches (first match == exit, see hififlow.cpp elide_regs).
        bool elided = false; uint64_t V[3] = {0, 0, 0};
        if (flow[fi] & 0x80000000u) {
            if (b.lmode == 2) {
                if (lppos + b.nr <= nlp) { elided = true; for (int q = 0; q < b.nr; q++) V[q] = lpc[lppos++]; elided_loops++; }
                else loop_exhausted++;
            } else { if (lppos < nlp) reps = lpc[lppos++] + 1; else loop_exhausted++; rle_iters += reps; }
        }
        if (elided)   // a register the loop never writes holds V at entry too: define it, check it
            for (int q = 0; q < b.nr; q++) if (!(b.wmask & (1u << b.rr[q]))) {
                Val v = vi.get_reg(GPOFF[b.rr[q]], 8);
                if (v.known() && v.u64() != V[q]) elide_inv_mismatch++;
                vi.set_reg(GPOFF[b.rr[q]], 8, V[q]);
            }
        bool unresolved = false; const uint64_t fi_insn0 = n_insn; uint64_t prevR[16]; bool havePrev = false; int stx = 0;
        uint64_t E[3] = {0, 0, 0}, D[3] = {0, 0, 0}, P[3] = {0, 0, 0}; int Ek[3] = {0, 0, 0}; bool affine = true; const uint64_t lppos_e = lppos;
        if (elided) for (int q = 0; q < b.nr; q++) { Val v = vi.get_reg(GPOFF[b.rr[q]], 8); Ek[q] = v.known(); E[q] = v.u64(); P[q] = E[q]; }
        const uint64_t after_addr = fi + 1 < nflow ? addr_of(fi + 1) : 0;
        for (uint64_t rep_i = 0; ; rep_i++) {
        bool last = !elided && rep_i + 1 >= reps;
        uint64_t next_addr = last ? after_addr : b.addr;
        if (!b.lifted) {
            for (auto& in : b.ins) {
                size_t o = in.ip - b.addr; in.bytes = b.bytes.data() + o;
                in.blk = lifter.lift_one(in.ip, in.bytes, b.bytes.size() - o);
            }
            b.lifted = true;
        }
        for (size_t k = 0; k < b.ins.size(); k++) {
            Ins& in = b.ins[k]; n_insn++;
            cur_fi = fi; cur_after = false;
            // A callee returns the callee-saved registers it was given (the ABI the static analysis
            // already assumes: the HiFi plan never re-logs them after a call).  The reconstruction
            // normally gets them back through the callee's push/pop of its stack shadow; a shadow
            // clear inside the callee (unknown-address store, unresolved elided loop) loses them, and
            // with them e.g. CPython's tstate in %r15 for the rest of an outer eval activation.
            // At the return site (same %rsp as at the call) refill only registers that are UNKNOWN
            // with the value they had at the call; a known value is never overridden.  Frames whose
            // %rsp has been popped past (longjmp, unwinding) are dropped without restoring.
            if (abi_restore && !calls.empty()) { Val spv = vi.get_reg(OFFSET_amd64_RSP, 8);
                if (spv.known()) { const uint64_t sp = spv.u64();
                    while (!calls.empty() && sp > calls.back().sp) { calls.pop_back(); abi_dropped++; }
                    if (!calls.empty() && sp == calls.back().sp && in.ip == calls.back().ret) {
                        for (int q = 0; q < 6; q++) if (calls.back().r[q].known() && !vi.get_reg(GPOFF[CSR[q]], 8).known()) { vi.set_reg(GPOFF[CSR[q]], 8, calls.back().r[q].u64()); abi_restored++; }
                        calls.pop_back(); } } }
            if (in.site && !in.site->before.empty()) apply(in, in.site->before);
            acc.clear();
            Val sp0v = vi.get_reg(OFFSET_amd64_RSP, 8); const bool sp_unk = !sp0v.known(); const uint64_t sp0 = sp0v.u64();
            // could a store of this instruction through an unknown address alias the own stack?  Decided on
            // the PRE-execution registers: the destination's base register (REP: %rdi) points into the own stack
            // (rsp-derived) or is unknown, or there is no base register / a segment / no explicit memory operand.
            // A known base outside the own stack with an unknown index cannot reach the stack without leaving its object.
            bool ua_alias = true;
            if (ua_rule && !sp_unk) {
                auto in_own = [&](uint64_t v) { return own_hi ? (v >= own_lo && v < own_hi) : (v + 4096 >= sp0 && v < sp0 + (16u << 20)); };
                int br = (in.m.flags & F_REP) ? 7 : ((in.m.flags & F_HASMEM) && in.m.seg == 0 && in.m.base < 16 ? in.m.base : -1);
                if ((in.m.flags & F_HASMEM) && !(in.m.flags & F_REP) && in.m.seg == 0 && in.m.base == 16) ua_alias = false;   // rip-relative: never the stack
                else if (br >= 0) { Val bv = vi.get_reg(GPOFF[br], 8); ua_alias = !bv.known() || in_own(bv.u64()); }
            }
            if (!in.blk->ok) {
                n_lift_fail++;
                for (int r = 0; r < 16; r++) if (in.m.gpw & (1u << r)) vi.set_reg_unknown(GPOFF[r], 8);
                if (in.m.flags & F_HASMEM) {
                    n_lift_fail_mem++; bool kn; uint64_t a = ea_of(in, &kn);
                    for (int q = 0; q < in.m.ngt; q++) acc.push_back(MemAccess{kn ? a : 0, kn, (int)in.m.msize, (in.m.flags & F_MEMW) ? 1 : 0});
                    if (in.m.flags & F_MEMW) { if (kn) vi.shadow_forget(a, in.m.msize); else vi.shadow_clear(); }
                }
            } else if (in.m.flags & F_REP) {
                Val rcx = vi.get_reg(OFFSET_amd64_RCX, 8);
                if (trp && in.ip == trip && trd < 4 && !vi.get_reg(OFFSET_amd64_RSI, 8).known()) {
                    std::string fn = std::string(trp) + "." + std::to_string(trd++); FILE* tf = fopen(fn.c_str(), "w");
                    uint64_t n0 = trn > trail.size() ? trn - trail.size() : 0;
                    for (uint64_t q = n0; q < trn; q++) { const TR& e = trail[q & (trail.size() - 1)];
                        fprintf(tf, "%lx %lu %u %x ", (unsigned long)e.ip, (unsigned long)e.g0, e.ng, e.unk); for (int z = 0; z < e.len && z < 16; z++) fprintf(tf, "%02x", e.by[z]); fprintf(tf, "\n"); }
                    fprintf(tf, "%lx %lu 0 0 ", (unsigned long)in.ip, (unsigned long)gtpos); for (int z = 0; z < in.m.len && z < 16; z++) fprintf(tf, "%02x", in.bytes[z]); fprintf(tf, "\n");
                    fclose(tf); }
                { static uint64_t dip = getenv("DIAG_IP") ? strtoull(getenv("DIAG_IP"), 0, 16) : 0; static int dn = 0;
                  if (in.ip == dip && dn++ < 8) { fprintf(stderr, "DIAG %#lx insn=%lu rsi=%d rdi=%d rcx=%d rsp=%d calls=%zu:", (unsigned long)in.ip, (unsigned long)n_insn,
                      (int)vi.get_reg(OFFSET_amd64_RSI,8).known(), (int)vi.get_reg(OFFSET_amd64_RDI,8).known(), (int)rcx.known(), (int)vi.get_reg(OFFSET_amd64_RSP,8).known(), calls.size());
                    for (size_t q = 0; q < 6 && q < calls.size(); q++) fprintf(stderr, " %#lx", (unsigned long)calls[calls.size()-1-q].ret);
                    fprintf(stderr, "\n"); for (unsigned q = 0; q < 16 && q < lue_n; q++) { const LUE& e = lue[(lue_n - 1 - q) & 15]; fprintf(stderr, "  LUE %#lx ea=%#lx insn=%lu\n", (unsigned long)e.ip, (unsigned long)e.ea, (unsigned long)e.insn); } } }
                if (!rcx.known() || rcx.u64() > (1u << 26)) {
                    if (verbose > 0 && repunk_seen[in.ip]++ < 4) { fprintf(stderr, "REPUNK2 %#lx insn=%lu gtpos=%lu rdx=%d\n", (unsigned long)in.ip, (unsigned long)n_insn, (unsigned long)gtpos, (int)vi.get_reg(OFFSET_amd64_RDX,8).known());
                        for (unsigned q = 0; q < 16 && q < lue_n; q++) { const LUE& e = lue[(lue_n - 1 - q) & 15]; fprintf(stderr, "  LUE %#lx ea=%#lx gtpos=%lu insn=%lu\n", (unsigned long)e.ip, (unsigned long)e.ea, (unsigned long)e.gtpos, (unsigned long)e.insn); } }
                    if (verbose > 1) fprintf(stderr, "REPUNK %#lx r8=%d rdx=%d rdi=%d rsi=%d fs=%d insn=%lu\n", (unsigned long)in.ip, (int)vi.get_reg(OFFSET_amd64_R8,8).known(), (int)vi.get_reg(OFFSET_amd64_RDX,8).known(), (int)vi.get_reg(OFFSET_amd64_RDI,8).known(), (int)vi.get_reg(OFFSET_amd64_RSI,8).known(), (int)vi.get_reg(OFFSET_amd64_FS_CONST,8).known(), (unsigned long)n_insn);
                    acc.push_back(MemAccess{0, false, 1, 2});
                    vi.set_reg_unknown(OFFSET_amd64_RSI, 8); vi.set_reg_unknown(OFFSET_amd64_RDI, 8);
                    vi.set_reg_unknown(OFFSET_amd64_RAX, 8); vi.set_reg_unknown(OFFSET_amd64_RCX, 8);
                } else {
                    uint64_t n = rcx.u64();
                    { static uint64_t rip = getenv("DIAG_REP") ? strtoull(getenv("DIAG_REP"), 0, 16) : 0; static int rn = 0;
                      if (in.ip == rip && rn++ < 12) fprintf(stderr, "DIAG-REP %#lx insn=%lu n=%lu rdi=%d rsi=%d dflag=%d items=%zu fi=%lu\n", (unsigned long)in.ip, (unsigned long)n_insn, (unsigned long)n,
                          (int)vi.get_reg(OFFSET_amd64_RDI, 8).known(), (int)vi.get_reg(OFFSET_amd64_RSI, 8).known(), (int)vi.get_reg(OFFSET_amd64_DFLAG, 8).known(),
                          in.site ? in.site->before.size() : (size_t)0, (unsigned long)fi); }
                    for (uint64_t i = 0; i < n; i++) { vi.exec(*in.blk, emitf); Val r2 = vi.get_reg(OFFSET_amd64_RCX, 8);
                        if (verbose > 2) fprintf(stderr, "rep %#lx i=%lu n=%lu acc=%zu rcx_known=%d rcx=%lu\n", (unsigned long)in.ip, (unsigned long)i, (unsigned long)n, acc.size(), (int)r2.known(), (unsigned long)r2.u64());
                        if (!r2.known() || r2.u64() == 0) break; }
                }
            } else {
                const bool is_sys = in.blk->jk == Ijk_Sys_syscall;
                Val sysno = is_sys ? vi.get_reg(OFFSET_amd64_RAX, 8) : Val::U(8);
                Val code = is_sys ? vi.get_reg(OFFSET_amd64_RDI, 8) : Val::U(8);
                if (abi_restore && in.blk->jk == Ijk_Call && !sp_unk) {
                    if (calls.size() >= (1u << 20)) { calls.erase(calls.begin(), calls.begin() + (1u << 19)); abi_dropped += 1u << 19; }
                    CallRec c; c.ret = in.ip + in.m.len; c.sp = sp0; for (int q = 0; q < 6; q++) c.r[q] = vi.get_reg(GPOFF[CSR[q]], 8); calls.push_back(c); }
                bool hk = false; uint64_t hea = 0; const bool hint = hint_ea && in.m.ngt && (in.m.flags & F_HASMEM);
                if (hint) hea = ea_of(in, &hk);
                // the direction flag is part of the SysV call ABI (DF clear at every call and return).  A
                // popf of an unknown stack word (or a context the replay cannot derive) would leave guest_DFLAG
                // unknown for the rest of the thread, so every later REP string op would lose all addresses after
                // its first element.  Re-establish DF = 0 at calls and returns, never overriding a known value.
                const bool df_known = vi.get_reg(OFFSET_amd64_DFLAG, 8).known();
                if (!df_known && (in.blk->jk == Ijk_Call || in.blk->jk == Ijk_Ret)) { vi.set_reg(OFFSET_amd64_DFLAG, 8, 1); dflag_abi++; }
                vi.exec(*in.blk, emitf);
                if (df_known && !vi.get_reg(OFFSET_amd64_DFLAG, 8).known() && dflag_lost++ < 5 && verbose > 0)
                    fprintf(stderr, "DFLAG-LOST ip=%#lx insn=%lu\n", (unsigned long)in.ip, (unsigned long)n_insn);
                if (hint && acc.empty()) { acc.push_back(MemAccess{hea, hk, (int)in.m.msize, 0}); n_hint_ea++; }
                if (is_sys) {
                    vi.set_reg_unknown(OFFSET_amd64_RAX, 8); vi.set_reg_unknown(OFFSET_amd64_RCX, 8); vi.set_reg_unknown(OFFSET_amd64_R11, 8);
                    if (sysno.known() && sysno.u64() == 158 && (!code.known() || code.u64() == 0x1002)) vi.set_reg_unknown(OFFSET_amd64_FS_CONST, 8);
                }
            }
            // A store through an UNKNOWN address while %rsp is unknown may have hit any stack slot the
            // shadow holds (an alloca left %rsp unknown, the callee's push %r13 went to an unknown
            // address, and its pop later "restored" the stale stack value the shadow still had at that
            // slot -> wrong addresses in ld.so).  Forget the shadow there, as the lift-fail path does.
            // With %rsp known, the stack slots a store can reach are rsp-relative and known; the
            // remaining case (an unknown pointer that targets the stack) is counted against the gt below.
            ua_store = false; for (auto& a : acc) if (!a.known && a.op != 0) { ua_store = true; break; }
            if (ua_store && (sp_unk || !vi.get_reg(OFFSET_amd64_RSP, 8).known())) { vi.shadow_clear(); shadow_clear_ua++; ua_store = false; ua_heapdefs.clear(); }
            else if (ua_store && ua_rule) {
                if (ua_alias) { vi.shadow_clear(); ua_clear_all++; ua_store = false; ua_heapdefs.clear(); }
                else { for (uint64_t q : ua_heapdefs) vi.shadow_forget(q << 3, 8); ua_forgot += ua_heapdefs.size(); ua_heapdefs.clear(); ua_clear_outside++; }
                // ua_store stays set after an outside-only clear: the UA-STORE-TO-STACK check below then counts any
                // such store the gt shows DID hit the stack (the rule's soundness check).
            }
            if (elided && k + 1 == b.ins.size()) {
                // exit test of this iteration: 1 = R == V (exit), 0 = continue, -1 = cannot decide
                int st = 1;
                for (int q = 0; q < b.nr && st == 1; q++) {
                    if (!(b.wmask & (1u << b.rr[q]))) continue;
                    Val v = vi.get_reg(GPOFF[b.rr[q]], 8);
                    if (!v.known()) st = -1; else if (v.u64() != V[q]) st = 0;
                }
                if (st == 0 && rep_i >= (1ull << elide_cap)) st = -1;
                // fixed point.  The loop rewrote every register it writes with the value it already held,
                // and it cannot write memory: the next iteration recomputes exactly this state, so R can never
                // reach V (a spin loop whose exit depends on another thread's store).  Unresolved now, not after
                // 2^elide_cap identical iterations.
                if (st == 0) { bool same = havePrev, allk = true; uint64_t cur[16];
                    for (int r = 0; r < 16; r++) if (b.wmask & (1u << r)) { Val v = vi.get_reg(GPOFF[r], 8); if (!v.known()) { allk = false; break; } cur[r] = v.u64(); if (havePrev && cur[r] != prevR[r]) same = false; }
                    if (allk) { bool wr = false; for (auto& i2 : b.ins) if (i2.m.flags & (F_MEMW | F_REP | F_SYSCALL) || !i2.blk || !i2.blk->ok) { wr = true; break; }
                        if (same && !wr) { st = -1; fixpt_hits++; } else { for (int r = 0; r < 16; r++) prevR[r] = cur[r]; havePrev = true; } } }
                // projected exit.  A written exit register that moved by the same nonzero stride d on every
                // iteration so far reaches V after (V-R)/d more iterations only if d divides V-R and the quotient is
                // within 2^elide_cap; otherwise the replay cannot meet V by stepping (the entry state disagrees with
                // the recorded exit).  Declaring it unresolved early loses nothing the cap would not lose later.
                if (st == 0) for (int q = 0; q < b.nr; q++) { if (!(b.wmask & (1u << b.rr[q]))) continue;
                    Val v = vi.get_reg(GPOFF[b.rr[q]], 8); if (!v.known()) break; uint64_t d = v.u64() - P[q]; P[q] = v.u64();
                    if (rep_i == 0) D[q] = d; else if (d != D[q]) affine = false;
                    if (affine && rep_i >= 15 && D[q]) { uint64_t gap = V[q] - v.u64(); bool ok = false;
                        if (D[q] == 1) ok = gap <= (1ull << elide_cap); else if (D[q] == ~0ull) ok = -gap <= (1ull << elide_cap);
                        else { int64_t sd = (int64_t)D[q]; int64_t sg = (int64_t)gap; ok = sd && sg % sd == 0 && sg / sd >= 0 && (uint64_t)(sg / sd) <= (1ull << elide_cap); }
                        if (!ok) { st = -1; proj_hits++; break; } } }
                stx = st;
                if (st != 0) last = true;
                if (st == -1) unresolved = true;
                next_addr = last ? after_addr : b.addr;
                elided_iters++;
            }
            if (shm_ps) { vi.shadow_forget(shm_pa, shm_ps); shm_ps = 0; }
            // IPOINT_AFTER values: hifitool inserts them only on instructions with a fall-through, and
            // Pin runs them only when the fall-through path is taken.
            if (in.site && !in.site->after.empty() && (in.m.flags & F_FALL)) {
                bool take = true;
                if ((in.m.flags & F_CONDBR) && k + 1 == b.ins.size()) {
                    after_on_branch++;
                    take = next_addr == in.ip + in.m.len;
                }
                cur_after = true;
                if (take) apply(in, in.site->after);
            }
            if (verbose > 0 && (in.m.flags & F_MEMR) && !(in.m.flags & F_REP) && in.m.gpw) { bool kl = false; uint64_t ea = 0;
                for (auto& a : acc) if (a.known && a.op == 0) { kl = true; ea = a.addr; break; }
                if (kl) for (int r = 0; r < 16; r++) if ((in.m.gpw & (1u << r)) && r != 4 && !vi.get_reg(GPOFF[r], 8).known()) { lue[lue_n++ & 15] = LUE{in.ip, ea, gtpos, n_insn}; { static FILE* df = getenv("DIAG_LUE") ? fopen(getenv("DIAG_LUE"), "w") : 0; if (df) fprintf(df, "%lx %lx %lu\n", (unsigned long)in.ip, (unsigned long)ea, (unsigned long)gtpos); } break; } }
            // ---- records + comparison (ground truth is read only here)
            Img* im = img_of(in.ip); if (!im) im = &other;
            for (auto& a : acc) { n_rec++; im->recs++; if (!a.known) { n_rec_unknown++; im->recunk++; } }
            if (gt) {
                size_t first = gtpos;
                // One instance of a non-REP instruction owns at most its own memory-operand count of
                // gt records: a tight loop whose only memory instruction is this one produces
                // back-to-back records at the same ip that belong to DIFFERENT executions.
                const size_t lim = (in.m.flags & F_REP) ? (size_t)-1 : (size_t)in.m.ngt;
                while (gtpos < ngt && gtpos - first < lim && (gt[gtpos].meta & 0xffffffffffffull) == in.ip) gtpos++;
                size_t ng = gtpos - first;
                if (ng == 0 && in.m.ngt && !acc.empty() && gtpos < ngt && verbose > 1)
                    fprintf(stderr, "no gt for %#lx (next gt ip %#lx)\n", (unsigned long)in.ip, (unsigned long)(gt[gtpos].meta & 0xffffffffffffull));
                bool any_unknown = false; for (auto& a : acc) if (!a.known) any_unknown = true;
                size_t matched = 0;
                // a long REP string op has one access and one gt record per element; matching each record
                // against every access is quadratic (a multi-MB REP on the Java main thread stalled recon for >20 min).
                // Same predicate, through a set of the known addresses.
                static std::unordered_set<uint64_t> kset; const bool big = acc.size() > 64;
                if (big) { kset.clear(); for (auto& a : acc) if (a.known) kset.insert(a.addr); }
                for (size_t q = first; q < gtpos; q++) {
                    uint64_t ea = gt[q].ea; bool hit = false;
                    if (big) hit = kset.count(ea) != 0;
                    else for (auto& a : acc) if (a.known && a.addr == ea) { hit = true; break; }
                    im->gt++;
                    if (hit) { g_ident++; im->ident++; matched++; }
                    else if (any_unknown) { g_unk++; im->unk++; g_unk_ip[in.ip]++; }
                    else { g_wrong++; im->wrong++;
                        if (reported++ < max_report) fprintf(stderr, "WRONG ip=%#lx gt_ea=%#lx recon=[%s%#lx] flow=%lu insn=%lu\n", (unsigned long)in.ip, (unsigned long)ea,
                            acc.empty() ? "none " : "", acc.empty() ? 0ul : (unsigned long)acc[0].addr, (unsigned long)fi, (unsigned long)n_insn); }
                }
                if (ua_store) for (size_t q = first; q < gtpos; q++) if (gt[q].ea + 4096 >= sp0 && gt[q].ea < sp0 + (16u << 20)) {
                    if (ua_store_stack_gt++ < 5) fprintf(stderr, "UA-STORE-TO-STACK ip=%#lx gt_ea=%#lx rsp=%#lx insn=%lu\n", (unsigned long)in.ip, (unsigned long)gt[q].ea, (unsigned long)sp0, (unsigned long)n_insn);
                    break; }
                // recon-only: a reconstructed KNOWN address that no gt record of this instance has
                // (a read-modify-write that VEX splits into load+store is one gt operand, not extra).
                for (auto& a : acc) { if (!a.known) continue; bool hit = false;
                    for (size_t q = first; q < gtpos; q++) if (gt[q].ea == a.addr) { hit = true; break; }
                    if (!hit) g_recon_only++; }
                if (trp) { TR& e = trail[trn++ & (trail.size() - 1)]; e.ip = in.ip; e.g0 = first; e.ng = (uint32_t)(gtpos - first); e.unk = 0;
                    for (int r = 0; r < 16; r++) if (!vi.get_reg(GPOFF[r], 8).known()) e.unk |= 1u << r;
                    e.len = in.m.len; memcpy(e.by, in.bytes, in.m.len < 16 ? in.m.len : 16);
                    // diag (DIAG_TRAIL_IPS=hex,hex,...): also dump the trail at the first DIAG_TRAIL_N (2) executions of each
                    // listed ip that produced an unknown-address record.
                    static std::map<uint64_t,int> tips; static bool tinit = false; if (!tinit) { tinit = true; const char* s = getenv("DIAG_TRAIL_IPS");
                        while (s && *s) { char* e2; uint64_t v = strtoull(s, &e2, 16); if (e2 == s) break; tips[v] = 0; s = *e2 ? e2 + 1 : e2; } }
                    static const int tn = getenv("DIAG_TRAIL_N") ? atoi(getenv("DIAG_TRAIL_N")) : 2;
                    if (any_unknown && !tips.empty()) { auto ti = tips.find(in.ip); if (ti != tips.end() && ti->second < tn) {
                        std::string fn = std::string(trp) + "." + std::to_string(in.ip) + "." + std::to_string(ti->second++); FILE* tf = fopen(fn.c_str(), "w");
                        uint64_t n0 = trn > trail.size() ? trn - trail.size() : 0;
                        for (uint64_t q = n0; q < trn; q++) { const TR& e3 = trail[q & (trail.size() - 1)];
                            fprintf(tf, "%lx %lu %u %x ", (unsigned long)e3.ip, (unsigned long)e3.g0, e3.ng, e3.unk); for (int z = 0; z < e3.len && z < 16; z++) fprintf(tf, "%02x", e3.by[z]); fprintf(tf, "\n"); }
                        fclose(tf); } } }
            }
        }
        if (!last && n_insn - fi_insn0 > max_insn_flow) {   // watchdog (see --max-insn-flow)
            wd_hits++; wd_insn += n_insn - fi_insn0; unresolved = true;
            fprintf(stderr, "WATCHDOG flow=%lu block=%#lx-%#lx lmode=%d elided=%d reps=%lu iters=%lu insns=%lu gtpos=%lu: remaining iterations unresolved\n",
                    (unsigned long)fi, (unsigned long)b.addr, (unsigned long)b.end, b.lmode, (int)elided, (unsigned long)reps, (unsigned long)rep_i + 1, (unsigned long)(n_insn - fi_insn0), (unsigned long)gtpos);
            break; }
        if (last) break;
        }   // rep_i
        if (verbose > 0 && n_insn - fi_insn0 > (1ull << 24) && heavy_n++ < 40)
            { fprintf(stderr, "HEAVY flow=%lu block=%#lx-%#lx ninsn=%zu lmode=%d elided=%d reps=%lu insns=%lu exit=%d unresolved=%d gtpos=%lu lppos=%lu/%lu", (unsigned long)fi, (unsigned long)b.addr, (unsigned long)b.end,
                    b.ins.size(), b.lmode, (int)elided, (unsigned long)reps, (unsigned long)(n_insn - fi_insn0), stx, (int)unresolved, (unsigned long)gtpos, (unsigned long)lppos_e, (unsigned long)nlp);
                for (int q = 0; q < b.nr; q++) fprintf(stderr, " r%d:entry=%s%#lx,V=%#lx,w=%d", b.rr[q], Ek[q] ? "" : "?", (unsigned long)E[q], (unsigned long)V[q], (int)!!(b.wmask & (1u << b.rr[q])));
                fprintf(stderr, "\n"); }
        if (verbose > 0 && n_insn >= next_prog) { next_prog = n_insn + (1ull << 30); fprintf(stderr, "PROGRESS flow %lu/%lu insn %lu gtpos %lu\n", (unsigned long)fi, (unsigned long)nflow, (unsigned long)n_insn, (unsigned long)gtpos); }
        if (unresolved) {
            // Loss, never a guess: the iterations after this one are unknown.  Their gt records are
            // the ones at this block's ips (the fall-through successor starts at b.end); the
            // registers the loop writes become unknown except R, which the exit log gives.
            elide_unresolved++;
            for (int r = 0; r < 16; r++) if (b.wmask & (1u << r)) vi.set_reg_unknown(GPOFF[r], 8);
            if (elided) for (int q = 0; q < b.nr; q++) vi.set_reg(GPOFF[b.rr[q]], 8, V[q]);
            else {   // watchdog stop of an RLE loop: no exit registers recorded; vector and flag state unknown too
                for (int q = 0; q < 16 * 4; q++) vi.set_reg_unknown(OFFSET_amd64_YMM0 + 8 * q, 8);
                vi.set_reg_unknown(OFFSET_amd64_CC_DEP1, 8); vi.set_reg_unknown(OFFSET_amd64_CC_DEP2, 8); vi.set_reg_unknown(OFFSET_amd64_CC_NDEP, 8); }
            // The skipped iterations run only this block.  If none of its instructions can write
            // memory, memory is what it was and the shadow stays valid (clearing it here lost every
            // callee-saved register spilled up the call chain, e.g. CPython's tstate in %r15 for the
            // rest of an outer _PyEval_EvalFrameDefault activation, 0.29 % of pyperf float).
            bool writes = false;
            for (auto& in : b.ins) if (in.m.flags & (F_MEMW | F_REP | F_SYSCALL) || !in.blk || !in.blk->ok) { writes = true; break; }
            if (writes) { vi.shadow_clear(); elide_unresolved_clear++; }
            if (gt) while (gtpos < ngt) { uint64_t ip = gt[gtpos].meta & 0xffffffffffffull; if (ip < b.addr || ip >= b.end) break;
                Img* im = img_of(ip); if (!im) im = &other; im->gt++; im->unk++; g_unk++; g_unk_ip[ip | (1ull<<63)]++; gtpos++; elide_unresolved_gt++; }
        }
    }
    if (verbose > 0) { std::vector<std::pair<uint64_t,uint64_t>> v(g_unk_ip.begin(), g_unk_ip.end());
        std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.second > b.second; });
        for (size_t i = 0; i < v.size() && i < (getenv("UNKTOP") ? strtoull(getenv("UNKTOP"),0,10) : 25); i++) fprintf(stderr, "UNK-IP %#lx%s %lu\n", (unsigned long)(v[i].first & ~(1ull<<63)), (v[i].first>>63) ? " (elide)" : "", (unsigned long)v[i].second); }
    uint64_t gt_tail = gt ? ngt - gtpos : 0;
    const uint64_t truncated = trunc_flow ? gt_tail : 0;   // gt records after the stop: neither unknown nor wrong
    fprintf(stderr, "hifirecon: direction flag lost %lu times, re-established at call/ret %lu\n", (unsigned long)dflag_lost, (unsigned long)dflag_abi);
    if (trunc_flow) fprintf(stderr, "hifirecon: TRUNCATED tail: stopped at flow %lu of %lu (insn %lu), cv %lu of %lu consumed; gt records truncated %lu\n",
            (unsigned long)(nflow - trunc_flow), (unsigned long)nflow, (unsigned long)trunc_insn_at, (unsigned long)cvpos, (unsigned long)ncv, (unsigned long)truncated);
    uint64_t cmp = g_ident + g_unk + g_wrong;
    fprintf(stderr, "hifirecon: flow=%lu blocks(versions)=%zu insns=%lu records=%lu unknown_records=%lu lift_fail=%lu (mem %lu) missing_block=%lu\n",
            (unsigned long)nflow, blocks.size(), (unsigned long)n_insn, (unsigned long)n_rec, (unsigned long)n_rec_unknown, (unsigned long)n_lift_fail,
            (unsigned long)n_lift_fail_mem, (unsigned long)missing_block);
    fprintf(stderr, "hifirecon: rle loop records %lu of %lu consumed (exhausted %lu), iterations %lu\n", (unsigned long)lppos, (unsigned long)nlp, (unsigned long)loop_exhausted, (unsigned long)rle_iters);
    fprintf(stderr, "hifirecon: elided loops %lu, iterations replayed %lu, unresolved %lu (shadow cleared %lu, gt records skipped as unknown %lu), invariant mismatches %lu\n",
            (unsigned long)elided_loops, (unsigned long)elided_iters, (unsigned long)elide_unresolved, (unsigned long)elide_unresolved_clear, (unsigned long)elide_unresolved_gt, (unsigned long)elide_inv_mismatch);
    fprintf(stderr, "hifirecon: watchdog hits %lu (insns %lu, limit %lu per flow record), elided fixed points %lu, projected-exit stops %lu\n", (unsigned long)wd_hits, (unsigned long)wd_insn, (unsigned long)max_insn_flow, (unsigned long)fixpt_hits, (unsigned long)proj_hits);
    if (tag) fprintf(stderr, "hifirecon: cvtag mismatches %lu (tag stream %lu values)\n", (unsigned long)tag_mismatch, (unsigned long)ntag);
    if (hint_ea) fprintf(stderr, "hifirecon: hint-ea address-only operands taken from registers %lu\n", (unsigned long)n_hint_ea);
    if (ua_rule) fprintf(stderr, "hifirecon: ua-rule unknown-address stores with %%rsp known: shadow cleared %lu, outside-own-stack only %lu (words forgotten %lu)\n",
            (unsigned long)ua_clear_all, (unsigned long)ua_clear_outside, (unsigned long)ua_forgot);
    if (own_hi) fprintf(stderr, "hifirecon: mt-shadow memop values outside own stack not kept %lu\n", (unsigned long)memop_dropped);
    if (nctx) fprintf(stderr, "hifirecon: context changes applied %lu of %lu\n", (unsigned long)ctx_applied, (unsigned long)nctx);
    fprintf(stderr, "hifirecon: cv consumed %lu of %lu (exhausted %lu) memop_ea_unknown=%lu after_on_branch=%lu shadow_clear_unknown_store=%lu unknown_store_to_stack_gt=%lu abi_restored=%lu abi_dropped=%lu\n",
            (unsigned long)cvpos, (unsigned long)ncv, (unsigned long)cv_exhausted, (unsigned long)memop_ea_unknown, (unsigned long)after_on_branch, (unsigned long)shadow_clear_ua, (unsigned long)ua_store_stack_gt, (unsigned long)abi_restored, (unsigned long)abi_dropped);
    fprintf(stderr, "hifirecon: thread fs_base seeded %#lx\n", (unsigned long)fs_seeded);
    if (gt) fprintf(stderr, "hifirecon: gt %lu records; compared %lu: identical %lu, unknown %lu (%.6f %%), wrong %lu (%.6f %%); recon_only %lu; gt_tail %lu\n",
            (unsigned long)ngt, (unsigned long)cmp, (unsigned long)g_ident, (unsigned long)g_unk, cmp ? 100.0 * g_unk / cmp : 0.0,
            (unsigned long)g_wrong, cmp ? 100.0 * g_wrong / cmp : 0.0, (unsigned long)g_recon_only, (unsigned long)gt_tail);
    else   /* a thread whose -gtstart window never opened is reported, not silently omitted */
        fprintf(stderr, "hifirecon: NO WINDOW: gt %lu records, pgtwin %s, flow %lu records; nothing compared\n", (unsigned long)ngt,
                gt_all ? (win_k || win_ns ? "present (window flow index not reached)" : "absent") : "absent (no gt file)", (unsigned long)nflow);
    std::vector<Img*> order; for (auto& m : imgs) order.push_back(&m); order.push_back(&other);
    if (getenv("JITRANGES") && gt) {   // "lo hi kind" lines (kind: interp | nmethod | stub); classify window gt + unknowns
        std::vector<std::tuple<uint64_t,uint64_t,std::string>> rg; FILE* rf = fopen(getenv("JITRANGES"), "r"); char kb[64]; unsigned long lo, hi;
        while (rf && fscanf(rf, "%lx %lx %63s", &lo, &hi, kb) == 3) rg.emplace_back(lo, hi, kb); if (rf) fclose(rf);
        std::sort(rg.begin(), rg.end());
        auto kind = [&](uint64_t ip) -> std::string { auto it = std::upper_bound(rg.begin(), rg.end(), std::make_tuple(ip, ~0ull, std::string("~")));
            if (it != rg.begin()) { --it; if (ip < std::get<1>(*it)) return std::get<2>(*it); } Img* m = img_of(ip); return m ? base_name(m->name) : "other"; };
        std::map<std::string, uint64_t> kg, ku;
        for (uint64_t q = 0; q < gtpos && q < ngt; q++) kg[kind(gt[q].meta & 0xffffffffffffull)]++;
        for (auto& e : g_unk_ip) ku[kind(e.first & ~(1ull << 63))] += e.second;
        for (auto& e : kg) fprintf(stderr, "region %-40s gt=%lu unknown=%lu\n", e.first.c_str(), (unsigned long)e.second, (unsigned long)ku[e.first]);
    }
    std::sort(order.begin(), order.end(), [](Img* a, Img* b) { return a->gt + a->recs > b->gt + b->recs; });
    for (Img* m : order) if (m->gt || m->recs)
        fprintf(stderr, "  image %-50s gt=%lu identical=%lu unknown=%lu wrong=%lu | records=%lu unknown=%lu\n", base_name(m->name).c_str(),
                (unsigned long)m->gt, (unsigned long)m->ident, (unsigned long)m->unk, (unsigned long)m->wrong, (unsigned long)m->recs, (unsigned long)m->recunk);
    if (!json.empty()) {
        FILE* f = fopen(json.c_str(), "w");
        if (f) {
            fprintf(f, "{\"flow\":%lu,\"insns\":%lu,\"records\":%lu,\"unknown_records\":%lu,\"lift_fail\":%lu,\"missing_block\":%lu,"
                       "\"cv_total\":%lu,\"cv_consumed\":%lu,\"cv_exhausted\":%lu,\"gt_records\":%lu,\"compared\":%lu,\"identical\":%lu,"
                       "\"unknown\":%lu,\"wrong\":%lu,\"recon_only\":%lu,\"gt_tail\":%lu,\"truncated\":%lu,\"truncated_flow\":%lu,\"elided_loops\":%lu,\"elided_iters\":%lu,\"elide_unresolved\":%lu,\"elide_inv_mismatch\":%lu,\"rle_iters\":%lu,\"shm\":%lu,\"shm_used\":%lu,\"shm_diff\":%lu,\"shm_same\":%lu,\"shm_own\":%lu,\"shm_ea_unknown\":%lu,\"images\":[",
                    (unsigned long)nflow, (unsigned long)n_insn, (unsigned long)n_rec, (unsigned long)n_rec_unknown, (unsigned long)n_lift_fail,
                    (unsigned long)missing_block, (unsigned long)ncv, (unsigned long)cvpos, (unsigned long)cv_exhausted, (unsigned long)ngt,
                    (unsigned long)cmp, (unsigned long)g_ident, (unsigned long)g_unk, (unsigned long)g_wrong, (unsigned long)g_recon_only, (unsigned long)gt_tail, (unsigned long)truncated, (unsigned long)trunc_flow,
                    (unsigned long)elided_loops, (unsigned long)elided_iters, (unsigned long)elide_unresolved, (unsigned long)elide_inv_mismatch, (unsigned long)rle_iters,
                    (unsigned long)shm_n, (unsigned long)shm_used, (unsigned long)shm_diff, (unsigned long)shm_same, (unsigned long)shm_own, (unsigned long)shm_ea_unknown);
            bool first = true;
            for (Img* m : order) if (m->gt || m->recs) { fprintf(f, "%s{\"image\":\"%s\",\"gt\":%lu,\"identical\":%lu,\"unknown\":%lu,\"wrong\":%lu,\"records\":%lu,\"unknown_records\":%lu}",
                    first ? "" : ",", m->name.c_str(), (unsigned long)m->gt, (unsigned long)m->ident, (unsigned long)m->unk, (unsigned long)m->wrong, (unsigned long)m->recs, (unsigned long)m->recunk); first = false; }
            fprintf(f, "]}\n"); fclose(f);
        }
    }
    return (cv_exhausted || missing_block) ? 1 : 0;   // a --stop-trunc stop exhausts nothing: rc 0
}
