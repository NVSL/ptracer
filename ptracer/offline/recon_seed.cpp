// recon_seed.cpp -- SEEDED --jobs chunks (anchor-free, exact).
//
// A chunk that does not start at an anchor needs the machine state a serial run has at its first
// instruction.  The state is produced in two parallel rounds (recon_par.cpp):
//   round 1  chunk 0 runs for real and exports its end state; every other chunk runs ONCE with a
//            tagged FAKE seed (every register lane = fake64(lane), every stack chunk it touches first
//            = fake64(128 + i)) and exports its final state;
//   parent   chains P[k+1] = resolve(pass-1 final of chunk k, P[k]), P[1] = end of chunk 0 -- a value
//            that is fake64(S) + d is "seed S plus d", anything else is seed-independent;
//   round 2  chunk k >= 1 runs for real, importing P[k] where its own range starts, and exports
//            its entry and end state;
//   verify   entry[k] must equal end[k-1] byte for byte (the whole state); a chunk whose prediction
//            was wrong is re-run with the true state (end[k-1]) and the check moves on.
// So the record stream is the serial one by construction; the fake run is only a predictor.
#include "recon.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <libvex_guest_offsets.h>

namespace {
struct SeedState {
    uint32_t magic = 0x44535450u, version = 1; uint8_t fake = 0;
    // history (value-stream state)
    uint64_t cv_pos = 0; uint8_t cv_unaligned = 0, pre_first_sync = 0, anchored = 0, sl_pending = 0;
    int64_t sync_arm = 0, sync_err = 0; uint32_t sync_mark_period = 0;
    int64_t sl_vals_rel = 0, sl_sync_rel = 0;
    std::vector<std::pair<uint32_t, uint64_t>> tramp_cv;
    std::vector<uint64_t> delta_val; std::vector<uint8_t> delta_valid;
    // instruction-stream state
    int64_t pending_delta = -1, pending_kf = -1, ptw_pend_ent = -1; uint64_t kf_fired_branch = 0;
    std::vector<std::pair<uint64_t, int32_t>> ptw_q; uint8_t ptw_pending = 0; int32_t ptw_kind = 0;
    uint64_t ptw_pend_ip = 0, ptw_pend_ea = 0; int32_t ptw_pend_off = -1, ptw_pend_size = 0, ptw_pend_memsize = 0; uint8_t ptw_pend_mem_legacy = 0;
    uint8_t tnt_on = 0; int32_t tnt_k = 0; uint64_t tnt_v = 0;
    uint64_t cursor = 0; uint8_t cursor_valid = 0, tramp_fallback = 0, in_ovf = 0; uint64_t cur_tramp = 0;
    uint64_t prev_end = 0, last_ip_orig = 0; uint8_t in_tramp = 0, eo = 0;
    std::vector<uint8_t> machine;          // VexInterp::dump_state
    // pass-1 extras (zero in a true state)
    uint8_t cleared = 0, dinv = 0, pages_ovf = 0; uint64_t cv_start = 0, n_sync_used = 0, n_switch_loss = 0, consumed = 0;
    std::vector<uint64_t> fab_keys, pages;
    // uncertainty (parent side): guest-state bytes whose value is seed-dependent but not expressible (prediction), or
    // that differed between the two real runs of a chunk with an uncertain entry (end state); empty = none
    std::vector<uint8_t> unc;
};
struct W { std::vector<uint8_t> b;
    template <class T> void pod(const T& x) { const uint8_t* p = (const uint8_t*)&x; b.insert(b.end(), p, p + sizeof x); }
    template <class T> void operator()(T& x) { pod(x); }
    template <class A, class B> void operator()(std::vector<std::pair<A, B>>& v) { uint64_t n = v.size(); pod(n); for (auto& e : v) { pod(e.first); pod(e.second); } }
    template <class T> void operator()(std::vector<T>& v) { uint64_t n = v.size(); pod(n); if (n) { const uint8_t* p = (const uint8_t*)v.data(); b.insert(b.end(), p, p + n * sizeof(T)); } } };
struct R { const uint8_t* p; const uint8_t* e; bool ok = true;
    template <class T> void pod(T& x) { if ((size_t)(e - p) < sizeof x) { ok = false; return; } memcpy(&x, p, sizeof x); p += sizeof x; }
    template <class T> void operator()(T& x) { pod(x); }
    template <class A, class B> void operator()(std::vector<std::pair<A, B>>& v) { uint64_t n = 0; pod(n); if (!ok || n > (uint64_t)(e - p)) { ok = false; return; } v.resize(n); for (auto& x : v) { pod(x.first); pod(x.second); } }
    template <class T> void operator()(std::vector<T>& v) { uint64_t n = 0; pod(n); if (!ok || n * sizeof(T) > (uint64_t)(e - p)) { ok = false; return; } v.resize(n); if (n) { memcpy(v.data(), p, n * sizeof(T)); p += n * sizeof(T); } } };
template <class IO> void visit(IO& io, SeedState& s) {
    io(s.magic); io(s.version); io(s.fake);
    io(s.cv_pos); io(s.cv_unaligned); io(s.pre_first_sync); io(s.anchored); io(s.sl_pending);
    io(s.sync_arm); io(s.sync_err); io(s.sync_mark_period); io(s.sl_vals_rel); io(s.sl_sync_rel);
    io(s.tramp_cv); io(s.delta_val); io(s.delta_valid);
    io(s.pending_delta); io(s.pending_kf); io(s.ptw_pend_ent); io(s.kf_fired_branch);
    io(s.ptw_q); io(s.ptw_pending); io(s.ptw_kind); io(s.ptw_pend_ip); io(s.ptw_pend_ea);
    io(s.ptw_pend_off); io(s.ptw_pend_size); io(s.ptw_pend_memsize); io(s.ptw_pend_mem_legacy);
    io(s.tnt_on); io(s.tnt_k); io(s.tnt_v);
    io(s.cursor); io(s.cursor_valid); io(s.tramp_fallback); io(s.in_ovf); io(s.cur_tramp);
    io(s.prev_end); io(s.last_ip_orig); io(s.in_tramp); io(s.eo);
    io(s.machine);
    io(s.cleared); io(s.dinv); io(s.pages_ovf); io(s.cv_start); io(s.n_sync_used); io(s.n_switch_loss); io(s.consumed);
    io(s.fab_keys); io(s.pages); io(s.unc);
}
bool save(const SeedState& s0, const std::string& path) {
    SeedState s = s0; W w; visit(w, s);
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb"); if (!f) return false;
    bool ok = fwrite(w.b.data(), 1, w.b.size(), f) == w.b.size(); ok = fclose(f) == 0 && ok;
    return ok && rename(tmp.c_str(), path.c_str()) == 0;
}
bool load(SeedState& s, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
    std::vector<uint8_t> b; uint8_t buf[1 << 16]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    fclose(f);
    R r{b.data(), b.data() + b.size()}; visit(r, s);
    return r.ok && r.p == r.e && s.magic == 0x44535450u && s.version == 1;
}
// the machine section, parsed
struct Mach { uint8_t g[1024], gk[1024]; std::map<uint64_t, std::pair<uint64_t, uint8_t>> sh;   // key -> (8 bytes, known mask)
    bool parse(const std::vector<uint8_t>& m) {
        if (m.size() < 2056) return false; memcpy(g, m.data(), 1024); memcpy(gk, m.data() + 1024, 1024);
        uint64_t n; memcpy(&n, m.data() + 2048, 8); if (m.size() != 2056 + n * 17) return false;
        const uint8_t* p = m.data() + 2056;
        for (uint64_t i = 0; i < n; i++, p += 17) { uint64_t k, v; memcpy(&k, p, 8); memcpy(&v, p + 8, 8); sh[k] = {v, p[16]}; }
        return true; }
    void dump(std::vector<uint8_t>& m) const {
        m.assign(g, g + 1024); m.insert(m.end(), gk, gk + 1024);
        uint64_t n = 0; for (auto& e : sh) if (e.second.second) n++;
        const uint8_t* np = (const uint8_t*)&n; m.insert(m.end(), np, np + 8);
        for (auto& e : sh) { if (!e.second.second) continue; uint64_t k = e.first, v = e.second.first; uint8_t kn = e.second.second;
            uint64_t vv = 0; for (int i = 0; i < 8; i++) if ((kn >> i) & 1) vv |= v & (0xffull << (8 * i));
            const uint8_t* kp = (const uint8_t*)&k; m.insert(m.end(), kp, kp + 8);
            const uint8_t* vp = (const uint8_t*)&vv; m.insert(m.end(), vp, vp + 8); m.push_back(kn); } }
    uint64_t lane(int l, uint8_t* kn) const { uint64_t v; memcpy(&v, g + 8 * l, 8); uint8_t m = 0; for (int i = 0; i < 8; i++) if (gk[8 * l + i]) m |= 1 << i; *kn = m; return v; }
    // 8 bytes of memory at a (any alignment) and their known mask
    uint64_t read8(uint64_t a, uint8_t* kn) const { uint64_t v = 0; uint8_t m = 0;
        for (int i = 0; i < 8; i++) { const uint64_t x = a + i; auto it = sh.find(x >> 3); if (it == sh.end()) continue; const int o = (int)(x & 7);
            if ((it->second.second >> o) & 1) { v |= ((it->second.first >> (8 * o)) & 0xff) << (8 * i); m |= 1 << i; } }
        *kn = m; return v; }
    void write_byte(uint64_t x, uint8_t b, bool known) {
        auto it = sh.find(x >> 3); const int o = (int)(x & 7);
        if (it == sh.end()) { if (!known) return; it = sh.emplace(x >> 3, std::make_pair((uint64_t)0, (uint8_t)0)).first; }
        if (known) { it->second.first = (it->second.first & ~(0xffull << (8 * o))) | ((uint64_t)b << (8 * o)); it->second.second |= 1 << o; }
        else { it->second.second &= ~(1 << o); if (!it->second.second) sh.erase(it); } }
};
// ---- two fake runs: run 1 seeds S with fake(S, 0), run 2 with fake(S, 1) = fake(S, 0) + delta(S) (both halves
// shifted, no carry).  A value equal in both runs does not depend on the seed; otherwise v2 - v1 = sum a_i delta(S_i)
// names the seeds it is a linear function of (one term with |a| <= 8, or two register terms with a = +-1), and its
// true value is v1 + sum a_i (T(S_i) - fake(S_i, 0)).  A 4-byte half is decoded the same way with one term.
struct Resolver {
    const Mach& T; const SeedState& p1; const SeedState& p2;
    uint64_t unres = 0; uint32_t nsym;
    std::function<bool(uint64_t, uint64_t*)> cv; uint64_t cv0 = 0;
    std::unordered_map<uint64_t, std::pair<uint32_t, int>> m64;   // a * delta(S) -> (S, a); a = 0: ambiguous
    std::unordered_map<uint32_t, std::pair<uint32_t, int>> m32;   // a * delta_half(S) -> (S, a * (half ? 2 : 1) sign-coded)
    mutable int depth = 0;
    std::unordered_map<uint32_t, std::pair<uint64_t, uint8_t>> memo;
    Resolver(const Mach& t, const SeedState& a, const SeedState& b) : T(t), p1(a), p2(b) {
        nsym = 128 + (uint32_t)p1.fab_keys.size();
        static const int co[] = {1, -1, 2, -2, 4, -4, 8, -8};
        (void)co;
        for (uint32_t S = 0; S < nsym; S++) {
            const uint64_t d = VexInterp::fake_delta(S);
            for (int a : {1, -1}) { auto r = m64.emplace(d * (uint64_t)(int64_t)a, std::make_pair(S, a)); if (!r.second) r.first->second.second = 0; }
            // 4-byte forms: registers and fabricated STACK chunks only (heap chunks and logged values are many and
            // the 32-bit delta space is small: their entries would make the stack ones ambiguous)
            if (S >= 128) { const uint64_t e = p1.fab_keys[S - 128]; if ((e >> 63) || (uint32_t)((e << 3) >> 32) < 0x5e000000u) continue; }
            for (int h = 0; h < 2; h++) for (int a : {1, -1}) {
                const uint32_t dh = (uint32_t)(h ? d >> 32 : d) * (uint32_t)a;
                auto r = m32.emplace(dh, std::make_pair(S, h ? 2 * a : a)); if (!r.second) r.first->second.second = 0; }
        }
    }
    // the truth's seed S (8 bytes, known mask)
    const uint8_t* tunc = nullptr; bool unc_hit = false;
    uint64_t seed(uint32_t S, uint8_t* kn) {
        if (S < 128) { if (tunc) for (int i = 0; i < 8; i++) if (tunc[8 * S + i]) unc_hit = true; return T.lane((int)S, kn); }
        auto it = memo.find(S); if (it != memo.end()) { *kn = it->second.second; return it->second.first; }
        uint64_t r = 0; uint8_t k = 0;
        const size_t i = S - 128;
        if (i < p1.fab_keys.size() && i < p2.fab_keys.size()) {
            const uint64_t e1 = p1.fab_keys[i], e2 = p2.fab_keys[i];
            if (e1 >> 63) { if (e1 == e2 && cv && cv(cv0 + (e1 & ~(1ull << 63)), &r)) k = 0xff; }
            else if (depth < 8) { depth++; uint8_t ak; uint64_t a = value(e1 << 3, 0xff, e2 << 3, 0xff, &ak); depth--;
                                  if (ak == 0xff) r = T.read8(a, &k); }
        }
        memo[S] = {r, k}; *kn = k; return r;
    }
    // n with n * delta(S) == R (mod 2^64), |n| < 2^24 (delta(S) = odd * 2^t: n is fixed mod 2^(64-t))
    std::vector<uint64_t> dinv; std::vector<int> dtz;
    void prep_inv() { if (!dinv.empty()) return; dinv.resize(nsym); dtz.resize(nsym);
        for (uint32_t S = 0; S < nsym; S++) { const uint64_t d = VexInterp::fake_delta(S); const int t = __builtin_ctzll(d); const uint64_t o = d >> t;
            uint64_t x = o; for (int i = 0; i < 6; i++) x *= 2 - o * x; dinv[S] = x; dtz[S] = t; } }
    bool coef(uint64_t R, uint32_t S, int64_t* n) const {
        const int t = dtz[S]; if (R & ((1ull << t) - 1)) return false;
        uint64_t q = (R >> t) * dinv[S]; if (t) { q &= (~0ull) >> t; if (q >> (63 - t)) q |= ~((~0ull) >> t); }   // sign-extend from 64-t bits
        const int64_t v = (int64_t)q; if (v == 0 || v <= -(1ll << 24) || v >= (1ll << 24)) return false; *n = v; return true; }
    // v2 - v1 = D = sum a_i delta(S_i): one term (any coefficient), or a +-1 term plus one more, or two +-1 register terms plus one more
    // (strides n * s are general-purpose registers or the flags thunk: lanes 2..21; deeper searches only for full=true)
    bool form64(uint64_t D, std::pair<uint32_t, int64_t>* t, int* nt, bool full) {
        prep_inv();
        const uint32_t lim = std::min<uint32_t>(nsym, 128 + 1024);
        int64_t n;
        { auto it = m64.find(D); if (it != m64.end() && it->second.second) { t[0] = {it->second.first, it->second.second}; *nt = 1; return true; } }   // +-1 * any seed
        for (uint32_t S = 0; S < 128; S++) if (coef(D, S, &n)) { t[0] = {S, n}; *nt = 1; return true; }
        if (!full || deep_left <= 0) return false;
        deep_left--;
        (void)lim;
        for (uint32_t S1 = 0; S1 < 128; S1++) for (int a : {1, -1}) {
            const uint64_t R = D - VexInterp::fake_delta(S1) * (uint64_t)(int64_t)a;
            for (uint32_t S2 = 2; S2 < 22; S2++) if (S2 != S1 && coef(R, S2, &n)) { t[0] = {S1, a}; t[1] = {S2, n}; *nt = 2; return true; } }
        if (!full3) return false;
        for (uint32_t S1 = 2; S1 < 18; S1++) for (int a1 : {1, -1}) for (uint32_t S2 = S1 + 1; S2 < 18; S2++) for (int a2 : {1, -1}) {
            const uint64_t R = D - VexInterp::fake_delta(S1) * (uint64_t)(int64_t)a1 - VexInterp::fake_delta(S2) * (uint64_t)(int64_t)a2;
            for (uint32_t S3 = 2; S3 < 22; S3++) if (S3 != S1 && S3 != S2 && coef(R, S3, &n)) { t[0] = {S1, a1}; t[1] = {S2, a2}; t[2] = {S3, n}; *nt = 3; return true; } }
        return false;
    }
    bool full_ = true, full3 = true; long deep_left = 20000;
    uint64_t value(uint64_t v1, uint8_t k1, uint64_t v2, uint8_t k2, uint8_t* okn) {
        if (v1 == v2 && k1 == k2) { *okn = k1; return v1; }
        if (k1 != k2) unc_hit = true;
        if (k1 == 0xff && k2 == 0xff) {
            std::pair<uint32_t, int64_t> t[3]; int nt = 0;
            if (form64(v2 - v1, t, &nt, full_)) {
                uint64_t r = v1; bool full = true;
                for (int i = 0; i < nt; i++) { uint8_t m; const uint64_t sv = seed(t[i].first, &m); if (m != 0xff) { full = false; break; }
                    r += (sv - VexInterp::fakev(t[i].first, 0)) * (uint64_t)t[i].second; }
                if (full) { *okn = 0xff; return r; }
                // a partly known seed: fall through to the 4-byte halves (e.g. a 32-bit counter in a stack chunk)
            }
        }
        uint64_t r = 0; uint8_t rk = 0;
        for (int h = 0; h < 2; h++) {
            const uint8_t m1 = (uint8_t)((k1 >> (4 * h)) & 0xf), m2 = (uint8_t)((k2 >> (4 * h)) & 0xf);
            const uint32_t h1 = (uint32_t)(v1 >> (32 * h)), h2 = (uint32_t)(v2 >> (32 * h));
            if (m1 != m2) { unres++; continue; }
            if (h1 == h2) { r |= (uint64_t)h1 << (32 * h); rk |= (uint8_t)(m1 << (4 * h)); continue; }
            if (m1 != 0xf) { unres++; continue; }
            auto it = m32.find(h2 - h1);
            if (it == m32.end() || !it->second.second) { unres++; unc_hit = true; continue; }   // seed-dependent, not expressible
            const uint32_t S = it->second.first; const int code = it->second.second; const int hh = (code == 2 || code == -2) ? 1 : 0; const int a = code > 0 ? 1 : -1;
            uint8_t m; const uint64_t sv = seed(S, &m);
            if (((m >> (4 * hh)) & 0xf) != 0xf) { unres++; continue; }
            const uint32_t th = (uint32_t)(sv >> (32 * hh)), fh = (uint32_t)(VexInterp::fakev(S, 0) >> (32 * hh));
            r |= (uint64_t)(uint32_t)(h1 + (th - fh) * (uint32_t)a) << (32 * h); rk |= (uint8_t)(0xf << (4 * h));
        }
        *okn = rk; return r;
    }
    // run 1's address a1 and the matching run-2 address (same seed form): the run-2 key set says which
    bool pair_addr(uint64_t k1, const std::function<bool(uint64_t)>& has2, uint64_t* k2) {
        if (has2(k1)) { *k2 = k1; return true; }
        // run 1's fake addresses are fake64(S) + d with |d| < 2^30: the high half names S directly
        const uint32_t hi = (uint32_t)((k1 << 3) >> 32);
        if (hi >= 0x5e000000u && hi < 0x5f000000u && !((hi - 0x5e000000u) & 0xff)) {
            const uint32_t S = (hi - 0x5e000000u) >> 8;
            if (S < nsym) { const uint64_t c = ((k1 << 3) + VexInterp::fake_delta(S)) >> 3; if (has2(c)) { *k2 = c; return true; } } }
        static const int pref[] = {OFFSET_amd64_RSP / 8, OFFSET_amd64_RBP / 8};
        for (int S : pref) { const uint64_t c = ((k1 << 3) + VexInterp::fake_delta((uint32_t)S)) >> 3; if (has2(c)) { *k2 = c; return true; } }
        return false;
    }
};
void fill(SeedState& s, const Mach& m) { m.dump(s.machine); }
const char* first_diff(const SeedState& a, const SeedState& b) {
#define D(f) if (a.f != b.f) return #f;
    if (a.prev_end != b.prev_end || a.last_ip_orig != b.last_ip_orig) { static char t[160]; snprintf(t, sizeof t, "prev_end %lx/%lx last_ip_orig %lx/%lx", (unsigned long)a.prev_end, (unsigned long)b.prev_end, (unsigned long)a.last_ip_orig, (unsigned long)b.last_ip_orig); return t; }
    D(cv_pos) D(cv_unaligned) D(pre_first_sync) D(anchored) D(sl_pending) D(sync_arm) D(sync_err) D(sync_mark_period) D(sl_vals_rel) D(sl_sync_rel)
    D(tramp_cv) D(delta_val) D(delta_valid) D(pending_delta) D(pending_kf) D(ptw_pend_ent) D(kf_fired_branch) D(ptw_q) D(ptw_pending) D(ptw_kind)
    D(ptw_pend_ip) D(ptw_pend_ea) D(ptw_pend_off) D(ptw_pend_size) D(ptw_pend_memsize) D(ptw_pend_mem_legacy) D(tnt_on) D(tnt_k) D(tnt_v)
    D(cursor) D(cursor_valid) D(tramp_fallback) D(in_ovf) D(cur_tramp) D(prev_end) D(last_ip_orig) D(in_tramp) D(eo) D(machine)
#undef D
    return nullptr;
}
}   // namespace

void Recon::seed_begin_fake(int var) {
    vi_.fake_begin(var);
    const bool have_cv = !cv_.empty() || !o_.cvfile.empty();
    seed_cv_start_ = have_cv ? (1ull << 40) : cv_pos_; cv_pos_ = (size_t)seed_cv_start_;
    cv_unaligned_ = have_cv; pre_first_sync_ = true; tramp_cv_.clear();
    anchored_ = false; switch_loss_pending_ = false; switch_loss_vals_ = 0; switch_loss_sync_ = 0;
    delta_valid_.assign(delta_valid_.size(), 0); seed_dinv_ = false;
    sync_arm_ = sync_rt_; sync_err_ = 0;
    seed_swl0_ = n_switch_loss_;
}

void Recon::seed_begin_unknown() {
    vi_.all_unknown();
    if (o_.seed_fs && o_.fs_base) vi_.set_reg(OFFSET_amd64_FS_CONST, 8, o_.fs_base);
    const bool have_cv = !cv_.empty() || !o_.cvfile.empty();
    seed_cv_start_ = have_cv ? (1ull << 40) : cv_pos_; cv_pos_ = (size_t)seed_cv_start_;
    cv_unaligned_ = have_cv; pre_first_sync_ = true; tramp_cv_.clear();
    anchored_ = false; switch_loss_pending_ = false; switch_loss_vals_ = 0; switch_loss_sync_ = 0;
    delta_valid_.assign(delta_valid_.size(), 0); seed_dinv_ = false;
    sync_arm_ = sync_rt_; sync_err_ = 0; seed_swl0_ = n_switch_loss_;
}

bool Recon::seed_export(const SeedLocals& L, const std::string& path, int fake_final) {
    SeedState s;
    const uint64_t consumed = n_ptw_used_ - n_sync_used_ + n_cv_used_ + n_delta_skipped_;
    s.cv_pos = cv_pos_; s.cv_unaligned = cv_unaligned_; s.pre_first_sync = pre_first_sync_; s.anchored = anchored_; s.sl_pending = switch_loss_pending_;
    s.sync_arm = sync_arm_; s.sync_err = sync_err_; s.sync_mark_period = sync_mark_period_;
    if (switch_loss_pending_) { s.sl_vals_rel = (int64_t)switch_loss_vals_ - (int64_t)consumed; s.sl_sync_rel = (int64_t)switch_loss_sync_ - (int64_t)n_sync_used_; }   // (meaningless otherwise)
    for (auto& p : tramp_cv_) s.tramp_cv.push_back({p.first, (uint64_t)p.second});
    if (has_delta()) { s.delta_val = delta_val_; s.delta_valid.assign(delta_valid_.begin(), delta_valid_.end()); }
    s.pending_delta = pending_delta_; s.pending_kf = pending_kf_; s.ptw_pend_ent = ptw_pend_ent_; s.kf_fired_branch = kf_fired_branch_;
    for (auto& q : ptw_q_) s.ptw_q.push_back({q.first, (int32_t)q.second});
    s.ptw_pending = ptw_pending_; s.ptw_kind = (int32_t)ptw_kind_; s.ptw_pend_ip = ptw_pend_ip_; s.ptw_pend_ea = ptw_pend_ea_;
    s.ptw_pend_off = ptw_pend_off_; s.ptw_pend_size = ptw_pend_size_; s.ptw_pend_memsize = ptw_pend_memsize_; s.ptw_pend_mem_legacy = ptw_pend_mem_legacy_;
    if (!ptw_pending_) { s.ptw_kind = 0; s.ptw_pend_ip = s.ptw_pend_ea = 0; s.ptw_pend_off = -1; s.ptw_pend_size = s.ptw_pend_memsize = 0; s.ptw_pend_mem_legacy = 0; }   // stale (read only while pending)
    s.tnt_on = tnt_.on; s.tnt_k = tnt_.on ? tnt_.k : 0; s.tnt_v = tnt_.on ? tnt_.v : 0;
    s.cursor = cursor_valid_ ? cursor_ : 0; s.cursor_valid = cursor_valid_; s.tramp_fallback = tramp_fallback_; s.in_ovf = in_ovf_; s.cur_tramp = cur_tramp_;
    s.prev_end = L.prev_end; s.last_ip_orig = L.last_ip_orig; s.in_tramp = L.in_tramp; s.eo = L.eo;
    vi_.dump_state(s.machine);
    if (fake_final) {
        s.fake = (uint8_t)fake_final; s.cleared = vi_.fake_.cleared; s.dinv = seed_dinv_; s.pages_ovf = vi_.fake_.pages_ovf;
        s.cv_start = seed_cv_start_; s.n_sync_used = n_sync_used_; s.n_switch_loss = n_switch_loss_ - seed_swl0_; s.consumed = consumed;
        s.fab_keys = vi_.fake_.keys;
        for (auto& pg : vi_.fake_.touched) for (int w = 0; w < 8; w++) for (uint64_t b = pg.second[w]; b; b &= b - 1)
            s.pages.push_back((pg.first << 9) | ((uint64_t)w << 6) | (uint64_t)__builtin_ctzll(b));      // an 8-byte chunk key
        std::sort(s.pages.begin(), s.pages.end()); s.pages_ovf = 0;
    }
    return save(s, path);
}

bool Recon::seed_import(const std::string& path, SeedLocals* L) {
    SeedState s; if (!load(s, path) || s.fake) { fprintf(stderr, "ptrecon: bad seed state %s\n", path.c_str()); return false; }
    if (!s.unc.empty()) {   // uncertain bytes: unknown (run a) or a distinct fake (run b, o_.seed_unc_fake)
        Mach m; if (!m.parse(s.machine)) return false;
        for (int i = 0; i < 1024 && i < (int)s.unc.size(); i++) if (s.unc[i]) {
            if (o_.seed_unc_fake) { m.g[i] = (uint8_t)(VexInterp::fake64(40000u + (uint32_t)i / 8) >> (8 * (i % 8))); m.gk[i] = 1; } else { m.g[i] = 0; m.gk[i] = 0; } }
        m.dump(s.machine);
    }
    const uint8_t* p = s.machine.data(); vi_.fake_.on = false;
    if (!vi_.load_state(p, s.machine.data() + s.machine.size())) { fprintf(stderr, "ptrecon: bad machine state in %s\n", path.c_str()); return false; }
    const uint64_t consumed = n_ptw_used_ - n_sync_used_ + n_cv_used_ + n_delta_skipped_;
    cv_pos_ = (size_t)s.cv_pos; cv_unaligned_ = s.cv_unaligned; pre_first_sync_ = s.pre_first_sync; anchored_ = s.anchored; switch_loss_pending_ = s.sl_pending;
    sync_arm_ = s.sync_arm; sync_err_ = s.sync_err; sync_mark_period_ = s.sync_mark_period;
    switch_loss_vals_ = (uint64_t)((int64_t)consumed + s.sl_vals_rel); switch_loss_sync_ = (uint64_t)((int64_t)n_sync_used_ + s.sl_sync_rel);
    tramp_cv_.clear(); for (auto& q : s.tramp_cv) tramp_cv_.emplace_back(q.first, (size_t)q.second);
    if (has_delta() && s.delta_val.size() == delta_val_.size()) { delta_val_ = s.delta_val; delta_valid_.assign(s.delta_valid.begin(), s.delta_valid.end()); }
    pending_delta_ = s.pending_delta; pending_kf_ = s.pending_kf; ptw_pend_ent_ = s.ptw_pend_ent; kf_fired_branch_ = s.kf_fired_branch;
    ptw_q_.clear(); for (auto& q : s.ptw_q) ptw_q_.push_back({q.first, (int)q.second});
    ptw_pending_ = s.ptw_pending; ptw_kind_ = (PtwKind)s.ptw_kind; ptw_pend_ip_ = s.ptw_pend_ip; ptw_pend_ea_ = s.ptw_pend_ea;
    ptw_pend_off_ = s.ptw_pend_off; ptw_pend_size_ = s.ptw_pend_size; ptw_pend_memsize_ = s.ptw_pend_memsize; ptw_pend_mem_legacy_ = s.ptw_pend_mem_legacy;
    tnt_.on = s.tnt_on; tnt_.k = s.tnt_k; tnt_.v = s.tnt_v;
    cursor_ = s.cursor; cursor_valid_ = s.cursor_valid; tramp_fallback_ = s.tramp_fallback; in_ovf_ = s.in_ovf; cur_tramp_ = s.cur_tramp;
    if (L) { L->prev_end = s.prev_end; L->last_ip_orig = s.last_ip_orig; L->in_tramp = s.in_tramp; L->eo = s.eo; }
    return true;
}

bool Recon::seed_resolve(const std::string& p1path, const std::string& entry, const std::string& out, std::string* why, const Recon* cvsrc, const std::string& p2path) {
    SeedState a, b, t;
    if (!load(a, p1path) || a.fake != 1) { if (why) *why = "no pass-1 state"; return false; }
    if (!load(b, p2path) || b.fake != 2) { if (why) *why = "no second pass-1 state"; return false; }
    if (!load(t, entry) || t.fake) { if (why) *why = "no entry state"; return false; }
    if (t.unc.size() != 1024) t.unc.assign(1024, 0);
    if (a.pages_ovf || b.pages_ovf || (a.pages.size() && (a.pages.back() >> 63))) { if (why) *why = "pass 1 noted too many keys"; return false; }
    if (a.fab_keys.size() != b.fab_keys.size() || a.cleared != b.cleared || a.cv_pos != b.cv_pos) { if (why) *why = "the two pass-1 runs diverged"; return false; }
    Mach T, A, B; if (!T.parse(t.machine) || !A.parse(a.machine) || !B.parse(b.machine)) { if (why) *why = "bad machine section"; return false; }
    Resolver rs(T, a, b);
    rs.cv0 = t.cv_pos; rs.tunc = t.unc.data();
    if (cvsrc) rs.cv = [cvsrc](uint64_t p, uint64_t* v) { if (!cvsrc->cv_have((size_t)p)) return false; *v = cvsrc->cv_at((size_t)p); return true; };
    SeedState P = t;     // history defaults: the entry state's
    P.unc.assign(1024, 0);
    // ---- registers
    Mach M; memset(M.g, 0, sizeof M.g); memset(M.gk, 0, sizeof M.gk);
    for (int l = 0; l < 128; l++) { uint8_t k1, k2, rk = 0; const uint64_t v1 = A.lane(l, &k1), v2 = B.lane(l, &k2); uint64_t r = 0;
        rs.full_ = 8 * l < OFFSET_amd64_YMM0;    // vector lanes: copies only
        rs.unc_hit = false;
        if (k1 | k2) r = rs.value(v1, k1, v2, k2, &rk);
        rs.full_ = true;
        if (rs.unc_hit) { for (int i = 0; i < 8; i++) P.unc[8 * l + i] = 1; r = 0; rk = 0; }
        memcpy(M.g + 8 * l, &r, 8); for (int i = 0; i < 8; i++) M.gk[8 * l + i] = (rk >> i) & 1;
        for (int i = 0; i < 8; i++) if (!M.gk[8 * l + i]) M.g[8 * l + i] = 0;
        if (getenv("PTRECON_SEED_DUMP") && l < 28) { uint8_t tk; uint64_t tv = T.lane(l, &tk); fprintf(stderr, "seedres lane%d %016lx/%016lx %02x/%02x -> %016lx/%02x (entry %016lx/%02x)\n", l, (unsigned long)v1, (unsigned long)v2, k1, k2, (unsigned long)r, rk, (unsigned long)tv, tk); } }
    // ---- shadow: the entry shadow (unless pass 1 cleared it), minus every chunk pass 1 removed/overwrote, plus pass 1's chunks
    if (!a.cleared) M.sh = T.sh;
    std::unordered_set<uint64_t> tb(b.pages.begin(), b.pages.end());
    for (uint64_t k1 : a.pages) {
        uint64_t k2; if (!rs.pair_addr(k1, [&](uint64_t k) { return tb.count(k) != 0; }, &k2)) { rs.unres++; continue; }
        uint8_t ak; const uint64_t ta = rs.value(k1 << 3, 0xff, k2 << 3, 0xff, &ak);
        if (ak != 0xff) continue;     // base unknown in the truth: the truth stored nothing there
        for (int i = 0; i < 8; i++) M.write_byte(ta + i, 0, false);
    }
    const bool dump = getenv("PTRECON_SEED_DUMP") != nullptr;
    rs.full3 = false; rs.deep_left = 2000;     // memory: no three-term search (registers only), few two-term ones
    // fabricated chunks the run never changed are the entry's own content: nothing to do
    std::unordered_map<uint64_t, uint32_t> fab1; fab1.reserve(a.fab_keys.size() * 2);
    for (size_t i = 0; i < a.fab_keys.size(); i++) if (!(a.fab_keys[i] >> 63)) fab1.emplace(a.fab_keys[i], (uint32_t)(128 + i));
    for (auto& e : A.sh) {
        if (e.second.second == 0xff) { auto f = fab1.find(e.first);
            if (f != fab1.end() && e.second.first == VexInterp::fakev(f->second, 0)) {
                const uint64_t k2 = b.fab_keys[f->second - 128]; auto g2 = B.sh.find(k2);
                if (g2 != B.sh.end() && g2->second.second == 0xff && g2->second.first == VexInterp::fakev(f->second, 1)) {
                    if (k2 == e.first) continue;                       // a real address: the entry's chunk is already in M
                } } }
        uint64_t k2; if (!rs.pair_addr(e.first, [&](uint64_t k) { return B.sh.count(k) != 0; }, &k2)) { rs.unres++; if (dump) fprintf(stderr, "seedsh %lx unpaired\n", (unsigned long)(e.first << 3)); continue; }
        uint8_t ak; const uint64_t ta = rs.value(e.first << 3, 0xff, k2 << 3, 0xff, &ak);
        if (ak != 0xff) { rs.unres++; if (dump) fprintf(stderr, "seedsh %lx/%lx addr unresolved\n", (unsigned long)(e.first << 3), (unsigned long)(k2 << 3)); continue; }
        const auto& e2 = B.sh.at(k2);
        uint8_t rk; const uint64_t r = rs.value(e.second.first, e.second.second, e2.first, e2.second, &rk);
        if (dump) fprintf(stderr, "seedsh %lx -> %lx: %016lx/%02x %016lx/%02x -> %016lx/%02x\n", (unsigned long)(e.first << 3), (unsigned long)ta, (unsigned long)e.second.first, e.second.second, (unsigned long)e2.first, e2.second, (unsigned long)r, rk);
        if (!(ta & 7)) { if (rk) { auto& c = M.sh[ta >> 3]; uint64_t m = 0; for (int i = 0; i < 8; i++) if ((rk >> i) & 1) m |= 0xffull << (8 * i); c.first = r & m; c.second = rk; } else M.sh.erase(ta >> 3); }
        else for (int i = 0; i < 8; i++) M.write_byte(ta + i, (uint8_t)(r >> (8 * i)), (rk >> i) & 1);
    }
    fill(P, M);
    // ---- the pending ptwrite address
    { uint8_t rk; uint64_t r = rs.value(a.ptw_pend_ea, 0xff, b.ptw_pend_ea, 0xff, &rk); P.ptw_pend_ea = rk == 0xff ? r : a.ptw_pend_ea; }
    // ---- value-stream history
    if (a.cv_unaligned) P.cv_pos = t.cv_pos + (a.cv_pos - a.cv_start); else { P.cv_pos = a.cv_pos; P.cv_unaligned = 0; }
    if (a.n_sync_used) {
        P.pre_first_sync = 0; P.tramp_cv = a.tramp_cv;
        P.sync_arm = t.sync_arm > 0 ? a.sync_arm : t.sync_arm; P.sync_err = t.sync_err + a.sync_err; P.sync_mark_period = a.sync_mark_period;
    } else if (t.pre_first_sync) { for (auto& q : a.tramp_cv) P.tramp_cv.push_back({q.first, q.second - a.cv_start + t.cv_pos}); }
    else P.tramp_cv.clear();
    if (a.anchored) P.anchored = 1;
    if (a.n_switch_loss) { P.sl_pending = a.sl_pending; P.sl_vals_rel = a.sl_vals_rel; P.sl_sync_rel = a.sl_sync_rel; }
    else if (t.sl_pending) { P.sl_vals_rel = t.sl_vals_rel - (int64_t)a.consumed; P.sl_sync_rel = t.sl_sync_rel - (int64_t)a.n_sync_used; if (a.consumed) { P.sl_pending = 0; P.sl_vals_rel = P.sl_sync_rel = 0; } }
    if (!t.delta_valid.empty() && a.delta_valid.size() == t.delta_valid.size() && b.delta_valid.size() == t.delta_valid.size()) {
        for (size_t i = 0; i < a.delta_valid.size(); i++) {
            if (!a.delta_valid[i]) { if (a.dinv) P.delta_valid[i] = 0; continue; }
            uint8_t rk; const uint64_t r = rs.value(a.delta_val[i], 0xff, b.delta_val[i], 0xff, &rk);
            P.delta_valid[i] = rk == 0xff; P.delta_val[i] = rk == 0xff ? r : 0;
        }
    }
    // ---- instruction-stream state: pass 1 ran the same instructions
    P.pending_delta = a.pending_delta; P.pending_kf = a.pending_kf; P.ptw_pend_ent = a.ptw_pend_ent; P.kf_fired_branch = a.kf_fired_branch;
    P.ptw_q = a.ptw_q; P.ptw_pending = a.ptw_pending; P.ptw_kind = a.ptw_kind; P.ptw_pend_ip = a.ptw_pend_ip;
    P.ptw_pend_off = a.ptw_pend_off; P.ptw_pend_size = a.ptw_pend_size; P.ptw_pend_memsize = a.ptw_pend_memsize; P.ptw_pend_mem_legacy = a.ptw_pend_mem_legacy;
    P.tnt_on = a.tnt_on; P.tnt_k = a.tnt_k; P.tnt_v = a.tnt_v;
    P.cursor = a.cursor; P.cursor_valid = a.cursor_valid; P.tramp_fallback = a.tramp_fallback; P.in_ovf = a.in_ovf; P.cur_tramp = a.cur_tramp;
    P.prev_end = a.prev_end; P.last_ip_orig = a.last_ip_orig; P.in_tramp = a.in_tramp; P.eo = a.eo;
    P.fake = 0; P.cleared = P.dinv = P.pages_ovf = 0; P.cv_start = P.n_sync_used = P.n_switch_loss = P.consumed = 0; P.fab_keys.clear(); P.pages.clear();
    { size_t nu = 0; for (uint8_t x : P.unc) nu += x; if (!nu) P.unc.clear(); if (why) *why = "unresolved " + std::to_string(rs.unres) + ", uncertain bytes " + std::to_string(nu); return save(P, out); }
    if (why) *why = "unresolved " + std::to_string(rs.unres);
    return save(P, out);
}

bool Recon::seed_has_unc(const std::string& p) { SeedState a; return load(a, p) && !a.unc.empty(); }
// the end state of run (a) of a chunk gets the bytes in which run (b) (uncertain entry bytes = fakes) ended differently
bool Recon::seed_mark_uncertain(const std::string& pa, const std::string& pb, std::string* why) {
    SeedState a, b; if (!load(a, pa) || !load(b, pb)) { if (why) *why = "unreadable"; return false; }
    Mach x, y; if (!x.parse(a.machine) || !y.parse(b.machine)) { if (why) *why = "bad machine"; return false; }
    if (x.sh != y.sh) { if (why) *why = "shadow depends on the uncertain bytes"; return false; }
    SeedState a2 = a, b2 = b; a2.machine.clear(); b2.machine.clear(); a2.unc.clear(); b2.unc.clear();
    const char* f = first_diff(a2, b2); if (f) { if (why) *why = std::string("depends on the uncertain bytes: ") + f; return false; }
    std::vector<uint8_t> u(1024, 0); size_t n = 0;
    for (int i = 0; i < 1024; i++) if (x.gk[i] != y.gk[i] || (x.gk[i] && x.g[i] != y.g[i])) { u[i] = 1; n++; }
    if (!a.unc.empty()) for (int i = 0; i < 1024; i++) if (a.unc[i] && !u[i]) { u[i] = 1; n++; }
    if (n) { for (int i = 0; i < 1024; i++) if (u[i]) { x.g[i] = 0; x.gk[i] = 0; } x.dump(a.machine); a.unc = u; } else a.unc.clear();
    return save(a, pa);
}
bool Recon::seed_same(const std::string& pa, const std::string& pb, std::string* why, const std::string& maskp) {
    SeedState a, b;
    if (!load(a, pa) || !load(b, pb)) { if (why) *why = "unreadable"; return false; }
    if (!maskp.empty()) { SeedState m; if (load(m, maskp)) a.unc = m.unc; }   // an entry state carries its prediction's uncertainty
    // pa = a chunk's entry (prediction), pb = the previous chunk's end: bytes uncertain in the prediction are covered by
    // the chunk's double run; bytes uncertain only in pb cannot be checked
    if (!a.unc.empty() || !b.unc.empty()) {
        Mach x, y; if (!x.parse(a.machine) || !y.parse(b.machine)) { if (why) *why = "bad machine"; return false; }
        std::vector<uint8_t> ua = a.unc, ub = b.unc; ua.resize(1024, 0); ub.resize(1024, 0);
        for (int i = 0; i < 1024; i++) {
            if (ua[i]) { x.g[i] = y.g[i] = 0; x.gk[i] = y.gk[i] = 0; continue; }
            if (ub[i]) { if (why) { char t[64]; snprintf(t, sizeof t, "byte %d uncertain in the truth only", i); *why = t; } return false; }
        }
        x.dump(a.machine); y.dump(b.machine); a.unc.clear(); b.unc.clear();
    }
    const char* f = first_diff(a, b);
    if (!f) return true;
    if (strcmp(f, "machine") && a.machine != b.machine && why) { *why = std::string(f) + " +machine"; f = "machine"; }
    if (why) {
        if (why->empty()) *why = f;
        if (!strcmp(f, "machine")) {
            Mach x, y; if (x.parse(a.machine) && y.parse(b.machine)) {
                std::string l;
                for (int i = 0; i < 128; i++) { uint8_t k1, k2; uint64_t v1 = x.lane(i, &k1), v2 = y.lane(i, &k2);
                    if (k1 != k2 || ((v1 ^ v2) & [&]{ uint64_t m = 0; for (int j = 0; j < 8; j++) if ((k1 >> j) & 1) m |= 0xffull << (8 * j); return m; }())) {
                        char t[96]; snprintf(t, sizeof t, " lane%d(off %d) %016lx/%02x vs %016lx/%02x", i, 8 * i, (unsigned long)v1, k1, (unsigned long)v2, k2); l += t; } }
                size_t nd = 0; std::string sx;
                for (auto& e : x.sh) { auto it = y.sh.find(e.first); if (it == y.sh.end() || it->second != e.second) { if (nd < 4) { char t[96]; snprintf(t, sizeof t, " sh[%lx]=%016lx/%02x", (unsigned long)(e.first << 3), (unsigned long)e.second.first, e.second.second); sx += t; } nd++; } }
                for (auto& e : y.sh) if (!x.sh.count(e.first)) { if (nd < 8) { char t[96]; snprintf(t, sizeof t, " sh2[%lx]=%016lx/%02x", (unsigned long)(e.first << 3), (unsigned long)e.second.first, e.second.second); sx += t; } nd++; }
                *why += l + " shadow_diffs " + std::to_string(nd) + " (" + std::to_string(x.sh.size()) + " vs " + std::to_string(y.sh.size()) + ")" + sx;
            }
        }
    }
    return false;
}
