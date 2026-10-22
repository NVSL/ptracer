// shadowmap.h -- the interpreter's shadow memory (8-byte chunks keyed by
// addr >> 3) as an open-addressing table (linear probing, backward-shift erase, generation-stamped
// slots so clear() is O(1)).  Same contents as the std::unordered_map it replaces: only find /
// insert / erase / clear / size were ever used (never iteration), so behaviour is identical.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

struct ShadowChunk { uint8_t v[8]; uint8_t kn; };

class ShadowMap {
public:
    ShadowMap() { rehash(10); }
    ShadowChunk* find(uint64_t key) {
        if (!pf_test(key)) return nullptr;      // no chunk was ever inserted in this 4 KB page (since clear)
        for (size_t i = idx(key);; i = (i + 1) & mask_) { Slot& s = t_[i];
            if (s.gen != gen_) return nullptr;
            if (s.key == key) return &s.c; } }
    const ShadowChunk* find(uint64_t key) const { return const_cast<ShadowMap*>(this)->find(key); }
    // existing chunk, or a new zeroed one
    ShadowChunk* insert(uint64_t key) {
        if ((n_ + 1) * 2 > cap_) rehash(bits_ + 1);
        size_t i = idx(key);
        for (;; i = (i + 1) & mask_) { Slot& s = t_[i];
            if (s.gen != gen_) { s.gen = gen_; s.key = key; memset(&s.c, 0, sizeof s.c); n_++; pf_set(key); return &s.c; }
            if (s.key == key) return &s.c; } }
    void erase(uint64_t key) {
        size_t i = idx(key);
        for (;; i = (i + 1) & mask_) { Slot& s = t_[i]; if (s.gen != gen_) return; if (s.key == key) break; }
        // backward-shift deletion
        size_t j = i;
        for (;;) {
            j = (j + 1) & mask_;
            Slot& sj = t_[j]; if (sj.gen != gen_) break;
            size_t h = idx(sj.key);
            // move sj into the hole i if h is cyclically outside (i, j]
            if ((j > i && (h <= i || h > j)) || (j < i && (h <= i && h > j))) { t_[i] = sj; i = j; }
        }
        t_[i].gen = gen_ - 1; n_--;
    }
    void clear() { if (!n_) return; n_ = 0; if (++gen_ == 0) { for (auto& s : t_) s.gen = 0; gen_ = 1; } memset(pf_, 0, sizeof pf_); }
    size_t size() const { return n_; }
    // chunk seeds: visit every chunk (unordered)
    template <class F> void for_each(F f) const { if (!n_) return; for (const auto& sl : t_) if (sl.gen == gen_) f(sl.key, sl.c); }
private:
    struct Slot { uint64_t key; uint32_t gen; ShadowChunk c; };
    // page filter -- one bit per hashed 4 KB page (key >> 9) that holds or held a chunk
    // since the last clear(); a clear bit proves find() would miss (heap stores forgetting nothing)
    uint64_t pf_[1u << 14] = {};
    static uint32_t pf_h(uint64_t key) { return (uint32_t)(((key >> 9) * 0x9E3779B97F4A7C15ull) >> 44); }   // 20 bits
    bool pf_test(uint64_t key) const { const uint32_t h = pf_h(key); return (pf_[h >> 6] >> (h & 63)) & 1; }
    void pf_set(uint64_t key) { const uint32_t h = pf_h(key); pf_[h >> 6] |= 1ull << (h & 63); }
    std::vector<Slot> t_; size_t cap_ = 0, mask_ = 0, n_ = 0; int bits_ = 0; uint32_t gen_ = 1;
    size_t idx(uint64_t key) const { return (size_t)((key * 0x9E3779B97F4A7C15ull) >> (64 - bits_)); }
    void rehash(int bits) {
        std::vector<Slot> old; old.swap(t_);
        const uint32_t og = gen_;
        bits_ = bits; cap_ = (size_t)1 << bits; mask_ = cap_ - 1; gen_ = 1;
        t_.assign(cap_, Slot{0, 0, {{0}, 0}}); n_ = 0;
        for (auto& s : old) if (s.gen == og) { ShadowChunk* c = insert(s.key); *c = s.c; }
    }
};
