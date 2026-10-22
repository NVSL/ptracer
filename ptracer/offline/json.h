// Minimal JSON reader/writer for PTracer v2 offline tools (spec, site map, sideband, summary).
// Supports objects, arrays, strings, numbers (int64/double), bool, null. No dependencies.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <cstring>
#include <vector>

struct Json {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double num = 0; int64_t inum = 0; bool is_int = false; std::string s;
    // an object is a flat vector of (key, value) SORTED by key with unique keys -- the
    // iteration order and lookup results of the std::map it replaces, with one allocation per object
    // instead of one node + one string per member (the whole-program site maps are ~75 MB of JSON).
    std::vector<std::pair<std::string, Json>> obj;
    std::vector<Json> arr;
    const Json* ofind(const std::string& k) const {
        size_t lo = 0, hi = obj.size();
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (obj[mid].first < k) lo = mid + 1; else hi = mid; }
        return lo < obj.size() && obj[lo].first == k ? &obj[lo].second : nullptr; }
    const Json& operator[](const std::string& k) const { static Json none; const Json* v = ofind(k); return v ? *v : none; }
    const Json& operator[](size_t i) const { static Json none; return i < arr.size() ? arr[i] : none; }
    bool has(const std::string& k) const { return ofind(k) != nullptr; }
    // after appending members in document order: sort by key, keep the LAST of duplicate keys (map assignment semantics)
    void ofinish() {
        bool sorted = true; for (size_t i = 1; i < obj.size(); i++) if (!(obj[i - 1].first < obj[i].first)) { sorted = false; break; }
        if (sorted) return;
        std::stable_sort(obj.begin(), obj.end(), [](const std::pair<std::string, Json>& a, const std::pair<std::string, Json>& b) { return a.first < b.first; });
        std::vector<std::pair<std::string, Json>> u; u.reserve(obj.size());
        for (size_t i = 0; i < obj.size(); i++) { if (i + 1 < obj.size() && obj[i + 1].first == obj[i].first) continue; u.emplace_back(std::move(obj[i])); }
        obj.swap(u); }

    int64_t i64(int64_t d = 0) const { return t == NUM ? (is_int ? inum : (int64_t)num) : d; }
    uint64_t u64(uint64_t d = 0) const { return t == NUM ? (is_int ? (uint64_t)inum : (uint64_t)num) : d; }
    double dbl(double d = 0) const { return t == NUM ? (is_int ? (double)inum : num) : d; }
    const std::string& str() const { return s; }
    bool boolean(bool d = false) const { return t == BOOL ? b : d; }
    size_t size() const { return t == ARR ? arr.size() : t == OBJ ? obj.size() : 0; }
    bool isnull() const { return t == NUL; }
};

struct JsonParser {
    const char* p; const char* e;
    JsonParser(const std::string& s) : p(s.data()), e(s.data() + s.size()) {}
    JsonParser(const char* b, const char* en) : p(b), e(en) {}
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++; }
    [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
    Json parse() {
        ws(); if (p >= e) fail("eof");
        Json j;
        if (*p == '{') { j.t = Json::OBJ; p++; ws(); if (*p == '}') { p++; return j; }
            for (;;) { ws(); if (*p != '"') fail("key"); std::string k = pstr(); ws(); if (*p != ':') fail(":"); p++;
                j.obj.emplace_back(std::move(k), parse()); ws(); if (*p == ',') { p++; continue; } if (*p == '}') { p++; j.ofinish(); return j; } fail("obj"); } }
        if (*p == '[') { j.t = Json::ARR; p++; ws(); if (*p == ']') { p++; return j; }
            for (;;) { j.arr.emplace_back(parse()); ws(); if (*p == ',') { p++; continue; } if (*p == ']') { p++; return j; } fail("arr"); } }
        if (*p == '"') { j.t = Json::STR; j.s = pstr(); return j; }
        if (!strncmp(p, "true", 4)) { p += 4; j.t = Json::BOOL; j.b = true; return j; }
        if (!strncmp(p, "false", 5)) { p += 5; j.t = Json::BOOL; j.b = false; return j; }
        if (!strncmp(p, "null", 4)) { p += 4; return j; }
        // number
        const char* s = p; bool isint = true;
        if (*p == '-') p++;
        while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) { if (*p == '.' || *p == 'e' || *p == 'E') isint = false; p++; }
        if (p == s) fail("value");
        j.t = Json::NUM; j.is_int = isint;
        // strtoll/strtod straight on the buffer: they stop inside [s, p), exactly where they stopped on
        // the copied token (JSON has no hex/inf/nan); a token at the very end of the buffer is copied.
        if (p < e) { if (isint) { j.inum = strtoll(s, nullptr, 10); j.num = (double)j.inum; }
                     else { j.num = strtod(s, nullptr); j.inum = (int64_t)j.num; } }
        else { std::string n(s, p);
               if (isint) { j.inum = strtoll(n.c_str(), nullptr, 10); j.num = (double)j.inum; }
               else { j.num = strtod(n.c_str(), nullptr); j.inum = (int64_t)j.num; } }
        return j;
    }
    std::string pstr() {
        p++; std::string r;
        {   const char* q = p; while (q < e && *q != '"' && *q != '\\') q++;   // fast path: no escapes
            if (q < e && *q == '"') { r.assign(p, q); p = q + 1; return r; } }
        while (p < e && *p != '"') {
            if (*p == '\\') { p++; char c = *p++; switch (c) { case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break;
                case 'u': { unsigned v = strtoul(std::string(p, p + 4).c_str(), nullptr, 16); p += 4; if (v < 0x80) r += (char)v; else if (v < 0x800) { r += (char)(0xC0 | (v >> 6)); r += (char)(0x80 | (v & 0x3F)); } else { r += (char)(0xE0 | (v >> 12)); r += (char)(0x80 | ((v >> 6) & 0x3F)); r += (char)(0x80 | (v & 0x3F)); } break; }
                default: r += c; } }
            else r += *p++;
        }
        if (p >= e) fail("string");
        p++;
        return r;
    }
};
// skip one value without building it (string-aware); returns the end pointer.
inline const char* json_skip(const char* p, const char* e) {
    int depth = 0;
    do {
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',' || *p == ':')) { if (depth == 0 && (*p == ',' || *p == ':')) break; p++; }
        if (p >= e) throw std::runtime_error("json: eof (skip)");
        char c = *p;
        if (c == '"') { p++; while (p < e && *p != '"') { if (*p == '\\') p++; p++; } if (p >= e) throw std::runtime_error("json: string (skip)"); p++; }
        else if (c == '{' || c == '[') { depth++; p++; }
        else if (c == '}' || c == ']') { depth--; p++; }
        else if (c == ',' || c == ':') { p++; }
        else { while (p < e && *p != ',' && *p != ']' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r' && *p != ':') p++; }
    } while (depth > 0);
    return p;
}
#include <thread>
// A top-level object whose big arrays (>= 1 MB) are parsed by several threads: each array's elements
// are delimited with json_skip, cut into byte-balanced groups, parsed independently and concatenated
// in order -- the same Json tree as JsonParser::parse().
inline Json json_parse_par(const std::string& str) {
    JsonParser jp(str); jp.ws();
    if (jp.p >= jp.e || *jp.p != '{') return jp.parse();
    const unsigned hw = std::thread::hardware_concurrency(); const unsigned T = hw < 2 ? 1 : hw > 8 ? 8 : hw;
    Json j; j.t = Json::OBJ; jp.p++; jp.ws(); if (*jp.p == '}') return j;
    for (;;) {
        jp.ws(); if (*jp.p != '"') jp.fail("key"); std::string k = jp.pstr(); jp.ws(); if (*jp.p != ':') jp.fail(":"); jp.p++; jp.ws();
        const char* vend = (*jp.p == '[') ? json_skip(jp.p, jp.e) : nullptr;
        if (vend && T > 1 && vend - jp.p >= (1 << 20)) {
            std::vector<const char*> el; const char* q = jp.p + 1;
            for (;;) { while (q < vend && (*q == ' ' || *q == '\n' || *q == '\t' || *q == '\r')) q++;
                       if (*q == ']') break; el.push_back(q); q = json_skip(q, vend);
                       while (q < vend && (*q == ' ' || *q == '\n' || *q == '\t' || *q == '\r')) q++;
                       if (*q == ',') { q++; continue; } if (*q == ']') break; jp.fail("arr"); }
            el.push_back(q);   // sentinel
            const size_t n = el.size() - 1; std::vector<size_t> cut{0};
            for (unsigned t = 1; t < T; t++) { const char* want = jp.p + (vend - jp.p) * t / T; size_t i = cut.back();
                while (i < n && el[i] < want) i++; cut.push_back(i); }
            cut.push_back(n);
            std::vector<std::vector<Json>> part(T); std::vector<std::string> err(T); std::vector<std::thread> th;
            for (unsigned t = 0; t < T; t++) th.emplace_back([&, t]() { try {
                for (size_t i = cut[t]; i < cut[t + 1]; i++) { JsonParser sp(el[i], el[i + 1]); part[t].emplace_back(sp.parse()); } }
                catch (std::exception& ex) { err[t] = ex.what(); } });
            for (auto& x : th) x.join();
            for (auto& x : err) if (!x.empty()) throw std::runtime_error(x);
            Json a; a.t = Json::ARR; a.arr.reserve(n);
            for (auto& v : part) for (auto& x : v) a.arr.emplace_back(std::move(x));
            j.obj.emplace_back(std::move(k), std::move(a)); jp.p = vend;
        } else j.obj.emplace_back(std::move(k), jp.parse());
        jp.ws(); if (*jp.p == ',') { jp.p++; continue; } if (*jp.p == '}') { jp.p++; j.ofinish(); return j; } jp.fail("obj");
    }
}
inline Json json_parse(const std::string& s) { if (s.size() >= (4u << 20)) return json_parse_par(s); JsonParser jp(s); return jp.parse(); }
inline Json json_load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) throw std::runtime_error("cannot open " + path);
    // A SHORT read must never be mistaken for end-of-file: a truncated buffer parses as a
    // syntax error at an arbitrary offset ("json: :"), which is what two of 475 `ptrecon
    // --mt' children reported while five of them read the same 8 MB of specs at once.
    // Read until feof, and retry an interrupted read instead of stopping at it.
    std::string s; char buf[65536]; size_t n;
    for (;;) {
        n = fread(buf, 1, sizeof buf, f);
        if (n) { s.append(buf, n); continue; }
        if (feof(f)) break;
        if (ferror(f) && errno == EINTR) { clearerr(f); continue; }
        int e = errno; fclose(f);
        throw std::runtime_error("short read on " + path + ": " + strerror(e));
    }
    fclose(f);
    // Name the file in the exception: a bare "json: :" from one of 475 `ptrecon --mt'
    // children says nothing about WHICH of the fifteen JSON inputs failed to parse.
    try { return json_parse(s); }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) + " while parsing " + path +
                                 " (" + std::to_string(s.size()) + " bytes read)");
    }
}
inline std::string json_quote(const std::string& s) { std::string r = "\""; for (char c : s) { if (c == '"' || c == '\\') { r += '\\'; r += c; } else if (c == '\n') r += "\\n"; else r += c; } return r + "\""; }

// Shared by the threaded and chunked summary writers.
inline std::string json_dump(const Json& j) {
    switch (j.t) {
        case Json::NUL: return "null";
        case Json::BOOL: return j.b ? "true" : "false";
        case Json::NUM: if (j.is_int) return std::to_string(j.inum); else { char b[64]; snprintf(b, sizeof b, "%.17g", j.num); return b; }
        case Json::STR: return json_quote(j.s);
        case Json::ARR: { std::string s = "["; bool first = true; for (const auto& x : j.arr) { if (!first) s += ","; s += json_dump(x); first = false; } return s + "]"; }
        case Json::OBJ: { std::string s = "{"; bool first = true; for (const auto& kv : j.obj) { if (!first) s += ","; s += json_quote(kv.first) + ":" + json_dump(kv.second); first = false; } return s + "}"; }
    }
    return "null";
}

// ---- fast loading of large FLAT arrays -----------------------------------------------
// A big site map is ~20 M small objects; the DOM above costs a heap node per value (node.e9: 659 MB, ~6 s even with
// json_parse_par).  JFlat is a view of ONE flat element (an object of scalar values, or a scalar) with the same
// accessors the converters use on Json (operator[], u64, i64, str, t, b).  json_flat_chunks() splits an array
// [ '[' .. ']' ] into chunks at element starts found after a newline (a raw newline is never inside a JSON string),
// and json_flat_each() parses one chunk.  Anything unexpected throws; callers then fall back to the DOM.
struct JFlat {
    struct V { Json::T t = Json::NUL; bool b = false, is_int = false; int64_t inum = 0; double num = 0; const char* s0 = nullptr; const char* s1 = nullptr; bool esc = false;
        int64_t i64(int64_t d = 0) const { return t == Json::NUM ? (is_int ? inum : (int64_t)num) : d; }
        uint64_t u64(uint64_t d = 0) const { return t == Json::NUM ? (is_int ? (uint64_t)inum : (uint64_t)num) : d; }
        std::string str() const { if (t != Json::STR) return std::string(); if (!esc) return std::string(s0, s1);
            JsonParser jp(s0 - 1, s1 + 1); return jp.pstr(); } };
    bool is_obj = false; V self; int n = 0; const char* k0[48]; const char* k1[48]; V v[48];
    const V& operator[](const char* key) const { static const V none; if (!is_obj) return none; const size_t L = strlen(key);
        for (int i = n - 1; i >= 0; i--) if ((size_t)(k1[i] - k0[i]) == L && !memcmp(k0[i], key, L)) return v[i]; return none; }
    Json::T t() const { return is_obj ? Json::OBJ : self.t; }
    int64_t i64(int64_t d = 0) const { return is_obj ? d : self.i64(d); }
    uint64_t u64(uint64_t d = 0) const { return is_obj ? d : self.u64(d); }
};
inline const char* jf_ws(const char* p, const char* e) { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++; return p; }
inline const char* jf_scalar(const char* p, const char* e, JFlat::V& v) {
    if (p >= e) throw std::runtime_error("jflat eof");
    if (*p == '"') { const char* q = p + 1; bool esc = false; while (q < e && *q != '"') { if (*q == '\\') { esc = true; q++; } q++; }
                     if (q >= e) throw std::runtime_error("jflat string"); v.t = Json::STR; v.s0 = p + 1; v.s1 = q; v.esc = esc; return q + 1; }
    if (*p == '[') {   // an array of scalars inside a flat object (spec `regs'): kept as text, read by jf_each_str
        const char* q = p + 1;
        for (;;) { q = jf_ws(q, e); if (q >= e) throw std::runtime_error("jflat arr");
                   if (*q == ']') break;
                   if (*q == '{' || *q == '[') throw std::runtime_error("jflat nested");
                   if (*q == '"') { q++; while (q < e && *q != '"') { if (*q == '\\') q++; q++; } if (q >= e) throw std::runtime_error("jflat arr string"); q++; }
                   else while (q < e && *q != ',' && *q != ']' && *q != ' ' && *q != '\n' && *q != '\t' && *q != '\r') q++;
                   q = jf_ws(q, e); if (q < e && *q == ',') { q++; continue; } if (q < e && *q == ']') break; throw std::runtime_error("jflat arr sep"); }
        v.t = Json::ARR; v.s0 = p; v.s1 = q + 1; return q + 1; }
    if (*p == '{') throw std::runtime_error("jflat nested");
    if (e - p >= 4 && !strncmp(p, "true", 4)) { v.t = Json::BOOL; v.b = true; return p + 4; }
    if (e - p >= 5 && !strncmp(p, "false", 5)) { v.t = Json::BOOL; v.b = false; return p + 5; }
    if (e - p >= 4 && !strncmp(p, "null", 4)) { v.t = Json::NUL; return p + 4; }
    const char* s = p; bool isint = true; if (*p == '-') p++;
    while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) { if (*p == '.' || *p == 'e' || *p == 'E') isint = false; p++; }
    if (p == s || p >= e) throw std::runtime_error("jflat value");   // (p < e always holds inside an array: strtoll stops at the delimiter)
    v.t = Json::NUM; v.is_int = isint;
    if (isint) { v.inum = strtoll(s, nullptr, 10); v.num = (double)v.inum; } else { v.num = strtod(s, nullptr); v.inum = (int64_t)v.num; }
    return p;
}
inline const char* jf_elem(const char* p, const char* e, JFlat& x) {
    x.n = 0; x.is_obj = false; x.self = JFlat::V();
    if (*p != '{') return jf_scalar(p, e, x.self);
    x.is_obj = true; p = jf_ws(p + 1, e); if (p < e && *p == '}') return p + 1;
    for (;;) {
        if (p >= e || *p != '"') throw std::runtime_error("jflat key");
        const char* q = p + 1; while (q < e && *q != '"') { if (*q == '\\') throw std::runtime_error("jflat key escape"); q++; }
        if (q >= e || x.n >= 48) throw std::runtime_error("jflat key/size");
        x.k0[x.n] = p + 1; x.k1[x.n] = q; p = jf_ws(q + 1, e);
        if (p >= e || *p != ':') throw std::runtime_error("jflat :");
        p = jf_ws(p + 1, e); x.v[x.n] = JFlat::V(); p = jf_scalar(p, e, x.v[x.n]); x.n++;
        p = jf_ws(p, e); if (p < e && *p == ',') { p = jf_ws(p + 1, e); continue; } if (p < e && *p == '}') return p + 1;
        throw std::runtime_error("jflat obj");
    }
}
// Fast skip of one value starting at p (strings, nesting); returns one past it.
inline const char* json_skip_fast(const char* p, const char* e) {
    int depth = 0;
    for (;;) {
        if (p >= e) throw std::runtime_error("json: eof (skip)");
        const char c = *p;
        if (c == '"') { const char* q = p + 1;
            for (;;) { q = (const char*)memchr(q, '"', (size_t)(e - q)); if (!q) throw std::runtime_error("json: string (skip)");
                       const char* r = q; size_t bs = 0; while (r > p + 1 && r[-1] == '\\') { r--; bs++; } if (bs % 2 == 0) break; q++; }
            p = q + 1; if (depth == 0) return p; continue; }
        if (c == '{' || c == '[') { depth++; p++; continue; }
        if (c == '}' || c == ']') { depth--; p++; if (depth <= 0) return p; continue; }
        if (depth == 0) { if (c == ' ' || c == '\n' || c == '\t' || c == '\r') { p++; continue; }
                          while (p < e && *p != ',' && *p != ']' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r' && *p != ':') p++; return p; }
        p++;
    }
}
// Element-start cut points of the array [ab, ae) (ab at '[', ae one past its ']'), about T chunks.  cuts.front() = first
// element (or the ']' of an empty array), cuts.back() = the ']'.
inline std::vector<const char*> json_flat_chunks(const char* ab, const char* ae, unsigned T) {
    std::vector<const char*> c; const char* first = jf_ws(ab + 1, ae); c.push_back(first);
    const char* close = ae - 1; if (*close != ']') throw std::runtime_error("jflat array end");
    if (first >= close) { c.push_back(close); return c; }
    const bool objs = *first == '{';
    for (unsigned t = 1; t < T; t++) {
        const char* q = first + (close - first) * (uint64_t)t / T; if (q <= c.back()) continue;
        for (;;) { q = (const char*)memchr(q, '\n', (size_t)(close - q)); if (!q) break; const char* r = jf_ws(q, close);
                   // an element of the same shape as the first: objects cut only at '{', scalars only at a scalar start
                   // (in an object array a line starting with '"' is a key)
                   if (r < close && (objs ? *r == '{' : (*r == '-' || (*r >= '0' && *r <= '9') || *r == '"'))) { q = r; break; } q++; }
        if (!q) break; if (q > c.back()) c.push_back(q);
    }
    c.push_back(close); return c;
}
// Parse the elements of [b, end) (b = an element start, end = the next chunk's start or the array's ']').
template<class F> inline void json_flat_each(const char* b, const char* end, bool last, F f) {
    const char* p = b; JFlat x;
    while (p < end) {
        p = jf_elem(p, end + (last ? 1 : 0), x); f((const JFlat&)x);
        p = jf_ws(p, end + 1);
        if (p < end && *p == ',') { p = jf_ws(p + 1, end + 1); continue; }
        if (last && p == end && *p == ']') return;
        throw std::runtime_error("jflat chunk boundary");
    }
    if (p != end) throw std::runtime_error("jflat chunk overrun");
}

// each element of an array value as a string (str() of a non-string = ""), for the DOM and for a JFlat value
template<class F> inline void jf_each_str(const Json& j, F f) { for (auto& r : j.arr) f(r.str()); }
template<class F> inline void jf_each_str(const JFlat::V& v, F f) {
    if (v.t != Json::ARR) return; JsonParser jp(v.s0, v.s1); Json a = jp.parse(); for (auto& r : a.arr) f(r.str()); }
