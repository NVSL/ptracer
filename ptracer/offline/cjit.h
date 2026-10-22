// cjit.h -- native code for hot fused skeleton programs.  VexInterp::cj_source
// writes C for one program; this service compiles batches of them with the system C compiler on a
// background thread (posix_spawn gcc -O2 -shared), dlopens the result and publishes each function
// pointer into its FProg.  The interpreter keeps running the program until the pointer appears, so
// the record stream does not depend on when compilation finishes.  PTRECON_CJIT=0 disables it;
// PTRECON_CJIT_HOT=N (default 4096) executions make a program hot; PTRECON_CJIT_CC overrides "gcc".
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

namespace cjit {
bool enabled();
// --jobs with more than PTRECON_CJIT_MAXCHUNKS (8) chunks: every chunk process would run its own
// compiler on the same hot programs (py_nbody n24, 48 chunks: 9.5 -> 27.6 s); off there unless
// PTRECON_CJIT=1.  A Rust --mt-split thread (2 long chunks) keeps it.
void off_for_chunks(int nchunks);
void back_to_default();         // a --jobs run that falls back to serial
uint32_t hot();
uint32_t benefit_runs(size_t nops);   // run count at which compiling F pays (0 = rule off; cjit.cpp)
int pid();                       // this process (updated in fork children)
const char* prelude();
void submit(const std::string& name, const std::string& src, std::atomic<void*>* slot, std::atomic<int>* state, uint64_t key = 0);
// a compile cache SHARED by the chunk processes of one --jobs run (PTRECON_CJIT_CACHE = a directory;
// run_parallel creates one per run when chunks > PTRECON_CJIT_MAXCHUNKS and PTRECON_CJIT_SHARE=1; default off: measured slower, cjit.cpp).
// A program is named by the hash of its source (key); one chunk claims it (O_EXCL file) and compiles it, the others
// pick the published .so up (f_<key>.so symlink) instead of running a compiler of their own.
bool cache_on();
extern bool g_cache;                       // == cache_on(), kept current by cache_create_for_run/cache_remove (hot-path test)
void* cache_lookup(uint64_t key);          // the published function, or null
bool cache_claim(uint64_t key);             // true: this process compiles it
void cache_create_for_run();               // run_parallel: make the directory, export PTRECON_CJIT_CACHE
void cache_remove();                       // run_parallel parent at the end
void quiesce();                  // drop queued jobs and wait for the batch in flight (before FProgs are freed)
struct Stats { uint64_t batches = 0, funcs = 0, failed = 0, dropped = 0; double cc_s = 0; };   // dropped: one-cpu compile budget
Stats stats();
}
