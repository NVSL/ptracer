// PTracer memory trace format. Single source of truth.
#pragma once
#include <stdint.h>
#define MTRACE_MAGIC 0x544D5450u /* "PTMT" little-endian */
#define MTRACE_VERSION 2u
struct mtrace_hdr { uint32_t magic; uint32_t version; uint32_t flags; uint32_t tid_count; uint8_t reserved[16]; };
enum { MTH_TIME_MERGED = 1u << 0, MTH_SAMPLED_OUTPUT = 1u << 1 };
enum { MT_LOAD = 0, MT_STORE = 1, MT_RMW = 2, MT_MARKER = 3,
       MT_SYNC = 4 };   // synchronization event: addr = object/peer, ip = call-site return
                        // address, size = PTSYNC_* op, flags >> 8 = result byte (0 = success)
enum { PTSYNC_CREATE = 1, PTSYNC_START, PTSYNC_EXIT, PTSYNC_JOIN, PTSYNC_FORK, PTSYNC_FORK_CHILD, PTSYNC_VFORK,
       PTSYNC_EXEC, PTSYNC_WAIT, PTSYNC_LOCK, PTSYNC_TRYLOCK, PTSYNC_UNLOCK, PTSYNC_TIMEDLOCK, PTSYNC_CWAIT_BEGIN,
       PTSYNC_CWAIT_END, PTSYNC_CSIGNAL, PTSYNC_CBROADCAST, PTSYNC_RDLOCK, PTSYNC_WRLOCK, PTSYNC_TRYRDLOCK,
       PTSYNC_TRYWRLOCK, PTSYNC_RWUNLOCK, PTSYNC_SEMWAIT, PTSYNC_SEMTRYWAIT, PTSYNC_SEMTIMEDWAIT, PTSYNC_SEMPOST,
       PTSYNC_SPAWN, PTSYNC_NOPS };
#ifdef __cplusplus
static const char* const ptsync_op_name[PTSYNC_NOPS] = {"?", "create", "start", "exit", "join", "fork", "fork_child",
    "vfork", "exec", "wait", "lock", "trylock", "unlock", "timedlock", "cwait_begin", "cwait_end", "csignal",
    "cbroadcast", "rdlock", "wrlock", "tryrdlock", "trywrlock", "rwunlock", "semwait", "semtrywait", "semtimedwait",
    "sempost", "spawn"};
#endif
enum { MTF_ADDR_UNKNOWN = 1u << 0, MTF_IN_OVERFLOW = 1u << 1 };
struct mtrace_rec { uint64_t addr; uint64_t ip; uint64_t ts; uint32_t tid; uint8_t op; uint8_t size; uint16_t flags; };
#ifdef __cplusplus
static_assert(sizeof(struct mtrace_hdr) == 32, "hdr"); static_assert(sizeof(struct mtrace_rec) == 32, "rec");
#else
_Static_assert(sizeof(struct mtrace_hdr) == 32, "hdr"); _Static_assert(sizeof(struct mtrace_rec) == 32, "rec");
#endif
