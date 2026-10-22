#!/usr/bin/env bash
# Offline reconstruction figure: per suite, reconstruction time / uninstrumented run time, split into Intel PT decoding
# and execution.
#
#   bash docker/run.sh run/04_offline.sh [--dry-run]
#
# Cells (one representative set per suite, as in the paper): PolyBench/C 2mm atax correlation doitgen fdtd-2d gemm
# jacobi-2d lu; pyperformance nbody float chaos richards; Rust Stream rayon; Memcached memslap (OFF_MC_OPS=20000 ops);
# Node.js Web Tooling acorn and Java Renaissance scrabble (one iteration each).  Per cell (run/lib/offline.py):
#   * native: the uninstrumented program, Figure 5's command line, OFF_NATIVE_REPS (5) runs, median wall clock;
#   * capture: one Fast run with Figure 5's command line whose PT packets, sideband, value streams and (Node.js/Java)
#     JIT code dump are kept.  PolyBench/C, pyperformance, Rust and Memcached use whole-process Fast images built for
#     this experiment like Figure 5's, but with keyframe period OFF_KEYFRAME (1024; a keyframe re-defines the state a
#     reconstruction chunk starts from) into $RUN_OUT/offline/img; Node.js/Java use the Figure 5 JIT stage
#     ($RUN_OUT/jitwp, built here if absent);
#   * reconstruction: ptrecon with OFF_JOBS workers (default: all cpus), no trace output, timed twice: full and
#     --decode-only; execution = full - decoding.  The capture is then deleted (OFF_KEEP=1 keeps it).
# Resumable: cells already reconstructed with the same worker count are skipped.
# Env: OFF_JOBS, OFF_CELLS ("suite" or "suite:cell" list, e.g. "poly:gemm pyperf rust"), OFF_NATIVE_REPS, OFF_KEYFRAME,
#      OFF_MC_OPS, OFF_TIMEOUT (per reconstruction, 10800 s), OFF_KEEP, JOBS (image build processes).
# Output: $RUN_OUT/offline.csv, offline_native.csv -> offline.png (top: this run; bottom: the paper), offline.txt.
set -u
. "$(dirname "$0")/lib/common.sh"
cd "$ROOT"
pick_python
pick_out
export OFF_DIR="$RUN_OUT/offline" JITWP_DIR="${JITWP_DIR:-$RUN_OUT/jitwp}"
mkdir -p "$OFF_DIR"
echo "== Offline reconstruction: reconstruction time / uninstrumented time, decoding vs execution =="
"$PYBIN" run/lib/offline.py plan
echo "  disk: up to ~30 GB for one capture (Java), deleted after the cell; ~3 GB for the images"
echo "  results: $RUN_OUT/offline.csv -> $RUN_OUT/offline.png"
[ "${1:-}" = "--dry-run" ] && exit 0

for t in ptracer/offline/ptrecon ptracer/offline/pt_capture2 ptracer/runtime/rt/ptlogmt.so; do
    [ -e "$t" ] || { echo "FATAL: $t not built (bash ptracer/build.sh)"; exit 2; }; done
grep -q 'intel_pt' /proc/cpuinfo || [ -d /sys/bus/event_source/devices/intel_pt ] || { echo "FATAL: no Intel PT on this CPU"; exit 2; }
CELLS=$("$PYBIN" run/lib/offline.py cells)
has() { [[ " $CELLS " == *" $1:"* ]]; }

# 1. whole-process Fast images of the non-JIT cells (only what is missing is built); their program interpreter
#    /ptld/fast/ld.so points into this tree for the run
T=""; has poly && T="$T poly"; has rust && T="$T rust"; has mc && T="$T mc"; has pyperf && T="$T py"
if [ -n "$T" ]; then
    PC=$(echo $CELLS | tr ' ' '\n' | sed -n 's/^poly://p' | tr '\n' ' ')
    RUN_OUT="$OFF_DIR/img" FAST_KEYFRAME="${OFF_KEYFRAME:-1024}" POLY_CELLS="$PC" bash run/lib/build_wp.sh ld libs $T interp \
        || echo "WARNING: some Fast images failed (their cells will report it)"
    ln -sfn "$OFF_DIR/img/fastwp/ptld" /ptld 2>/dev/null || [ "$(readlink /ptld)" = "$OFF_DIR/img/fastwp/ptld" ] \
        || { echo "FATAL: cannot create /ptld -> $OFF_DIR/img/fastwp/ptld (run in the container)"; exit 2; }
fi
# 2. the Figure 5 JIT stage (Node.js/Java), if absent
if has node || has java; then
    JB=""; has java && JB="java"; has node && [ -e suites/node/bin/node ] && JB="$JB node"
    [ -f "$JITWP_DIR/stage/manifest.json" ] || bash run/lib/jitwp/build_stage.sh $JB \
        || echo "WARNING: JIT stage not built: the Node.js/Java cells will fail"
fi

# 3. cells, one after the other (each reconstruction uses all OFF_JOBS workers)
fail=0
for sc in $CELLS; do
    "$PYBIN" run/lib/offline.py cell "${sc%%:*}" "${sc#*:}" || fail=$((fail + 1))
    "$PYBIN" figures/offline.py --outdir "$RUN_OUT" > /dev/null 2>&1 || true
done

# 4. figure + table
"$PYBIN" figures/offline.py --outdir "$RUN_OUT"
[ $fail = 0 ] || echo "WARNING: $fail cell(s) failed (logs: $OFF_DIR/repslice.log, $OFF_DIR/logs); rerun to retry them"
echo "== offline reconstruction done: $RUN_OUT/offline.png =="
