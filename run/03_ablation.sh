#!/usr/bin/env bash
# Figure 6 (ablation of PTracer's new techniques), all six suites:
#
#   bash docker/run.sh run/03_ablation.sh [--dry-run]       # after run/fig5.sh
#
#   PTracer                     = the Figure 5 Fast bar (rows of run/fig5.sh, not measured again)
#   PTracer w/o PT              = abl_blk: the Fast images and plans, plus a block identifier logged at every basic-block
#                                 leader (control flow logged by instrumentation instead of Intel PT); no PT capture.
#                                 Node.js/Java: the whole-program Fast images rebuilt the same way
#                                 (run/lib/jitwp/build_abl.sh) and the runtime hooks put block ids into the JIT code.
#   PTracer w/o PT and Static   = abl_memt: every memory access (stack and string accesses included) logged by a plain
#                                 logging call (no static analysis, no PTracer instrumentation optimizations, no PT).
#                                 Node.js/Java: the whole process under Pin with ablmemtool.so (JIT code included).
# The logged values go to the software buffer and are discarded, as in Figure 5.  Each arm is relative to the vanilla
# rows of Figure 5 (ABL_VANILLA=1 measures vanilla again alongside).  Output: out/experiments_v1/ablation.png (+ .json),
# rows in ablation.csv (redrawn as rows arrive).  Idempotent: rows already there are kept, built images are kept; after
# an update only the rows of configurations whose measurement changed are measured again (run/lib/row_versions.py).
#
# Defaults (every one can be set on the host, e.g. REPS=1 bash docker/run.sh run/03_ablation.sh):
#   REPS=3        PolyBench (30), pyperformance (94), Rust Stream (6), Memcached; lanes TIMED_LANES / HELPER_LANES as
#                 run/fig5.sh; a run stopped at 200x its vanilla time counts as 200x
#   JIT_REPS=1    Node.js (18 Web Tooling benchmarks; ABL_NODE_CELLS=a,b,.. narrows)
#   JREPS=1       Java (25 Renaissance benchmarks, JAVA_LANES=2 lanes; ABL_JAVA_CELLS=a,b,.. narrows); abl_memt runs
#                 ABLMEMT_JAVA_ITERS=4 iterations and times the last (ABLMEMT_JAVA_DROP=3)
#   ABL_CONFIGS=abl_blk,abl_memt  ABL_SUITES="poly pyperf rust mc node java"  POLY_CELLS  RUST_CELLS  PYPERF_LIST
set -eu
. "$(dirname "$0")/lib/common.sh"
cd "$ROOT"
quick_mode
REPS="${REPS:-3}"; JIT_REPS="${JIT_REPS:-1}"; JREPS="${JREPS:-1}"
read -r -a TL <<< "${TIMED_LANES:-1 4 7 10 13 16 19 22}"
read -r -a HL <<< "${HELPER_LANES:-2 5 8 11 14 17 20 23}"
[ "${#TL[@]}" = "${#HL[@]}" ] || { echo "FATAL: TIMED_LANES and HELPER_LANES differ in length"; exit 2; }
CFGS="${ABL_CONFIGS:-abl_blk,abl_memt}"
[ "${ABL_VANILLA:-0}" = 1 ] && CFGS="vanilla,$CFGS"
SUITES=" ${ABL_SUITES:-poly pyperf rust mc node java} "
has() { case "$SUITES" in *" $1 "*) return 0;; esac; return 1; }
hascfg() { case ",$CFGS," in *",$1,"*) return 0;; esac; return 1; }
POLY_ALL="2mm 3mm adi atax bicg cholesky correlation covariance deriche doitgen durbin fdtd-2d floyd-warshall gemm gemver gesummv gramschmidt heat-3d jacobi-1d jacobi-2d lu ludcmp mvt nussinov seidel-2d symm syr2k syrk trisolv trmm"
PYD="$ROOT/suites/pyperformance"
read -r -a POLYCELLS <<< "${POLY_CELLS:-$POLY_ALL}"
read -r -a PYCELLS <<< "$(tr '\n' ' ' < "${PYPERF_LIST:-$PYD/pyperf/bench94.lst}")"
read -r -a RUSTCELLS <<< "${RUST_CELLS:-seq rayon rust-ssp std-threads tokio pipeliner}"
NODE_CELLS="${ABL_NODE_CELLS:-acorn,babel,babel-minify,babylon,buble,chai,coffeescript,espree,esprima,jshint,lebab,postcss,prepack,prettier,source-map,terser,typescript,uglify-js}"
JAVA_CELLS="${ABL_JAVA_CELLS:-$(grep -v '^#' run/lib/jitwp/java_iters.txt | awk 'NF {print $1}' | paste -sd,)}"
JN="${JAVA_LANES:-2}"
NODE_PRESENT=0
{ [ -e "$ROOT/suites/node/bin/node" ] || [ -e "$ROOT/suites/node/bin/node.part-00" ]; } && NODE_PRESENT=1
[ "$NODE_PRESENT" = 1 ] || SUITES="${SUITES/ node / }"

if [ "${1:-}" = "--dry-run" ]; then
    # serial minutes per cell and repetition on a c7i.metal-24xl (4th-gen Xeon Scalable), measured
    "${PYBIN:-python3}" - "$SUITES" "$CFGS" "$REPS" "$JIT_REPS" "$JREPS" "${#TL[@]}" "$JN" "${#POLYCELLS[@]}" \
        "${#PYCELLS[@]}" "${#RUSTCELLS[@]}" "$(echo "$NODE_CELLS" | tr ',' '\n' | grep -c .)" \
        "$(echo "$JAVA_CELLS" | tr ',' '\n' | grep -c .)" <<'PY'
import sys
suites, cfgs = sys.argv[1].split(), sys.argv[2].split(",")
reps, jreps_node, jreps_java, lanes, jlanes = (int(x) for x in sys.argv[3:8])
n = dict(zip(("poly", "pyperf", "rust", "node", "java"), (int(x) for x in sys.argv[8:13])))
n["mc"] = 1
MIN = {"poly": dict(vanilla=0.08, abl_blk=0.11, abl_memt=1.4), "pyperf": dict(vanilla=0.03, abl_blk=0.14, abl_memt=1.2),
       "rust": dict(vanilla=0.02, abl_blk=0.04, abl_memt=0.1), "mc": dict(vanilla=0.05, abl_blk=0.05, abl_memt=0.1),
       "node": dict(vanilla=0.05, abl_blk=7.5, abl_memt=1.0), "java": dict(vanilla=0.9, abl_blk=13.0, abl_memt=4.1)}
WARM = {"node": 10.0 / 6, "java": 10.0 / 4}            # abl_blk's untimed warm-up processes, in parallel slots
tot = 0.0
print("== Figure 6 (ablation), plan ==")
print("  PTracer bar: the Figure 5 Fast rows (run/fig5.sh); configurations measured here: %s" % ",".join(cfgs))
for s in ("poly", "pyperf", "rust", "mc", "node", "java"):
    if s not in suites:
        continue
    r = reps if s in ("poly", "pyperf", "rust", "mc") else jreps_node if s == "node" else jreps_java
    div = lanes if s in ("poly", "pyperf") else jlanes if s == "java" else 1
    m = n[s] * r * sum(MIN[s].get(c, 0) for c in cfgs) / div
    if s in WARM and "abl_blk" in cfgs:
        m += n[s] * WARM[s]
    tot += m
    print("  %-7s %3d cells x %d rep(s)%s: ~%.1f h" % (s, n[s], r, " on %d lanes" % div if div > 1 else "", m / 60))
print("  image builds (first run only): ~2.5 h on all cores")
print("  estimated: ~%.0f h measuring (+ the builds on a first run)" % (tot / 60))
PY
    pick_python; pick_out
    echo "  results: $RUN_OUT/ablation.csv (rows already there are kept, the missing ones are measured)"
    "$PYBIN" run/lib/row_versions.py prune "$RUN_OUT" --set ablation --dry-run
    exit 0
fi

pick_python
pick_out
CSV="$RUN_OUT/ablation.csv"
"$PYBIN" run/lib/row_versions.py prune "$RUN_OUT" --set ablation
echo "== Figure 6 (ablation): $CFGS; suites:$SUITES; $REPS repetitions (Node.js $JIT_REPS, Java $JREPS) =="
"$PYBIN" - "$RUN_OUT/overhead.csv" <<'PY' || echo "WARNING: no Figure 5 Fast rows yet: the PTracer bars and the vanilla baselines come from run/fig5.sh (or set ABL_VANILLA=1)"
import csv, sys
try:
    ok = any(r["config"] in ("fast", "e9fast") and r["rc"] == "0" for r in csv.DictReader(open(sys.argv[1])))
except OSError:
    ok = False
sys.exit(0 if ok else 1)
PY
echo "Results -> $CSV, figure -> $RUN_OUT/ablation.png (redrawn as results arrive)"
progress_figure ablation
quiet_machine_hints
PIN="${PIN_ROOT:-$ROOT/third_party/pin-4.4}"
[ -x "$ROOT/ptracer/offline/pt_capture2" ] || { echo "FATAL: not built (bash ptracer/build.sh)"; exit 2; }
export PTLOG_DIR=/dev/null ABL_DIR="$RUN_OUT/abl"
PINX="-pin_memory_range 0x1000000000:0x1400000000 -enforce_pin_range_allocations 1 -xyzzy -inter_trace_liveness 1 -spillslot_memop 0 -assert_on_memory_conflict FIXED_ONLY"

# ---- image builds (first run only, untimed, before any timed row) ----------------------------------------------------
# Node.js/Java (log: ablation.build_jit.log) alongside the other suites': the Fast images and stage of run/fig5.sh are
# kept (abl_blk starts from their plans), the ablation image sets are built next to them.
T=""; for s in poly rust mc pyperf; do has $s && T="$T ${s/pyperf/py}"; done
JSU=""; for s in node java; do has $s && JSU="$JSU $s"; done
JITWP_DIR="${JITWP_DIR:-$RUN_OUT/jitwp}"; export JITWP_DIR
rm -f "$RUN_OUT/.ablation_jit_ok"
if [ -n "$JSU" ]; then
    ( ok=1
      if [ ! -f "$JITWP_DIR/stage/manifest.json" ]; then   # the whole-program Fast stage of run/fig5.sh
          bash run/lib/jitwp/build_stage.sh $JSU || { ok=0; echo "ERROR: JIT whole-program Fast stage not built -- Node.js/Java rows SKIPPED"; }
      fi
      if [ "$ok" = 1 ] && hascfg abl_blk; then
          SRC="$JITWP_DIR" W="$ABL_DIR/blk/jitwp" bash run/lib/jitwp/build_abl.sh common $JSU \
              || { ok=0; echo "ERROR: Node.js/Java w/o-PT images not built -- Node.js/Java rows SKIPPED"; }
      fi
      [ "$ok" = 0 ] || touch "$RUN_OUT/.ablation_jit_ok" ) > "$RUN_OUT/ablation.build_jit.log" 2>&1 &
fi
if [ -n "$T" ]; then
    bash run/lib/build_wp.sh ld libs $T interp || echo "WARNING: some whole-process Fast images were not built (see [build_wp] above)"
    for c in abl_blk abl_memt; do
        hascfg $c || continue
        ABL=${c#abl_} bash run/lib/build_wp.sh ld libs $T interp || echo "WARNING: some $c images were not built (see [build_wp] above); their cells fail"
    done
fi
wait_jobs
[ -z "$JSU" ] || grep -E '^\[abl\]|ERROR|FATAL' "$RUN_OUT/ablation.build_jit.log" || true

# ---- PolyBench, pyperformance, Rust Stream, Memcached ----------------------------------------------------------------
if [ -n "$T" ]; then
    FWP="$RUN_OUT/fastwp"
    if [ "$(readlink /ptld 2>/dev/null)" != "$FWP/ptld" ]; then
        ln -sfn "$FWP/ptld" /ptld 2>/dev/null || { echo "FATAL: cannot create /ptld -> $FWP/ptld (run as root, e.g. in the container)"; exit 2; }
    fi
    [ -x /ptld/fast/ld.so ] || { echo "FATAL: /ptld/fast/ld.so missing"; exit 2; }
    ln -sfn "$ABL_DIR/ptld" /ptlda 2>/dev/null || { echo "FATAL: cannot create /ptlda -> $ABL_DIR/ptld"; exit 2; }
    [ -x /ptlda/ld.so ] || { echo "FATAL: /ptlda/ld.so (the ablation images' loader) missing"; exit 2; }
    export FIG5_POLY="${POLYCELLS[*]}" FIG5_RUST="${RUSTCELLS[*]}" FIG5_PYPERF_LIST="${PYPERF_LIST:-$PYD/pyperf/bench94.lst}" \
           FIG5_PYRUN="$PYD/pyperf/run_one_full.py" FIG5_PYLOOPS="${FIG5_PYLOOPS-$PYD/pyperf/loops.tsv}" \
           FIG5_PYPATH_VAN="$PYD/wp/extmods:$PYD/wp/site"
    RS_OUT="$RUN_OUT/ablation_rs"; mkdir -p "$RS_OUT"
    rs() {  # rs <csv> <suite> [repslice args...]
        local csv="$1" su="$2"; shift 2
        FIG5_OUT="${LANE_OUT:-$RS_OUT}" "$PYBIN" run/lib/repslice.py --suites "$su" --configs "$CFGS" --reps "$REPS" \
            --label abl --csv "$csv" --cap-x 200 --cap-from "$RUN_OUT/overhead.csv" ${MC_OPS:+--mc-ops "$MC_OPS"} "$@"
    }
    lanes() {  # lanes <suite> <cells...>: cells dealt round-robin over the lanes, one repslice per lane
        local su="$1"; shift; local cells=("$@") n=${#TL[@]} i j parts=()
        if [ "$n" = 1 ]; then TIMED_CORE="${TL[0]}" HELPER_CORES="${HL[0]}" rs "$CSV" "$su" --cells "$(IFS=,; echo "${cells[*]}")"; return; fi
        for i in $(seq 0 $((n-1))); do
            local mine=()
            for j in $(seq "$i" "$n" $((${#cells[@]}-1))); do mine+=("${cells[$j]}"); done
            [ ${#mine[@]} -gt 0 ] || continue
            parts+=("$RUN_OUT/ablation.lane$i.$su.csv"); mkdir -p "$RS_OUT/lane$i"
            TIMED_CORE="${TL[$i]}" HELPER_CORES="${HL[$i]}" LANE_OUT="$RS_OUT/lane$i" \
                rs "$RUN_OUT/ablation.lane$i.$su.csv" "$su" --cells "$(IFS=,; echo "${mine[*]}")" \
                --no-lock --mutex-inherit --load-gate 0 > "$RUN_OUT/ablation.lane$i.$su.log" 2>&1 &
        done
        wait_jobs
        "$PYBIN" run/lib/merge_csv.py "$CSV" "${parts[@]}"
    }
    if has poly; then lanes poly "${POLYCELLS[@]}"; fi
    if has pyperf; then lanes pyperf "${PYCELLS[@]}"; fi
    if has rust; then TIMED_CORE="${TL[0]}" HELPER_CORES="${HL[0]}" rs "$CSV" rust; fi
    if has mc; then TIMED_CORE="${TL[0]}" HELPER_CORES="${HL[0]}" rs "$CSV" mc; fi
fi

# ---- Node.js and Java -----------------------------------------------------------------------------------------------
if [ -n "$JSU" ]; then
    JOK=0; [ -f "$RUN_OUT/.ablation_jit_ok" ] && JOK=1
    if [ "$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)" != 0 ] && hascfg abl_memt; then
        echo "WARNING: kernel.yama.ptrace_scope != 0: the Node.js/Java abl_memt rows (Pin) may fail (docker/run.sh sets it)"
    fi
    export ABLMEMT_PIN_EXTRA="$PINX" ABLMEMT_JAVA_ITERS="${ABLMEMT_JAVA_ITERS:-4}" ABLMEMT_JAVA_DROP="${ABLMEMT_JAVA_DROP:-3}" \
           ABLMEMT_PAGES="${ABLMEMT_PAGES:-1024}" PIN_ROOT="$PIN"
    AJ="$RUN_OUT/ablation_jit"
    seed_cache() {  # <out dir> <suite>: the w/o-PT arm analyzes JIT code exactly as Fast does: start from Fast's plan cache
        local d="$1/jitcache/$2" c
        [ -d "$d" ] && return 0
        mkdir -p "$d"
        for c in "$RUN_OUT/jitfast/jitcache/$2" "$RUN_OUT"/jitfast/jlane*/jitcache/"$2"; do
            [ -d "$c" ] && cp -rn "$c/." "$d/" 2>/dev/null
        done
        return 0
    }
    NWS="${NODE_WARM_SLOTS-1/2,3,4,5 6/7,8,9,10 12/13,14,15,20 24/25,26,27,28 30/31,32,33,34 36/37,38,39,40}"
    JWS="${JAVA_WARM_SLOTS-0-3/4,5,6,7,8,9,10,11 24-27/28,29,30,31,32,33,34,35 12-15/16,17,18,19,20,21,22,23 36-39/40,41,42,43,44,45,46,47}"
    hascfg abl_blk || { NWS=""; JWS=""; }       # untimed warm-up processes: abl_blk only
    if [ "$JOK" = 1 ] && has node; then
        seed_cache "$AJ" node
        [ -z "$NWS" ] || "$PYBIN" run/lib/jitwp/sweep_jit_fast.py --stage "$JITWP_DIR/stage" --out "$AJ" --csv "$CSV" \
            --suites node --configs "$CFGS" --warm-only --warm-slots "$NWS" --node-cells "$NODE_CELLS" \
            || echo "WARNING: Node.js warm-up pass incomplete (the sweep warms the rest inline)"
        "$PYBIN" run/lib/jitwp/sweep_jit_fast.py --stage "$JITWP_DIR/stage" --out "$AJ" --csv "$CSV" --suites node \
            --node-reps "$JIT_REPS" --configs "$CFGS" --node-cells "$NODE_CELLS" --timeout "${ABL_TIMEOUT:-7200}" \
            || echo "ERROR: the Node.js sweep did not finish (rows so far are kept; rerun to resume)"
    fi
    if [ "$JOK" = 1 ] && has java; then
        read -r -a JLC <<< "${JAVA_LANE_CORES:-0-3 24-27}"; read -r -a JLJ <<< "${JAVA_LANE_JIT:-8,9,10,11,16,17,18,19 28,29,30,31,32,33,34,35}"
        [ "$JN" -le "${#JLC[@]}" ] || JN=${#JLC[@]}
        read -r -a JC <<< "$(echo "$JAVA_CELLS" | tr ',' ' ')"
        read -r -a JWA <<< "$JWS"
        for i in $(seq 0 $((JN-1))); do   # warm-up pass: lane i's benchmarks on slots i, i+JN, ...
            mine=(); for j in $(seq "$i" "$JN" $((${#JC[@]}-1))); do mine+=("${JC[$j]}"); done
            ws=(); for j in $(seq "$i" "$JN" $((${#JWA[@]}-1))); do ws+=("${JWA[$j]}"); done
            [ ${#mine[@]} -gt 0 ] || continue
            seed_cache "$AJ/jlane$i" java
            [ ${#ws[@]} -gt 0 ] || continue
            "$PYBIN" run/lib/jitwp/sweep_jit_fast.py --stage "$JITWP_DIR/stage" --out "$AJ/jlane$i" \
                --csv "$RUN_OUT/ablation.jlane$i.csv" --suites java --configs "$CFGS" --warm-only \
                --warm-slots "${ws[*]}" --java-cells "$(IFS=,; echo "${mine[*]}")" > "$RUN_OUT/ablation.jwarm$i.log" 2>&1 &
        done
        wait_jobs
        parts=()
        for i in $(seq 0 $((JN-1))); do
            mine=(); for j in $(seq "$i" "$JN" $((${#JC[@]}-1))); do mine+=("${JC[$j]}"); done
            [ ${#mine[@]} -gt 0 ] || continue
            parts+=("$RUN_OUT/ablation.jlane$i.csv")
            JAVA_CORES="${JLC[$i]}" JAVA_JIT_CORES="${JLJ[$i]}" "$PYBIN" run/lib/jitwp/sweep_jit_fast.py \
                --stage "$JITWP_DIR/stage" --out "$AJ/jlane$i" --csv "$RUN_OUT/ablation.jlane$i.csv" --suites java \
                --java-reps "$JREPS" --mutex-inherit --configs "$CFGS" --timeout "${ABL_TIMEOUT:-7200}" \
                --java-cells "$(IFS=,; echo "${mine[*]}")" > "$RUN_OUT/ablation.jlane$i.log" 2>&1 &
        done
        wait_jobs
        "$PYBIN" run/lib/merge_csv.py "$CSV" "${parts[@]}"
    fi
fi

# every (suite, configuration) must have valid rows; a silently dropped cell is an error
"$PYBIN" - "$CSV" <<'PYCHK'
import csv, sys
ok, seen = set(), set()
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
except OSError:
    rows = []
for r in rows:
    if (r.get("label") or "") == "wpfast_r0":
        continue
    seen.add((r["suite"], r["config"]))
    if r["rc"] == "0" and r["ktime"]:
        ok.add((r["suite"], r["config"]))
bad = sorted(seen - ok)
if bad:
    print("ERROR: no valid rows for " + ", ".join("%s/%s" % b for b in bad) + " (the figure shows them as Err; rerun to retry)")
PYCHK
progress_stop
"$PYBIN" figures/ablation.py --run-out "$RUN_OUT"
echo "== Figure 6 done: $RUN_OUT/ablation.png =="
