#!/usr/bin/env bash
# HiFi offline reconstruction accuracy.
#
#   bash docker/run.sh run/05_hifi_recon.sh [--dry-run]
#
# HiFi mode runs the program under Pin's JIT, so Intel PT records the code-cache addresses rather
# than the original program and the Fast-mode PT decoder cannot map them back.  Instead, the HiFi
# Pintool (runtime/pinjit/hififlow.so = the HiFi tool plus a software control-flow log) records, per
# executed basic block, a block-version id, and captures each block's original bytes at translation
# time.  The reconstructor (ptracer/offline/hifirecon) replays that control-flow log and the logged
# critical values through the same VEX interpreter as Fast mode -- no PT decode -- and emits the
# memory trace.  Each run also logs a same-process ground truth (one record per executed memory
# operand) that is compared AFTER each instruction is interpreted and never feeds the machine state,
# so "wrong" and "unknown" are an honest per-record accuracy.
#
# Cells: PolyBench/C and pyperformance (CPython).
#
# Per cell (one iteration): assemble a whole-program HiFi plan (the program plus the loader and the
# C library); run the program under hififlow.so keeping the flow log, critical-value log and ground
# truth; reconstruct with hifirecon; record the ground-truth record count, wrong records (expected
# 0) and unknown records (marked, never guessed).  The streams are deleted after the cell
# (HIFI_RECON_KEEP=1 keeps them).
#
# Env: HIFI_RECON_CELLS ("suite:cell" list; default below), HIFI_RECON_SIZE (PolyBench dataset:
#      small [default], medium), HIFI_RECON_CORE (timed core, default 1), HIFI_JOBS (analyzer
#      workers, default all cpus), HIFI_RECON_PAGES (flow/value buffer pages), HIFI_RECON_KEEP.
# Output: $RUN_OUT/hifi_recon.csv -> hifi_recon.png (per-cell accuracy).
set -eu
. "$(dirname "$0")/lib/common.sh"
cd "$ROOT"
pick_python
pick_out

CELLS="${HIFI_RECON_CELLS:-poly:gemm poly:2mm poly:atax pyperf:nbody pyperf:float}"
SIZE="${HIFI_RECON_SIZE:-small}"
CORE="${HIFI_RECON_CORE:-1}"
JOBS="${HIFI_JOBS:-$(nproc)}"
PIN="${PIN_ROOT:-$ROOT/third_party/pin-4.4}/pin"
NO_CLONE3="$ROOT/third_party/no_clone3/no_clone3"
HIFLOW="$ROOT/ptracer/runtime/pinjit/obj-intel64/hififlow.so"
HIREC="$ROOT/ptracer/offline/hifirecon"
PYVAN="$ROOT/suites/pyperformance/cpython-cg/bin/python3.12"
PYRUN="$ROOT/suites/pyperformance/pyperf/run_one.py"
ANALYZE="$ROOT/ptracer/static/analyze.py"
MKPLAN="$ROOT/ptracer/runtime/pinjit/mkplan.py"
LDSO_PLAN="$ROOT/data/plans/wp/ld.so.lib.plan"
D="$RUN_OUT/hifi_recon"; mkdir -p "$D/plans"
CSV="$RUN_OUT/hifi_recon.csv"

echo "== HiFi offline reconstruction accuracy (PolyBench/C, pyperformance) =="
echo "  cells: $CELLS"
echo "  PolyBench dataset: $SIZE; 1 iteration per cell; wrong records expected 0"
echo "  results: $CSV -> $RUN_OUT/hifi_recon.png"
[ "${1:-}" = "--dry-run" ] && exit 0

[ -x "$HIREC" ] || { echo "FATAL: $HIREC not built (bash ptracer/build.sh)"; exit 2; }
[ -f "$HIFLOW" ] || { echo "FATAL: $HIFLOW not built (needs the Pin 4.4 kit; bash ptracer/build.sh)"; exit 2; }
[ -x "$PIN" ] || { echo "FATAL: Pin 4.4 kit not found at $PIN (set PIN_ROOT)"; exit 2; }

[ -f "$CSV" ] || echo "suite,cell,size,gt_records,compared,identical,unknown,wrong,unknown_pct,wrong_pct" > "$CSV"
done_cell() { awk -F, -v s="$1" -v c="$2" 'NR>1 && $1==s && $2==c {found=1} END{exit !found}' "$CSV"; }
absplan() { sed -E "s#^I (suites/)#I $ROOT/\1#" "$1"; }   # make relative image paths absolute

# Whole-program HiFi plan for one PolyBench binary: static analysis of the binary, flattened to a
# PLAN, with the loader + C library sites appended.  No keyframe/resync sites are needed under Pin.
poly_plan() {   # $1 image  $2 plan-out
    local img="$1" out="$2" spec main
    [ -s "$out" ] && return 0
    spec="$D/plans/$(basename "$img").spec.json"; main="$D/plans/$(basename "$img").main.plan"
    "$PYBIN" "$ANALYZE" "$img" -o "$spec" --mode hifi --jobs "$JOBS" --cache "$D/plans/cache" \
        > "$D/plans/$(basename "$img").analyze.log" 2>&1 \
        || { echo "FATAL: HiFi analysis of $img failed (see $D/plans/$(basename "$img").analyze.log)"; return 2; }
    "$PYBIN" "$MKPLAN" -o "$main" --auto "$spec" >> "$D/plans/$(basename "$img").analyze.log" 2>&1
    { cat "$main"; tail -n +2 "$LDSO_PLAN"; } > "$out"
}

run_cell() {   # $1 suite  $2 cell
    local suite="$1" cell="$2" cdir plan pid tid
    cdir="$D/$suite.$cell"; rm -rf "$cdir"; mkdir -p "$cdir"
    local -a prog; local -a envp=()
    if [ "$suite" = poly ]; then
        local img="$ROOT/suites/polybench/targets/${cell}_${SIZE}"
        [ -x "$img" ] || { echo "  skip $suite:$cell (no $img)"; return 1; }
        plan="$D/plans/poly.${cell}.${SIZE}.wp.plan"
        poly_plan "$img" "$plan" || return 1
        prog=("$img")
    elif [ "$suite" = pyperf ]; then
        plan="$D/plans/cpython.wp.plan"
        [ -s "$plan" ] || { absplan "$ROOT/data/plans/wp/cpython_wp.plan"; tail -n +2 "$LDSO_PLAN"; } > "$plan"
        prog=("$PYVAN" "$PYRUN" "$cell")
        envp=(PPF_LOOPS=1 PPF_WARM=1)
    else
        echo "  skip $suite:$cell (unsupported suite)"; return 1
    fi
    echo "-- $suite:$cell --"
    local -a pages=(); [ -n "${HIFI_RECON_PAGES:-}" ] && pages=(-pages "$HIFI_RECON_PAGES")
    taskset -c "$CORE" setarch -R env "${envp[@]}" \
        "$NO_CLONE3" "$PIN" -ifeellucky -t "$HIFLOW" -plan "$plan" -cvdir "$cdir" -flow 1 -gt 1 "${pages[@]}" \
        -- "${prog[@]}" > "$cdir/run.log" 2>&1 || { echo "  FATAL: hififlow run failed (see $cdir/run.log)"; return 1; }
    pid=$(ls "$cdir"/code.*.bin 2>/dev/null | head -1 | sed -E 's#.*/code\.([0-9]+)\.bin#\1#')
    tid=$(ls "$cdir"/flow."$pid".*.bin 2>/dev/null | head -1 | sed -E 's#.*/flow\.[0-9]+\.([0-9]+)\.bin#\1#')
    [ -n "$pid" ] && [ -n "$tid" ] || { echo "  FATAL: no flow log produced (see $cdir/run.log)"; return 1; }
    "$HIREC" --dir "$cdir" --pid "$pid" --tid "$tid" --plan "$plan" --gt --json "$cdir/recon.json" > "$cdir/recon.log" 2>&1 \
        || { echo "  FATAL: hifirecon failed (see $cdir/recon.log)"; return 1; }
    "$PYBIN" - "$cdir/recon.json" "$suite" "$cell" "$SIZE" "$CSV" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); suite, cell, size, csv = sys.argv[2:6]
cmp = j["compared"]; unk = j["unknown"]; wr = j["wrong"]
up = 100.0 * unk / cmp if cmp else 0.0; wp = 100.0 * wr / cmp if cmp else 0.0
with open(csv, "a") as f:
    f.write("%s,%s,%s,%d,%d,%d,%d,%d,%.6f,%.6f\n" % (suite, cell, size, j["gt_records"], cmp, j["identical"], unk, wr, up, wp))
print("  %s:%s  gt=%d compared=%d  wrong=%d (%.6f%%)  unknown=%d (%.6f%%)" % (suite, cell, j["gt_records"], cmp, wr, wp, unk, up))
PY
    [ "${HIFI_RECON_KEEP:-0}" = 1 ] || rm -rf "$cdir"
}

fail=0
for sc in $CELLS; do
    s="${sc%%:*}"; c="${sc#*:}"
    if done_cell "$s" "$c"; then echo "-- $s:$c (done, kept) --"; continue; fi
    run_cell "$s" "$c" || fail=$((fail + 1))
done
"$PYBIN" figures/hifi_recon.py --outdir "$RUN_OUT" > /dev/null 2>&1 || true
"$PYBIN" figures/hifi_recon.py --outdir "$RUN_OUT" || true
[ "$fail" = 0 ] || echo "WARNING: $fail cell(s) failed (rerun to retry them)"
echo "== HiFi reconstruction done: $RUN_OUT/hifi_recon.png =="
