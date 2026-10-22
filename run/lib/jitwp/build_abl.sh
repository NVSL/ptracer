#!/usr/bin/env bash
# build_abl.sh -- the native ELF images of the JIT suites for the "PTracer w/o PT" bar of Figure 6 (run/03_ablation.sh).
#
#   SRC=<whole-program Fast build dir> W=<out dir> bash run/lib/jitwp/build_abl.sh [common java node]
#
# SRC is the run/lib/jitwp/build_stage.sh output ($JITWP_DIR): every image is rewritten from the Fast build's own
# converged spec, with the same keyframe period and --kf-gs counter base, plus `--log-blocks --sync 0' (a block id at
# every basic-block leader, logged into the same software buffer as the values; no sync markers, no Intel PT).  Block
# ids E9Patch cannot place are left out with --keep-going and counted in <image>.e9.unpatched.  PLT stubs and the
# rt_sigreturn restorers get no block id (no instruction starts there / the buffer sink's guard-page handler returns
# through them while it swaps buffers).  Output: W/<group>/{build,logs}, W/stage (stage.py; manifest for repslice.py).
# The runtime hooks then run with PTJIT_ABL=blk (ptracer/runtime/jit/jitpatch.h: a block id at every JIT block leader).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/../../.." && pwd)}"
SRC=${SRC:?SRC=<Fast JITWP_DIR>}; W=${W:?W=<out dir>}
VPY=${PYBIN:-python3}
RW=$ROOT/ptracer/runtime/jit_toolchain/rewrite.py
KF=${KF:-${FAST_KEYFRAME:-128}}
export E9PATCH_DIR=${E9PATCH_DIR:-$ROOT/third_party/e9patch}
is_shared () {
  local t; t=$(readelf -h "$1" | awk '/Type:/{print $2}')
  [ "$t" = DYN ] && ! readelf -d "$1" 2>/dev/null | grep -q '(FLAGS_1).*PIE' && return 0
  return 1
}
excl_ranges () {   # PLT stubs + every rt_sigreturn restorer (`mov $15,%rax; syscall')
  "$VPY" - "$1" <<'PY'
import re, subprocess, sys
r = []
for l in subprocess.run(["readelf", "-SW", sys.argv[1]], capture_output=True, text=True).stdout.splitlines():
    m = re.search(r"\]\s+\.plt(\.got|\.sec)?\s+\S+\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m:
        a = int(m.group(2), 16); r.append("0x%x-0x%x" % (a, a + int(m.group(3), 16)))
    m = re.search(r"\]\s+\.text\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", l)
    if m:
        va, off, sz = (int(v, 16) for v in m.groups())
        data = open(sys.argv[1], "rb").read()[off:off + sz]
        pat = b"\x48\xc7\xc0\x0f\x00\x00\x00\x0f\x05"; i = data.find(pat)
        while i >= 0:
            r.append("0x%x-0x%x" % (va + i, va + i + 9)); i = data.find(pat, i + 1)
print(",".join(r))
PY
}
group () {  # <group>: rewrite its images, then its keyframe plan; returns 1 if an image failed
  local g=$1 fail=0 n base cnt img out log sh x t0 u RWA
  [ -f "$SRC/$g/build/kfplan.json" ] || { echo "[abl] no $SRC/$g/build/kfplan.json (skip $g)"; return 0; }
  mkdir -p "$W/$g/build" "$W/$g/logs"; [ -e "$W/$g/specs" ] || ln -s "$SRC/$g/specs" "$W/$g/specs"
  case $g in common) RWA="--reserve-cursor";; *) RWA="--reserve-cursor --dead-regs off";; esac
  "$VPY" - "$SRC/$g/build/kfplan.json" > "$W/$g/images.txt" <<'PY'
import json, sys
for n, im in json.load(open(sys.argv[1]))["images"].items():
    print(n, im["base"], im["counters"], im["image"])
PY
  while read -r n base cnt img; do
    out=$W/$g/build/$n.e9; log=$W/$g/logs/$n.blk.log
    if [ -s "$out.sitemap.json" ] && grep -q '^ABL_OK' "$log" 2>/dev/null; then echo "[abl] $n: exists"; continue; fi
    sh=""; is_shared "$img" && sh="--shared"
    x=$(excl_ranges "$img"); t0=$(date +%s)
    "$VPY" "$RW" "$SRC/$g/specs/$n.spec.json" "$img" -o "$out" --sink buffer --sync 0 --keyframe "$KF" --kf-gs \
        --kf-base "$base" --log-blocks --count-blocks-retry 3 ${x:+--exclude-ranges "$x"} $sh $RWA \
        $(cat "$SRC/$g/build/$n.dic.e9excl" 2>/dev/null) --keep-going > "$log" 2>&1
    if [ $? -ne 0 ] || [ ! -s "$out.sitemap.json" ]; then echo "[abl] $n: REWRITE FAILED ($log)"; fail=1; continue; fi
    u=$(grep -c '^0x' "$out.unpatched" 2>/dev/null); echo "ABL_OK unpatched=${u:-0} $(( $(date +%s) - t0 )) s" >> "$log"
    echo "[abl] $n: OK, ${u:-0} site(s)/block id(s) left unpatched, $(( $(date +%s) - t0 )) s"
  done < "$W/$g/images.txt"
  # the keyframe plan of the Fast build (same counter bases), re-pointed at these images
  "$VPY" - "$SRC/$g/build/kfplan.json" "$W/$g/build/kfplan.json" "$W/$g/build" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
for n, im in d["images"].items():
    im["e9"] = os.path.realpath(os.path.join(sys.argv[3], n + ".e9"))
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
  return $fail
}
# common first; java and node then rewrite concurrently (one E9Patch process per image)
fail=0; set -- ${*:-common java node}
case " $* " in *" common "*) group common || fail=1;; esac
pids=(); for g in "$@"; do [ "$g" = common ] || { group "$g" & pids+=($!); }; done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "[abl] INCOMPLETE"; exit 1; }
"$VPY" "$HERE/stage.py" "$W" || exit 1
echo "[abl] staged under $W/stage"
