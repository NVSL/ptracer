#!/usr/bin/env python3
"""Measurement versions of the Figure 5 configurations.

When an update changes what a configuration measures, its entry in VERSION is bumped: "<config>" for every suite, or
"<suite>/<config>" for one suite only (it then overrides "<config>" for that suite).  The version of the rows kept in
$RUN_OUT/overhead*.csv is recorded per suite and configuration in the ledger $RUN_OUT/overhead.versions.json (a
directory without a ledger holds version-1 rows; a ledger entry "<config>" covers every suite without its own entry).
Before a run, `prune' drops the rows of every (suite, configuration) whose recorded version is older than VERSION (they
are moved to overhead.stale.csv), so the resume logic re-measures exactly those rows and keeps all others; the figure
ignores stale rows as well.  The configurations whose rows were dropped are listed in
$RUN_OUT/overhead.remeasure.json: run/01_overhead.sh re-measures those with a reduced plan (see PTW_REMEASURE_* there;
PTW_REMEASURE_FULL=1 re-measures them in full).

    row_versions.py prune <RUN_OUT> [--dry-run]
    row_versions.py remeasure <RUN_OUT> <config>     # exit 0 iff <config> (of any suite) is being re-measured after an update

The Figure 6 rows (run/03_ablation.sh, $RUN_OUT/ablation*.csv) have their own table ABL_VERSION and ledger
ablation.versions.json:  row_versions.py prune <RUN_OUT> --set ablation [--dry-run]
"""
import csv
import glob
import json
import os
import sys

# configuration -> version of its measurement (absent = 1)
VERSION = {
    "pinhifi_ptw": 2,   # 2: mixed PTWRITE sink with a per-cell budget (1: every site logged with ptwrite)
    "fast": 2,          # 2: keyframe period 128 on every image but ld.so, analyzer v2.38 (1: 1024 / none, v2.37)
    "fast_ptw": 2,      # 2: as fast, and PTWRITE spacing 12 (1: 3)
    "e9fast": 2,        # 2: keyframe period 128 in the images and the JIT-code hooks, analyzer v2.38
    "e9fast_ptw": 4,    # 4: as e9fast, PTWRITE spacing 12, mixed sink in node/libjvm.so (budgeted PTWRITE sites),
                        #    JIT-code sites on the buffer
    "pyperf/fast_ptw": 3,   # PTWRITE sites budgeted on all 94 benchmarks
    "rust/fast_ptw": 3,     # spacing 12
    "mc/fast_ptw": 3,       # PTWRITE lists budgeted on Memcached (mc/lib)
}
# Figure 6 configurations (run/03_ablation.sh), same rules
ABL_VERSION = {
    "abl_blk": 1,
    "abl_memt": 1,
}
TABLES = {"overhead": VERSION, "ablation": ABL_VERSION}
SUITES = ("poly", "pyperf", "rust", "mc", "node", "java")
LEDGER = "overhead.versions.json"
REMEASURE = "overhead.remeasure.json"


def current(suite, cfg, kind="overhead"):
    t = TABLES[kind]
    return t.get("%s/%s" % (suite, cfg), t.get(cfg, 1))


def configs(kind="overhead"):
    return sorted({k.split("/")[-1] for k in TABLES[kind]})


def version_of(rec, suite, cfg):
    """the version a ledger records for (suite, cfg)"""
    return rec.get("%s/%s" % (suite, cfg), rec.get(cfg, 1))


def recorded(run_out, kind="overhead"):
    """configuration -> version of the rows stored in run_out (default 1)."""
    p = os.path.join(run_out, LEDGER if kind == "overhead" else kind + ".versions.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def stale_configs(run_out, kind="overhead"):
    """{(suite, config)} whose rows in run_out are of an older measurement version."""
    rec = recorded(run_out, kind)
    return {(s, c) for s in SUITES for c in configs(kind) if version_of(rec, s, c) < current(s, c, kind)}


def csvs(run_out, kind="overhead"):
    return sorted(p for p in glob.glob(os.path.join(run_out, kind + "*.csv"))
                  if not p.endswith(kind + ".stale.csv"))


def kidx(hdr):
    return [hdr.index(k) for k in ("suite", "cell", "config", "rep", "label") if k in hdr]


def prune(run_out, dry=False, kind="overhead"):
    stale = stale_configs(run_out, kind)
    drop, keep = {}, {}
    for p in csvs(run_out, kind):
        with open(p, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        hdr, body = rows[0], rows[1:]
        try:
            ci, si = hdr.index("config"), hdr.index("suite")
        except ValueError:
            continue
        old = [r for r in body if len(r) > max(ci, si) and (r[si], r[ci]) in stale]
        new = [r for r in body if not (len(r) > max(ci, si) and (r[si], r[ci]) in stale)]
        for r in old:
            drop[(r[si], r[ci])] = drop.get((r[si], r[ci]), 0) + 1
        for r in new:                       # a row can be both in a lane's CSV and in the merged one
            if len(r) > max(ci, si):
                keep.setdefault((r[si], r[ci]), set()).add(tuple(r[i] if i < len(r) else "" for i in kidx(hdr)))
        if old and not dry:
            sp = os.path.join(run_out, kind + ".stale.csv")
            fresh = not os.path.exists(sp)
            with open(sp, "a", newline="") as f:
                w = csv.writer(f)
                if fresh:
                    w.writerow(hdr + ["from"])
                for r in old:
                    w.writerow(r + [os.path.basename(p)])
            tmp = p + ".tmp"
            with open(tmp, "w", newline="") as f:
                csv.writer(f).writerows([hdr] + new)
            os.replace(tmp, p)
    if drop:
        print("[versions] %s rows of an older measurement version: %s -> re-measured"
              % ("would drop" if dry else "dropped", ", ".join(
                  "%s/%s (%d)" % (s, c, n) for (s, c), n in sorted(drop.items()))))
    if dry:
        print("[versions] rows kept (%s.csv and lane files): " % kind + (", ".join(
            "%s/%s (%d)" % (s, c, len(n)) for (s, c), n in sorted(keep.items())) or "none"))
    elif os.path.isdir(run_out):
        rec = recorded(run_out, kind)
        rec.update({c: TABLES[kind].get(c, 1) for c in configs(kind)})
        rec.update({"%s/%s" % (s, c): current(s, c, kind) for s in SUITES for c in configs(kind)})
        with open(os.path.join(run_out, LEDGER if kind == "overhead" else kind + ".versions.json"), "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        if drop and kind == "overhead":
            rm = remeasured(run_out)
            rm.update({"%s/%s" % (s, c): current(s, c) for s, c in drop})
            with open(os.path.join(run_out, REMEASURE), "w") as f:
                json.dump(rm, f, indent=1, sort_keys=True)
    return drop


def remeasured(run_out):
    """"<suite>/<config>" (or "<config>") -> version, for the rows prune() dropped in this directory."""
    try:
        return json.load(open(os.path.join(run_out, REMEASURE)))
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "remeasure":
        c = sys.argv[3]
        sys.exit(0 if any(k == c or k.endswith("/" + c) for k in remeasured(sys.argv[2])) else 1)
    if len(sys.argv) < 3 or sys.argv[1] != "prune":
        sys.exit(__doc__)
    kind = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv[3:] else "overhead"
    prune(sys.argv[2], dry="--dry-run" in sys.argv[3:], kind=kind)
