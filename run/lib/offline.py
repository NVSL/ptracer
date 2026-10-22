#!/usr/bin/env python3
"""Offline-reconstruction experiment (run/04_offline.sh): one cell = native time, capture, timed reconstructions.

    offline.py plan                 the cells and the time estimate (printed by --dry-run)
    offline.py cell SUITE CELL      native runs, one capture, `ptrecon' full and decode-only, rows; the capture is deleted
    offline.py table                per-suite summary of the rows (also written next to the figure)

Per cell:
  native   the uninstrumented program (repslice.py configuration `vanilla', the Figure 5 command line), OFF_NATIVE_REPS
           runs; native time = median whole-process wall clock (Memcached: wall clock of the fixed-op-count load run);
  capture  one Fast run (repslice.py `fast'; Node.js/Java `e9fast') with FIG5_KEEP set: the same command as the Figure 5
           Fast bar, but the PT packets, the sideband, the value streams and (JIT) the code dump are kept.  The non-JIT
           images are this experiment's (keyframe period OFF_KEYFRAME, see run/04_offline.sh); the JIT suites use the
           Figure 5 stage;
  recon    `ptrecon' with OFF_JOBS workers (default: all cpus), no trace output: once in full and once --decode-only
           (Intel PT decoding alone); execution = full - decode.  Single-threaded cells (PolyBench/C, pyperformance):
           `--jobs N' (pyperformance `--chunks-per-job 2'); multi-threaded and JIT cells: `--mt --mt-jobs N --mt-split N'.
Rows: $RUN_OUT/offline.csv (reconstructions), $RUN_OUT/offline_native.csv (repslice rows); a row of another VERSION is
ignored and measured again.  Overhead = reconstruction wall time / native time; a suite's value = geometric mean.
"""
import csv, glob, hashlib, json, math, os, re, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VERSION = 1
SUITES = ["poly", "mc", "pyperf", "rust", "node", "java"]
NAMES = {"poly": "PolyBench/C", "mc": "Memcached", "pyperf": "pyperformance", "rust": "Rust Stream",
         "node": "Node.js (Web Tooling)", "java": "Java (Renaissance)"}
# one representative set of cells per suite (the paper's figure)
CELLS = {"poly": "2mm atax correlation doitgen fdtd-2d gemm jacobi-2d lu", "pyperf": "nbody float chaos richards",
         "rust": "rayon", "mc": "memslap", "node": "acorn", "java": "scrabble"}
MT = {"rust", "mc", "node", "java"}
# reconstruction seconds of a cell at 96 workers (full + decode-only), capture + native seconds: for the estimate only
EST = {"2mm": (40, 30), "atax": (3, 10), "correlation": (45, 30), "doitgen": (26, 20), "fdtd-2d": (83, 20),
       "gemm": (57, 15), "jacobi-2d": (80, 20), "lu": (500, 120), "nbody": (8, 30), "float": (8, 30),
       "chaos": (10, 30), "richards": (13, 40), "rayon": (55, 40), "memslap": (5, 60), "acorn": (64, 240),
       "scrabble": (235, 600)}
E = os.environ
RUN_OUT = E.get("RUN_OUT") or os.path.join(E.get("OUT", os.path.join(ROOT, "out")), "experiments_v1")
OFF = E.get("OFF_DIR") or os.path.join(RUN_OUT, "offline")
# the whole-process Fast images of this experiment (run/lib/build_wp.sh with keyframe period OFF_KEYFRAME)
IMG = os.path.join(OFF, "img")
JITWP = E.get("JITWP_DIR") or os.path.join(RUN_OUT, "jitwp")    # the Figure 5 JIT stage
RES = os.path.join(RUN_OUT, "offline.csv")
NAT = os.path.join(RUN_OUT, "offline_native.csv")
FIELDS = ["version", "suite", "cell", "mode", "workers", "wall_s", "rc", "records", "unknown", "trace_mb", "ts"]
PY = E.get("PYBIN") or sys.executable
RECON = os.path.join(ROOT, "ptracer", "offline", "ptrecon")
PYD = os.path.join(ROOT, "suites", "pyperformance")


def jobs():
    return int(E.get("OFF_JOBS") or os.cpu_count())


def cells():
    want = set(filter(None, E.get("OFF_CELLS", "").replace(",", " ").split()))   # "suite:cell" or "suite"
    out = []
    for s in SUITES:
        for c in CELLS[s].split():
            if not want or s in want or "%s:%s" % (s, c) in want:
                out.append((s, c))
    return out


def log(m):
    print("[offline %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# ------------------------------------------------------------------------------------------------ rows
def rows():
    try:
        return [r for r in csv.DictReader(open(RES)) if r.get("version") == str(VERSION)]
    except OSError:
        return []


def done(s, c, mode, n):
    return any(r["suite"] == s and r["cell"] == c and r["mode"] == mode and r["workers"] == str(n) and r["rc"] == "0"
               for r in rows())


def add_row(**kw):
    new = not os.path.exists(RES)
    with open(RES, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(kw, version=VERSION, ts=int(time.time())))


def native_values(s, c):
    """the vanilla rows of a cell: whole-process wall (Memcached: the load run's wall clock = ktime)"""
    v = []
    try:
        for r in csv.DictReader(open(NAT)):
            if r["suite"] == s and r["cell"] == c and r["config"] == "vanilla" and r["rc"] == "0" \
                    and r.get("label") == "offv%d" % VERSION:
                x = r["ktime"] if s == "mc" else r["wall"]
                if x:
                    v.append(float(x))
    except (OSError, KeyError):
        pass
    return v


# ------------------------------------------------------------------------------------------------ repslice
def img(s):
    return IMG


def fast_image(s, c):
    f = os.path.join(img(s), "fastwp")
    return {"poly": os.path.join(f, "poly", "%s_large" % c), "pyperf": os.path.join(f, "py", "pyfast", "bin", "python3.12"),
            "rust": os.path.join(f, "rust", "micro-bench"), "mc": os.path.join(f, "mc", "memcached_sym")}.get(s)


def repslice(s, c, cfg, reps, csvp, keep=None, timeout=10800):
    env = dict(E)
    env.pop("FIG5_KEEP", None)
    if keep:
        env["FIG5_KEEP"] = keep
    FWP = os.path.join(img(s), "fastwp")
    fd, lib = FWP, os.path.join(FWP, "lib")
    if s == "pyperf":
        fd, lib = os.path.join(FWP, "py"), os.path.join(FWP, "py", "lib")
    env.update(FAST_DIR=fd, FIG5_FASTLIB=lib, FIG5_OUT=os.path.join(OFF, "rs"), FIG5_POLY=c, FIG5_RUST=c,
               FIG5_PYPERF_LIST=os.path.join(PYD, "pyperf", "bench94.lst"),
               FIG5_PYRUN=os.path.join(PYD, "pyperf", "run_one_full.py"), FIG5_PYLOOPS=os.path.join(PYD, "pyperf", "loops.tsv"),
               FIG5_PYPATH_VAN="%s/wp/extmods:%s/wp/site" % (PYD, PYD),
               FIG5_PYPATH_FAST="%s/py/pyext:%s/py/pysite" % (FWP, FWP))
    os.makedirs(env["FIG5_OUT"], exist_ok=True)
    a = [PY, os.path.join(HERE, "repslice.py"), "--suites", s, "--configs", cfg, "--cells", c, "--reps", str(reps),
         "--java-reps", str(reps), "--label", "offv%d" % VERSION, "--csv", csvp, "--mutex-inherit", "--timeout", str(timeout),
         "--mc-ops", E.get("OFF_MC_OPS", "20000")]
    if s in ("node", "java"):
        env.update(PTJ_NODE_V8FLAGS=E.get("PTJ_NODE_V8FLAGS", "--no-short-builtin-calls --max-semi-space-size=128 "
                                                               "--single-threaded-gc"),
                   WTB_WARMUP="0", WTB_ITERS="1", WTB_JS=os.path.join(HERE, "jitwp", "wtb.js"), PTJ_JAVA_ITERS_ALL="1")
        a += ["--java-pin-iters", "1", "--java-pin-drop", "0"]
        if cfg == "e9fast":
            node = s == "node"
            a += ["--jit-cache", os.path.join(OFF, "jitcache", s), "--jit-workers", "4" if node else "8",
                  "--jit-cores", E.get("NODE_JIT_CORES", "16,17,18,19") if node else E.get("JAVA_JIT_CORES", "8,9,10,11,16,17,18,19"),
                  "--jitwp-stage", os.path.join(JITWP, "stage"), "--jit-dir", os.path.join(OFF, "jd_%s" % s)]
    with open(os.path.join(OFF, "repslice.log"), "a") as lg:
        lg.write("\n== %s %s %s keep=%s\n" % (s, c, cfg, keep or "-")); lg.flush()
        rc = subprocess.call(a, env=env, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT)
    try:
        last = [r for r in csv.DictReader(open(csvp)) if r["suite"] == s and r["cell"] == c and r["config"] == cfg]
    except OSError:
        last = []
    return rc, (last[-1] if last else None)


# ------------------------------------------------------------------------------------------------ ptrecon arguments
def spec_of(s, orig):
    """the analyzer spec run/lib/build_wp.sh wrote for the original image `orig'"""
    n = "%s_%s" % (hashlib.md5((orig + "\n").encode()).hexdigest()[:6], os.path.basename(orig))
    for d in ("kf", "nokf"):
        p = os.path.join(img(s), "build", "spec_wp", d, n + ".spec.json")
        if os.path.exists(p):
            return p
    return None


def recon_args(s, d):
    if s in ("node", "java"):
        man = json.load(open(os.path.join(JITWP, "stage", "manifest.json")))
        a = ["--sideband", os.path.join(d, "sb.json"), "--cv-dir", os.path.join(d, "cv")]
        for sm in man["sitemaps"]:
            if sm.get("group", s) == s:
                a += ["--spec", sm["spec"], "--sitemap", sm["sitemap"]]
        return a + ["--jitdump", os.path.join(d, "code.jitdump"), "--sitemap", os.path.join(d, "sitemap.json")]
    seen, groups = set(), []
    for p in re.findall(r'"(/[^"]+)"', open(os.path.join(d, "sb.json")).read()):
        if p in seen:
            continue
        for q in (p, os.path.realpath(p)):
            if os.path.exists(q + ".sitemap.json"):
                seen.add(p)
                groups.append((q, json.load(open(q + ".sitemap.json")).get("orig_image")))
                break
    a = ["--sideband", os.path.join(d, "sb.json")]
    if s in MT:
        a += ["--cv-dir", os.path.join(d, "cv")]
    else:
        cv = sorted(glob.glob(os.path.join(d, "cv", "cv.*.bin")), key=os.path.getsize)
        pid = os.path.basename(cv[-1]).split(".")[1] if cv else "0"
        a += ["--aux", os.path.join(d, "aux"), "--cv", os.path.join(d, "cv", "cv.%s.%s.bin" % (pid, pid))]
    for img, orig in groups:
        sp = spec_of(s, orig) if s in MT and orig else None
        a += (["--spec", sp] if sp else []) + ["--sitemap", img + ".sitemap.json"] + (["--orig-image", orig] if orig else [])
    return a


def workers_args(s, c, n):
    if s in MT:
        return ["--mt", "--mt-jobs", str(n), "--mt-split", str(n)]
    return ["--jobs", str(n)] + (["--chunks-per-job", "2"] if s == "pyperf" else [])


def du_mb(d):
    t = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try:
                t += os.lstat(os.path.join(r, f)).st_blocks * 512
            except OSError:
                pass
    return round(t / 2 ** 20)


def free_gb(p):
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 2 ** 30


# ------------------------------------------------------------------------------------------------ one cell
def cell(s, c):
    n = jobs()
    os.makedirs(OFF, exist_ok=True)
    have = len(native_values(s, c))
    reps = int(E.get("OFF_NATIVE_REPS", "5"))
    if have < reps:
        log("%s %s: native, %d runs" % (s, c, reps - have))
        repslice(s, c, "vanilla", reps, NAT, timeout=3600)          # (repslice.py resumes: only the missing runs)
    if done(s, c, "full", n) and done(s, c, "decode", n):
        log("%s %s: reconstructed with %d workers already" % (s, c, n)); return 0
    d = os.path.join(OFF, "cells", "%s.%s" % (s, c))
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "cv"))
    need = float(E.get("OFF_CELL_GB", "40" if s == "java" else "15"))
    if free_gb(d) < need:
        log("%s %s: SKIPPED, %.0f GB free under %s (needs ~%.0f GB)" % (s, c, free_gb(d), d, need)); return 1
    cfg = "e9fast" if s in ("node", "java") else "fast"
    warm = os.path.join(OFF, "jitcache", "%s.%s.warm" % (s, c))
    if cfg == "e9fast" and not os.path.exists(warm):
        log("%s %s: untimed warm-up run (fills the JIT plan cache)" % (s, c))
        _, wr = repslice(s, c, cfg, 1, os.path.join(d, "warm.csv"))
        if wr and wr.get("rc") == "0":
            open(warm, "w").close()
    log("%s %s: capture" % (s, c))
    rc, row = repslice(s, c, cfg, 1, os.path.join(d, "capture.csv"), keep=d)
    if row:
        new = not os.path.exists(os.path.join(OFF, "capture.csv"))
        with open(os.path.join(OFF, "capture.csv"), "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys())); new and w.writeheader(); w.writerow(row)
    crc = int(row["rc"]) if row and row.get("rc") not in (None, "") else 1
    if crc != 0 or not os.path.exists(os.path.join(d, "sb.json")):
        log("%s %s: capture FAILED (rc %s; %s)" % (s, c, row and row.get("rc"), os.path.join(OFF, "repslice.log")))
        add_row(suite=s, cell=c, mode="capture", workers=n, wall_s="", rc=crc or 1, records="", unknown="", trace_mb="")
        if E.get("OFF_KEEP") != "1":
            shutil.rmtree(d, ignore_errors=True)
        return 1
    mb = du_mb(d)
    args = recon_args(s, d)
    env = dict(E, PTRECON_CJIT_DIR=os.path.join(OFF, "tmp"))
    if s == "poly" and c in ("gemm", "2mm"):    # seeded (anchor-free) chunks: exact for these kernels
        env.setdefault("PTRECON_SEEDS", "1")
    os.makedirs(env["PTRECON_CJIT_DIR"], exist_ok=True)
    bad = 0
    for mode in ("full", "decode"):
        if done(s, c, mode, n):
            continue
        sm = os.path.join(d, "recon.%s.json" % mode)
        a = [RECON] + args + workers_args(s, c, n) + (["--decode-only"] if mode == "decode" else []) + ["--summary", sm]
        log("%s %s: reconstruct (%s, %d workers; trace %d MB)" % (s, c, mode, n, mb))
        with open(os.path.join(OFF, "logs", "%s.%s.%s.log" % (s, c, mode)), "w") as lg:
            lg.write(" ".join(a) + "\n\n"); lg.flush()
            t0 = time.monotonic()
            try:
                rrc = subprocess.call(a, env=env, stdout=lg, stderr=subprocess.STDOUT, timeout=int(E.get("OFF_TIMEOUT", "10800")))
            except subprocess.TimeoutExpired:
                rrc = 124
            w = time.monotonic() - t0
        try:
            j = json.load(open(sm))
        except (OSError, ValueError):
            j = {}
        add_row(suite=s, cell=c, mode=mode, workers=n, wall_s="%.3f" % w, rc=rrc, records=j.get("records", ""),
                unknown=j.get("unknown_addr", ""), trace_mb=mb)
        log("%s %s: %s %.1f s rc %d" % (s, c, mode, w, rrc))
        bad |= rrc != 0
    if E.get("OFF_KEEP") != "1":
        shutil.rmtree(d, ignore_errors=True)
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------ summary
def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def summary(n=None):
    """suite -> dict(cells, total, decode, exec) for the rows of `n' workers (default OFF_JOBS)"""
    n = n or jobs()
    by = {}
    for r in rows():
        if r["workers"] == str(n) and r["rc"] == "0" and r["mode"] in ("full", "decode"):
            by[(r["suite"], r["cell"], r["mode"])] = float(r["wall_s"])
    out = {}
    for s in SUITES:
        xt, xd, cs = [], [], []
        for c in CELLS[s].split():
            nat = native_values(s, c)
            if not nat or (s, c, "full") not in by or (s, c, "decode") not in by:
                continue
            m = statistics.median(nat)
            xt.append(by[(s, c, "full")] / m); xd.append(by[(s, c, "decode")] / m); cs.append(c)
        if cs:
            t, dd = geomean(xt), geomean(xd)
            out[s] = dict(cells=cs, total=t, decode=min(dd, t), exec=t - min(dd, t))
    return out


def table():
    n = jobs()
    sm = summary(n)
    lines = ["Offline reconstruction, %d workers: reconstruction time / native time (geometric mean over the cells)" % n,
             "%-24s %6s %9s %9s %9s" % ("suite", "cells", "total", "decode", "exec")]
    for s in SUITES:
        if s in sm:
            v = sm[s]
            lines.append("%-24s %6d %8.1fx %8.1fx %8.1fx" % (NAMES[s], len(v["cells"]), v["total"], v["decode"], v["exec"]))
        else:
            lines.append("%-24s %6s %9s %9s %9s" % (NAMES[s], 0, "-", "-", "-"))
    if sm:
        lines.append("%-24s %6s %8.1fx %8.1fx" % ("average of the suites", "", statistics.mean(v["total"] for v in sm.values()),
                                                    statistics.mean(v["decode"] for v in sm.values())))
    return "\n".join(lines)


def plan():
    n = jobs(); cs = cells()
    sc = 96.0 / n if n < 96 else 1.0
    rec = sum(EST[c][0] for _, c in cs) * sc
    oth = sum(EST[c][1] for _, c in cs)
    print("  cells (%d): %s" % (len(cs), ", ".join("%s:%s" % x for x in cs)))
    print("  workers: %d (OFF_JOBS; default = all %d cpus); native runs per cell: %s" % (n, os.cpu_count(), E.get("OFF_NATIVE_REPS", "5")))
    print("  estimated time on a 96-vCPU machine: cells ~%d min (reconstructions ~%d min at %d workers, captures and"
          % (round((rec + oth) / 60), round(rec / 60), n))
    print("  native runs ~%d min), plus the Fast images of this experiment if absent (~30 min) and the Figure 5 JIT stage"
          % round(oth / 60))
    print("  if absent (~20-40 min); finished cells are skipped")
    done_n = sum(1 for s, c in cs if done(s, c, "full", n) and done(s, c, "decode", n))
    print("  already reconstructed with %d workers: %d of %d cells" % (n, done_n, len(cs)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(os.path.join(OFF, "logs"), exist_ok=True)
    if cmd == "plan":
        plan()
    elif cmd == "cells":
        print(" ".join("%s:%s" % x for x in cells()))
    elif cmd == "cell":
        sys.exit(cell(sys.argv[2], sys.argv[3]))
    elif cmd == "table":
        print(table())
    else:
        sys.exit(__doc__)
