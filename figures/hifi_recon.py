#!/usr/bin/env python3
"""HiFi offline reconstruction accuracy figure.

Reads <outdir>/hifi_recon.csv (written by run/05_hifi_recon.sh) and draws, per cell, the per-record
accuracy of HiFi offline reconstruction.  Every bar is annotated with its wrong-record count (0 when
the reconstruction is exact).  Outputs hifi_recon.png (+ .svg, .json) in --outdir.
"""
import argparse, csv, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def default_outdir():
    out = os.environ.get("RUN_OUT")
    if out:
        return Path(out)
    return Path(os.environ.get("OUT", "out")) / "experiments_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=default_outdir())
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    csvp = a.outdir / "hifi_recon.csv"
    if not csvp.exists():
        if not a.progress:
            print("no %s yet" % csvp)
        return
    rows = [r for r in csv.DictReader(open(csvp)) if r.get("compared")]
    if not rows:
        return
    labels = ["%s:%s" % (r["suite"], r["cell"]) for r in rows]
    unk = [float(r["unknown_pct"]) for r in rows]
    wrong = [int(r["wrong"]) for r in rows]

    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(rows) + 2), 3.4))
    x = range(len(rows))
    bars = ax.bar(x, [max(v, 1e-6) for v in unk], color="#4878a8", width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("unknown records (% of ground truth)")
    ax.set_title("HiFi offline reconstruction accuracy (wrong records = 0)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    for i, b in enumerate(bars):
        tag = "wrong=%d" % wrong[i] if wrong[i] else "0 wrong"
        ax.annotate("%s\n%.4f%%" % (tag, unk[i]), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
    ax.margins(y=0.25)
    fig.tight_layout()
    a.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.outdir / "hifi_recon.png", dpi=150)
    fig.savefig(a.outdir / "hifi_recon.svg")
    plt.close(fig)
    out = dict(cells=[dict(suite=r["suite"], cell=r["cell"], size=r["size"],
                           gt_records=int(r["gt_records"]), unknown=int(r["unknown"]),
                           wrong=int(r["wrong"]), unknown_pct=float(r["unknown_pct"]),
                           wrong_pct=float(r["wrong_pct"])) for r in rows])
    (a.outdir / "hifi_recon.json").write_text(json.dumps(out, indent=1) + "\n")
    if not a.progress:
        print("Wrote hifi_recon.png (+ .svg, .json) to %s" % a.outdir)


if __name__ == "__main__":
    main()
