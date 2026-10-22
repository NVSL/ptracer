#!/usr/bin/env python3
"""offline.py -- the offline-reconstruction figure, from the rows of run/04_offline.sh.

    python3 figures/offline.py --outdir DIR      (reads $RUN_OUT/offline.csv and offline_native.csv)

Two panels: this run (top) and the paper's bars (bottom, data/paper_offline_bars.json).  A bar = reconstruction wall
time / uninstrumented run time, geometric mean over the suite's cells, split into Intel PT decoding (a --decode-only
reconstruction) and execution (full - decoding).  "pending" = no rows yet.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'run' / 'lib'))
import offline  # noqa: E402

PAPER = ROOT / 'data' / 'paper_offline_bars.json'
LABEL = {'poly': 'PolyBench(C)', 'mc': 'Memcached(C)', 'pyperf': 'pyperformance(Python)',
         'rust': 'Rust Stream(Rust)', 'node': 'Web Tooling(Node.js)', 'java': 'Renaissance(Java)'}
C_DEC, C_EXEC, INK, INK3 = '#387E22', '#1A1AE6', '#1a1a1a', '#8a8a8a'


def panel(ax, vals, title, ylabel):
    from matplotlib.patches import Patch
    tops = [v['total'] for v in vals.values() if v]
    top = max(tops) if tops else 1.0
    for i, s in enumerate(offline.SUITES):
        v = vals.get(s)
        if not v:
            ax.text(i, top * 0.02, 'pending', ha='center', va='bottom', fontsize=8, color=INK3, rotation=90)
            continue
        dec, tot = v['decode'], v['total']
        ax.bar(i, dec, 0.6, color=C_DEC, edgecolor=INK, lw=0.5, zorder=3)
        ax.bar(i, tot - dec, 0.6, bottom=dec, color=C_EXEC, edgecolor=INK, lw=0.5, zorder=3)
        for y0, h in ((0, dec), (dec, tot - dec)):
            if h > 0.13 * top:
                ax.text(i, y0 + h / 2, '%.1f' % h, ha='center', va='center', fontsize=8, color='white')
        ax.text(i, tot + top * 0.015, '%.1fx' % tot, ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(range(len(offline.SUITES)))
    ax.set_xticklabels([LABEL[s] for s in offline.SUITES], rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', color='#dddddd', zorder=0)
    ax.legend(handles=[Patch(facecolor=C_DEC, edgecolor=INK, lw=0.5, label='Decoding'),
                       Patch(facecolor=C_EXEC, edgecolor=INK, lw=0.5, label='Execution')],
              loc='upper left', fontsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path(offline.RUN_OUT))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = offline.jobs()
    ours = offline.summary(n)
    p = json.loads(PAPER.read_text())
    paper = {s: dict(total=p['total'][s], decode=p['decode'][s]) for s in p['suites']}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7.0))
    avg = ('; average %.1fx' % statistics.mean(v['total'] for v in ours.values())) if ours else ''
    panel(a1, ours, 'This run%s' % avg, 'Reconstruction time /\nuninstrumented time')
    panel(a2, paper, 'The paper', 'Reconstruction time /\nuninstrumented time')
    fig.tight_layout()
    a.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.outdir / 'offline.png', dpi=150)
    (a.outdir / 'offline.txt').write_text(offline.table() + '\n')
    print(offline.table())
    print('figure: %s' % (a.outdir / 'offline.png'))


if __name__ == '__main__':
    main()
