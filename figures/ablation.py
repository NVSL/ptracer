#!/usr/bin/env python3
"""ablation.py -- Figure 6 (ablation of PTracer's new techniques), from the rows of run/fig5.sh (overhead.csv) and
run/03_ablation.sh (ablation.csv).

    python3 figures/ablation.py --run-out DIR [--progress] [--data-only]

Two panels: this run (top) and the paper's published bars (bottom, data/paper_ablation_bars.json).  Per suite:
  * PTracer                   = the Figure 5 Fast bar, computed exactly as figures/six_suite_overhead.py does;
  * PTracer w/o PT            = abl_blk: control flow logged by instrumentation (a block id per basic block) instead of
                                Intel PT, the Fast plan otherwise;
  * PTracer w/o PT and Static = abl_memt: every memory access logged by a plain logging call, no static analysis.
Per cell the median of the repetitions over the median of the vanilla rows of the same cell (the ablation run's own
vanilla rows if it has any, else Figure 5's: on Node.js/Java those of the JIT Fast sweep), then the geometric mean over
the suite's cells.  A run stopped at 200x its vanilla time counts as 200x.  "pending" = no rows yet.
Outputs: DIR/ablation.png (+ .svg, .json).
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import six_suite_overhead as F5  # noqa: E402

PAPER = ROOT / 'data' / 'paper_ablation_bars.json'
BARS = [('ptracer', 'PTracer', 'fast'), ('wo_pt', 'PTracer w/o PT', 'abl_blk'),
        ('wo_pt_static', 'PTracer w/o PT and Static', 'abl_memt')]
COLORS = {'native': '#3d8c40', 'ptracer': '#7e3f9e', 'wo_pt': '#a5392f', 'wo_pt_static': '#e8000b'}


def abl_value(f5, ab, s, cfg):
    """(value, n_ok, n_cells, status, reps) of an ablation configuration, as six_suite_overhead.suite_value."""
    jit = s in ('node', 'java')
    key, base = ('wp:' + cfg, 'wp:vanilla') if jit else (cfg, 'vanilla')
    ratios, failed, cells, reps = [], 0, 0, 0
    for (su, cell, c), v in ab.items():
        if su != s or c != key:
            continue
        cells += 1
        good = [k for k, _, _ in v if k is not None]
        reps = max(reps, len(good))
        van = [k for k, _, _ in ab.get((su, cell, base), []) if k is not None] or \
              [k for k, _, _ in f5.get((su, cell, base), []) if k is not None]
        if good and van:
            ratios.append(statistics.median(good) / statistics.median(van))
        elif any(cp for _, cp, _ in v):
            ratios.append(F5.CAP)
        else:
            failed += 1
    if cells == 0:
        return None, 0, 0, 'none', 0
    if not ratios:
        return None, 0, cells, 'Err', reps
    return F5.geomean(ratios), len(ratios), cells, 'ok', reps


def load(run_out, paper, progress=False):
    d = Path(run_out)
    f5p = [d / 'overhead.csv']
    abp = [d / 'ablation.csv']
    if progress:      # rows of the lanes not merged yet
        f5p = sorted(set(d.glob('overhead*.csv')) - {d / 'overhead.stale.csv'})
        abp = sorted(set(d.glob('ablation*.csv')) - {d / 'ablation.stale.csv'})
    f5 = F5.read_rows([str(p) for p in f5p])
    ab = F5.read_rows([str(p) for p in abp])
    pb = json.loads(Path(paper).read_text())['series']
    rows = []
    for s in F5.SUITES:
        row = dict(suite=s, name=F5.NAMES[s], ours={}, paper={}, cells={}, reps={})
        for key, _, cfg in BARS:
            row['paper'][key] = pb[key][s]
            if key == 'ptracer':
                v, n, tot, st, nr = F5.suite_value(f5, s, 'fast', 'fast')
            else:
                v, n, tot, st, nr = abl_value(f5, ab, s, cfg)
            row['ours'][key] = 'pending' if st == 'none' else st if st != 'ok' else v
            row['cells'][key] = '%d/%d' % (n, tot)
            row['reps'][key] = nr
        rows.append(row)
    return rows


def _panel(ax, rows, which, title):
    import numpy as np
    n = len(BARS) + 1
    w = 0.84 / n
    x = np.arange(len(rows))
    ax.set_yscale('log')
    ax.set_ylim(0.8, 250)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200], labels=['1X', '2X', '5X', '10X', '20X', '50X', '100X', '200X'])
    ax.axhline(1, color=F5.INK3, lw=0.8, ls=':')
    ax.grid(axis='y', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    for xi, r in enumerate(rows):
        x0 = x[xi] - 0.42 + w / 2
        ax.bar(x0, 1 - 0.8, bottom=0.8, width=w * 0.92, color=COLORS['native'], edgecolor='black', lw=0.5,
               label='Uninstrumented' if xi == 0 else None, zorder=3)
        for k, (key, label, _) in enumerate(BARS):
            xp = x0 + (k + 1) * w
            v = r[which].get(key)
            if isinstance(v, (int, float)):
                ax.bar(xp, v - 0.8, bottom=0.8, width=w * 0.92, color=COLORS[key], edgecolor='black', lw=0.5,
                       label=label if xi == 0 else None, zorder=3)
                ax.annotate(('%.2f' % v) if v < 10 else ('%.0f' % v), (xp, v), ha='center', va='bottom',
                            fontsize=6.5, color=F5.INK, xytext=(0, 1.5), textcoords='offset points')
            else:
                ax.bar(xp, 0, bottom=0.8, width=w * 0.92, color=COLORS[key], label=label if xi == 0 else None)
                ax.annotate(str(v if v is not None else 'n/a'), (xp, 0.85), rotation=90, ha='center', va='bottom',
                            fontsize=6.5, color='#c00000' if v == 'Err' else F5.INK3)
    ax.set_xticks(x, labels=[r['name'] for r in rows], fontsize=8.5)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylabel('Runtime overhead (times, log)', fontsize=9)
    ax.set_title(title, loc='left', fontsize=10.5)


def plot(rows, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    _panel(a1, rows, 'ours', 'Figure 6, this run (geometric mean per suite; PTracer = the Figure 5 Fast bar)')
    _panel(a2, rows, 'paper', 'Figure 6, as published in the paper')
    a1.legend(ncol=4, fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.16), frameon=False)
    fig.tight_layout()
    for ext, kw in (('png', dict(dpi=170)), ('svg', {})):   # written next to the final name, then renamed
        tmp = outdir / ('.ablation.tmp.' + ext)
        fig.savefig(tmp, bbox_inches='tight', **kw)
        os.replace(tmp, outdir / ('ablation.' + ext))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-out', type=Path, required=True, help='the experiments directory (overhead.csv, ablation.csv)')
    ap.add_argument('--outdir', type=Path, default=None, help='default: --run-out')
    ap.add_argument('--paper', type=Path, default=PAPER)
    ap.add_argument('--data-only', action='store_true')
    ap.add_argument('--progress', action='store_true', help='during a run: also read the per-lane CSVs, print nothing')
    a = ap.parse_args()
    out = a.outdir or a.run_out
    if a.progress:
        sys.stdout = open(os.devnull, 'w')
    rows = load(a.run_out, a.paper, a.progress)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ablation.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
    print('%-22s' % 'suite' + ''.join('%28s' % b[1] for b in BARS))
    for r in rows:
        print('%-22s' % r['name'] + ''.join('%28s' % F5.fmt(r['ours'][b[0]]) for b in BARS))
        print('%-22s' % '  cells/reps' + ''.join('%28s' % ('%s r%d' % (r['cells'][b[0]], r['reps'][b[0]]))
                                                   for b in BARS))
        print('%-22s' % '  paper' + ''.join('%28s' % F5.fmt(r['paper'][b[0]]) for b in BARS))
    if not a.data_only:
        plot(rows, out)
        print('Wrote ablation.png (+ .svg, .json) to %s' % out)


if __name__ == '__main__':
    main()
