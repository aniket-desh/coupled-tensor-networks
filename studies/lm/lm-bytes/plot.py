"""Figures for lm-bytes, from saved results only (PDF).
Usage: PYTHONPATH=. python studies/lm/lm-bytes/plot.py <results> <figures>
  curves.pdf  validation bits/byte (2^17 fixed validation positions) against updates: left, the LR pilots on the 10 MB
              pilot set; right, the 40k-update runs and the 160k run (median over seeds, band = range), with the
              100 MB Kneser-Ney orders 4 and 8 as horizontal references.
  TTN and chain runs use the identity init; the 'ttn_var' group (grey) uses the variance init.
  params.pdf  final test bits/byte against parameter count, one marker per seed, with the n-gram orders as lines."""
import json
import re
import sys
from pathlib import Path

import numpy as np

from ctn import style
from ctn.style import plt

COLOR = {'ttn': style.C1, 'ttn_var': style.MUTED, 'ttn_frozenW': '#6da7ec', 'ttn_tied': '#104281', 'mlp': style.C2, 'chain': style.C3, 'transformer': '#8a5cd1'}
DASH = {'ttn_tied': '--', 'ttn_r32_long': ':', 'ttn_var': '-.', 'ttn_frozenW': (0, (1, 1))}
INIT_NOTE = 'TTN and chain: identity init; ttn_var: variance init; ttn_frozenW: fusion cores frozen'
GROUP_COLOR = {'ttn_r16': style.RAMP[2], 'ttn_r64': style.RAMP[5], 'ttn_r128': style.RAMP[6], 'chain_r32': '#8fd6b9'}


def color(group, family):
    return GROUP_COLOR.get(group, COLOR[family])


def load(res):
    return {r['cfg']['name']: r for r in (json.load(open(f)) for f in sorted(Path(res, 'runs').glob('*.json')))}


def ngram_big(runs):
    for k in ('ngram_100M', 'ngram_large'):
        if k in runs and runs[k]['status'] == 'done':
            return runs[k]


def main(res, figs):
    runs = load(res)
    Path(figs).mkdir(parents=True, exist_ok=True)
    ng = ngram_big(runs)
    groups = {}
    for n, r in runs.items():
        if r['cfg'].get('family') != 'ngram' and not n.startswith('pilot_') and r.get('curve', {}).get('step'):
            groups.setdefault(re.sub(r'_s\d+$', '', n), []).append(r)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for n, r in sorted(runs.items()):
        if n.startswith('pilot_') and r.get('curve', {}).get('step'):
            fam, lr = r['cfg']['family'], r['cfg']['lr']
            ax0.plot(r['curve']['step'], r['curve']['val_bits'], color=COLOR[fam], ls='-' if lr > 5e-4 else '--',
                     label=f'{fam} {lr:g}')
    ax0.set(title='LR pilots (pilot set; solid 1e-3, dashed 3e-4)', xlabel='updates', ylabel='validation bits/byte')
    ax0.legend(ncol=2, fontsize=6.5)
    for g, rs in sorted(groups.items()):
        steps = min((r['curve']['step'] for r in rs), key=len)
        V = np.array([r['curve']['val_bits'][:len(steps)] for r in rs])
        c = color(g, rs[0]['cfg']['family'])
        ax1.plot(steps, np.median(V, 0), color=c, ls=DASH.get(g, DASH.get(rs[0]['cfg']['family'], '-')),
                 lw=1.0 if g in ('ttn_r16', 'chain_r32') else 1.6, label=f'{g} (n={len(rs)})')
        if len(rs) > 1:
            ax1.fill_between(steps, V.min(0), V.max(0), color=c, alpha=0.15, lw=0)
    if ng:
        for o, ls in ((4, '--'), (8, '-')):
            ax1.axhline(ng['val_bits'][o - 1], color=style.MUTED, ls=ls, lw=0.9)
            ax1.text(1, ng['val_bits'][o - 1], f' KN order {o}', color=style.MUTED,
                     fontsize=7, va='bottom', transform=ax1.get_yaxis_transform())
    ax1.text(0.98, 0.98, INIT_NOTE, transform=ax1.transAxes, ha='right', va='top', fontsize=6.5, color=style.MUTED)
    ax1.set(title='main runs (median, band = seed range)', xlabel='updates', xscale='log')
    ax1.legend(fontsize=6.5, ncol=2)
    ymax = max([max(r['curve']['val_bits']) for rs in groups.values() for r in rs if np.isfinite(max(r['curve']['val_bits']))]
               + [1.0])
    ax0.set_ylim(top=min(ymax, 8.5))
    fig.tight_layout()
    fig.savefig(Path(figs) / 'curves.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    seen = set()
    for g, rs in sorted(groups.items()):
        for r in rs:
            if r['status'] != 'done':
                continue
            fam = r['cfg']['family']
            lab = g if g not in seen else None
            seen.add(g)
            ax.scatter(r['params'], r['test_bits'], color=color(g, fam), marker='s' if 'long' in g else 'D' if 'tied' in g else 'x' if 'var' in g else 'v' if 'frozen' in g else 'o', s=18,
                       label=lab, zorder=3)
    if ng:
        for o in range(2, len(ng['test_bits']) + 1):
            ax.axhline(ng['test_bits'][o - 1], color=style.MUTED, lw=0.6, ls=':')
            if o not in (2, 4, 8):
                continue
            ax.text(1.0, ng['test_bits'][o - 1], f' KN {o}', color=style.MUTED, fontsize=6.5, va='center',
                    transform=ax.get_yaxis_transform())
    ax.set(xscale='log', xlabel='parameters', ylabel='test bits/byte', title='final test bits/byte (one marker per seed)')
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    # legend outside the axes so it doesn't hide markers (the frozen-fusion TTN sits mid-plot)
    ax.legend(fontsize=6.5, title=INIT_NOTE.replace('; ', '\n'), title_fontsize=6.5, loc='upper left', bbox_to_anchor=(1.08, 1))
    fig.tight_layout()
    fig.savefig(Path(figs) / 'params.pdf')
    plt.close(fig)
    print(f'wrote {figs}/curves.pdf, {figs}/params.pdf')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
