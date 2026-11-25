"""Draw the mathematical ceilings and the spectra of the saved legal rule tables."""
import json
from pathlib import Path

import numpy as np

from studies.rhm.plotting import CATEGORICAL, plt, save


HERE = Path(__file__).resolve().parent


def main():
    r = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), layout='constrained')
    ax, heat = axes
    ax.plot(r, r**4, color=CATEGORICAL[0], linewidth=2)
    threshold = 0.5**0.25
    ax.plot([0, threshold, threshold], [.5, .5, 0], '--', color='#626262')
    ax.scatter([threshold], [.5], color=CATEGORICAL[0], zorder=3)
    ax.annotate('0.841', (threshold, .02), xytext=(-5, 6), textcoords='offset points', ha='right')
    ax.set(title='Tied child tables', xlabel='Child alignment $r$',
           ylabel='Maximum fraction explained', xlim=(0, 1), ylim=(0, 1))
    upper = (r[:, None] * r[None, :])**2
    levels = [.0625, .25, .5, .75]
    heat.contourf(r, r, upper, levels=np.linspace(0, 1, 21), cmap='Blues', vmin=0, vmax=1)
    contours = heat.contour(r, r, upper, levels=levels, colors='#16354d', linewidths=.8)
    heat.clabel(contours, fmt={.0625: '1/16', .25: '1/4', .5: '1/2', .75: '3/4'}, fontsize=7)
    heat.plot(r, r, '--', color='#585858', linewidth=.8)
    heat.set(title='Separate child tables', xlabel='Left alignment', ylabel='Right alignment',
             xlim=(0, 1), ylim=(0, 1), aspect='equal')
    heat.grid(False)
    save(fig, HERE / 'figures' / 'frozen-root.pdf')

    tables = json.loads((HERE / 'results' / 'grammar_audit.json').read_text())['tables']
    fig, ax = plt.subplots(figsize=(5.7, 2.65), layout='constrained')
    markers = ['o', 's', '^']
    for i, table in enumerate(tables):
        fractions = table['additive_fractions']
        ax.plot(np.arange(1, len(fractions) + 1), fractions, marker=markers[i], ms=3.3,
                color=CATEGORICAL[i], label=f"Grammar {table['seed']}")
    ax.set(xlabel='Root contrast, ordered within each grammar',
           ylabel='Fraction explained by additive fit', ylim=(0, 1), xlim=(.6, 15.4),
           xticks=[1, 3, 5, 7, 9, 11, 13, 15])
    ax.legend()
    save(fig, HERE / 'figures' / 'grammar-spectra.pdf')


if __name__ == '__main__':
    main()
