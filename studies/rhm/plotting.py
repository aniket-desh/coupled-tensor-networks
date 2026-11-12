"""Shared figure style for the RHM experiments (PDF figures read only saved results)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

WIDTH_COLORS = {8: '#86b6ef', 16: '#3987e5', 32: '#1c5cab', 64: '#0d366b'}  # one hue, light -> dark with r
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
                     'legend.frameon': False, 'pdf.fonttype': 42})


def steps_x(steps):
    """Log-scale x positions: update 0 is drawn at 0.5 so the initial evaluation stays visible."""
    return [max(s, 0.5) for s in steps]


def save(fig, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)
