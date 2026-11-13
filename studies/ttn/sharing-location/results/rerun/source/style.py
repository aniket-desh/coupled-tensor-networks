"""Shared figure style. Import before plotting; series identity (C1, C2, C3) is fixed across figures."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#0b0b0b', '#8a8984'
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#104281']  # sequential, light to dark

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e8e7e3', 'grid.linewidth': 0.6, 'axes.edgecolor': MUTED, 'lines.linewidth': 1.6,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'legend.fontsize': 8,
                     'savefig.dpi': 170, 'figure.facecolor': 'white'})
