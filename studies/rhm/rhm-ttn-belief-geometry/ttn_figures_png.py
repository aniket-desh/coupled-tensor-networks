"""PNG copies of the figures NOTE.md embeds (GitHub does not render PDFs inline).

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/ttn_figures_png.py RESULTS_DIR FIGURES_DIR PROJ_DIR [cell ...]

Writes FIGURES_DIR/png/<name>.png at 200 dpi with the plotting functions of ttn_plot.py on the same saved results, so
each PNG shows what the PDF of the same name shows. The PDFs are not touched. positions, beliefsplit, summary and
training are always written; geometry and triangles are written for the cells named on the command line (seed 0), e.g.
v8m2s2L5_r32. PROJ_DIR holds the projection files (<run>/proj_ckpt*.npz), which live in the data folder
runs/rhm-ttn-belief-geometry/proj and not next to analysis.json."""
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttn_plot as TP  # noqa: E402

matplotlib.rcParams['savefig.dpi'] = 200


def main():
    runs = TP.PL.load(sys.argv[1])
    out = Path(sys.argv[2]) / 'png'
    proj = Path(sys.argv[3])
    for c in sys.argv[4:]:
        a = dict(runs[f'{c}_s0'], dir=proj / f'{c}_s0')
        TP.geometry(a, out / f'geometry_{c}.png')
        TP.triangles(a, out / f'triangles_{c}.png')
    TP.positions(runs, out / 'positions.png')
    TP.positions(runs, out / 'beliefsplit.png', TP.BELIEF_LINES, 'R² on unseen beliefs', 'bel')
    TP.summary(runs, out / 'summary.png')
    TP.training(runs, out / 'training.png')


if __name__ == '__main__':
    main()
