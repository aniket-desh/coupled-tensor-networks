"""PNG copies of selected figures, for embedding in NOTE.md (GitHub does not render PDFs inline).

    PYTHONPATH=. python studies/rhm/rhm-belief-geometry/figures_png.py RESULTS_DIR FIGURES_DIR [grammar ...]

Writes FIGURES_DIR/png/<name>.png with the same plotting functions and the same saved results as plot.py, so each PNG
shows exactly what the PDF of the same name shows. The PDFs are not touched. Per-grammar figures (geometry, triangles,
distances) are written for the grammars named on the command line (seed 0); summary, positions and training always."""
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot as PL  # noqa: E402

matplotlib.rcParams['savefig.dpi'] = 150


def main():
    runs = PL.load(sys.argv[1])
    out = Path(sys.argv[2]) / 'png'
    out.mkdir(parents=True, exist_ok=True)
    for g in sys.argv[3:]:
        a = runs[f'{g}_s0']
        PL.geometry(a, out / f'geometry_{g}.png')
        PL.triangles(a, out / f'triangles_{g}.png')
        PL.distances(a, out / f'distances_{g}.png')
    PL.summary(runs, out / 'summary.png')
    PL.positions(runs, out / 'positions.png')
    PL.training(runs, out / 'training.png')


if __name__ == '__main__':
    main()
