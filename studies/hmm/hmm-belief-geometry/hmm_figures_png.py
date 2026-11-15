"""PNG copies of the figures, for embedding in NOTE.md (GitHub does not render PDFs inline).

    PYTHONPATH=. python studies/hmm/hmm-belief-geometry/hmm_figures_png.py RESULTS_DIR FIGURES_DIR

Writes FIGURES_DIR/png/<name>.png at 200 dpi with the plotting functions of hmm_plot.py and the same saved results,
so each PNG shows what the PDF of the same name shows. The PDFs are not touched."""
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm_plot as HP  # noqa: E402

matplotlib.rcParams['savefig.dpi'] = 200


def main():
    runs = HP.PL.load(sys.argv[1])
    out = Path(sys.argv[2]) / 'png'
    out.mkdir(parents=True, exist_ok=True)
    for proc in HP.PROCESSES:
        if any(f'{proc}_{m}_s0' in runs for m in HP.MODELS):
            S = next(a for r, a in runs.items() if r.startswith(proc + '_'))['cfg']['S']
            HP.geometry(runs, proc, out / f"{'triangle' if S == 3 else 'pca'}_{proc}.png")
    HP.positions(runs, out / 'positions.png')
    HP.summary(runs, out / 'summary.png')
    HP.training(runs, out / 'training.png')


if __name__ == '__main__':
    main()
