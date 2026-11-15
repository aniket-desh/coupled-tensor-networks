"""PNG copies of selected figures, for embedding in NOTE.md (GitHub does not render PDFs inline).

    PYTHONPATH=. python studies/rhm/rhm-belief-architectures/arch_figures_png.py RESULTS_DIR FIGURES_DIR [config_arm ...]

Writes FIGURES_DIR/png/<name>.png with the plotting functions of arch_plot.py on the same saved results, so each PNG
shows exactly what the PDF of the same name shows. The PDFs are not touched. summary, positions and training are always
written; geometry_<config>_<arm>.png for each <config>_<arm> named on the command line (seed 0), which needs that run's
proj_ckpt*.npz in RESULTS_DIR (see README, "Where the files are")."""
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_plot as AP  # noqa: E402

matplotlib.rcParams['savefig.dpi'] = 200


def main():
    runs = AP.PL.load(sys.argv[1])
    out = Path(sys.argv[2]) / 'png'
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[3:]:
        AP.geometry(runs[f'{name}_s0'], out / f'geometry_{name}.png')
    AP.summary(runs, out / 'summary.png')
    AP.positions(runs, out / 'positions.png')
    AP.training(runs, out / 'training.png')


if __name__ == '__main__':
    main()
