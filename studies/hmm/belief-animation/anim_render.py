"""Render the belief-geometry animations (MP4, H.264) from saved frames only.

    PYTHONPATH=. python studies/hmm/belief-animation/anim_render.py FRAMES_DIR OUT_DIR [mess3 rrxor rhm]

FRAMES_DIR holds <run>.npz and <run>.json from anim_frames.py. One video per process, 1920 x 1080, 12 frames per
second, one frame per saved checkpoint (the checkpoints are log-spaced in the update, so the video runs at constant
speed in log time), with the first frame held for 1 s and the last for 3 s; all.mp4 is the three in sequence.

Every panel shows the same held-out rows (the first 1000 test sequences of the analysis set), colored by their true
belief, at the position the probe puts them; only the positions move between frames. The probe is refit at each
checkpoint, so the motion is what a linear (or projective) readout can recover from the network at that update.
  mess3  the 2-simplex; all positions t = 1 .. 15 overlaid; color = belief as RGB.
  rrxor  the top two principal axes of the distinct true beliefs; small points = rows, large = their mean per
         distinct belief; color = the truth on axes 1-3, rank-normalized (the layout of hmm_plot.geometry).
  rhm    grammar v8m2s2L4; one row per position (the position with the most distinct beliefs in each half of the
         sequence); axes = the top two principal axes of the true beliefs at that position; points and colors as rrxor.
Under each panel: R2 of the rows of that panel's probe on the whole test half (pooled over positions for the HMMs).
Bottom: excess test loss over Bayes and the pooled probe R2 against the update, with a cursor at the current frame.
An arm with fewer updates than the longest run of its video (RHM: 30,000 against the transformer's 50,000) holds its
last checkpoint once it has ended."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'rhm' / 'rhm-belief-geometry'))
import plot as PL  # noqa: E402  (rgb, tri_xy; importing it sets that study's rcParams, so the style below comes after)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

INK, INK2 = '#171717', '#545454'
COLORS = ['#0F4D92', '#B64342', '#42949E', '#9A4D8E', '#C8862B']  # figures4papers palette, plus an ochre for a fifth arm
FPS, HOLD_FIRST, HOLD_LAST = 12, 12, 36
for f in Path('/usr/share/texmf/fonts/opentype/public/tex-gyre').glob('texgyreheros-*.otf'):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    'font.family': ['TeX Gyre Heros', 'DejaVu Sans'], 'font.size': 11,
    'axes.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False,
    'axes.grid': False, 'lines.linewidth': 1.6, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

HMM_ARMS = [('{p}_transformer_s0', 'transformer', 'affine probe, last layer'),
            ('{p}_chain_s0', 'chain, r = 32', 'projective probe, state'),
            ('{p}_tree_s0', 'tree, r = 32', 'projective probe, top bonds'),
            ('{p}_tree8_s0', 'tree, r = 8', 'projective probe, top bonds')]
RHM = 'v8m2s2L4'
VIDEOS = {
    'mess3': dict(title='Mess3', arms=[(r.format(p='mess3'), n, d) for r, n, d in HMM_ARMS]),
    'rrxor': dict(title='RRXOR', arms=[(r.format(p='rrxor'), n, d) for r, n, d in HMM_ARMS]),
    'rhm': dict(title='RHM (v = 8, m = 2, L = 4)',
                arms=[(f'{RHM}_s0', 'transformer', 'affine probe, last layer'),
                      (f'{RHM}_tree_s0', 'aligned tree, r = 32', 'bilinear probe, top bonds'),
                      (f'{RHM}_r16_s0', 'aligned tree, r = 16', 'bilinear probe, top bonds'),
                      (f'{RHM}_perm_s0', 'permuted tree, r = 32', 'bilinear probe, top bonds'),
                      (f'{RHM}_chain_s0', 'chain', 'affine probe, state')]),
}
OUTLINE = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])


def load(frames_dir, run):
    z = dict(np.load(frames_dir / f'{run}.npz'))
    z.update(json.load(open(frames_dir / f'{run}.json')))
    z['pred'] = z['pred'].astype(np.float32)
    return z


def rows_hmm(arms):
    """One panel row: every position overlaid. Returns [(label, truth xy, colors, uid or None, lims, pick)], where
    pick(arm, frame) gives that arm's predicted xy and R2 at a frame."""
    true = arms[0]['true'].reshape(-1, arms[0]['true'].shape[-1])  # (P * N, S); the same rows for every arm
    if true.shape[1] == 3:
        proj = lambda b: np.stack(PL.tri_xy(b), 1)
        col, uid, lims = np.clip(true, 0, 1), None, ((-0.08, 1.08), (-0.08, np.sqrt(3) / 2 + 0.08))
    else:
        uniq, uid = np.unique(true.round(5), axis=0, return_inverse=True)
        uid = uid.ravel()
        mu = uniq.mean(0)
        Vh = np.linalg.svd(uniq - mu, full_matrices=False)[2][:3]
        proj = lambda b: (b - mu) @ Vh[:2].T
        col = PL.rgb((true - mu) @ Vh.T, uid)
        z = proj(true)
        pad = 0.25 * (z.max(0) - z.min(0)).max()
        lims = ((z[:, 0].min() - pad, z[:, 0].max() + pad), (z[:, 1].min() - pad, z[:, 1].max() + pad))

    def pick(a, f):
        return proj(a['pred'][f].reshape(-1, true.shape[1])), 1 - a['sse'][f].sum() / a['sst'].sum()
    return [('', proj(true), col, uid, lims, pick)]


def rows_rhm(arms):
    """Two panel rows: the position with the most distinct beliefs in each half of the sequence."""
    a0 = arms[0]
    pos, T = list(a0['positions']), a0['cfg']['T']
    out = []
    for half in (lambda t: t <= T // 2, lambda t: t > T // 2):
        i = max((i for i, t in enumerate(pos) if half(t)), key=lambda i: a0['n_unique'][i])
        true, uid = a0['true'][i], a0['uid'][i]
        col = PL.rgb(true, uid)
        z = true[:, :2]
        pad = 0.25 * (z.max(0) - z.min(0)).max()
        lims = ((z[:, 0].min() - pad, z[:, 0].max() + pad), (z[:, 1].min() - pad, z[:, 1].max() + pad))
        pick = lambda a, f, i=i: (a['pred'][f][i][:, :2], 1 - a['sse'][f][i] / a['sst'][i])
        out.append((f"$t = {pos[i]}$\n{a0['n_unique'][i]} beliefs", z, col, uid, lims, pick))
    return out


def centroids(xy, uid):
    ids = np.unique(uid)
    return np.stack([xy[uid == k].mean(0) for k in ids])


def render(name, frames_dir, out_dir):
    spec = VIDEOS[name]
    arms = [load(frames_dir, r) for r, _, _ in spec['arms']]
    rows = rows_hmm(arms) if name != 'rhm' else rows_rhm(arms)
    steps = sorted(set(int(s) for a in arms for s in a['steps']))
    n_arm, n_row = len(arms), len(rows)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    gs = fig.add_gridspec(n_row + 1, n_arm + 1, left=0.05, right=0.985, top=0.83 if n_row == 1 else 0.845, bottom=0.085,
                          height_ratios=[1] * n_row + [0.72 if n_row == 1 else 0.8], hspace=0.3 if n_row == 1 else 0.36, wspace=0.06)
    fig.text(0.05, 0.955, f"{spec['title']}: beliefs decoded from the hidden state during training", fontsize=17, weight='bold',
             va='center')
    step_txt = fig.text(0.985, 0.955, '', fontsize=17, va='center', ha='right', family='DejaVu Sans Mono')
    few = rows[0][3] is not None
    panels = []  # (row, arm index, scatter, centroid scatter or None, R2 text)
    for i, (label, zt, col, uid, lims, pick) in enumerate(rows):
        for j in range(n_arm + 1):
            ax = fig.add_subplot(gs[i, j])
            ax.set(xlim=lims[0], ylim=lims[1], xticks=[], yticks=[], aspect='equal')
            for sp in ax.spines.values():
                sp.set_visible(False)
            if not few:
                ax.plot(OUTLINE[:, 0], OUTLINE[:, 1], color='0.65', lw=0.8, zorder=0)
            if j == 0:
                if few:
                    ax.scatter(*centroids(zt, uid).T, c=np.stack([col[uid == k][0] for k in np.unique(uid)]), s=42,
                               edgecolors='k', linewidths=0.4)
                else:
                    ax.scatter(zt[:, 0], zt[:, 1], c=col, s=2.5, alpha=0.8, edgecolors='none')
                ax.set_ylabel(label, fontsize=11, rotation=0, ha='right', va='center', labelpad=8)
                if i == 0:
                    ax.set_title('Bayes belief\n', fontsize=12.5, weight='bold', pad=4)
                continue
            sc = ax.scatter(zt[:, 0], zt[:, 1], c=col, s=2.2 if few else 1.2, alpha=0.2 if few else 0.6, edgecolors='none')
            cen = ax.scatter(*centroids(zt, uid).T, c=np.stack([col[uid == k][0] for k in np.unique(uid)]), s=42,
                             edgecolors='k', linewidths=0.4) if few else None
            if i == 0:
                ax.set_title(spec['arms'][j - 1][1], fontsize=12.5, weight='bold', color=COLORS[j - 1], pad=20)
                ax.text(0.5, 1.03, spec['arms'][j - 1][2], transform=ax.transAxes, ha='center', va='bottom', fontsize=9.5, color=INK2)
            txt = ax.text(0.5, -0.03, '', transform=ax.transAxes, ha='center', va='top', fontsize=11, family='DejaVu Sans Mono')
            panels.append((i, j - 1, sc, cen, txt))
    # bottom strips: excess loss and pooled probe R2 against the update; update 0 is drawn at x = X0
    X0 = 4
    xs = lambda s: np.maximum(np.asarray(s, dtype=float), X0)
    axl = fig.add_subplot(gs[n_row, 1:1 + n_arm // 2 + n_arm % 2])
    axr = fig.add_subplot(gs[n_row, 1 + n_arm // 2 + n_arm % 2:])
    for k, a in enumerate(arms):
        h = a['hist']
        axl.plot(xs([q['step'] for q in h]), np.maximum([q['excess'] for q in h], 1e-4), color=COLORS[k])
        axr.plot(xs(a['steps']), np.clip(1 - a['sse'].sum(1) / a['sst'].sum(), 0, 1), color=COLORS[k])
    axl.set(xscale='log', yscale='log', ylabel='excess loss (nats / token)')
    axr.set(xscale='log', ylim=(-0.03, 1.03), ylabel='probe R²' + (', all positions' if n_row > 1 else ''))
    cap = fig.add_subplot(gs[n_row, 0])
    cap.axis('off')
    cap.text(0, 0.98, 'Seed 0. The probe is refit at every\ncheckpoint on 30,000 sequences.\n\n'
             + ('Points: 1,000 held-out sequences,\nall 15 positions overlaid,\ncolored by the true belief.' if n_row == 1 else
                'Small points: 1,000 held-out\nsequences at position t.\nLarge: their mean per true belief.\nColor: the true belief.')
             + '\n\nCurves: panel-title colors.', transform=cap.transAxes, fontsize=9.5 if n_row == 1 else 9, color=INK2, va='top',
             linespacing=1.35)
    ticks = [X0] + [t for t in (10, 100, 1000, 10_000) if t <= steps[-1]]
    cursors = []
    for ax in (axl, axr):
        ax.set_xticks(ticks, ['0'] + [f'{t:,d}' for t in ticks[1:]])
        ax.set_xticks([], minor=True)
        ax.set_xlim(X0 * 0.85, steps[-1] * 1.15)
        ax.set_xlabel('update', labelpad=1)
        ax.tick_params(labelsize=9.5)
        cursors.append(ax.axvline(X0, color=INK, lw=1.0, alpha=0.7))
    pos = axl.get_position()
    dx = 0.045 if n_row == 1 else 0.085  # room for the caption on the left
    axl.set_position([pos.x0 + dx, pos.y0, pos.width - dx - 0.025, pos.height])
    axl.yaxis.set_minor_formatter(NullFormatter())
    pos = axr.get_position()
    axr.set_position([pos.x0 + 0.035, pos.y0, pos.width - 0.035, pos.height])

    def draw(step):
        for i, k, sc, cen, txt in panels:
            a = arms[k]
            f = int(np.searchsorted(a['steps'], step, side='right')) - 1
            xy, r2 = rows[i][5](a, f)
            sc.set_offsets(xy)
            if cen is not None:
                cen.set_offsets(centroids(xy, rows[i][3]))
            done = step > a['steps'][-1]
            txt.set_text(f'R² = {r2:6.3f}' + (' (ended)' if done else ''))
        for c in cursors:
            c.set_xdata([max(step, X0)] * 2)
        step_txt.set_text(f'update {step:>6,d}')

    out = out_dir / f'{name}.mp4'
    writer = FFMpegWriter(fps=FPS, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                                                                '-profile:v', 'high', '-movflags', '+faststart'])
    with writer.saving(fig, str(out), dpi=150):
        for n, step in enumerate(steps):
            draw(step)
            for _ in range(HOLD_FIRST if n == 0 else HOLD_LAST if n == len(steps) - 1 else 1):
                writer.grab_frame()
    plt.close(fig)
    print(f'wrote {out}: {len(steps)} checkpoints, {len(steps) - 2 + HOLD_FIRST + HOLD_LAST} frames', flush=True)
    return out


def main():
    frames_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or list(VIDEOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [render(n, frames_dir, out_dir) for n in names]
    if len(outs) > 1:  # the same encoder settings and size, so the streams are joined without re-encoding
        lst = out_dir / 'all.txt'
        lst.write_text(''.join(f"file '{o.name}'\n" for o in outs))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy',
                        '-movflags', '+faststart', str(out_dir / 'all.mp4')], check=True)
        lst.unlink()
        print('wrote', out_dir / 'all.mp4')


if __name__ == '__main__':
    main()
