"""Check interactive/core.js against model.py, which produced the published numbers.

Each check targets an error that would make the page show a wrong trajectory: a wrong gradient on a general
(editable) target, integrator drift on the conserved balances, or a wrong closed-form loss floor.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here.parent))
import model as m

JS = r"""
const C = require(process.argv[1]);
const out = {};
const S = (s1, s2) => { const T = new Float64Array(8); T[0] = s1; T[5] = s2; return T; };

// Gradient at fixed random points, returned for comparison with model.grad and with differences.
const pts = JSON.parse(process.argv[2]);
out.grads = pts.map(p => { const g = new Float64Array(p.th.length);
  const L = C.flowGrad(Float64Array.from(p.th), (p.th.length - 8) / 2, Float64Array.from(p.T), g); return {L, g: [...g]}; });

// Published trajectories.
out.runs = [];
for (const [rho, delta] of [[0.5, 0.01], [0.2, 0.01], [0.1, 0.01], [0.8, 0.01], [0.2, 0.03]]) for (const K of [1, 2]) {
  const r = C.simulate(S(1, rho), K, C.initAligned(delta, K)), s = C.summary(r);
  out.runs.push({rho, delta, K, over: s.overshoot, half: [r.half[0], r.half[5]], steps: r.steps, done: r.done});
}

// Balances along the shared flow on the aligned sector: a^2 - r1^2 - r2^2 and b_k^2 - r_k^2 stay zero.
const r = C.simulate(S(1, 0.5), 1, C.initAligned(0.01, 1)), y = r.ys[r.ys.length - 1];
out.balance = [y[0]*y[0] - y[2]*y[2] - y[5]*y[5], y[6]*y[6] - y[2]*y[2], y[9]*y[9] - y[5]*y[5]];

// Loss floors: start at random and compare where the flow ends with the closed form.
out.floors = JSON.parse(process.argv[3]).map(([name, T]) => { const Ts = Float64Array.from(T);
  return [1, 2].map(K => { const r = C.simulate(Ts, K, C.initRandom(0.1, K, 7));
    return {name, K, floor: r.floor, end: r.Lend, done: r.done, steps: r.steps}; }); }).flat();

// Timing for one recompute on a slow case, since it runs on every drag event.
const t0 = Date.now(); C.simulate(S(1, 0.5), 2, C.initAligned(0.001, 2)); out.ms_slow_case = Date.now() - t0;
console.log(JSON.stringify(out));
"""

rng = np.random.default_rng(0)


def fd_grad(th, T):
    def L(x):
        U, R, B = m.unpack(x)
        return 0.5 * ((m.tensor(U, R, B) - T) ** 2).sum()
    return np.array([(L(th + 1e-6 * e) - L(th - 1e-6 * e)) / 2e-6 for e in np.eye(th.size)])


pts = []
for P in (10, 12):
    for T in (m.target([1, 0.3]), rng.standard_normal((2, 2, 2))):
        pts.append({'th': (0.5 * rng.standard_normal(P)).tolist(), 'T': T.ravel().tolist()})
AND = [1, 1, 1, 0, 0, 0, 0, 1]
floors = [['and gate', AND], ['two input-1 directions', [1, 0, 0, 0, 0, 0, 0, 0.5]],
          ['rotated', (np.array([1, 0, 1, 0, 0, 0.5, 0, 0.5]) / np.sqrt(2)).tolist()]]
res = json.loads(subprocess.run(['node', '-e', JS, str(here / 'core.js'), json.dumps(pts), json.dumps(floors)],
                                capture_output=True, text=True, check=True).stdout)

err = max(np.abs(np.array(r['g']) - fd_grad(np.array(p['th']), np.array(p['T']).reshape(2, 2, 2))).max()
          for p, r in zip(pts, res['grads']))
print(f'gradient vs centered differences, general targets: {err:.2e}')

print('rho  delta  K  overshoot % (js, model.py)   half-times (js)      half-times (model.py)     steps')
for r in res['runs']:
    s = np.array([1.0, r['rho']])
    sol = m.flow(m.init_aligned(s, r['delta'], r['K'] == 1), s, 2000, stop_loss=1e-16)
    t = m.refine(sol)
    c = np.array([m.coeffs(y) for y in sol.sol(t).T]).T
    over, half = m.overshoot(c, s)[0], [m.half_time(t, c[k], s[k]) for k in range(2)]
    print(f"{r['rho']:.1f}  {r['delta']:.2f}  {r['K']}  {100*r['over']:9.5f} {100*over:9.5f}     "
          f"{r['half'][0]:9.4f} {r['half'][1]:9.4f}  {half[0]:9.4f} {half[1]:9.4f}  {r['steps']}")
print('balance drift', np.abs(res['balance']).max())
for f in res['floors']:
    print(f"{f['name']:24s} K={f['K']} floor {f['floor']:.8f} flow ends {f['end']:.8f} done={f['done']} steps={f['steps']}")
print('slowest recompute (ms)', res['ms_slow_case'])
