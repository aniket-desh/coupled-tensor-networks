// Reduced population gradient flow of the planted two-level model: the equations of
// studies/rhm/child_feature_enabling/planted.py (notes toy.tex and untied.tex), integrated with an adaptive
// Dormand-Prince 5(4) step. Checked against results/planted/planted.json by explorer/check.js.
//
// Tied state (r, b, w); untied state (rL, rR, b, w). s = 1 adds the easy target U, s = 0 leaves the interaction alone.
const PHI = (1 + Math.sqrt(5)) / 2;
const LAMBDA_1 = 1 / PHI;
const LAMBDA_2 = (Math.sqrt(PHI * PHI + 4 * PHI) - PHI) / 2;

function rhs(tied, s, y) {
  if (tied) {
    const r = y[0], b = y[1], w = y[2];
    const R = 1 + b * b + 2 * w * w;
    return [(1 - r * r) * (s * b / Math.sqrt(R) + 2 * w * r), s * r * Math.sqrt(R) - b * R, R * r * r - w * R * R];
  }
  const rl = y[0], rr = y[1], b = y[2], w = y[3];
  const Rl = 1 + b * b + w * w, Rr = 1 + w * w;
  const al = rl * Math.sqrt(Rl), ar = rr * Math.sqrt(Rr);
  return [(1 - rl * rl) * (s * b + w * ar) / Math.sqrt(Rl), (1 - rr * rr) * w * al / Math.sqrt(Rr),
          s * al - b * Rl, al * ar - w * Rl * Rr];
}

// [rL, rR, D_h, A_h]: alignments, the trained hard fit and its frozen-child ceiling.
function observe(tied, y) {
  if (tied) {
    const r = y[0], b = y[1], w = y[2];
    const R = 1 + b * b + 2 * w * w;
    const d = w * R - r * r;
    return [r, r, r ** 4 - d * d, r ** 4];
  }
  const rl = y[0], rr = y[1], b = y[2], w = y[3];
  const Rl = 1 + b * b + w * w, Rr = 1 + w * w;
  const d = w * Math.sqrt(Rl * Rr) - rl * rr;
  return [rl, rr, (rl * rr) ** 2 - d * d, (rl * rr) ** 2];
}

// One Dormand-Prince step from y of size h: returns [5th-order state, error estimate].
const C = [[1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
           [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
           [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
           [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
const E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];

function step(f, y, h) {
  const k = [f(y)];
  let next = y;
  for (const row of C) {
    next = y.map((v, i) => v + h * row.reduce((acc, c, j) => acc + c * k[j][i], 0));
    k.push(f(next));
  }
  const err = y.map((_, i) => h * E.reduce((acc, e, j) => acc + e * k[j][i], 0));
  return [next, err];
}

// Integrate from alignment delta until the last of the three first upward crossings of 1/2
// (rL, rR, D_h), or t = 4/delta^2 + 300, then `tail` further. Returns the event times and the sampled path.
function integrate(tied, s, delta, opts) {
  const o = Object.assign({ rtol: 1e-9, atol: 1e-12, tail: 15, hmax: 2, keep: true }, opts || {});
  const f = (y) => rhs(tied, s, y);
  const cap = 4 / (delta * delta) + 300;
  let y = tied ? [delta, 0, 0] : [delta, delta, 0, 0];
  let t = 0, h = 1e-3, end = cap;
  const events = [null, null, null];  // rL, rR, D_h
  const watched = [0, 1, 2];
  let obs = observe(tied, y);
  const path = o.keep ? [[t, ...obs]] : null;
  let guard = 0;
  while (t < end && guard++ < 5e6) {
    h = Math.min(h, o.hmax, end - t);
    const [next, err] = step(f, y, h);
    let ratio = 0;
    for (let i = 0; i < y.length; i++) {
      ratio = Math.max(ratio, Math.abs(err[i]) / (o.atol + o.rtol * Math.max(Math.abs(y[i]), Math.abs(next[i]))));
    }
    if (ratio > 1) { h *= Math.max(0.2, 0.9 * ratio ** -0.2); continue; }
    const nextObs = observe(tied, next);
    for (const k of watched) {
      if (events[k] === null && obs[k] < 0.5 && nextObs[k] >= 0.5) {
        let lo = 0, hi = 1;  // bisect the fraction of this step at which the observable crosses 1/2
        for (let it = 0; it < 50; it++) {
          const mid = (lo + hi) / 2;
          if (observe(tied, step(f, y, mid * h)[0])[k] >= 0.5) hi = mid; else lo = mid;
        }
        events[k] = t + hi * h;
      }
    }
    t += h; y = next; obs = nextObs;
    if (path) path.push([t, ...obs]);
    if (end === cap && events.every((e) => e !== null)) end = Math.min(cap, t + o.tail);
    h *= Math.min(5, Math.max(0.2, 0.9 * Math.max(ratio, 1e-10) ** -0.2));
  }
  return { tied, s, delta, events: { left: events[0], right: events[1], hard: events[2] }, tFinal: t, path, cap };
}

// Leading terms of the half-fit time T(D_h = 1/2) as delta -> 0; each law has an O(1) remainder.
function law(tied, s, delta) {
  const L = Math.log(1 / delta);
  if (s === 1) return tied ? L / LAMBDA_1 : (1 / LAMBDA_1 + 1 / LAMBDA_2) * L;
  return tied ? 1 / (4 * delta * delta) + 2.5 * L : 1 / (2 * delta * delta) + 3 * L;
}

if (typeof module !== 'undefined') module.exports = { rhs, observe, integrate, law, LAMBDA_1, LAMBDA_2, PHI };
