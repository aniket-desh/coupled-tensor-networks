/* Gradient flow on T_oij = U_k(o)i (RB)_oj, the same model as ../model.py.
 *
 * K = 1: one u shared by both outputs, 10 parameters. K = 2: each output owns its u, 12 parameters.
 * theta = [U (K x 2), R (2 x 2), B (2 x 2)]. The tensor is stored flat at index 4 o + 2 i + j.
 * Loss is 1/2 ||T - T*||^2, which is the population loss for whitened inputs. Time is gradient-flow time.
 */
(function (root) {
  'use strict';

  function flowGrad(th, K, Ts, g, T) {
    // Returns the loss. g receives dL/dtheta and T receives the tensor when passed.
    var r = 2 * K, b = r + 4, L = 0, o, i, j;
    var M = [th[r] * th[b] + th[r + 1] * th[b + 2], th[r] * th[b + 1] + th[r + 1] * th[b + 3],
             th[r + 2] * th[b] + th[r + 3] * th[b + 2], th[r + 2] * th[b + 1] + th[r + 3] * th[b + 3]];
    var dM = [0, 0, 0, 0];
    for (i = 0; i < r; i++) g[i] = 0;
    for (o = 0; o < 2; o++) {
      var k = K === 1 ? 0 : o;
      for (i = 0; i < 2; i++) for (j = 0; j < 2; j++) {
        var v = th[2 * k + i] * M[2 * o + j], d = v - Ts[4 * o + 2 * i + j];
        if (T) T[4 * o + 2 * i + j] = v;
        L += d * d;
        g[2 * k + i] += d * M[2 * o + j];
        dM[2 * o + j] += d * th[2 * k + i];
      }
    }
    // Chain rule through M = R B: dR = dM B^T, dB = R^T dM.
    g[r] = dM[0] * th[b] + dM[1] * th[b + 1];
    g[r + 1] = dM[0] * th[b + 2] + dM[1] * th[b + 3];
    g[r + 2] = dM[2] * th[b] + dM[3] * th[b + 1];
    g[r + 3] = dM[2] * th[b + 2] + dM[3] * th[b + 3];
    g[b] = th[r] * dM[0] + th[r + 2] * dM[2];
    g[b + 1] = th[r] * dM[1] + th[r + 2] * dM[3];
    g[b + 2] = th[r + 1] * dM[0] + th[r + 3] * dM[2];
    g[b + 3] = th[r + 1] * dM[1] + th[r + 3] * dM[3];
    return 0.5 * L;
  }

  function tensor(th, K) {
    var T = new Float64Array(8), g = new Float64Array(th.length);
    flowGrad(th, K, new Float64Array(8), g, T);
    return T;
  }

  function normSq(Ts) { var s = 0; for (var n = 0; n < 8; n++) s += Ts[n] * Ts[n]; return s; }

  // Smallest loss the network can reach. U is rank one across input 1 and M = RB is unconstrained, so
  // Eckart-Young gives the optimum: shared u unfolds T* as input-1 x (output, input-2), and per-output u
  // treats each output's 2 x 2 slice on its own. Either way the floor is half the discarded sigma^2.
  function smallEig(a, b, c) { // [[a, b], [b, c]]
    var m = (a + c) / 2, d = Math.sqrt((a - c) * (a - c) / 4 + b * b);
    return Math.max(m - d, 0);
  }
  function lossFloor(Ts, K) {
    var a = 0, b = 0, c = 0, o, j, f;
    if (K === 1) {
      for (o = 0; o < 2; o++) for (j = 0; j < 2; j++) {
        var x = Ts[4 * o + j], y = Ts[4 * o + 2 + j];
        a += x * x; b += x * y; c += y * y;
      }
      return 0.5 * smallEig(a, b, c);
    }
    f = 0;
    for (o = 0; o < 2; o++) {
      var p = Ts[4 * o], q = Ts[4 * o + 1], r = Ts[4 * o + 2], s = Ts[4 * o + 3];
      f += 0.5 * smallEig(p * p + q * q, p * r + q * s, r * r + s * s);
    }
    return f;
  }

  // The aligned start stays on the pattern T_111, T_212 with c_k > 0, so it can only reach targets of that form.
  function alignedCanReach(Ts) {
    for (var n = 0; n < 8; n++) if (n !== 0 && n !== 5 && Math.abs(Ts[n]) > 1e-12) return false;
    return Ts[0] >= 0 && Ts[5] >= 0;
  }

  function initAligned(delta, K) {
    var U = K === 1 ? [Math.SQRT2 * delta, 0] : [delta, 0, delta, 0];
    return Float64Array.from(U.concat([delta, 0, 0, delta, delta, 0, 0, delta]));
  }

  function rng(seed) { // mulberry32
    return function () {
      seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function initRandom(delta, K, seed) {
    // R and B are drawn first, so both networks start from the same R, B and the same first row of U.
    var u = rng(seed), z = [];
    for (var n = 0; n < 12; n++) z.push(delta * Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u()));
    return Float64Array.from(z.slice(8, 8 + 2 * K).concat(z.slice(0, 8)));
  }

  // Dormand-Prince 5(4), with a cap on how far theta moves in relative terms per step (as in ../hierarchy.py),
  // so long plateaus cost little and fast transitions stay resolved.
  var C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
  var A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
  var E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];

  function simulate(Ts, K, th0, opt) {
    opt = opt || {};
    var rtol = opt.rtol || 1e-9, atol = opt.atol || 1e-13, cap = opt.cap || 0.05;
    var maxSteps = opt.maxSteps || 150000, tMax = opt.tMax || 1e6, tolStop = opt.tolStop || 1e-11;
    var P = th0.length, y = Float64Array.from(th0), w = new Float64Array(P), yn = new Float64Array(P);
    var norm = Math.max(normSq(Ts), 1e-4), floor = lossFloor(Ts, K);
    var kk = []; for (var s = 0; s < 7; s++) kk.push(new Float64Array(P));
    var T = new Float64Array(8), L = flowGrad(y, K, Ts, kk[0], T);

    // Every accepted step is kept, so peaks and half-times are read from the integrator's own points.
    var ts = [0], ys = [Float64Array.from(y)], Ls = [L], Tprev = Float64Array.from(T);
    var sign = [], half = new Array(8), peak = new Array(8), tPeak = new Array(8);
    for (var n = 0; n < 8; n++) {
      sign.push(Ts[n] >= 0 ? 1 : -1); half[n] = null; peak[n] = sign[n] * T[n]; tPeak[n] = 0;
    }
    var t = 0, dt = 1e-2, steps = 0, done = false;

    while (steps < maxSteps && t < tMax) {
      var fn = 0, yy = 0, i, j;
      for (i = 0; i < P; i++) { fn += kk[0][i] * kk[0][i]; yy += y[i] * y[i]; }
      dt = Math.min(dt, cap * Math.sqrt(yy / Math.max(fn, 1e-300)));
      for (s = 1; s < 7; s++) {
        for (i = 0; i < P; i++) {
          var inc = 0;
          for (j = 0; j < s; j++) inc += A[s][j] * kk[j][i];
          w[i] = y[i] - dt * inc; // the flow is minus the gradient
        }
        flowGrad(w, K, Ts, kk[s]);
      }
      var err = 0;
      for (i = 0; i < P; i++) {
        var e = 0;
        for (s = 0; s < 7; s++) e += E[s] * kk[s][i];
        var sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(w[i]));
        err = Math.max(err, Math.abs(dt * e) / sc);
      }
      steps++;
      if (err > 1) { dt *= Math.max(0.2, 0.9 * Math.pow(err, -0.2)); continue; }
      t += dt;
      y.set(w);
      L = flowGrad(y, K, Ts, yn, T);
      var tmp = kk[0]; kk[0] = yn; yn = tmp;
      for (n = 0; n < 8; n++) {
        var v = sign[n] * T[n];
        if (v > peak[n]) { peak[n] = v; tPeak[n] = t; }
        var h = 0.5 * Math.abs(Ts[n]);
        if (half[n] === null && h > 0 && v >= h) {
          var vp = sign[n] * Tprev[n];
          half[n] = t - dt + dt * (h - vp) / (v - vp);
        }
      }
      Tprev.set(T);
      ts.push(t); ys.push(Float64Array.from(y)); Ls.push(L);
      if (L - floor < tolStop * norm) { done = true; break; }
      dt *= Math.min(5, Math.max(0.2, 0.9 * Math.pow(Math.max(err, 1e-10), -0.2)));
    }

    // End of the interesting part: first time the loss is within 1e-7 of where it finishes.
    var Lend = done ? floor : L, tSettle = t;
    for (n = 0; n < Ls.length; n++) if (Ls[n] - Lend < 1e-7 * norm) { tSettle = ts[n]; break; }
    return { K: K, Ts: Ts, ts: ts, ys: ys, Ls: Ls, steps: steps, done: done, floor: floor, tSettle: tSettle,
             peak: peak, tPeak: tPeak, half: half, sign: sign, tEnd: t, Lend: L };
  }

  // Parameters at time t by linear interpolation; after the run ends the final state holds.
  function stateAt(run, t, hint) {
    var ts = run.ts, a = hint || 0, b = ts.length - 1;
    if (t >= ts[b]) return { i: b, f: 0, hint: b };
    if (t <= 0) return { i: 0, f: 0, hint: 0 };
    if (ts[a] > t) a = 0;
    while (b - a > 1) { var m = (a + b) >> 1; if (ts[m] <= t) a = m; else b = m; }
    return { i: a, f: (t - ts[a]) / (ts[a + 1] - ts[a]), hint: a };
  }
  function thetaAt(run, t) {
    var s = stateAt(run, t), y = run.ys[s.i];
    if (!s.f) return y;
    var z = run.ys[s.i + 1], out = new Float64Array(y.length);
    for (var n = 0; n < y.length; n++) out[n] = y[n] + s.f * (z[n] - y[n]);
    return out;
  }
  function lossAt(run, t) {
    var s = stateAt(run, t);
    return s.f ? run.Ls[s.i] + s.f * (run.Ls[s.i + 1] - run.Ls[s.i]) : run.Ls[s.i];
  }

  // Entries with a real target. Overshoot only means something if the network can reach the target.
  function targetEntries(Ts) {
    var mx = 0, out = [], n;
    for (n = 0; n < 8; n++) mx = Math.max(mx, Math.abs(Ts[n]));
    for (n = 0; n < 8; n++) if (mx > 0 && Math.abs(Ts[n]) > 0.01 * mx) out.push(n);
    return out;
  }
  function summary(run) {
    var Ts = run.Ts, list = targetEntries(Ts), reach = run.done && run.floor < 1e-9 * Math.max(normSq(Ts), 1e-4);
    var over = 0, entry = null, lastHalf = null, firstHalf = null, allHalf = list.length > 0;
    list.forEach(function (n) {
      var o = run.peak[n] / Math.abs(Ts[n]) - 1;
      if (reach && o > over) { over = o; entry = n; }
      if (run.half[n] === null) { allHalf = false; return; }
      if (lastHalf === null || run.half[n] > lastHalf) lastHalf = run.half[n];
      if (firstHalf === null || run.half[n] < firstHalf) firstHalf = run.half[n];
    });
    return { reach: reach, overshoot: over, overshootEntry: entry, firstHalf: firstHalf,
             lastHalf: allHalf ? lastHalf : null, list: list };
  }

  var api = { flowGrad: flowGrad, tensor: tensor, lossFloor: lossFloor, alignedCanReach: alignedCanReach,
    initAligned: initAligned, initRandom: initRandom, simulate: simulate, thetaAt: thetaAt, lossAt: lossAt,
    targetEntries: targetEntries, summary: summary, normSq: normSq };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TTN = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
