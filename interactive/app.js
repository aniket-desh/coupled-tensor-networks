/* View layer. All numbers come from core.js. Flat tensor index n = 4 o + 2 i + j, shown 1-based as "oij". */
(function () {
  'use strict';
  var C = window.TTN;
  var $ = function (id) { return document.getElementById(id); };
  var SVGNS = 'http://www.w3.org/2000/svg';
  var ENTRY_HUE = [1, 3, 4, 5, 6, 2, 7, 8]; // n -> palette slot, so T111 and T212 get the first two hues on the default target
  var name = function (n) { return '' + ((n >> 2) + 1) + (((n >> 1) & 1) + 1) + ((n & 1) + 1); };
  var hue = function (n) { return 'var(--s' + ENTRY_HUE[n] + ')'; };
  var netName = function (K) { return K === 1 ? 'shared u' : 'one u per output'; };

  var r2 = Math.SQRT1_2;
  var PRESETS = [
    { id: 'p1', label: 'Strong and weak', T: [1, 0, 0, 0, 0, 0.5, 0, 0] },
    { id: 'p2', label: 'Much weaker second', T: [1, 0, 0, 0, 0, 0.2, 0, 0] },
    { id: 'p3', label: 'Nearly equal', T: [1, 0, 0, 0, 0, 0.9, 0, 0] },
    { id: 'p4', label: 'Rotated input 1', T: [r2, 0, r2, 0, 0, r2 / 2, 0, r2 / 2] },
    { id: 'p5', label: 'Two input-1 directions', T: [1, 0, 0, 0, 0, 0, 0, 0.5] },
    { id: 'p6', label: 'AND gate', T: [1, 1, 1, 0, 0, 0, 0, 1] }
  ];

  var S = { T: Float64Array.from(PRESETS[0].T), preset: 'p1', K: 1, delta: 0.03, init: 'aligned', seed: 1, overlay: true,
            t: 0, playing: !matchMedia('(prefers-reduced-motion: reduce)').matches, speed: 1, hl: null };
  var R = {}; // current runs, samples and scales
  var SECONDS = 14; // length of one full playback at 1x

  // ------------------------------------------------------------ formatting
  function f3(v) { var s = Math.abs(v) < 0.0005 ? 0 : v; return s.toFixed(3); }
  function f2(v) { return (Math.abs(v) < 0.005 ? 0 : v).toFixed(2); }
  function fT(t) { return t >= 100 ? t.toFixed(0) : t.toFixed(1); }
  function fL(v) { if (v === 0) return '0'; if (v >= 0.01) return v.toFixed(3); var e = Math.floor(Math.log10(v)); return (v / Math.pow(10, e)).toFixed(1) + 'e' + e; }
  function tone(el, v, vmax) {
    el.style.setProperty('--v', Math.min(1, Math.abs(v) / vmax).toFixed(3));
    el.style.setProperty('--tone', v < 0 ? 'var(--neg)' : 'var(--pos)');
  }
  function colorScale() { var m = 1; for (var n = 0; n < 8; n++) m = Math.max(m, Math.abs(S.T[n])); return m; }

  // ------------------------------------------------------------ the two stacked-matrix views
  // Oblique layout: input 2 runs right, input 1 runs down, output runs back and up, as in the reference sketch.
  var CUBE = { ox: 66, oy: 58, jx: 90, iy: 78, ox2: 42, oy2: -39, w: 72, h: 37 };
  function cellPos(n) {
    var o = n >> 2, i = (n >> 1) & 1, j = n & 1;
    return { x: CUBE.ox + j * CUBE.jx + o * CUBE.ox2, y: CUBE.oy + i * CUBE.iy + o * CUBE.oy2 };
  }
  function bracket(x0, y0, x1, y1, c) {
    var d = 5;
    return '<path d="M' + (x0 + d) + ' ' + y0 + 'h' + (-d) + 'V' + y1 + 'h' + d + 'M' + (x1 - d) + ' ' + y0 + 'h' + d + 'V' + y1 + 'h' + (-d) +
      '" fill="none" stroke="' + c + '" stroke-width="1.3"/>';
  }
  function cubeFrame() {
    var c = CUBE, x1 = c.ox + c.jx + c.w, y1 = c.oy + c.iy + c.h;
    var s = '<svg viewBox="0 0 300 196" aria-hidden="true"><defs><marker id="ah" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6z" fill="var(--ink2)"/></marker></defs>';
    s += bracket(c.ox + c.ox2 - 5, c.oy + c.oy2 - 4, x1 + c.ox2 + 5, y1 + c.oy2 + 4, 'var(--ink3)');
    s += bracket(c.ox - 5, c.oy - 4, x1 + 5, y1 + 4, 'var(--ink)');
    s += '<text x="' + (c.ox + c.ox2 + 8) + '" y="' + (c.oy + c.oy2 - 8) + '">output 2</text>';
    s += '<text x="' + (c.ox + 8) + '" y="' + (y1 + 16) + '">output 1</text>';
    // Axis triad: names and directions of the three indices.
    var ax = 12, ay = 34;
    s += '<line x1="' + ax + '" y1="' + ay + '" x2="' + (ax + 20) + '" y2="' + (ay - 18) + '" stroke="var(--ink2)" marker-end="url(#ah)"/>';
    s += '<line x1="' + ax + '" y1="' + ay + '" x2="' + (ax + 28) + '" y2="' + ay + '" stroke="var(--ink2)" marker-end="url(#ah)"/>';
    s += '<line x1="' + ax + '" y1="' + ay + '" x2="' + ax + '" y2="' + (ay + 28) + '" stroke="var(--ink2)" marker-end="url(#ah)"/>';
    s += '<text x="' + (ax + 24) + '" y="' + (ay - 18) + '">output</text>';
    s += '<text x="' + (ax + 33) + '" y="' + (ay + 4) + '">input 2</text>';
    s += '<text x="' + (ax - 12) + '" y="' + (ay + 42) + '">input 1</text>';
    return s + '</svg>';
  }
  var targetCells = [], modelCells = [];
  function buildCubes() {
    var tg = $('cube-target'), md = $('cube-model');
    tg.innerHTML = cubeFrame(); md.innerHTML = cubeFrame().replace(/id="ah"/, 'id="ah2"').replace(/#ah\)/g, '#ah2)');
    [4, 5, 6, 7, 0, 1, 2, 3].forEach(function (n) { // back slice first so the front slice sits on top
      var p = cellPos(n);
      var a = document.createElement('div');
      a.className = 'cell edit'; a.style.left = p.x + 'px'; a.style.top = p.y + 'px'; a.dataset.entry = n;
      a.innerHTML = '<small>' + name(n) + '</small>';
      var inp = document.createElement('input');
      inp.type = 'text'; inp.inputMode = 'decimal'; inp.id = 'target-' + name(n);
      inp.setAttribute('aria-label', 'Target for output ' + ((n >> 2) + 1) + ', input 1 index ' + (((n >> 1) & 1) + 1) + ', input 2 index ' + ((n & 1) + 1));
      a.appendChild(inp); tg.appendChild(a); targetCells[n] = { el: a, input: inp };
      wireEdit(n, a, inp);
      a.addEventListener('pointerenter', function () { setHl(n); }); a.addEventListener('pointerleave', function () { setHl(null); });

      var b = document.createElement('div');
      b.className = 'cell'; b.style.left = p.x + 'px'; b.style.top = p.y + 'px'; b.dataset.entry = n;
      b.innerHTML = '<small>' + name(n) + '</small><span class="v num"></span><span class="up" aria-hidden="true">▲</span>';
      md.appendChild(b); modelCells[n] = { el: b, v: b.querySelector('.v') };
      b.addEventListener('pointerenter', function () { setHl(n); }); b.addEventListener('pointerleave', function () { setHl(null); });
    });
  }
  function clampT(v) { return Math.max(-1.5, Math.min(1.5, v)); }
  function wireEdit(n, cell, input) {
    var drag = null;
    cell.addEventListener('pointerdown', function (e) {
      if (document.activeElement === input) return; // already typing: let the caret work
      e.preventDefault(); cell.setPointerCapture(e.pointerId);
      drag = { y: e.clientY, v: S.T[n], moved: false };
    });
    cell.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var dy = drag.y - e.clientY;
      if (Math.abs(dy) > 2) drag.moved = true;
      if (drag.moved) { setTarget(n, Math.round(clampT(drag.v + dy * 0.01) * 100) / 100, true); }
    });
    function end(e) {
      if (!drag) return;
      var moved = drag.moved; drag = null;
      if (moved) restart(); else { input.focus(); input.select(); }
    }
    cell.addEventListener('pointerup', end); cell.addEventListener('pointercancel', end);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        var st = (e.shiftKey ? 0.01 : 0.05) * (e.key === 'ArrowUp' ? 1 : -1);
        setTarget(n, Math.round(clampT(S.T[n] + st) * 100) / 100, true); restart();
      } else if (e.key === 'Enter') { input.blur(); }
    });
    input.addEventListener('blur', function () {
      var v = parseFloat(input.value);
      if (isFinite(v) && Math.abs(clampT(v) - S.T[n]) > 1e-9) { setTarget(n, clampT(v), false); restart(); }
      else showTarget();
    });
  }
  function setTarget(n, v) { S.T[n] = v; S.preset = null; schedule(); }
  function showTarget() {
    var vm = colorScale();
    for (var n = 0; n < 8; n++) {
      var c = targetCells[n];
      if (document.activeElement !== c.input) c.input.value = f2(S.T[n]);
      tone(c.el, S.T[n], vm);
    }
  }

  // ------------------------------------------------------------ the tensor network with its entries
  var tnCells = [];
  function bubble(cx, cy, rx, ry, title, rowsCols, thStart, rows, cols, rowLab, colLab) {
    var w = 54, h = 26, x0 = cx - cols * w / 2, y0 = cy - rows * h / 2 + 9, s = '';
    s += '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + rx + '" ry="' + ry + '" fill="var(--panel)" stroke="var(--ink3)" stroke-dasharray="1.5 3"/>';
    s += '<text x="' + cx + '" y="' + (y0 - 18) + '" text-anchor="middle" class="lab">' + title + '</text>';
    s += '<text x="' + cx + '" y="' + (y0 - 3) + '" text-anchor="middle" class="lab2" style="font-size:10.5px">' + rowsCols + '</text>';
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var idx = thStart + r * cols + c;
      s += '<rect data-p="' + idx + '" x="' + (x0 + c * w + 2) + '" y="' + (y0 + r * h + 1) + '" width="' + (w - 4) + '" height="' + (h - 2) + '" rx="2" stroke="var(--rule)"/>';
      s += '<text data-v="' + idx + '" class="val" x="' + (x0 + c * w + w / 2) + '" y="' + (y0 + r * h + h / 2) + '"></text>';
    }
    var yb0 = y0 - 1, yb1 = y0 + rows * h + 1, xb0 = x0 - 2, xb1 = x0 + cols * w + 2;
    s += bracket(xb0, yb0, xb1, yb1, 'var(--ink)');
    return s;
  }
  function buildTN() {
    var K = S.K, svg = $('tn'), s = '';
    var Rn = { x: 300, y: 92 }, Un = { x: 186, y: 222 }, Bn = { x: 414, y: 222 };
    s += '<defs><marker id="tnarrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7z" fill="currentColor"/></marker></defs>';
    // Open legs.
    s += '<line x1="' + Rn.x + '" y1="' + (Rn.y - 24) + '" x2="' + Rn.x + '" y2="22" stroke="currentColor" stroke-width="1.6" marker-end="url(#tnarrow)"/>';
    s += '<text class="lab2" x="' + (Rn.x + 8) + '" y="20">output, index <tspan class="lab">o</tspan> (dim 2)</text>';
    s += '<line x1="' + Un.x + '" y1="316" x2="' + Un.x + '" y2="' + (Un.y + 26) + '" stroke="currentColor" stroke-width="1.6" marker-end="url(#tnarrow)"/>';
    s += '<text class="lab2" x="' + (Un.x - 8) + '" y="312" text-anchor="end">input 1, index <tspan class="lab">i</tspan> (dim 2)</text>';
    s += '<line x1="' + Bn.x + '" y1="316" x2="' + Bn.x + '" y2="' + (Bn.y + 26) + '" stroke="currentColor" stroke-width="1.6" marker-end="url(#tnarrow)"/>';
    s += '<text class="lab2" x="' + (Bn.x + 8) + '" y="312">input 2, index <tspan class="lab">j</tspan> (dim 2)</text>';
    // Bonds. The one difference between the two networks is drawn here.
    s += '<line x1="' + Rn.x + '" y1="' + Rn.y + '" x2="' + Bn.x + '" y2="' + Bn.y + '" stroke="currentColor" stroke-width="1.6"/>';
    s += '<text class="lab2" x="' + ((Rn.x + Bn.x) / 2 + 12) + '" y="' + ((Rn.y + Bn.y) / 2 - 2) + '"><tspan class="lab">β</tspan> (dim 2)</text>';
    if (K === 1) {
      s += '<line x1="' + Rn.x + '" y1="' + Rn.y + '" x2="' + Un.x + '" y2="' + Un.y + '" stroke="currentColor" stroke-width="1.6"/>';
      s += '<text class="lab2" x="' + ((Rn.x + Un.x) / 2 - 12) + '" y="' + ((Rn.y + Un.y) / 2 - 2) + '" text-anchor="end"><tspan class="lab">α</tspan> (dim 1)</text>';
    } else {
      var dot = { x: Rn.x, y: 46 };
      s += '<line x1="' + dot.x + '" y1="' + dot.y + '" x2="' + Un.x + '" y2="' + Un.y + '" stroke="currentColor" stroke-width="1.6"/>';
      s += '<circle cx="' + dot.x + '" cy="' + dot.y + '" r="4.5" fill="currentColor"/>';
      s += '<text class="lab2" x="' + (dot.x - 10) + '" y="' + (dot.y - 2) + '" text-anchor="end">output index copied</text>';
    }
    // Bubbles with dashed leaders, as in the reference sketch.
    var bu = { x: 76, y: 196 }, br = { x: 498, y: 76 }, bb = { x: 536, y: 224 };
    s += '<line x1="' + (Un.x - 24) + '" y1="' + Un.y + '" x2="' + (bu.x + 64) + '" y2="' + (bu.y + 16) + '" stroke="var(--ink3)" stroke-dasharray="5 4"/>';
    s += '<line x1="' + (Rn.x + 24) + '" y1="' + Rn.y + '" x2="' + (br.x - 84) + '" y2="' + br.y + '" stroke="var(--ink3)" stroke-dasharray="5 4"/>';
    s += '<line x1="' + (Bn.x + 24) + '" y1="' + Bn.y + '" x2="' + (bb.x - 88) + '" y2="' + bb.y + '" stroke="var(--ink3)" stroke-dasharray="5 4"/>';
    if (K === 1) s += bubble(bu.x, bu.y, 72, 54, 'u ∈ ℝ<tspan dy="-5" style="font-size:11px">1×2</tspan>', 'one row, columns i', 0, 1, 2);
    else s += bubble(bu.x, bu.y, 72, 66, 'U ∈ ℝ<tspan dy="-5" style="font-size:11px">2×2</tspan>', 'rows o, columns i', 0, 2, 2);
    var r0 = 2 * K, b0 = r0 + 4;
    s += bubble(br.x, br.y, 84, 72, 'R ∈ ℝ<tspan dy="-5" style="font-size:11px">2×1×2</tspan>', 'rows o, columns β', r0, 2, 2);
    s += bubble(bb.x, bb.y, 88, 72, 'B ∈ ℝ<tspan dy="-5" style="font-size:11px">2×2</tspan>', 'rows β, columns j', b0, 2, 2);
    [[Un, K === 1 ? 'u' : 'U'], [Rn, 'R'], [Bn, 'B']].forEach(function (p) {
      s += '<circle cx="' + p[0].x + '" cy="' + p[0].y + '" r="24" fill="var(--node)" stroke="currentColor" stroke-width="1.4"/>';
      s += '<text class="nodel" x="' + p[0].x + '" y="' + p[0].y + '">' + p[1] + '</text>';
    });
    svg.innerHTML = s;
    tnCells = [];
    var P = 8 + 2 * K;
    for (var i = 0; i < P; i++) {
      tnCells.push({ r: svg.querySelector('rect[data-p="' + i + '"]'), t: svg.querySelector('text[data-v="' + i + '"]') });
    }
    $('tn-sub').innerHTML = K === 1
      ? 'A single row <i class="m">u</i> reads input 1 for both outputs. Multiply the three tensors and you get the model tensor above.'
      : 'Each output now has its own row of <i class="m">U</i>. Multiply the three tensors and you get the model tensor above.';
  }
  function tnFill(th) {
    for (var i = 0; i < tnCells.length; i++) {
      var c = tnCells[i], v = th[i], m = Math.min(1, Math.abs(v) / 1.3);
      c.t.textContent = f3(v);
      c.r.setAttribute('style', 'fill:color-mix(in srgb,' + (v < 0 ? 'var(--neg)' : 'var(--pos)') + ' ' + (m * 62).toFixed(1) + '%,var(--panel))');
    }
  }

  // ------------------------------------------------------------ charts
  function niceStep(span, n) {
    var raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  function ticks(a, b, n) {
    var st = niceStep(b - a, n), out = [], v = Math.ceil(a / st) * st;
    for (; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-6 ? 0 : v);
    return out;
  }
  function interp(ts, vs, t) {
    var a = 0, b = ts.length - 1;
    if (t >= ts[b]) return vs[b];
    if (t <= ts[0]) return vs[0];
    while (b - a > 1) { var m = (a + b) >> 1; if (ts[m] <= t) a = m; else b = m; }
    return vs[a] + (vs[b] - vs[a]) * (t - ts[a]) / (ts[b] - ts[a]);
  }
  var charts = {};
  function drawChart(id, cfg) {
    var el = $(id), W = Math.max(280, el.clientWidth), H = 238, m = { l: 46, r: cfg.labels ? 62 : 14, t: 8, b: 30 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b, xMax = R.xMax;
    var X = function (t) { return m.l + pw * t / xMax; };
    var tr = cfg.log ? Math.log10 : function (v) { return v; };
    var y0 = tr(cfg.yMin), y1 = tr(cfg.yMax);
    var Y = function (v) { return m.t + ph * (1 - (tr(Math.max(v, cfg.log ? 1e-300 : -Infinity)) - y0) / (y1 - y0)); };
    var clip = id + '-clip', s = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + cfg.aria + '">';
    s += '<defs><clipPath id="' + clip + '"><rect x="' + m.l + '" y="' + m.t + '" width="' + pw + '" height="' + ph + '"/></clipPath></defs>';
    var yt = cfg.log ? (function () { var o = [], a = Math.ceil(y0), b = Math.floor(y1), st = Math.max(1, Math.ceil((b - a) / 5)); for (var e = a; e <= b; e += st) o.push(Math.pow(10, e)); return o; })() : ticks(cfg.yMin, cfg.yMax, 5);
    yt.forEach(function (v) {
      var y = Y(v);
      s += '<line x1="' + m.l + '" x2="' + (m.l + pw) + '" y1="' + y + '" y2="' + y + '" stroke="var(--rule)" stroke-width="' + (v === 0 ? 1.4 : 1) + '"/>';
      s += '<text x="' + (m.l - 6) + '" y="' + (y + 4) + '" text-anchor="end">' + cfg.yFmt(v) + '</text>';
    });
    ticks(0, xMax, Math.max(3, Math.floor(pw / 70))).forEach(function (v) {
      s += '<text x="' + X(v) + '" y="' + (m.t + ph + 16) + '" text-anchor="middle">' + v + '</text>';
    });
    s += '<text x="' + (m.l + pw) + '" y="' + (m.t + ph + 28) + '" text-anchor="end">training time t</text>';
    s += '<g clip-path="url(#' + clip + ')">';
    (cfg.hlines || []).forEach(function (h) {
      var y = Y(h.y);
      s += '<line ' + (h.entry != null ? 'data-entry="' + h.entry + '" ' : '') + 'x1="' + m.l + '" x2="' + (m.l + pw) + '" y1="' + y + '" y2="' + y + '" stroke="' + h.c + '" stroke-width="1.2" stroke-dasharray="1.5 3"/>';
    });
    cfg.series.forEach(function (q) {
      var pts = '';
      for (var i = 0; i < q.t.length; i++) pts += X(q.t[i]).toFixed(1) + ',' + Y(q.v[i]).toFixed(1) + ' ';
      s += '<polyline ' + (q.entry != null ? 'data-entry="' + q.entry + '" ' : '') + 'points="' + pts + '" fill="none" stroke="' + q.c + '" stroke-width="' + (q.w || 2) + '" stroke-linejoin="round" ' +
        (q.dash ? 'stroke-dasharray="' + q.dash + '" ' : '') + (q.op ? 'opacity="' + q.op + '" ' : '') + '/>';
    });
    s += '</g>';
    // Direct labels at the right edge, spread so they do not collide.
    if (cfg.labels) {
      var labs = cfg.labels.map(function (l) { return { y: Y(l.v), t: l.t, entry: l.entry }; }).sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < labs.length; i++) if (labs[i].y - labs[i - 1].y < 13) labs[i].y = labs[i - 1].y + 13;
      labs.forEach(function (l) { s += '<text class="dl" ' + (l.entry != null ? 'data-entry="' + l.entry + '" ' : '') + 'x="' + (m.l + pw + 5) + '" y="' + (l.y + 4) + '">' + l.t + '</text>'; });
    }
    s += '<line class="cur" x1="0" x2="0" y1="' + m.t + '" y2="' + (m.t + ph) + '" stroke="var(--ink)" stroke-width="1"/>';
    var dots = cfg.series.filter(function (q) { return q.dot; });
    dots.forEach(function (q, i) { s += '<circle class="dot" data-d="' + i + '" ' + (q.entry != null ? 'data-entry="' + q.entry + '" ' : '') + 'r="4" fill="' + q.c + '" stroke="var(--panel)" stroke-width="2"/>'; });
    s += '</svg><div class="tip"></div>';
    el.innerHTML = s;
    charts[id] = { el: el, X: X, Y: Y, dots: dots, cur: el.querySelector('.cur'), dotEls: el.querySelectorAll('.dot'), m: m, pw: pw, tip: el.querySelector('.tip'), rows: cfg.tip, ph: ph, W: W, clamp: [m.t, m.t + ph] };
    if (!el.dataset.wired) {
      el.dataset.wired = 1;
      var scrubbing = false;
      var toT = function (e) { var c = charts[id], b = el.getBoundingClientRect(); return Math.max(0, Math.min(R.xMax, (e.clientX - b.left - c.m.l) / c.pw * R.xMax)); };
      el.addEventListener('pointerdown', function (e) { el.setPointerCapture(e.pointerId); scrubbing = true; pause(); setT(toT(e)); });
      el.addEventListener('pointerup', function () { scrubbing = false; });
      el.addEventListener('pointermove', function (e) {
        var t = toT(e), c = charts[id]; if (scrubbing) setT(t);
        c.tip.innerHTML = c.rows(t); c.tip.style.visibility = 'visible';
        var b = el.getBoundingClientRect(), x = e.clientX - b.left, tw = c.tip.offsetWidth;
        c.tip.style.left = (x + 12 + tw > c.W ? x - tw - 12 : x + 12) + 'px'; c.tip.style.top = (c.m.t + 4) + 'px';
      });
      el.addEventListener('pointerleave', function () { charts[id].tip.style.visibility = 'hidden'; });
    }
    applyHl();
  }
  function moveCursors(t) {
    Object.keys(charts).forEach(function (id) {
      var c = charts[id], x = c.X(t);
      c.cur.setAttribute('x1', x); c.cur.setAttribute('x2', x);
      c.dots.forEach(function (q, i) {
        var y = c.Y(interp(q.t, q.v, t)), d = c.dotEls[i];
        d.setAttribute('cx', x); d.setAttribute('cy', y);
        d.style.visibility = (y < c.clamp[0] || y > c.clamp[1]) ? 'hidden' : 'visible';
      });
    });
  }
  function setHl(n) { S.hl = n; applyHl(); }
  function applyHl() {
    document.querySelectorAll('.chart [data-entry]').forEach(function (e) {
      e.classList.toggle('dim', S.hl != null && +e.dataset.entry !== S.hl);
    });
  }
  function legend(id, items) {
    $(id).innerHTML = items.map(function (it) {
      return '<span' + (it.entry != null ? ' data-entry-l="' + it.entry + '"' : '') + '><i style="--c:' + it.c + '"' + (it.dash ? ' class="dash"' : '') + '></i>' + it.t + '</span>';
    }).join('');
  }

  // ------------------------------------------------------------ one run into plot series
  function sample(run, xMax) {
    var n = run.ts.length, stride = Math.max(1, Math.ceil(n / 1400)), t = [], th = [], L = [];
    for (var i = 0; i < n; i += stride) { if (run.ts[i] > xMax) break; t.push(run.ts[i]); th.push(run.ys[i]); L.push(run.Ls[i]); }
    // Hold the final state to the right edge so a run that finished early stays visible.
    var last = n - 1;
    if (run.ts[last] <= xMax && t[t.length - 1] !== run.ts[last]) { t.push(run.ts[last]); th.push(run.ys[last]); L.push(run.Ls[last]); }
    if (t[t.length - 1] < xMax) {
      var end = run.ts[last] <= xMax ? last : -1;
      var yy = end >= 0 ? run.ys[last] : C.thetaAt(run, xMax), ll = end >= 0 ? run.Ls[last] : C.lossAt(run, xMax);
      t.push(xMax); th.push(yy); L.push(ll);
    }
    var E = []; for (var k = 0; k < 8; k++) E.push(new Array(t.length));
    for (i = 0; i < t.length; i++) { var T = C.tensor(th[i], run.K); for (k = 0; k < 8; k++) E[k][i] = T[k]; }
    return { t: t, th: th, L: L, E: E };
  }
  function column(smp, p) { return smp.th.map(function (y) { return y[p]; }); }

  // ------------------------------------------------------------ recompute and redraw
  function theta0(K, aligned) { return aligned ? C.initAligned(S.delta, K) : C.initRandom(S.delta, K, S.seed); }
  function recompute() {
    var canAlign = C.alignedCanReach(S.T);
    var aligned = canAlign && S.init === 'aligned';
    R.aligned = aligned; R.canAlign = canAlign;
    R.main = C.simulate(S.T, S.K, theta0(S.K, aligned));
    R.other = C.simulate(S.T, 3 - S.K, theta0(3 - S.K, aligned));
    R.sm = C.summary(R.main); R.so = C.summary(R.other);
    // The other network can sit on a saddle for thousands of time units, so it may not stretch the axis
    // past 8 times this network's own run. The legend says when its curve is cut off.
    R.xMax = Math.max(R.main.tSettle, S.overlay ? Math.min(R.other.tSettle, 8 * R.main.tSettle) : 0, 1) * 1.1;
    R.tMain = sample(R.main, R.xMax); R.tOther = sample(R.other, R.xMax);
    renderControls(); renderStats(); renderCharts();
    S.t = Math.min(S.t, R.xMax);
    frame();
  }
  function significant(smp, Ts) {
    var mx = 0, out = [], n, i;
    for (n = 0; n < 8; n++) mx = Math.max(mx, Math.abs(Ts[n]));
    for (n = 0; n < 8; n++) {
      var pk = 0; for (i = 0; i < smp.t.length; i++) pk = Math.max(pk, Math.abs(smp.E[n][i]));
      if (Math.abs(Ts[n]) > 0.01 * mx || pk > 0.05 * Math.max(mx, 1e-6)) out.push(n);
    }
    return out;
  }
  function renderCharts() {
    var a = R.tMain, b = R.tOther, Ts = S.T, sig = significant(a, Ts), list = C.targetEntries(Ts);
    var otherLab = netName(3 - S.K) + (R.other.tSettle > R.xMax ? ', settles past the chart at t = ' + fT(R.other.tSettle) : ''), thisLab = netName(S.K);
    var tRows = function (t) { return 't = ' + fT(t); };

    // A: entries of T.
    var ser = [], lo = 0, hi = 0.1, hl = [], labs = [];
    for (var n = 0; n < 8; n++) {
      var on = sig.indexOf(n) >= 0;
      a.E[n].forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      lo = Math.min(lo, Ts[n]); hi = Math.max(hi, Ts[n]);
      if (S.overlay && on) {
        ser.push({ entry: n, t: b.t, v: b.E[n], c: hue(n), dash: '4 3', w: 1.4, op: 0.75 });
        b.E[n].forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      }
      ser.push({ entry: n, t: a.t, v: a.E[n], c: on ? hue(n) : 'var(--rule)', w: on ? 2 : 1, dot: on });
      if (Math.abs(Ts[n]) > 1e-9) hl.push({ y: Ts[n], c: hue(n), entry: n });
      if (on && sig.length <= 4) labs.push({ v: a.E[n][a.E[n].length - 1], t: name(n), entry: n });
    }
    var pad = 0.08 * (hi - lo);
    drawChart('ch-a', { aria: 'Model tensor entries against training time', series: ser, hlines: hl, labels: labs.length ? labs : null,
      yMin: lo < -1e-9 ? lo - pad : 0, yMax: hi + pad, yFmt: function (v) { return +v.toFixed(2); },
      tip: function (t) { var th = C.thetaAt(R.main, t), T = C.tensor(th, S.K); return tRows(t) + sig.map(function (n) { return '<br>T' + name(n) + ' ' + f3(T[n]) + ' (target ' + f2(Ts[n]) + ')'; }).join(''); } });
    legend('lg-a', sig.map(function (n) { return { c: hue(n), t: 'T' + name(n) }; }).concat(S.overlay ? [{ c: 'var(--ink3)', dash: 1, t: otherLab }] : []));

    // B: relative distance from target, in percent.
    var pct = function (arr, Tn) { return arr.map(function (v) { return 100 * (v - Tn) / Math.abs(Tn); }); };
    var O = 100 * Math.max(R.sm.overshoot, S.overlay ? R.so.overshoot : 0), serB = [], labB = [];
    list.forEach(function (n) {
      if (S.overlay) serB.push({ entry: n, t: b.t, v: pct(b.E[n], Ts[n]), c: hue(n), dash: '4 3', w: 1.4, op: 0.75 });
      serB.push({ entry: n, t: a.t, v: pct(a.E[n], Ts[n]), c: hue(n), dot: true });
      if (list.length <= 4 && R.sm.reach) labB.push({ v: 0, t: name(n), entry: n });
    });
    var bLo = R.sm.reach ? -Math.max(3 * O, 2) : -105, bHi = R.sm.reach ? Math.max(1.6 * O, 0.5) : 105;
    if (!R.sm.reach) list.forEach(function (n) { a.E[n].forEach(function (v) { bHi = Math.max(bHi, Math.min(300, 100 * (v - Ts[n]) / Math.abs(Ts[n]) * 1.1)); }); });
    drawChart('ch-b', { aria: 'Percent distance of each target entry from its target against training time', series: serB, labels: null,
      hlines: [{ y: 0, c: 'var(--ink)' }], yMin: bLo, yMax: bHi, yFmt: function (v) { return +v.toFixed(2) + '%'; },
      tip: function (t) { var T = C.tensor(C.thetaAt(R.main, t), S.K); return tRows(t) + list.map(function (n) { var d = 100 * (T[n] - Ts[n]) / Math.abs(Ts[n]); return '<br>T' + name(n) + ' ' + (d >= 0 ? '+' : '') + d.toFixed(2) + '%'; }).join(''); } });
    legend('lg-b', list.map(function (n) { return { c: hue(n), t: 'T' + name(n) }; }).concat(S.overlay ? [{ c: 'var(--ink3)', dash: 1, t: otherLab }] : []));

    // C: factor entries.
    var P = 8 + 2 * S.K, r0 = 2 * S.K, b0 = r0 + 4, serC = [], labC = [], loC = 0, hiC = 0.1, mxF = 0;
    var cols = []; for (var p = 0; p < P; p++) { var col = column(a, p); cols.push(col); col.forEach(function (v) { mxF = Math.max(mxF, Math.abs(v)); loC = Math.min(loC, v); hiC = Math.max(hiC, v); }); }
    var hueOut = ['var(--s1)', 'var(--s2)'], peak = function (c) { return c.reduce(function (m, v) { return Math.max(m, Math.abs(v)); }, 0); };
    var names = [], ends = {};
    for (p = 0; p < P; p++) {
      var nm, c, dash = null;
      if (p < r0) { nm = (S.K === 1 ? 'u' : 'U' + ((p >> 1) + 1)) + ((p & 1) + 1); c = 'var(--ink)'; dash = (S.K === 2 && p >= 2) ? '2 3' : null; }
      else if (p < b0) { var q = p - r0; nm = 'R' + ((q >> 1) + 1) + ((q & 1) + 1); c = hueOut[q >> 1]; }
      else { q = p - b0; nm = 'B' + ((q >> 1) + 1) + ((q & 1) + 1); c = hueOut[q >> 1]; dash = '5 4'; }
      var big = peak(cols[p]) > 0.08 * mxF;
      serC.push({ t: a.t, v: cols[p], c: big ? c : 'var(--rule)', dash: big ? dash : null, w: big ? 2 : 1, dot: big });
      if (big) names.push({ v: cols[p][cols[p].length - 1], t: nm });
    }
    // Merge labels of curves that end at the same value, such as R11 and B11 on a balanced run.
    var merged = [];
    names.sort(function (x, y) { return x.v - y.v; }).forEach(function (l) {
      var m = merged[merged.length - 1];
      if (m && Math.abs(m.v - l.v) < 0.002 * mxF && m.t.indexOf('=') < 0) m.t += '=' + l.t; else merged.push({ v: l.v, t: l.t });
    });
    if (merged.length <= 6) labC = merged;
    var padC = 0.08 * (hiC - loC);
    drawChart('ch-c', { aria: 'Entries of u, R and B against training time', series: serC, labels: labC.length ? labC : null,
      yMin: loC < -1e-9 ? loC - padC : 0, yMax: hiC + padC, yFmt: function (v) { return +v.toFixed(2); },
      tip: function (t) { var th = C.thetaAt(R.main, t), o = tRows(t); names.forEach(function (l) { }); serC.forEach(function (q, i) { if (q.dot) { var nmz = i < r0 ? (S.K === 1 ? 'u' + (i + 1) : 'U' + ((i >> 1) + 1) + ((i & 1) + 1)) : i < b0 ? 'R' + (((i - r0) >> 1) + 1) + (((i - r0) & 1) + 1) : 'B' + (((i - b0) >> 1) + 1) + (((i - b0) & 1) + 1); o += '<br>' + nmz + ' ' + f3(th[i]); } }); return o; } });
    legend('lg-c', [{ c: 'var(--ink)', t: S.K === 1 ? 'u' : 'U' }, { c: 'var(--s1)', t: 'output 1 rows' }, { c: 'var(--s2)', t: 'output 2 rows' }]);

    // D: loss.
    var norm = Math.max(C.normSq(Ts), 1e-4), f = R.main.floor, fo = R.other.floor;
    var yMinL = 1e-6 * norm; if (f > 1e-9 * norm) yMinL = f * 0.6; if (S.overlay && fo > 1e-9 * norm) yMinL = Math.min(yMinL, fo * 0.6); if (S.overlay && fo <= 1e-9 * norm) yMinL = 1e-6 * norm;
    var yMaxL = Math.max(R.main.Ls[0], R.other.Ls[0], yMinL * 10) * 1.5;
    var serD = []; if (S.overlay) serD.push({ t: b.t, v: b.L, c: 'var(--ink2)', dash: '4 3', w: 1.4, op: 0.8 });
    serD.push({ t: a.t, v: a.L, c: 'var(--ink)', dot: true });
    var hlD = f > 1e-9 * norm ? [{ y: f, c: 'var(--warn)' }] : [];
    drawChart('ch-d', { aria: 'Loss against training time on a log scale', log: true, series: serD, hlines: hlD, labels: null,
      yMin: yMinL, yMax: yMaxL, yFmt: function (v) { return v >= 0.001 ? '' + +v.toPrecision(1) : '1e' + Math.round(Math.log10(v)); },
      tip: function (t) { return tRows(t) + '<br>' + thisLab + ' ' + fL(C.lossAt(R.main, t)) + (S.overlay ? '<br>' + otherLab + ' ' + fL(C.lossAt(R.other, t)) : ''); } });
    legend('lg-d', [{ c: 'var(--ink)', t: thisLab }].concat(S.overlay ? [{ c: 'var(--ink2)', dash: 1, t: otherLab }] : []).concat(hlD.length ? [{ c: 'var(--warn)', dash: 1, t: 'best possible for ' + thisLab }] : []));

    // Stripes tie each cell to its line.
    for (n = 0; n < 8; n++) {
      var col2 = sig.indexOf(n) >= 0 ? hue(n) : 'transparent';
      targetCells[n].el.style.setProperty('--stripe', col2); modelCells[n].el.style.setProperty('--stripe', col2);
    }
    R.sig = sig;
  }
  function renderStats() {
    var m = R.sm, o = R.so, otherLab = netName(3 - S.K);
    var halfLabel = m.list.length > 1 ? 'When the slowest target entry reaches half its value' : 'When the target entry reaches half its value';
    var s = '';
    s += '<div class="stat"><b>' + (m.reach ? (100 * m.overshoot).toFixed(2) + '%' : 'n/a') + '</b><span>' + (m.reach
      ? 'Largest overshoot' + (m.overshootEntry != null ? ', on T' + name(m.overshootEntry) : '') + '. ' + (o.reach ? (100 * o.overshoot).toFixed(2) + '% with ' + otherLab + '.' : '')
      : 'Overshoot is undefined because this network cannot reach the target.') + '</span></div>';
    s += '<div class="stat"><b>' + (m.lastHalf != null ? 't = ' + fT(m.lastHalf) : 'never') + '</b><span>' + halfLabel + '. ' + (o.lastHalf != null ? 't = ' + fT(o.lastHalf) : 'Never') + ' with ' + otherLab + '.</span></div>';
    s += '<div class="stat"><b>' + fL(R.main.floor < 1e-12 ? 0 : R.main.floor) + '</b><span>Smallest loss this network can reach. ' + fL(R.other.floor < 1e-12 ? 0 : R.other.floor) + ' with ' + otherLab + '.</span></div>';
    s += '<div class="stat"><b id="liveL"></b><span>Loss at the current time.</span></div>';
    $('stats').innerHTML = s;
  }
  function renderControls() {
    document.querySelectorAll('#presets button').forEach(function (b) { b.classList.toggle('on', b.dataset.id === S.preset); });
    document.querySelectorAll('#net button').forEach(function (b) { b.classList.toggle('on', +b.dataset.k === S.K); });
    document.querySelectorAll('#init button').forEach(function (b) {
      b.classList.toggle('on', b.dataset.i === (R.aligned ? 'aligned' : 'random'));
      if (b.dataset.i === 'aligned') b.disabled = !R.canAlign;
    });
    $('reseed').disabled = R.aligned;
    $('deltaval').textContent = S.delta < 0.01 ? S.delta.toFixed(4) : S.delta.toFixed(3);
    var note = '';
    if (!R.canAlign) note = 'The aligned start keeps every off-pattern entry at zero and keeps T111 and T212 positive, so it cannot reach this target. This run uses a random start.';
    if (!R.main.done && R.sm.list.length) note += (note ? ' ' : '') + 'This run stopped before settling at the smallest possible loss, at t = ' + fT(R.main.tEnd) + '.';
    $('note').textContent = note;
    showTarget();
  }

  // ------------------------------------------------------------ what is happening now
  function phaseText(t) {
    var Ts = S.T, list = R.sm.list, el = $('phases'), sig;
    if (!R.sm.reach) {
      var why = S.K === 1
        ? 'Sharing one <i class="m">u</i> gives every output the same direction in input 1. This target needs more than one direction there, so the entries settle at the closest tensor the network can represent and the loss stops at the floor.'
        : 'Each output gets one direction in input 1, and at least one output of this target needs two. The entries settle at the closest tensor the network can represent and the loss stops at the floor.';
      setPhaseHtml('<p class="phases-note">' + why + '</p>'); return;
    }
    if (!R.aligned || list.length !== 2) {
      setPhaseHtml('<p class="phases-note">This run starts from small random numbers, so every entry of every tensor moves a little before the target pattern takes over. Entries whose target is zero can leave zero and come back.</p>' +
        '<p class="phases-note">With a random start an entry can also pass its target briefly because its row of <i class="m">u</i> or <i class="m">B</i> still points slightly off the right direction. On the rotated target this explains the small overshoot with one <i class="m">u</i> per output: projected onto the target direction, that overshoot is zero, while with shared <i class="m">u</i> about 1.7% remains.</p>'); return;
    }
    if (S.K === 2) {
      setPhaseHtml('<p class="phases-note">Each output has its own row of <i class="m">U</i>, so nothing couples the two entries. Each one rises on its own clock, depending only on its own target, and stops at the target with no overshoot. Compare when the weaker entry arrives with the dashed curve for the shared network.</p>'); return;
    }
    var strong = Ts[0] >= Ts[5] ? 0 : 5, weak = strong === 0 ? 5 : 0;
    var over = R.main.peak[strong] / Ts[strong] - 1 > 1e-6;
    var tCross = crossTime(strong), tPk = R.main.tPeak[strong];
    var ph = t < tCross ? 0 : t < tPk ? 1 : 2;
    var a = 'T' + name(strong), b = 'T' + name(weak), oS = strong === 0 ? 1 : 2;
    var items = [
      'Both entries grow from almost nothing. Both errors push <i class="m">u</i> up, so each entry speeds up the other. Most of this stretch looks flat because growth is exponentially slow from a small start.',
      a + ' has passed its target and is still climbing. <i class="m">u</i> keeps growing to serve ' + b + ', and because <i class="m">u</i> is shared, that growth raises ' + a + ' as well.',
      a + ' is coming back down. Since it passed its target its error has been negative, so row ' + oS + ' of <i class="m">R</i> and <i class="m">B</i> has been shrinking, and that now outpaces the growth of <i class="m">u</i>. ' + a + ' lands from above and ' + b + ' from below.'
    ];
    if (!over) { setPhaseHtml('<p class="phases-note">With equal targets the two entries rise together and neither one overshoots.</p>'); return; }
    var key = 'ph' + strong + ph;
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.innerHTML = '<ol class="phases">' + items.map(function (x, i) { return '<li' + (i === ph ? ' class="now"' : '') + '>' + x + '</li>'; }).join('') + '</ol>';
    }
  }
  function setPhaseHtml(h) { var el = $('phases'); if (el.dataset.key !== h) { el.dataset.key = h; el.innerHTML = h; } }
  function crossTime(n) {
    var ts = R.tMain.t, e = R.tMain.E[n], Tn = S.T[n];
    for (var i = 0; i < ts.length; i++) if (e[i] >= Tn) return ts[i];
    return Infinity;
  }

  // ------------------------------------------------------------ per-frame update
  function frame() {
    var t = S.t, th = C.thetaAt(R.main, t), T = C.tensor(th, S.K), vm = colorScale();
    for (var n = 0; n < 8; n++) {
      var c = modelCells[n];
      c.v.textContent = f3(T[n]);
      tone(c.el, T[n], vm);
      var above = R.sm.reach && Math.abs(S.T[n]) > 1e-9 && R.sm.list.indexOf(n) >= 0 && (T[n] - S.T[n]) * R.main.sign[n] > 1e-3 * Math.abs(S.T[n]);
      c.el.classList.toggle('above', above);
    }
    tnFill(th);
    moveCursors(t);
    $('clock').textContent = 't = ' + fT(t) + ' / ' + fT(R.xMax);
    $('scrub').value = Math.round(1000 * t / R.xMax);
    var ll = $('liveL'); if (ll) ll.textContent = fL(C.lossAt(R.main, t));
    var list = R.sm.list;
    $('pcts').innerHTML = (list.length <= 4 && R.sm.reach ? list.map(function (n) { return 'T' + name(n) + ' is at ' + (100 * T[n] / S.T[n]).toFixed(1) + '% of target'; }).join(', ') + '.' : '') +
      (document.querySelector('#cube-model .above') ? ' <span style="color:var(--warn)">▲ above target</span>' : '');
    phaseText(t);
  }
  function setT(t) { S.t = t; frame(); }

  // ------------------------------------------------------------ playback
  var last = null;
  function tick(now) {
    if (S.playing) {
      if (last != null) {
        var t = S.t + (now - last) / 1000 * S.speed * R.xMax / SECONDS;
        if (t >= R.xMax) { t = R.xMax; pause(); }
        S.t = t; frame();
      }
      last = now;
    } else last = null;
    requestAnimationFrame(tick);
  }
  function pause() { S.playing = false; $('play').textContent = 'Play'; }
  function play() { if (S.t >= R.xMax) S.t = 0; S.playing = true; $('play').textContent = 'Pause'; }
  function restart() { S.t = 0; if (!matchMedia('(prefers-reduced-motion: reduce)').matches) play(); else { pause(); frame(); } }

  var pending = false;
  function schedule() {
    if (pending) return; pending = true;
    requestAnimationFrame(function () { pending = false; recompute(); });
  }

  // ------------------------------------------------------------ wiring
  function deltaFromSlider(v) { return Math.pow(10, -3 + 2.5 * v / 100); } // 0.001 to 0.316
  function sliderFromDelta(d) { return 100 * (Math.log10(d) + 3) / 2.5; }
  function wire() {
    $('presets').innerHTML = PRESETS.map(function (p) { return '<button type="button" data-id="' + p.id + '">' + p.label + '</button>'; }).join('');
    $('presets').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      var p = PRESETS.filter(function (x) { return x.id === b.dataset.id; })[0];
      S.T = Float64Array.from(p.T); S.preset = p.id; recompute(); restart();
    });
    $('net').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      S.K = +b.dataset.k; buildTN(); recompute(); restart();
    });
    $('init').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b || b.disabled) return;
      S.init = b.dataset.i; recompute(); restart();
    });
    $('reseed').addEventListener('click', function () { S.seed += 1; recompute(); restart(); });
    $('overlay').addEventListener('change', function (e) { S.overlay = e.target.checked; recompute(); });
    $('delta').value = sliderFromDelta(S.delta);
    $('delta').addEventListener('input', function (e) { S.delta = deltaFromSlider(+e.target.value); schedule(); });
    $('delta').addEventListener('change', restart);
    $('play').addEventListener('click', function () { if (S.playing) pause(); else play(); });
    $('scrub').addEventListener('input', function (e) { pause(); setT(+e.target.value / 1000 * R.xMax); });
    $('speed').addEventListener('change', function (e) { S.speed = +e.target.value; });
    var rw; new ResizeObserver(function () { clearTimeout(rw); rw = setTimeout(function () { if (R.main) { renderCharts(); frame(); } }, 80); }).observe(document.querySelector('.charts'));
    document.fonts && document.fonts.ready && document.fonts.ready.then(function () { if (R.main) { renderCharts(); frame(); } });
  }

  wire(); buildCubes(); buildTN();
  recompute();
  // At rest a reader who has motion turned off sees the overshoot peak rather than an empty start.
  if (!S.playing) { S.t = Math.min(R.main.tPeak[0] || 0, R.xMax); $('play').textContent = 'Play'; frame(); }
  requestAnimationFrame(tick);
})();
