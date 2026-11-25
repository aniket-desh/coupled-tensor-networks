// node notes/child-feature-enabling/explorer/check.js
// Compares the page's integrator with the Radau event times saved by planted.py, for all twenty arms.
const fs = require('fs');
const path = require('path');
const { integrate } = require('./planted.js');
const saved = JSON.parse(fs.readFileSync(path.join(__dirname,
  '../../../studies/rhm/child_feature_enabling/results/planted/planted.json')));
let worst = 0;
for (const arm of saved.arms) {
  const t0 = Date.now();
  const mine = integrate(arm.tied, arm.s, arm.delta, { keep: false }).events;
  const diffs = [['left', 'left_half'], ['right', 'right_half'], ['hard', 'hard_half']]
    .map(([a, b]) => Math.abs(mine[a] - arm.events[b]) / arm.events[b]);
  worst = Math.max(worst, ...diffs);
  console.log(`${arm.tied ? 'tied  ' : 'untied'} s=${arm.s} delta=${arm.delta}  T_h ${mine.hard.toFixed(4)} ` +
              `(saved ${arm.events.hard_half.toFixed(4)})  max rel diff ${Math.max(...diffs).toExponential(1)}  ${Date.now() - t0} ms`);
}
console.log('largest relative difference in any event time:', worst.toExponential(2));
if (!(worst < 1e-6)) { console.error('FAIL'); process.exit(1); }
console.log('PASS');
