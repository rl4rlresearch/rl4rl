const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync('experiments/stagnation_dashboard.html', 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const script = scripts.at(-1)[1];
new vm.Script(script);

const context = { console };
vm.createContext(context);
vm.runInContext(
  script.slice(script.indexOf('function finite'), script.indexOf('const plateauPlugin')),
  context,
);

const minimize = context.summarizeRun(
  {
    points: [
      { proposal: 0, valid: true, raw_objective: 100, best_objective: 100 },
      { proposal: 1, valid: true, raw_objective: 102, best_objective: 100 },
      { proposal: 2, valid: false, raw_objective: 50, best_objective: 100 },
      { proposal: 3, valid: true, raw_objective: 80, best_objective: 80 },
      { proposal: 4, valid: true, raw_objective: 81, best_objective: 80 },
      { proposal: 5, valid: false, raw_objective: null, best_objective: 80 },
    ],
  },
  'minimize',
);
assert.deepEqual(JSON.parse(JSON.stringify(minimize.breakthroughs)), [
  { proposal: 3, previous: 100, current: 80 },
]);
assert.deepEqual(JSON.parse(JSON.stringify(minimize.plateaus)), [
  { start: 1, end: 2, length: 2 },
  { start: 4, end: 5, length: 2 },
]);
assert.equal(minimize.longest, 2);
assert.equal(minimize.current, 2);

const maximize = context.summarizeRun(
  { points: [
    { proposal: 0, valid: true, raw_objective: 0.5, best_objective: 0.5 },
    { proposal: 1, valid: true, raw_objective: 0.6, best_objective: 0.6 },
  ] },
  'maximize',
);
assert.equal(maximize.breakthroughs.length, 1);
assert.equal(maximize.breakthroughs[0].proposal, 1);
console.log('PASS: performance-only breakthrough and plateau calculation.');
