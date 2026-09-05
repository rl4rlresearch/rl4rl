const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('experiments/live_trajectory_dashboard.html', 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
for (const [, script] of scripts) new vm.Script(script);
const context = {
  proposalBounds: s => ({start: Number(s.proposalStart || 0), end: s.proposalEnd == null ? null : Number(s.proposalEnd)}),
  interventionWindowSize: s => Number(s.interventionWindow),
  interventionStarts: p => p.transition_opportunities,
  mean: xs => xs.reduce((a,b) => a+b,0)/xs.length,
  median: xs => xs.sort((a,b)=>a-b)[Math.floor(xs.length/2)],
  conditionColor: () => '#fff', pointRadius: () => 2,
};
vm.createContext(context);
for (const [start, end] of [
  ['function interventionCandidates(', 'function interventionNormalizationRanges('],
  ['function equalCampaignAggregate(', 'function aggregateDataset('],
  ['function interventionAggregateDataset(', 'function proposalObjectiveTooltip('],
]) vm.runInContext(html.slice(html.indexOf(start), html.indexOf(end)), context);
const payload = {transition_opportunities: [10,20]};
const run = {points: Array.from({length:29},(_,proposal)=>({proposal}))};
const state = {interventionWindow:7};
assert.equal(context.interventionCandidates(run,state,payload).length,2);
state.includeFirstWindow = true;
let windows = context.interventionCandidates(run,state,payload);
assert.equal(windows.length,3);
assert.equal(windows[0].before.proposal,0);
const longer=context.interventionCandidates(run,{...state,interventionWindow:12},payload)[0];
assert.equal(longer.points.at(-1).proposal,13);
assert.ok(longer.points.some(p=>p.proposal===10));
assert.deepEqual(Array.from(windows[0].points,p=>p.proposal),[1,2,3,4,5,6,7,8]);
assert.equal(context.interventionCandidates({points:run.points.filter(p=>p.proposal!==5)},state,payload).length,2);
assert.equal(context.interventionCandidates({points:run.points.slice(1)},state,payload).length,2);
assert.equal(context.interventionCandidates(run,{...state,proposalStart:10},payload).length,2);
assert.equal(context.interventionCandidates(run,state,{transition_opportunities:[1]}).filter(w=>w.initial).length,0);
const groups = [
  {run:{run_id:'a'},data:[{x:0,y:0},{x:1,y:1},{x:2,y:2}]},
  {run:{run_id:'b'},data:[{x:0,y:0},{x:1,y:3}]},
];
const aggregate=context.interventionAggregateDataset('C0',groups,state,{},'mean');
assert.equal(aggregate.data[1].y,2);
assert.equal(aggregate.data[2].y,null);
assert.ok(html.includes('includeFirstWindow:false'));
assert.ok(html.includes('data-control="includeFirstWindow"'));
console.log('PASS: initial-window opt-in, complete interval, seed requirement, bounds, and stable aggregate cohort.');
