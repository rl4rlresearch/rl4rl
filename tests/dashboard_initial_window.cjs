const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('experiments/live_trajectory_dashboard.html', 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
for (const [, script] of scripts) new vm.Script(script);
const context = {
  finite: value => value == null || !Number.isFinite(Number(value)) ? null : Number(value),
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

const nanoPayload = {...payload, objective_metric:'val_bpb'};
const thresholdState = {...state, nanogptWindowPhase:'reached'};
const nanoRun = {points:run.points.map(p=>({...p,best_objective:p.proposal<10?0.99:0.988}))};
assert.deepEqual(Array.from(context.interventionCandidates(nanoRun,thresholdState,nanoPayload),w=>w.interventionStart),[20]);
// Equality qualifies before the intervention, but crossing during it does not.
nanoRun.points[9].best_objective=0.988;
assert.deepEqual(Array.from(context.interventionCandidates(nanoRun,thresholdState,nanoPayload),w=>w.interventionStart),[10,20]);
nanoRun.points[0].best_objective=0.987;
assert.equal(context.interventionCandidates(nanoRun,thresholdState,nanoPayload)[0].initial,true);
nanoRun.points[9].best_objective=null;
assert.deepEqual(Array.from(context.interventionCandidates(nanoRun,thresholdState,nanoPayload),w=>w.interventionStart),[1,20]);
assert.equal(context.interventionCandidates(nanoRun,{...thresholdState,nanogptWindowPhase:'all'},nanoPayload).length,3);
assert.equal(context.interventionCandidates(nanoRun,thresholdState,payload).length,3);
assert.ok(html.includes("nanogptWindowPhase:'all'"));
console.log('PASS: nanoGPT baseline BPB threshold, equality, no look-ahead, seed, missing metric, opt-out, other tasks.');

const crossingRun = {points:run.points.map(p=>({...p,best_objective:p.proposal<12?0.99:0.988}))};
const noCrossing={...state,nanogptWindowPhase:'noCrossing'};
// Initial window 1–8 is before the crossing; P10–17 crosses; P20–27 is after.
assert.deepEqual(Array.from(context.interventionCandidates(crossingRun,noCrossing,nanoPayload),w=>w.interventionStart),[1,20]);
// Overlapping initial windows that contain the crossing are excluded too.
assert.equal(context.interventionCandidates(crossingRun,{...noCrossing,interventionWindow:12},nanoPayload).length,0);
const edgeRun = {points:run.points.map(p=>({...p,best_objective:p.proposal<17?0.99:0.988}))};
assert.deepEqual(Array.from(context.interventionCandidates(edgeRun,noCrossing,nanoPayload),w=>w.interventionStart),[1,20]);
const beforeRun = {points:run.points.map(p=>({...p,best_objective:p.proposal<9?0.99:0.988}))};
assert.equal(context.interventionCandidates(beforeRun,noCrossing,nanoPayload).length,3);
const neverCrosses = {points:run.points.map(p=>({...p,best_objective:0.99}))};
assert.equal(context.interventionCandidates(neverCrosses,noCrossing,nanoPayload).length,3);
console.log('PASS: exclude crossing windows only, retain windows before/after, equality, endpoint and overlapping-window crossings.');

const rollingState={...state,overlappingControls:true};
for(const condition of ['C0','C2']){
 const controlRun={...run,condition};
 const candidates=context.interventionCandidates(controlRun,rollingState,payload);
 assert.equal(candidates.length,19);
 assert.deepEqual(Array.from(candidates,c=>c.interventionStart),Array.from({length:19},(_,i)=>i+1));
 assert.ok(candidates.every(c=>c.points.length===10&&c.before.proposal===c.interventionStart-1));
 assert.equal(context.interventionCandidates(controlRun,{...rollingState,includeFirstWindow:false},payload).length,19);
 assert.equal(context.interventionCandidates(controlRun,{...rollingState,overlappingControls:false},payload).length,3);
}
for(const condition of ['C1','C3','C4'])assert.equal(context.interventionCandidates({...run,condition},rollingState,payload).length,3);
assert.deepEqual(Array.from(context.interventionCandidates({...run,condition:'C0'},{...rollingState,proposalStart:3,proposalEnd:5},payload),c=>c.interventionStart),[3,4,5]);
const rollingNano={...crossingRun,condition:'C2'};
assert.deepEqual(Array.from(context.interventionCandidates(rollingNano,{...rollingState,nanogptWindowPhase:'reached'},nanoPayload),c=>c.interventionStart),[13,14,15,16,17,18,19]);
assert.deepEqual(Array.from(context.interventionCandidates(rollingNano,{...rollingState,nanogptWindowPhase:'noCrossing'},nanoPayload),c=>c.interventionStart),[1,2,13,14,15,16,17,18,19]);
const gapRun={condition:'C0',points:run.points.filter(p=>p.proposal!==5)};
assert.equal(context.interventionCandidates(gapRun,rollingState,payload)[0].interventionStart,7);
console.log('PASS: overlapping control windows, fixed length, no duplicates, unchanged treatments, gaps, bounds and off-cadence BPB filters.');
