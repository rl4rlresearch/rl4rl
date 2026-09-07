const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);
const makeRun = values => ({run_id:'test',label:'Test',condition:'C1',points:values.map((best_objective,proposal)=>({proposal,best_objective,raw_objective:best_objective,is_seed:proposal===0,valid:true}))});
const normalizedKey = 'campaign_normalized_improvement_percent';

for (const file of ['experiments/live_trajectory_dashboard.html','experiments/ontology_dashboard.html']) {
  const source = fs.readFileSync(file,'utf8');
  for (const [,script] of source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(script);
  const context = {normalizedImprovementKey:normalizedKey,color:()=>'',replicateDash:()=>[],conditionColor:()=>'',interventionStarts:p=>p.transition_opportunities};
  vm.createContext(context);
  for (const [start,end] of [
    ['function html(', 'function stableIndex('],
    ['function proposalBounds(', 'function fmt('],
    ['function median(', 'function conditionColor('],
    ['function normalizationGroupKey(', 'function isInterventionPoint('],
    ['function pointAllowed(', 'function proposalObjectiveTooltip('],
  ]) {
    assert.ok(source.indexOf(start)>=0 && source.indexOf(end)>source.indexOf(start));
    vm.runInContext(source.slice(source.indexOf(start),source.indexOf(end)),context);
  }
  // The three-step jump ends at P5; only subsequent change belongs to the view.
  const run = makeRun([1,.999,.999,.994,.990,.987,.986,.985]);
  const unchanged = JSON.stringify(run);
  const payload = {objective_metric:'val_bpb',objective_direction:'minimize',runs:[run],transition_opportunities:[3,6]};
  const state = {nanogptJumpMode:'all',proposalStart:'0',proposalEnd:'',filter:'all',proposalType:'all',showSeed:true,connect:true,seriesMode:'conditionMean',interventionWindow:'1'};
  const stateFor = mode => ({...state,nanogptJumpMode:mode});
  const span = r => {const j=context.nanogptJump(r);return j?[j.start,j.end]:null;};
  assert.deepEqual(span(run),[3,5]);
  assert.deepEqual(span(makeRun([1,1,.988,.98])),[2,2]);
  assert.deepEqual(span(makeRun([1,1,.991,.987,.98])),[2,3]);
  assert.deepEqual(span(makeRun([1,.998,.995,.991,.987])),[2,4]);
  assert.equal(span(makeRun([1,.99,.989])),null);
  assert.equal(span(makeRun([.987,.985])),null);
  assert.equal(span({...run,points:run.points.filter(p=>p.proposal!==4)}),null);

  function plot(mode,key='objective_improvement',changes={},sourceRun=run,sourcePayload=payload) {
    const s={...stateFor(mode),...changes},ranges=context.normalizationRanges(sourcePayload,s);
    return context.dataset(sourceRun,sourceRun.points.filter(p=>context.pointAllowed(p,s)),'proposal',key,s,sourcePayload,ranges.get(context.normalizationGroupKey(sourceRun,sourcePayload))).data;
  }
  const after=plot('afterJump');
  assert.deepEqual(Array.from(after,p=>p.x),[5,6,7]);
  after.forEach((p,i)=>close(p.y,[0,-.001,-.002][i]));
  // Legacy exclusion selections migrate to the post-jump view.
  assert.deepEqual(Array.from(plot('excludeJump'),p=>[p.x,p.y]),Array.from(after,p=>[p.x,p.y]));
  plot('all').forEach((p,i)=>close(p.y,run.points[i].best_objective-1));
  for (const mode of ['afterJump','excludeJump']) {
    assert.deepEqual(Array.from(plot(mode,'best_objective'),p=>p.y),run.points.map(p=>p.best_objective));
    assert.deepEqual(Array.from(plot(mode,'raw_objective'),p=>p.y),run.points.map(p=>p.raw_objective));
    const normalized=plot(mode,normalizedKey);
    close(Math.max(...normalized.map(p=>Math.abs(p.y))),100);
  }
  plot('afterJump','objective_improvement',{proposalStart:'4'}).forEach((p,i)=>close(p.y,[0,-.001,-.002][i]));
  plot('afterJump','objective_improvement',{proposalStart:'6'}).forEach((p,i)=>close(p.y,[0,-.001][i]));
  assert.equal(plot('afterJump','objective_improvement',{proposalEnd:'4'}).length,0);
  const never=makeRun([1,.999,.999,.999,.999,.999,.998,.997]);
  assert.equal(plot('afterJump','objective_improvement',{},never).length,0);
  assert.equal(plot('excludeJump','objective_improvement',{},never).length,0);
  const other={...payload,objective_metric:'accuracy'};
  assert.strictEqual(context.progressRun(run,other,stateFor('excludeJump')),run);

  // Means use adjusted scores and a consistent denominator for no-crossing runs.
  const meanPayload={...payload,runs:[run,never]};
  function aggregate(mode,key,strict=false) {
    const s={...stateFor(mode),requireCompleteConditionMeans:strict};
    return context.aggregateDataset('C1',meanPayload.runs,'proposal',key,s,meanPayload,'mean',context.normalizationRanges(meanPayload,s)).data;
  }
  for(const strict of [false,true]) {
    close(aggregate('afterJump','objective_improvement',strict).at(-1).y,-.002);
    close(aggregate('excludeJump','objective_improvement',strict).at(-1).y,-.002);
    close(aggregate('afterJump',normalizedKey,strict).at(-1).y,-100);
    close(aggregate('afterJump','best_objective',strict).at(-1).y,(.985+.997)/2);
  }
  const windowState={...stateFor('all'),interventionWindow:'2'};
  let windows=context.interventionWindows(payload,windowState,[run],context.interventionNormalizationRanges(payload,windowState));
  assert.equal(windows.length,1);
  assert.ok(windows[0].data.at(-1).y<0,'include-jump mode preserves the original crossing window');
  // The pre-existing BPB filter uses recorded BPB, even when plotting adjusted gains.
  assert.equal(context.interventionCandidates(context.progressRun(run,payload,windowState),{...windowState,nanogptWindowPhase:'noCrossing'},payload).length,0);
  const afterState=stateFor('afterJump');
  windows=context.interventionWindows(payload,afterState,[run],context.interventionNormalizationRanges(payload,afterState));
  assert.deepEqual(Array.from(windows,w=>w.interventionStart),[6]);
  close(windows[0].data.at(-1).y,-100);
  assert.deepEqual(Array.from(context.interventionCandidates(run,stateFor('excludeJump'),payload),w=>w.interventionStart),[6]);
  assert.equal(JSON.stringify(run),unchanged,'raw campaign points must remain untouched');
  console.log(`PASS: ${file}: jump spans, adjusted curves, baselines, normalization, means, raw scores and intervention windows.`);

  // Optional local campaign audit; this ignored fixture is not required by CI.
  const fixture='outputs/nanogpt-jump-view-check.json';
  if(fs.existsSync(fixture)) {
    const real=JSON.parse(fs.readFileSync(fixture,'utf8'));
    const spans=real.runs.map(r=>({run:r.label,span:span(r)}));
    assert.equal(spans.filter(r=>r.span).length,11);
    assert.deepEqual(spans.find(r=>r.run==='B02-C1').span,[17,19]);
    assert.deepEqual(spans.find(r=>r.run==='B03-C2').span,[2,3]);
    assert.equal(spans.find(r=>r.run==='B03-C3').span,null);
    for(const r of real.runs) {
      const adjusted=context.progressRun(r,real,stateFor('afterJump')),jump=context.nanogptJump(r);
      const baseline=context.baselineObjective(adjusted,state);
      const change=context.valueAt(adjusted.points.at(-1),'objective_improvement',baseline,'minimize');
      if(jump)close(change,r.points.at(-1).best_objective-jump.after);
      else assert.equal(change,null);
    }
    console.log('PASS: 12 real nanoGPT trajectories; 11 post-jump baselines; native signed change.');
  }
}
