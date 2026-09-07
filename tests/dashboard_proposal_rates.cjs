const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

for (const file of ['experiments/live_trajectory_dashboard.html','experiments/ontology_dashboard.html']) {
  const source=fs.readFileSync(file,'utf8');
  const context={normalizedImprovementKey:'campaign_normalized_improvement_percent',color:()=>'',replicateDash:()=>[],conditionColor:()=>'',interventionStarts:p=>p.transition_opportunities};
  vm.createContext(context);
  for(const [start,end] of [
    ['function html(','function stableIndex('],
    ['function proposalBounds(','function fmt('],
    ['function median(','function conditionColor('],
    ['function axisOptions(','function xAxisOptions('],
    ['function normalizationGroupKey(','function isInterventionPoint('],
    ['function pointAllowed(','function proposalObjectiveTooltip('],
  ]) vm.runInContext(source.slice(source.indexOf(start),source.indexOf(end)),context);
  const run=(id,outcomes)=>({run_id:id,label:id,condition:'C1',points:[
    {proposal:0,is_seed:true,valid:true,retained:true,valid_rate:null,retained_rate:null},
    ...outcomes.map(([valid,retained],index)=>({proposal:index+1,valid,retained,valid_rate:Number(valid),retained_rate:Number(retained)})),
  ]});
  const a=run('a',[[false,false],[true,false],[true,true]]);
  const b=run('b',[[true,true],[true,true],[false,false],[true,false]]);
  const original=JSON.stringify([a,b]);
  const state={proposalStart:'0',proposalEnd:'',filter:'all',proposalType:'all',showSeed:true,seriesMode:'conditionMean',connect:true,interventionWindow:'1',nanogptJumpMode:'all'};
  const payload={objective_metric:'accuracy',objective_direction:'maximize',runs:[a,b],transition_opportunities:[1,2],axis_catalog:[{key:'valid_rate',label:'Valid rate'},{key:'retained_rate',label:'Retained rate'}]};
  const values=series=>Array.from(series.data,p=>[p.x,p.y]);
  for(const key of ['valid_rate','retained_rate']) {
    const individual=context.dataset(a,a.points,'proposal',key,state,payload);
    assert.equal(individual.data.length,3,'seed must not add a fabricated successful proposal');
    assert.ok(individual.data.every(p=>p.y===0||p.y===1));
    const options=context.axisOptions(payload,key,{...state,trajectoryMode:'interventions'});
    assert.ok(options.includes(`value="${key}" selected`));
  }
  const aggregate=(key,s=state,p=payload)=>context.aggregateDataset('C1',p.runs,'proposal',key,s,p,'mean',new Map());
  assert.deepEqual(values(aggregate('valid_rate')),[[1,.5],[2,1],[3,.5],[4,1]]);
  assert.deepEqual(values(aggregate('retained_rate')),[[1,.5],[2,.5],[3,.5],[4,0]]);
  assert.equal(aggregate('valid_rate',{...state,requireCompleteConditionMeans:true}).data.at(-1).y,null);
  assert.deepEqual(values(aggregate('retained_rate',{...state,filter:'valid'})),[[1,1],[2,.5],[3,1],[4,0]]);
  // Bounds change the observed interval, not the meaning of a binary rate.
  assert.deepEqual(values(aggregate('retained_rate',{...state,proposalStart:'2',proposalEnd:'3'})),[[2,.5],[3,.5]]);
  const combined={...payload,combined_semantic:true,runs:[{...a,source_campaign_key:'A'},{...a,run_id:'a2',source_campaign_key:'A'},{...b,source_campaign_key:'B'}]};
  assert.equal(aggregate('valid_rate',state,combined).data[0].y,.5,'campaigns retain equal weighting');
  assert.equal(aggregate('valid_rate',state,combined).data.at(-1).point.member_count,1,'ended runs are not filled forward for rates');
  const windowState={...state,y:'valid_rate'};
  const windows=context.interventionWindows(payload,windowState,[a,b],new Map());
  assert.equal(windows.length,4);
  assert.ok(windows.every(w=>w.data[0].x===1),'rates must not include a synthetic zero baseline');
  assert.deepEqual(Array.from(windows[0].data,p=>p.y),[0,1]);
  const means=context.interventionAggregateDataset('C1',windows,windowState,payload,'mean');
  assert.deepEqual(values(means),[[1,.75],[2,.75]]);
  assert.equal(JSON.stringify([a,b]),original);
  const timingKey='incremental_active_seconds';
  const timedRun=(r,durations)=>({...r,points:r.points.map((p,i)=>({...p,[timingKey]:durations[i]}))});
  const timedA=timedRun(a,[null,60,120,45]),timedB=timedRun(b,[null,120,null,75,30]);
  const timedPayload={...payload,runs:[timedA,timedB],axis_catalog:[...payload.axis_catalog,{key:timingKey,label:'Wall-clock time per proposal (seconds)'}]};
  assert.deepEqual(values(aggregate(timingKey,state,timedPayload)),[[1,90],[2,120],[3,60],[4,30]]);
  assert.equal(aggregate(timingKey,{...state,requireCompleteConditionMeans:true},timedPayload).data.find(p=>p.x===2).y,null);
  const timeState={...state,y:timingKey,trajectoryMode:'interventions'};
  assert.ok(context.axisOptions(timedPayload,timingKey,timeState).includes(`value="${timingKey}" selected`));
  assert.ok(context.isWindowMetric(timingKey));
  const timeWindows=context.interventionWindows(timedPayload,timeState,timedPayload.runs,new Map());
  assert.equal(timeWindows.length,2,'windows with missing timing stay unavailable');
  assert.deepEqual(values(timeWindows[0]),[[1,60],[2,120]],'durations are not baseline-subtracted');
  assert.equal(context.metricInterventionCandidates(timedB,timeState,timedPayload).length,0);
  assert.deepEqual(values(context.interventionAggregateDataset('C1',timeWindows,timeState,timedPayload,'mean')),[[1,90],[2,82.5]]);
  console.log(`PASS: ${file}: binary outcomes, seed exclusion, rate means, filters, bounds, campaign weights and intervention windows.`);
  console.log('PASS: per-proposal wall time, available-duration means, direct window values and missing-timestamp windows.');
}
