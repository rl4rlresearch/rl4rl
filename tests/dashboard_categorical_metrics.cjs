const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

for (const file of ['experiments/live_trajectory_dashboard.html','experiments/ontology_dashboard.html']) {
  const source=fs.readFileSync(file,'utf8');
  for(const [,script] of source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g))new vm.Script(script);
  const context={normalizedImprovementKey:'campaign_normalized_improvement_percent',color:()=>'',replicateDash:()=>[],conditionColor:()=>'',interventionStarts:p=>p.transition_opportunities,states:{},allConditions:()=>['C1'],allFamilies:()=>['factorial'],defaultConditions:()=>['C1'],hasInterventions:()=>true};
  vm.createContext(context);
  for(const [start,end] of [
    ['function html(','function stableIndex('],
    ['function proposalBounds(','function fmt('],
    ['function median(','function conditionColor('],
    ['function defaultState(','function xAxisOptions('],
    ['function normalizationGroupKey(','function isInterventionPoint('],
    ['function pointAllowed(','function proposalObjectiveTooltip('],
  ])vm.runInContext(source.slice(source.indexOf(start),source.indexOf(end)),context);
  const marginals={component_edits:[0,2,0,1,2,0],new_component_states:[0,2,0,1,0,0],family_switches:[0,1,0,1,1,0],new_families:[0,1,0,1,0,0]};
  const keys=Object.keys(marginals).flatMap(k=>['implemented','retained'].flatMap(stage=>[stage+':'+k+'_marginal',stage+':'+k+'_cumulative']));
  const run={run_id:'r',label:'r',condition:'C1',points:Array.from({length:6},(_,proposal)=>({proposal,candidate_id:'p'+proposal,is_seed:proposal===0,valid:proposal!==4,retained:proposal!==4,best_objective:null,ontology_metrics:Object.fromEntries(Object.entries(marginals).flatMap(([key,values])=>['implemented','retained'].flatMap(stage=>{const contributions=values.map((value,index)=>stage==='retained'&&index===4?0:value);return [[stage+':'+key+'_marginal',contributions[proposal]],[stage+':'+key+'_cumulative',contributions.slice(0,proposal+1).reduce((a,b)=>a+b,0)]];})))}))};
  const payload={objective_metric:'val_bpb',objective_direction:'minimize',runs:[run],transition_opportunities:[3],axis_catalog:keys.map(key=>({key:'ontology:'+key,label:key}))};
  const state={proposalStart:'0',proposalEnd:'',filter:'all',proposalType:'all',showSeed:true,seriesMode:'conditionMean',connect:true,interventionWindow:'2',nanogptJumpMode:'all',trajectoryMode:'interventions'};
  const before=JSON.stringify(run);
  for(const key of keys){
    const y='ontology:'+key,s={...state,y};
    assert.ok(context.axisOptions(payload,y,s).includes(`value="${y}" selected`));
    assert.ok(context.axisOptions(payload,y,{...s,trajectoryMode:'full'}).includes(`value="${y}" selected`));
    context.states.test={...context.defaultState(payload),...s};
    assert.equal(context.stateFor('test',payload).y,y,'Selected metric survives rerenders');
    const full=context.dataset(run,run.points,'proposal',y,s,payload);
    assert.deepEqual(Array.from(full.data,p=>p.y),run.points.map(p=>p.ontology_metrics[key]));
    const windows=context.interventionWindows(payload,s,[run],new Map());
    assert.equal(windows.length,1,'Ontology windows work even without objective scores');
    const cumulative=key.endsWith('_cumulative');
    const expected=run.points.slice(3).map(p=>p.ontology_metrics[key]-(cumulative?run.points[2].ontology_metrics[key]:0));
    assert.deepEqual(Array.from(windows[0].data,p=>p.y),cumulative?[0,...expected]:expected);
    assert.equal(windows[0].data[0].x,cumulative?0:1);
    assert.equal(context.interventionAggregateDataset('C1',windows,s,payload,'mean').data.at(-1).y,expected.at(-1));
    assert.equal(context.interventionAggregateDataset('C1',windows,s,payload,'median').data.at(-1).y,expected.at(-1));
    // An ontology count must not be filtered by a performance-jump adjustment.
    assert.equal(context.interventionWindows(payload,{...s,nanogptJumpMode:'afterJump'},[run],new Map()).length,1);
  }
  context.states.legacy={...context.defaultState(payload),...state,y:'ontology:component_edits_cumulative'};
  assert.equal(context.stateFor('legacy',payload).y,'ontology:implemented:component_edits_cumulative','Old selections migrate to implemented');
  assert.equal(run.points[4].ontology_metrics['retained:component_edits_marginal'],0);
  assert.equal(run.points[4].ontology_metrics['retained:component_edits_cumulative'],3);
  const unknownRetention=structuredClone(run);
  unknownRetention.points[4].ontology_metrics['retained:component_edits_cumulative']=null;
  assert.equal(context.valueAt(unknownRetention.points[4],'ontology:retained:component_edits_cumulative'),null);
  assert.equal(context.interventionWindows(payload,{...state,y:'ontology:retained:component_edits_cumulative'},[unknownRetention],new Map()).length,0);
  const missing=structuredClone(run);missing.points[4].ontology_metrics=null;
  const y='ontology:implemented:component_edits_cumulative';
  assert.equal(context.valueAt(missing.points[4],y),null);
  const series=context.dataset(missing,missing.points,'proposal',y,state,payload);
  assert.equal(series.data[4].y,null);assert.equal(series.spanGaps,false);
  assert.equal(context.interventionWindows(payload,{...state,y},[missing],new Map()).length,0);
  assert.equal(context.metricInterventionCandidates(missing,{...state,y},payload).length,0,'Window legends exclude incomplete metric windows');
  const average=context.aggregateDataset('C1',[run,missing],'proposal',y,{...state,trajectoryMode:'full'},payload,'mean',new Map());
  assert.equal(average.data.find(p=>p.x===4).y,5,'Missing review is not averaged as zero');
  assert.equal(JSON.stringify(run),before,'Graphing never mutates published metric values');
  // C2/C3 comparison choices use the precomputed full-history references.
  for(const condition of ['C2','C3']){
    const portfolio=structuredClone(run);portfolio.condition=condition;
    const previous=[0,1,3,2,4,1],minimum=[0,0,1,0,2,0];
    for(const point of portfolio.points){
      point.ontology_comparisons={condition};
      for(const [mode,counts] of [['previous',previous],['minimum_parents',minimum]]){
        const changes=counts[point.proposal],family=Number(changes>0);
        point.ontology_comparisons[mode]={
          'implemented:component_edits_marginal':changes,
          'implemented:family_switches_marginal':family,
          'retained:component_edits_marginal':point.retained?changes:0,
          'retained:family_switches_marginal':point.retained?family:0,
        };
      }
    }
    const portfolioPayload={...payload,runs:[portfolio]},original=JSON.stringify(portfolio);
    for(const mode of ['parent','previous','minimum_parents'])for(const stage of ['implemented','retained'])for(const kind of ['component_edits','family_switches']){
      const key=stage+':'+kind+'_marginal',y='ontology:'+key,s={...state,y,ontologyComparison:mode};
      const expected=portfolio.points.map(p=>mode==='parent'?p.ontology_metrics[key]:p.ontology_comparisons[mode][key]);
      const plotted=context.dataset(portfolio,portfolio.points,'proposal',y,s,portfolioPayload);
      assert.deepEqual(Array.from(plotted.data,p=>p.y),expected);
      const window=context.interventionWindows(portfolioPayload,s,[portfolio],new Map())[0];
      assert.deepEqual(Array.from(window.data,p=>p.y),expected.slice(3));
      const mean=context.aggregateDataset(condition,[portfolio],'proposal',y,{...s,trajectoryMode:'full'},portfolioPayload,'mean',new Map());
      assert.deepEqual(Array.from(mean.data,p=>p.y),expected);
      assert.ok(context.ontologyComparisonControl(portfolioPayload,s).includes(`value="${mode}" selected`));
      // Filtering the visible proposals must not redefine the previous reference.
      const bounded=context.dataset(portfolio,portfolio.points.slice(3),'proposal',y,{...s,proposalStart:'3'},portfolioPayload);
      assert.deepEqual(Array.from(bounded.data,p=>p.y),expected.slice(3));
      context.states.comparison={...context.defaultState(portfolioPayload),...s};
      assert.equal(context.stateFor('comparison',portfolioPayload).ontologyComparison,mode);
    }
    for(const key of keys.filter(key=>!/(component_edits|family_switches)_marginal$/.test(key))){
      const y='ontology:'+key,s={...state,y,ontologyComparison:'minimum_parents'};
      const plotted=context.dataset(portfolio,portfolio.points,'proposal',y,s,portfolioPayload);
      assert.deepEqual(Array.from(plotted.data,p=>p.y),portfolio.points.map(p=>p.ontology_metrics[key]),'Totals and novelty keep their existing definitions');
      assert.equal(context.ontologyComparisonControl(portfolioPayload,s),'');
    }
    const absent=structuredClone(portfolio);
    absent.points[4].ontology_comparisons.minimum_parents['implemented:component_edits_marginal']=null;
    const absentState={...state,y:'ontology:implemented:component_edits_marginal',ontologyComparison:'minimum_parents'};
    assert.equal(context.metricInterventionCandidates(absent,absentState,portfolioPayload).length,0);
    assert.equal(context.interventionWindows(portfolioPayload,absentState,[absent],new Map()).length,0);
    assert.equal(context.interventionWindows(portfolioPayload,{...absentState,ontologyComparison:'parent'},[absent],new Map()).length,1);
    assert.equal(JSON.stringify(portfolio),original,'Comparison selection never edits fingerprints or base metrics');
    for(const condition of ['C0','C1']){
      const control=structuredClone(portfolio);control.condition=condition;
      for(const point of control.points)point.ontology_comparisons.condition=condition;
      const y='ontology:implemented:component_edits_marginal',s={...state,y,ontologyComparison:'minimum_parents'};
      assert.deepEqual(Array.from(context.dataset(control,control.points,'proposal',y,s,payload).data,p=>p.y),marginals.component_edits);
      assert.equal(context.ontologyComparisonControl({...payload,runs:[control]},s),'');
    }
  }
  console.log('PASS:',file,'C2/C3 parent, previous and portfolio minimum comparisons; both stages, windows, means, bounds and missing references');
  console.log('PASS:',file,'all 16 implemented/retained axes, window baselines, full-history novelty, means/medians, missing reviews and immutable data');
}
