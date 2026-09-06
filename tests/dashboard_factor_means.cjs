const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('experiments/live_trajectory_dashboard.html','utf8');
const context={};
vm.createContext(context);
vm.runInContext(html.slice(html.indexOf('function factorSeries('),html.indexOf('function visibleRuns(')),context);
const state={seriesMode:'portfolioMean',conditions:['C0','C1','C2','C3','C4'],y:'best_objective',x:'proposal'};
const groups=context.seriesGroups(state);
assert.deepEqual(Array.from(groups,g=>Array.from(g.conditions)),[['C0','C1'],['C2','C3']]);
assert.deepEqual(Array.from(context.seriesGroups({...state,seriesMode:'interventionMean'}),g=>Array.from(g.conditions)),[['C0','C2'],['C1','C3']]);
assert.deepEqual(Array.from(context.seriesGroups({...state,conditions:['C1','C3','C4']}),g=>Array.from(g.conditions)),[['C1'],['C3']]);
let chart;
const runs=['C0','C1','C2','C3','C4'].map((condition,i)=>({condition,value:i+1}));
const summarize=(label,members,method)=>{assert.equal(method,'mean');return {label,value:members.reduce((sum,r)=>sum+r.value,0)/members.length};};
Object.assign(context,{
  Chart:function(_canvas,config){chart=config;this.destroy=()=>{};},states:{test:state},charts:new Map(),
  visibleRuns:()=>runs,interventionView:()=>false,normalizationRanges:()=>new Map(),
  interventionNormalizationRanges:()=>new Map(),normalizedImprovementKey:'normalized',
  document:{getElementById:()=>({})},chartOptions:()=>({}),
  aggregateDataset:(label,members,x,y,s,p,method)=>summarize(label,members,method),
  interventionWindows:()=>runs.map(run=>({run})),
  interventionAggregateDataset:(label,members,s,p,method)=>summarize(label,members.map(w=>w.run),method),
});
vm.runInContext(html.slice(html.indexOf('function drawCharts('),html.indexOf('function destroySectionCharts(')),context);
for(const intervention of [false,true]){
  context.interventionView=()=>intervention;
  state.seriesMode='portfolioMean';context.drawCharts('test',{});
  assert.deepEqual(Array.from(chart.data.datasets,d=>d.value),[1.5,3.5]);
  assert.notEqual(chart.data.datasets[0].borderColor,chart.data.datasets[1].borderColor);
  state.seriesMode='interventionMean';context.drawCharts('test',{});
  assert.deepEqual(Array.from(chart.data.datasets,d=>d.value),[2,3]);
}
console.log('PASS: factorial group membership, condition filters, C4 exclusion, distinct colors, and mean dispatch in both views.');
