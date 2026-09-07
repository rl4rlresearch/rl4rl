const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('experiments/live_trajectory_dashboard.html','utf8');
const code = html.slice(html.indexOf('function equalCampaignAggregate('),html.indexOf('function interventionAggregateDataset('));
const context = {
 baselineObjective:()=>0,pointAllowed:()=>true,
 chartPoint:(p,x,y)=>p[y]==null?null:{x:p[x],y:p[y],point:p},
 mean:v=>v.length?v.reduce((a,b)=>a+b,0)/v.length:null,
 median:v=>{v=[...v].sort((a,b)=>a-b);return v.length?(v[Math.floor((v.length-1)/2)]+v[Math.floor(v.length/2)])/2:null;},
 conditionColor:()=>'',pointRadius:()=>2
};
vm.createContext(context);vm.runInContext(code,context);
const run=points=>({points:points.map(([proposal,best_objective])=>({proposal,best_objective}))});
const a=run([[164,10],[165,10],[166,10],[168,10],[169,10]]);
const b=run([[164,30],[166,30],[168,30]]);
const loose=context.aggregateDataset(
 'C2',[a,b],'proposal','best_objective',
 {connect:true,seriesMode:'conditionMean',requireCompleteConditionMeans:false},{},'mean',new Map()
);
assert.equal(loose.spanGaps,false);
assert.deepEqual(
 Array.from(loose.data,p=>[p.point.proposal,p.y]),
 [[164,20],[165,10],[166,20],[167,null],[168,20],[169,10]]
);

const strict=context.aggregateDataset(
 'C2',[a,b],'proposal','best_objective',
 {connect:true,seriesMode:'conditionMean',requireCompleteConditionMeans:true},{},'mean',new Map()
);
assert.equal(strict.spanGaps,true);
assert.deepEqual(
 Array.from(strict.data,p=>[p.point.proposal,p.y]),
 [[164,20],[165,null],[166,20],[167,null],[168,20],[169,null]]
);

b.points.splice(1,0,{proposal:165,best_objective:30});
const restored=context.aggregateDataset(
 'C2',[a,b],'proposal','best_objective',
 {connect:true,seriesMode:'conditionMean',requireCompleteConditionMeans:true},{},'mean',new Map()
);
assert.equal(restored.data.find(p=>p.point.proposal===165).y,20);

console.log('PASS: available-value condition means, optional strict completeness, and bridged strict gaps.');
