const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const context={document:{createElement:()=>({}),head:{appendChild(){}},getElementById:()=>null},
  sectionHtml(){},drawCharts(){},redraw(){},destroySectionCharts(){},attachSection(){},campaignsFrom(){},
  stableIndex:()=>0,URLSearchParams,campaignDefs:[['live','Test','key']]};
vm.createContext(context);
vm.runInContext(fs.readFileSync('experiments/ontology_dashboard.js','utf8'),context);
const run={run_id:'r',points:[{proposal:0,candidate_id:'seed',is_seed:true,parent_ids:[]}]};
const payload={campaign:'test',task_display_name:'addition',runs:[run]};
let document=context.ontologyTemplate(payload);
document.rows[0].notes='first review';
let revision='1',requests=[];
context.fetch=async url=>{requests.push(url);return {ok:true,json:async()=>({review:document,revision})}};
const current=()=>vm.runInContext("ontologyReviews.get('live')",context);
const expire=()=>vm.runInContext("ontologyAutoLoads.get('live').checkedAt=0",context);
(async()=>{
  await context.loadSavedOntologyReviews('live',payload);
  assert.equal(current().rows[0].notes,'first review');
  document=structuredClone(document);document.rows[0].notes='new reviewed result';revision='2';expire();
  await context.loadSavedOntologyReviews('live',payload);
  assert.equal(current().rows[0].notes,'new reviewed result','Server review changes must reach an open tab');
  assert.ok(requests.at(-1).includes('revision=1'));
  // A review may arrive before the corresponding dashboard point. A new point
  // must invalidate the conditional request even when review revision is equal.
  const second={proposal:1,candidate_id:'child',parent_ids:['seed']};
  const enlarged={...payload,runs:[{...run,points:[...run.points,second]}]};
  document=context.ontologyTemplate(enlarged);revision='3';expire();
  await context.loadSavedOntologyReviews('live',payload);
  assert.equal(current().rows.length,1);
  expire();await context.loadSavedOntologyReviews('live',enlarged);
  assert.equal(current().rows.length,2);
  assert.ok(!requests.at(-1).includes('revision='),'New candidate inventory must fetch full review');
  vm.runInContext("ontologyLocalOverrides.add('live')",context);expire();
  const count=requests.length;
  await context.loadSavedOntologyReviews('live',enlarged);
  assert.equal(requests.length,count,'Manual imports/clear suppress server replacement until reload');
  console.log('PASS: live review refresh, new proposal inventory, and manual override preservation');
})().catch(error=>{console.error(error);process.exitCode=1});
