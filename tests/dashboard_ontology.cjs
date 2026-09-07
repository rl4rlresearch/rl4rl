const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = {document:{createElement:()=>({}),head:{appendChild(){}}},
  sectionHtml(){},drawCharts(){},redraw(){},destroySectionCharts(){},attachSection(){},campaignsFrom(){},
  stableIndex:()=>0};
vm.createContext(context);
vm.runInContext(fs.readFileSync('experiments/ontology_dashboard.js','utf8'),context);
for (const [task,key] of [['four-digit addition','carry_representation'],['language-model pretraining','kv_memory'],['grayscale image classification','spatial_operator'],['keyword spotting','state_update'],['activity recognition','sensor_fusion']]) {
  const schema=context.ontologySchema({task_display_name:task});
  assert.ok(schema[key],task);assert.ok(schema.other);assert.ok(schema.output);assert.ok(schema.activation);
}
const run={run_id:'r',points:[
  {proposal:0,candidate_id:'seed',is_seed:true,valid:true,retained:true,parent_ids:[]},
  {proposal:1,candidate_id:'a',parent_ids:['seed'],valid:false,failure_kind:'nonqualification'},
  {proposal:2,candidate_id:'b',parent_ids:['a'],valid:true,retained:true},
  {proposal:3,candidate_id:'c',parent_ids:['seed'],valid:true,retained:true},
]};
context.run=run;
assert.ok(context.ontologyDiscovery('test',run,'implemented').every(p=>p.y===null),'Unreviewed is not zero');
assert.equal(context.ontologyStage(run.points[1],null).executable,true);
assert.equal(context.ontologyStage({valid:false,failure_kind:'unmatched_diff'},null).implemented,null);
const payload={campaign:'test',task_display_name:'addition',runs:[run]};
const doc=context.ontologyTemplate(payload);
doc.rows[0].family='table';doc.rows[0].fingerprint.embedding='table';
doc.rows[1].family='functional';doc.rows[1].fingerprint.embedding='function';doc.rows[1].classification='changing';doc.rows[1].implemented=true;
doc.rows[2].family='functional';doc.rows[2].fingerprint.embedding='function';doc.rows[2].implemented=true;
assert.equal(context.ontologyValidate(doc,payload),doc);
context.doc=doc;vm.runInContext("ontologyReviews.set('test',doc)",context);
assert.deepEqual(Array.from(context.ontologyDiscovery('test',run,'implemented'),p=>p.y),[0,1,1,1]);
const bins=context.ontologyPersistence('test',run);
assert.equal(bins.get(1).assessable,1,'Sibling of crossing is not a descendant');
assert.equal(bins.get(1).retained,1);
assert.deepEqual(Array.from(context.ontologyDiff({a:null,b:'x'},{a:'new',b:'x'})),[],'Unknown is not a boundary');
assert.throws(()=>context.ontologyValidate({...doc,campaign:'other'},payload));
assert.throws(()=>context.ontologyValidate({...doc,rows:[...doc.rows,doc.rows[0]]},payload));
assert.throws(()=>context.ontologyValidate({...doc,rows:[{...doc.rows[0],candidate_id:'stale'}]},payload));
assert.throws(()=>context.ontologyValidate({...doc,rows:[{...doc.rows[0],fingerprint_complete:true}]},payload),'A partial fingerprint cannot claim completeness');
assert.equal(context.ontologyValidate({...doc,rows:[{...doc.rows[0],classification:'invalid_source',implemented:false,executable:false}]},payload).rows[0].classification,'invalid_source');
assert.equal(context.ontologyValidate({...doc,rows:[{...doc.rows[0],classification:'source_unavailable'}]},payload).rows[0].classification,'source_unavailable');
assert.equal(context.ontologyValidate({...doc,rows:[{...doc.rows[0],classification:'parent_source_unavailable'}]},payload).rows[0].classification,'parent_source_unavailable');
const html=fs.readFileSync('experiments/ontology_dashboard.html','utf8');
for(const [,script] of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g))new vm.Script(script);
assert.ok(html.includes('function interventionCandidates('),'Retains original explorer implementation');
console.log('PASS: task schemas, unknowns, candidate binding, ancestry, discovery, persistence, and copied-page syntax');
const meanContext = {document:{createElement:()=>({}),head:{appendChild(){}}},sectionHtml(){},drawCharts(){},redraw(){},destroySectionCharts(){},attachSection(){},campaignsFrom(){},stableIndex:()=>0,conditionColor:()=> '#fff'};
vm.createContext(meanContext);
vm.runInContext(fs.readFileSync('experiments/ontology_dashboard.js','utf8'),meanContext);
meanContext.curves={a:[{x:0,y:0},{x:1,y:1},{x:2,y:2}],b:[{x:0,y:0},{x:1,y:3}],c:[{x:0,y:null},{x:1,y:5}]};
vm.runInContext('ontologyDiscovery=(_id,run)=>curves[run.run_id]',meanContext);
const meanRuns=[{run_id:'a',condition:'C0'},{run_id:'b',condition:'C0'},{run_id:'c',condition:'C1'}];
const means=meanContext.ontologyDiscoveryDatasets('mean',meanRuns,{discoverySeries:'mean',mode:'implemented'},{},()=>true);
assert.equal(means.length,2);
assert.equal(means[0].data[1].y,2);
assert.equal(means[0].data[2].y,null,'Do not shrink the cohort when a run ends');
assert.equal(means[1].data[0].y,null,'Unknown review is not zero');
assert.equal(meanContext.ontologyDiscoveryDatasets('mean',[],{discoverySeries:'mean'},{},()=>true).length,0);
assert.equal(meanContext.ontologyDiscoveryDatasets('mean',meanRuns,{discoverySeries:'mean'},{},p=>p.proposal===1)[0].data.length,1);
console.log('PASS: condition discovery means, fixed cohorts, unknown values, and proposal bounds');
const familyDoc={rows:Array.from({length:30},(_,i)=>({family:'family-'+String(i).padStart(2,'0')}))};
context.familyDoc=familyDoc;vm.runInContext("ontologyReviews.set('palette',familyDoc)",context);
const familyStyles=familyDoc.rows.map(r=>context.ontologyFamilyStyle('palette',r.family));
assert.equal(new Set(familyStyles.map(s=>s.color)).size,30,'No family color collisions in current campaign sizes');
assert.equal(new Set(familyStyles.map(s=>s.tag)).size,30);
assert.equal(context.ontologyFamilyStyle('palette','unannotated').tag,'?');
assert.equal(context.ontologyFamilyStyle('palette','family-00').color,familyStyles[0].color);
console.log('PASS: distinct family shades, exact family IDs, unknown family fallback');
