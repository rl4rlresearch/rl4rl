/* The page is a copy of the trajectory explorer. Extend its rendering and controls. */
'use strict';
const ontologyCommon = {
  input_units:'Atomic input units and representation: tokens, pixels, patches, frames, channels; distinguish recoding from changing what a unit represents.',
  input_transform:'Feature construction before the model: raw, spectral, engineered or learned; include channel decomposition and joint versus separate inputs.',
  embedding:'Mapping inputs into features: lookup, factorized lookup, functional, continuous or structured encoding.',
  position:'How position/order enters computation: additive, score bias, rotary, implicit recurrence or none.',
  mixing:'Core interaction operator: attention, convolution, recurrence, dense mixing, spectral mixing or a hybrid.',
  routing:'Who communicates with whom and how routing weights arise: fixed, content-dependent, position-dependent, sparse or conditional.',
  state:'Persistent state and memory: none, recurrent state, external memory, multiple timescales or structured state.',
  feedforward:'Feature transformation: ungated MLP, multiplicative gating, expert mixture, polynomial or other construction.',
  parameter_construction:'How weights are represented: free matrices, low-rank factors, algebraic derivation, generated or constrained weights.',
  sharing:'Dependency/tying topology between weights, modules, depths or time steps. Ordinary tying alone is preserving under the strict rubric.',
  normalization:'Normalization mechanism, axes and placement; ordinary changes in scale count are preserving under the strict rubric.',
  connectivity:'Residual, skip, parallel, multiscale and hierarchical paths; depth changes alone are numerical settings.',
  aggregation:'How distributed features become an output representation: terminal state, mean/max pooling, learned queries or attention pooling.',
  output:'Readout representation and factorization: class head, token head, tied readout, functional output or structured prediction.',
  symmetry:'How invariance/equivariance is built into the architecture: unconstrained, group-constrained, canonicalized or other construction.',
  conditional_compute:'Input-dependent execution, adaptive depth, early exit, routing or fixed compute.',
  activation:'Nonlinearity family and its placement: scalar, gated, periodic, piecewise, rational or learned; distinguish tuning within a family.',
  stochasticity:'Stochastic versus deterministic representations, latent variables, sampling and noise injection.',
  bottleneck:'Compression/expansion mechanism: projection, pooling, latent tokens, discrete codes or information bottleneck.',
  iteration:'Single pass, unrolled iterations, equilibrium/implicit computation or recurrent refinement.',
  other:'Any additional representational primitive. Name and explain it; add a new schema component for recurring discoveries.'
};
const ontologyTaskComponents = {
  addition:{operand_encoding:'Separate digit streams, paired digits, numeric features; token identity and operand alignment.',digit_order:'Direction and organization of operand/result digits; distinguish order convention from new representation.',carry_representation:'Explicit learned carry state, distributed carry representation or another implemented mechanism; do not infer from accuracy.',attention_scores:'Content QK scores, learned distance scores, algebraic scores or hybrid.',projection_relations:'Q/K/V/O dependency structure: independent, tied, rotated/transposed or functionally generated.',output_factorization:'Joint/independent output units, vocabulary semantics and readout construction.'},
  nanogpt:{context_topology:'Local/global/hierarchical/context-compressed communication; window size within an existing scheme is a setting.',attention_scores:'Softmax dot-product, kernel/linear, position-only or another score mechanism.',kv_memory:'KV organization, grouped/shared heads, compressed or recurrent context memory.',projection_relations:'Q/K/V/O construction and relationships.',vocabulary_representation:'Token units, embedding and output vocabulary construction within the frozen task interface.',block_composition:'Attention/state/convolution/expert composition and alternation.'},
  fashion:{spatial_units:'Pixels, patches, regions or learned spatial units.',spatial_operator:'Dense convolution, separable convolution, attention, spectral or hybrid.',scale_representation:'Single-scale, pyramid, multiresolution or hierarchical representation.',spatial_readout:'Flattened head, global pooling, learned spatial pooling or class token.',channel_interaction:'Joint channels, grouped/separable paths or learned channel routing.'},
  kws:{acoustic_representation:'Frozen acoustic features versus permitted learned transforms; frame/frequency organization.',state_update:'GRU, LSTM, ungated, structured/linear or other causal state update.',state_structure:'Single state, factored state, multiscale memory or coupled states.',temporal_schedule:'Fixed stride/subsampling versus content-dependent frame selection.',readout_history:'Final state, online average, learned accumulation or other causal history summary.',exit_policy:'Fixed horizon versus input-dependent stopping; record the decision mechanism.'},
  har:{sensor_fusion:'Early joint processing, independent sensor streams with late fusion, cross-sensor attention or other fusion.',temporal_operator:'Recurrent, temporal convolution, attention, spectral or hybrid.',directionality:'Causal, bidirectional or separate forward/backward representations.',state_update:'GRU/LSTM/other state-transition mechanism when recurrence exists.',frequency_representation:'Time-domain, spectral, time-frequency or hybrid features.',temporal_readout:'Terminal states, mean/max pooling, learned temporal attention or other aggregation.'}
};
const ontologyProcedure = {
  training:{objective:'Loss/objective family and supervision structure.',optimizer:'Update-rule family; learning-rate values belong in settings.',curriculum:'Sampling/curriculum organization.',augmentation:'Training transformations/invariance induction.',teacher:'Distillation, teacher or auxiliary supervision.',parameterization:'Training-only reparameterization and constraints.',schedule:'Training phases, freezing, optimizer transitions.',other:'Other training mechanism.'},
  inference:{decoding:'Decoding/search procedure within task constraints.',ensemble:'Ensembling, averaging and test-time augmentation.',adaptation:'Test-time updates or adaptation.',precision:'Quantization/arithmetic representation.',execution:'Caching, compilation and execution transformations.',other:'Other inference mechanism.'}
};
function ontologyTask(payload){const s=String(payload.task_display_name||payload.task_id||payload.objective_metric||'').toLowerCase();return /speech|keyword|kws/.test(s)?'kws':/activity|har/.test(s)?'har':/fashion|image|grayscale/.test(s)?'fashion':/nanogpt|language|bpb|pretraining/.test(s)?'nanogpt':/addition|adder|digit/.test(s)?'addition':'generic';}
function ontologySchema(payload){return {...ontologyCommon,...(ontologyTaskComponents[ontologyTask(payload)]||{})};}
const ontologyReviews=new Map(),ontologyViews=new Map(),ontologyReviewIndexes=new WeakMap();
const ontologyPalette=['#58a6ff','#e3b341','#bc8cff','#3fb950','#f778ba','#79c0ff','#ffa657','#56d4dd','#d2a8ff','#ff7b72'];
function ontologyColor(value){return value&&value!=='unannotated'?ontologyPalette[stableIndex(value,ontologyPalette.length)]:'#677586';}
// Reserve rose, yellow and cyan for families; conditions use blue/orange/purple/green.
const ontologyFamilyIndexes=new WeakMap();
function ontologyFamilyEntries(id){const doc=ontologyReviews.get(id);if(!doc)return [];let families=ontologyFamilyIndexes.get(doc);if(!families){families=[...new Set(doc.rows.map(r=>r.family).filter(f=>f&&f!=='unannotated'))].sort();ontologyFamilyIndexes.set(doc,families);}return families;}
function ontologyFamilyStyle(id,family){
  const index=ontologyFamilyEntries(id).indexOf(family);
  if(index<0)return {color:'#89939f',tag:'?',family:'No assigned family'};
  const hue=[345,55,180][index%3],variant=Math.floor(index/3);
  return {color:`hsl(${hue} ${90-(variant%3)*12}% ${58+Math.floor(variant/3)*8}%)`,tag:`F${index+1}`,family};
}
function ontologyFamilyColor(id,family){return ontologyFamilyStyle(id,family).color;}
function ontologyKey(run,point){return `${run.run_id}:${point.proposal}`;}
function ontologyReview(id,run,point){const doc=ontologyReviews.get(id);if(!doc)return null;let index=ontologyReviewIndexes.get(doc);if(!index){index=new Map(doc.rows.map(r=>[`${r.run_id}:${r.proposal}`,r]));ontologyReviewIndexes.set(doc,index);}const row=index.get(ontologyKey(run,point));return row&&row.candidate_id===point.candidate_id?row:null;}
function ontologyFingerprint(row){return row?.fingerprint||{};}
function ontologyFamily(row){return row?.family||'unannotated';}
function ontologyChanged(id,run,point){const row=ontologyReview(id,run,point);return row?.classification||'unannotated';}
function ontologyParents(run,point){return (point.parent_ids||[]).map(cid=>run.points.filter(p=>p.candidate_id===cid&&p.proposal<point.proposal).at(-1)).filter(Boolean);}
function ontologyDiff(a,b){return [...new Set([...Object.keys(a),...Object.keys(b)])].filter(k=>a[k]!=null&&b[k]!=null&&a[k]!==b[k]);}
function ontologyStage(point,row){return {proposed:row?.proposed_change??null,implemented:row?.implemented??null,executable:point.valid||point.failure_kind==='nonqualification'?true:row?.executable??null,qualified:!!point.valid,retained:!!point.retained};}
function ontologyDiscovery(id,run,mode){const seen=new Set(),seed=run.points.find(p=>p.is_seed),seedFamily=seed?ontologyReview(id,run,seed)?.family:null; if(seedFamily)seen.add(seedFamily);let known=!!seedFamily;return run.points.map(p=>{const row=ontologyReview(id,run,p),eligible=mode==='implemented'?row?.implemented===true:mode==='qualified'?p.valid:p.retained;if(row?.family&&eligible){seen.add(row.family);known=true;}return {x:p.proposal,y:known?seen.size-(seedFamily?1:0):null,point:p};});}
function ontologyDiscoveryDatasets(id,runs,view,payload,inWindow){
  const series=runs.map(run=>({run,data:ontologyDiscovery(id,run,view.mode)}));
  if(view.discoverySeries!=='mean')return series.map(({run,data})=>({label:run.label,data:data.filter(d=>inWindow(d.point)),borderColor:color(run),backgroundColor:color(run),borderDash:replicateDash(run),pointRadius:2,stepped:true,spanGaps:false}));
  return [...new Set(runs.map(r=>r.condition))].map(condition=>{
    const members=series.filter(s=>s.run.condition===condition),indexes=members.map(s=>new Map(s.data.map(d=>[d.x,d.y])));
    const proposals=[...new Set(members.flatMap(s=>s.data.map(d=>d.x)))].sort((a,b)=>a-b);
    const data=proposals.filter(proposal=>inWindow({proposal})).map(x=>{const values=indexes.map((index,i)=>{if(index.has(x))return index.get(x);const last=members[i].data.at(-1);return payload.combined_semantic&&last&&x>last.x?last.y:undefined;});const available=members.map((member,i)=>({run:member.run,value:values[i]})).filter(item=>Number.isFinite(item.value));return {x,y:payload.combined_semantic?equalCampaignAggregate(available,item=>item.value,mean,payload):values.every(v=>Number.isFinite(v))?values.reduce((a,b)=>a+b,0)/members.length:null};});
    return {label:`${condition} mean · ${members.length} runs`,data,borderColor:conditionColor(condition,payload),backgroundColor:conditionColor(condition,payload),borderWidth:2,pointRadius:2,stepped:true,spanGaps:false};
  });
}
function ontologyView(id){if(!ontologyViews.has(id))ontologyViews.set(id,{run:'',mode:'implemented',cohort:'all',selected:null});return ontologyViews.get(id);}
function ontologyValidate(doc,payload){
  if(doc.schema_version!=='1.0'||doc.campaign!==payload.campaign||!Array.isArray(doc.rows)||typeof doc.rubric!=='string')throw new Error('Expected schema_version 1.0, matching campaign, rubric, and rows.');
  const seen=new Set();for(const row of doc.rows){const run=payload.runs.find(r=>r.run_id===row.run_id),point=run?.points.find(p=>p.proposal===row.proposal),key=`${row.run_id}:${row.proposal}`;
    if(!point||!point.candidate_id||row.candidate_id!==point.candidate_id||seen.has(key))throw new Error(`Unknown, stale or duplicate candidate: ${key}`);seen.add(key);
    if(!['changing','preserving','mixed','uncertain','unannotated','invalid_source','source_unavailable','parent_source_unavailable'].includes(row.classification))throw new Error(`Invalid classification: ${key}`);
    for(const flag of ['proposed_change','implemented','executable'])if(row[flag]!=null&&typeof row[flag]!=='boolean')throw new Error(`Expected boolean or null: ${flag}`);
    for(const group of ['fingerprint','training','inference']){const values=row[group];if(!values||typeof values!=='object'||Array.isArray(values)||Object.values(values).some(v=>v!==null&&(typeof v!=='string'||!v.trim())))throw new Error(`Invalid ${group}: ${key}`);}
    if(row.fingerprint_complete!=null&&typeof row.fingerprint_complete!=='boolean')throw new Error(`Invalid fingerprint_complete: ${key}`);
    if(row.fingerprint_complete&&Object.keys(ontologySchema(run.source_payload||payload)).some(component=>typeof row.fingerprint[component]!=='string'||!row.fingerprint[component].trim()))throw new Error(`Incomplete fingerprint marked complete: ${key}`);
    for(const field of ['family','notes','reviewer'])if(row[field]!=null&&typeof row[field]!=='string')throw new Error(`Invalid ${field}: ${key}`);
  }return doc;
}
function ontologyTemplate(payload){return {schema_version:'1.0',campaign:payload.campaign,rubric:'Task-specific ontology review v1 (edit and freeze before condition comparisons)',component_definitions:ontologySchema(payload),procedure_definitions:ontologyProcedure,rows:payload.runs.flatMap(run=>run.points.filter(p=>p.candidate_id).map(p=>({run_id:run.run_id,proposal:p.proposal,candidate_id:p.candidate_id,family:null,classification:'unannotated',proposed_change:null,implemented:p.is_seed?true:null,executable:p.valid?true:null,fingerprint:Object.fromEntries(Object.keys(ontologySchema(run.source_payload||payload)).map(k=>[k,null])),training:Object.fromEntries(Object.keys(ontologyProcedure.training).map(k=>[k,null])),inference:Object.fromEntries(Object.keys(ontologyProcedure.inference).map(k=>[k,null])),settings:{},reviewer:'',notes:''})))};}
const trajectorySectionHtml=sectionHtml;
sectionHtml=function(id,title,payload){const base=trajectorySectionHtml(id,title,payload);if(!payload?.available)return base;const schema=ontologySchema(payload);return base.replace('<div class="charts">',`<div class="ontology-panel" id="${id}-ontology"><div class="analysis"><h3>Post-campaign ontology diagnostics</h3><p>Unreviewed mechanisms stay unknown. Agent claims and scheduled interventions are not ontology labels. Saved source reviews load automatically. Imported JSON overrides them for this browser tab. Review data is never sent to researchers or written to campaign artifacts.</p><div class="controls"><button data-ontology="template">Download review template</button><label>Import reviewed JSON <input type="file" accept=".json,application/json" data-ontology="import"></label><button data-ontology="export">Export current reviews</button><button data-ontology="clear">Clear reviews</button></div><p role="status" id="${id}-ontology-status"></p><details><summary>Fingerprint definitions · ${html(ontologyTask(payload))} · ${Object.keys(schema).length} architecture components + training / inference</summary><p>Null = not reviewed; “absent” = reviewed and not present. Categories describe mechanisms; numeric dimensions, ranks and rates go in settings. Family names and strict/broad boundary decisions belong to the frozen rubric. Additional component keys are supported.</p><div class="ontology-definitions">${Object.entries({architecture:schema,...ontologyProcedure}).map(([group,items])=>`<div><h4>${html(group)}</h4>${Object.entries(items).map(([key,desc])=>`<p><b>${html(key)}</b> — ${html(desc)}</p>`).join('')}</div>`).join('')}</div></details></div><div class="controls"><label>Run detail <select data-ontology="run"></select></label><label>Outcome cohort <select data-ontology="cohort"><option value="all">All proposals</option><option value="changing">Reviewed changing / mixed</option><option value="preserving">Reviewed preserving</option></select></label></div><p class="note">Ontology panels use the condition/run legend and proposal bounds; Family discovery has its own condition toggles. Outcome/type filters and aggregation controls above affect the inherited performance chart only. Discovery counts include earlier history; new families exclude the reviewed seed family. Performance chart: lines identify runs; marker colors identify ontology family; triangles mark reviewed changing/mixed edits. Gray = no assigned family.</p><div id="${id}-ontology-summary" class="stats"></div><div class="ontology-grid"><div class="chart-card"><h3>Family discovery</h3><div class="ontology-discovery-controls"><span data-ontology="discovery-conditions" class="ontology-condition-toggles"></span><label>Lines <select data-ontology="discovery-series"><option value="runs">Runs</option><option value="mean">Condition mean</option></select></label><label>Stage <select data-ontology="mode"><option value="implemented">Implemented</option><option value="qualified">Qualified</option><option value="retained">Retained</option></select></label></div><p class="note">X: proposal · Y: reviewed families reached (lower bound while review is incomplete). Local condition toggles apply here; hidden runs and proposal bounds still apply. Single-campaign means require every included run. Aggregate means retain each ended run’s last observed count and weight contributing campaigns equally.</p><div class="ontology-chart"><canvas id="${id}-ontology-discovery"></canvas></div></div><div class="chart-card"><h3>Transition outcomes</h3><p class="note">Known yes / no / unknown per stage; selected cohort in visible runs/window. Missing review remains unknown.</p><div class="ontology-chart"><canvas id="${id}-ontology-stages"></canvas></div></div></div><div class="chart-card"><h3>Component fingerprint timeline</h3><p class="note">Selected run · columns: proposals · rows: components. Orange border = known category differs from a recorded parent. Click a cell to inspect.</p><div class="table-wrap" id="${id}-ontology-fingerprint"></div></div><div class="chart-card"><h3>Lineage</h3><p class="note">X: proposal · Y: branch layout (not a metric). Color: reviewed family; orange edge: reviewed crossing. Click a node. Unavailable parents remain disconnected.</p><div class="table-wrap" id="${id}-ontology-lineage"></div></div><div class="ontology-grid"><div class="chart-card"><h3>Portfolio composition</h3><p class="note">X: proposal · Y: occupied slots by reviewed family, including unknowns.</p><div class="ontology-chart"><canvas id="${id}-ontology-portfolio"></canvas></div></div><div class="chart-card"><h3>Mechanism persistence</h3><p class="note">Selected run · reviewed crossing edges only. Fraction of assessable descendant candidates retaining all introduced category values; all branches included. Depth is ancestry distance, not elapsed proposals.</p><div id="${id}-ontology-persistence"></div></div></div><div class="analysis" id="${id}-ontology-evidence"><h3>Candidate evidence</h3><p>Select a fingerprint cell or lineage node.</p></div></div><div class="charts">`);};
const ontologyStyle=document.createElement('style');ontologyStyle.textContent='.ontology-family-key{font-size:12px;margin-top:10px;padding:10px;border:1px solid #394553;border-radius:6px}.ontology-family-key p{margin:5px 0;color:#aab7c6}.ontology-family-swatches{display:flex;flex-wrap:wrap;gap:6px 14px;max-height:140px;overflow:auto}.ontology-family-swatches span{display:inline-flex;align-items:center;gap:5px}.ontology-family-swatches i{width:11px;height:11px;border-radius:50%;flex-shrink:0}'+'.ontology-discovery-controls{display:flex;align-items:center;flex-wrap:wrap;gap:6px 10px;margin:6px 0}.ontology-discovery-controls label{display:inline-flex;align-items:center;gap:4px;font-size:11px;margin:0}.ontology-discovery-controls select{font-size:11px;padding:2px 4px;min-height:24px;width:auto}.ontology-condition-toggles{display:inline-flex;gap:6px}.ontology-condition-toggles input{margin:0;width:12px;height:12px}.ontology-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}.ontology-disclosure>summary{cursor:pointer;font-size:15px;font-weight:650;padding:4px 0}.ontology-disclosure[open]>summary{margin-bottom:10px}.ontology-grid{align-items:start}.ontology-chart{height:280px;position:relative}.ontology-panel .chart-card{margin-top:12px}.ontology-definitions{display:grid;grid-template-columns:2fr 1fr 1fr;gap:18px}.ontology-panel .table-wrap{max-height:480px;overflow:auto}.ontology-panel thead th{position:sticky;top:0;background:#111821;z-index:2}.ontology-panel pre{max-height:360px;overflow:auto;white-space:pre-wrap;font-size:12px}.ontology-panel h3{font-size:15px}.fingerprint-cell{font-size:10px;min-width:68px;max-width:130px;overflow:hidden;text-overflow:ellipsis}.ontology-panel th:first-child{position:sticky;left:0;background:#111821;z-index:1}.ontology-node{cursor:pointer}.ontology-node:focus{outline:2px solid white}@media(max-width:900px){.ontology-grid,.ontology-definitions{grid-template-columns:1fr}}';document.head.appendChild(ontologyStyle);
function ontologyChart(id,suffix,type,data,x,y,extra={}){const key=`${id}-ontology-${suffix}`;charts.get(key)?.destroy();const target=document.getElementById(key);if(!target||typeof Chart==='undefined')return;charts.set(key,new Chart(target,{type,data,options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:'#b8c5d5',boxWidth:10}}},scales:{x:{type:type==='line'?'linear':'category',title:{display:true,text:x,color:'#b8c5d5'},ticks:{color:'#8b98a9'},grid:{color:'#222d3b'}},y:{beginAtZero:true,title:{display:true,text:y,color:'#b8c5d5'},ticks:{color:'#8b98a9',precision:0},grid:{color:'#222d3b'}}},...extra}}));}
function ontologyPersistence(id,run){const result=new Map();for(const origin of run.points){const review=ontologyReview(id,run,origin);if(!['changing','mixed'].includes(review?.classification)||review.implemented!==true)continue;const parents=ontologyParents(run,origin);if(parents.length!==1)continue;const before=ontologyFingerprint(ontologyReview(id,run,parents[0])),after=ontologyFingerprint(review),keys=ontologyDiff(before,after);if(!keys.length)continue;const depths=new Map([[origin.candidate_id,0]]);for(const p of run.points.filter(p=>p.proposal>origin.proposal)){const ds=(p.parent_ids||[]).filter(cid=>depths.has(cid)).map(cid=>depths.get(cid)+1);if(!ds.length||!p.candidate_id)continue;const depth=Math.min(...ds);if(p.candidate_id===origin.candidate_id)continue;depths.set(p.candidate_id,depth);const row=ontologyReview(id,run,p),fp=ontologyFingerprint(row),bin=result.get(depth)||{eligible:0,assessable:0,retained:0};bin.eligible++;if(row?.implemented===true&&keys.every(k=>fp[k]!=null)){bin.assessable++;if(keys.every(k=>fp[k]===after[k]))bin.retained++;}result.set(depth,bin);}}return result;}
function drawOntology(id,payload){
  const state=states[id],view=ontologyView(id),runs=orderedRuns(payload,payload.runs.filter(r=>runVisible(state,r))),bounds=proposalBounds(state),inWindow=p=>p.proposal>=bounds.start&&(bounds.end===null||p.proposal<=bounds.end),points=runs.flatMap(run=>run.points.filter(p=>!p.is_seed&&inWindow(p)).map(point=>({run,point,row:ontologyReview(id,run,point)}))),reviewed=points.filter(x=>x.row?.classification!=='unannotated'&&x.row);
  const status=document.getElementById(`${id}-ontology-status`);if(!status)return;
  status.textContent=`${payload.runs.every(r=>String(r.status).replace(/^saved:\s*/,'')==='completed')?'Campaign complete':'Provisional · campaign incomplete'} · ${reviewed.length}/${points.length} visible proposals assessed · ${reviewed.filter(x=>!['uncertain','unannotated'].includes(x.row.classification)).length} resolved · ${reviewed.filter(x=>x.row.fingerprint_complete).length} complete fingerprints · ${ontologyReviews.get(id)?.rubric||'No reviewed annotations loaded'}`;
  const select=document.querySelector(`#${id}-ontology [data-ontology="run"]`);if(!runs.some(r=>r.run_id===view.run))view.run=runs[0]?.run_id||'';select.innerHTML=runs.map(r=>`<option value="${html(r.run_id)}"${r.run_id===view.run?' selected':''}>${html(r.label)}</option>`).join('');
  document.querySelector(`#${id}-ontology [data-ontology="mode"]`).value=view.mode;
  document.getElementById(`${id}-ontology-summary`).innerHTML=[['Visible proposals',points.length],['Assessed',reviewed.length],['Resolved',reviewed.filter(x=>!['uncertain','unannotated'].includes(x.row.classification)).length],['Complete fingerprints',reviewed.filter(x=>x.row.fingerprint_complete).length],['Changing / mixed',reviewed.filter(x=>['changing','mixed'].includes(x.row.classification)).length],['Preserving',reviewed.filter(x=>x.row.classification==='preserving').length],['Uncertain',reviewed.filter(x=>x.row.classification==='uncertain').length],['Invalid source',reviewed.filter(x=>x.row.classification==='invalid_source').length],['Source unavailable',reviewed.filter(x=>x.row.classification==='source_unavailable').length],['Parent source unavailable',reviewed.filter(x=>x.row.classification==='parent_source_unavailable').length],['Unannotated',points.length-reviewed.length]].map(([label,n])=>`<div class="stat"><span>${label}</span><b>${n}</b></div>`).join('');
  const conditions=allConditions(payload);
  view.discoveryConditions??=[...conditions];
  view.discoverySeries??='runs';
  const toggles=document.querySelector(`#${id}-ontology [data-ontology="discovery-conditions"]`);
  toggles.innerHTML=conditions.map(c=>`<label style="color:${conditionColor(c,payload)}"><input type="checkbox" data-discovery-condition="${html(c)}" ${view.discoveryConditions.includes(c)?'checked':''}>${html(c)}</label>`).join('');
  toggles.onchange=e=>{const c=e.target.dataset.discoveryCondition;if(!c)return;view.discoveryConditions=e.target.checked?[...new Set([...view.discoveryConditions,c])]:view.discoveryConditions.filter(v=>v!==c);drawOntology(id,payload);};
  const series=document.querySelector(`#${id}-ontology [data-ontology="discovery-series"]`);
  series.value=view.discoverySeries;
  series.onchange=e=>{view.discoverySeries=e.target.value;drawOntology(id,payload);};
  const discoveryRuns=orderedRuns(payload,payload.runs.filter(r=>(!payload.combined_semantic||campaignIncluded(r,state))&&view.discoveryConditions.includes(r.condition)&&!state.hiddenRuns.includes(r.run_id)));
  ontologyChart(id,'discovery','line',{datasets:ontologyDiscoveryDatasets(id,discoveryRuns,view,payload,inWindow)},'Proposal','Reviewed families');
  document.querySelector(`#${id}-ontology [data-ontology="cohort"]`).value=view.cohort;
  const cohort=points.filter(x=>view.cohort==='all'||(view.cohort==='changing'?['changing','mixed'].includes(x.row?.classification):x.row?.classification==='preserving'));
  const stages=['proposed','implemented','executable','qualified','retained'];
  ontologyChart(id,'stages','bar',{labels:stages.map(s=>s==='proposed'?'Proposed ontology change':s==='implemented'?'Edit implemented':s),datasets:[true,false,null].map((value,i)=>({label:['Yes','No','Unknown'][i],backgroundColor:['#3fb950','#f78166','#677586'][i],data:stages.map(s=>cohort.filter(({point,row})=>ontologyStage(point,row)[s]===value).length)}))},'Stage (not necessarily nested)','Proposals');
  const run=runs.find(r=>r.run_id===view.run),fpTarget=document.getElementById(`${id}-ontology-fingerprint`),lineTarget=document.getElementById(`${id}-ontology-lineage`),persistTarget=document.getElementById(`${id}-ontology-persistence`);
  if(!run){fpTarget.textContent=lineTarget.textContent=persistTarget.textContent='No visible run selected.';ontologyChart(id,'portfolio','bar',{labels:[],datasets:[]},'Proposal','Slots');return;}
  const visible=run.points.filter(inWindow),schema=ontologySchema(run.source_payload||payload),keys=[...new Set([...Object.keys(schema),...visible.flatMap(p=>Object.keys(ontologyFingerprint(ontologyReview(id,run,p))))])];
  fpTarget.innerHTML=`<table><thead><tr><th>Component</th>${visible.map(p=>`<th>${p.is_seed?'Seed':p.proposal}</th>`).join('')}</tr></thead><tbody>${keys.map(k=>`<tr><th title="${html(schema[k]||'Additional reviewed component')}">${html(k)}</th>${visible.map(p=>{const fp=ontologyFingerprint(ontologyReview(id,run,p)),value=fp[k],changed=ontologyParents(run,p).some(parent=>ontologyDiff(ontologyFingerprint(ontologyReview(id,run,parent)),fp).includes(k));return `<td><button class="fingerprint-cell" data-proposal="${p.proposal}" title="${html(`${k}: ${value??'Not reviewed'}`)}" style="border-color:${changed?'#ffa657':ontologyColor(value)};color:${ontologyColor(value)}">${html(value??'?')}</button></td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
  const lanes=new Map(),children=new Map();let nextLane=0;for(const p of run.points){const parent=ontologyParents(run,p)[0],count=parent?(children.get(parent.proposal)||0):0;lanes.set(p.proposal,parent&&count===0?lanes.get(parent.proposal):nextLane++);if(parent)children.set(parent.proposal,count+1);}
  const min=visible[0]?.proposal||0,max=visible.at(-1)?.proposal||1,w=Math.max(900,visible.length*18),h=Math.max(150,nextLane*23+50),x=p=>40+(p.proposal-min)/Math.max(1,max-min)*(w-80),y=p=>35+lanes.get(p.proposal)*23,shown=new Set(visible.map(p=>p.proposal));
  lineTarget.innerHTML=`<svg role="img" aria-label="Candidate ancestry graph" width="${w}" height="${h}">${visible.flatMap(p=>ontologyParents(run,p).filter(parent=>shown.has(parent.proposal)).map(parent=>`<line x1="${x(parent)}" y1="${y(parent)}" x2="${x(p)}" y2="${y(p)}" stroke="${['changing','mixed'].includes(ontologyChanged(id,run,p))?'#ffa657':'#35465c'}"/>`)).join('')}${visible.map(p=>`<g class="ontology-node" role="button" tabindex="0" data-proposal="${p.proposal}" aria-label="Proposal ${p.proposal}, ${html(ontologyFamily(ontologyReview(id,run,p)))}"><title>${html(`Proposal ${p.proposal} · ${ontologyFamily(ontologyReview(id,run,p))} · ${ontologyChanged(id,run,p)} · ${p.retained?'retained':p.failure_kind||'not retained'}`)}</title><circle cx="${x(p)}" cy="${y(p)}" r="${view.selected===p.proposal?7:5}" fill="${ontologyFamilyColor(id,ontologyFamily(ontologyReview(id,run,p)))}" stroke="${p.retained?'#e6edf3':'#0d1117'}"/><text x="${x(p)}" y="${y(p)-10}" text-anchor="middle" fill="#8b98a9" font-size="9">${p.proposal}</text></g>`).join('')}</svg>`;
  const occupancy=visible.map(p=>{if(!Array.isArray(p.portfolio_after))return null;const bins=Object.create(null);for(const cid of p.portfolio_after||[]){const member=run.points.filter(q=>q.candidate_id===cid&&q.proposal<=p.proposal).at(-1),family=member?ontologyFamily(ontologyReview(id,run,member)):'unannotated';bins[family]=(bins[family]||0)+1;}return bins;}),families=[...new Set(occupancy.filter(Boolean).flatMap(Object.keys))];
  ontologyChart(id,'portfolio','bar',{labels:visible.map(p=>p.proposal),datasets:families.map(f=>({label:f,data:occupancy.map(bin=>bin===null?null:bin[f]||0),backgroundColor:ontologyFamilyColor(id,f),stack:'slots'}))},'Proposal','Occupied slots');
  const persistence=ontologyPersistence(id,run);persistTarget.innerHTML=persistence.size?`<table><thead><tr><th>Depth</th><th>Retaining / assessed</th><th>Unknown</th><th>Persistence</th></tr></thead><tbody>${[...persistence].sort((a,b)=>a[0]-b[0]).map(([depth,b])=>`<tr><td>${depth}</td><td>${b.retained} / ${b.assessable}</td><td>${b.eligible-b.assessable}</td><td>${b.assessable?Math.round(100*b.retained/b.assessable)+'%':'—'}</td></tr>`).join('')}</tbody></table>`:'<p class="empty">Requires a reviewed implemented crossing, reviewed parent components, and recorded descendants. No qualifying observations yet.</p>';
  for(const target of [fpTarget,lineTarget])target.querySelectorAll('[data-proposal]').forEach(el=>{const activate=()=>{view.selected=Number(el.dataset.proposal);ontologyEvidence(id,payload,run,run.points.find(p=>p.proposal===view.selected));};el.onclick=activate;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};});
  if(view.selected!==null&&visible.some(p=>p.proposal===view.selected))ontologyEvidence(id,payload,run,run.points.find(p=>p.proposal===view.selected));
  else document.getElementById(`${id}-ontology-evidence`).innerHTML='<h3>Candidate evidence</h3><p>Select a fingerprint cell or lineage node.</p>';
}
let ontologyEvidenceRequest=0;
async function ontologyEvidence(id,payload,run,point){const request=++ontologyEvidenceRequest,target=document.getElementById(`${id}-ontology-evidence`),row=ontologyReview(id,run,point);target.innerHTML=`<h3>${html(run.label)} · proposal ${point.proposal}</h3><p>${html(ontologyFamily(row))} · ${html(row?.classification||'unannotated')} · ${html(point.failure_kind|| (point.valid?'qualified':'unknown outcome'))} · ${point.retained?'retained':'not retained'}</p><p><b>Review:</b> ${html(row?.notes||'No reviewed explanation.')} ${html(row?.reviewer||'')}</p><p><b>Fingerprint coverage:</b> ${row?.fingerprint_complete?'Complete audited reference or preserving proof':row?.classification==='invalid_source'?'Invalid implementation; no source-defined executable architecture':row?.classification==='source_unavailable'?'Source unavailable; no program exists to fingerprint':row?.classification==='parent_source_unavailable'?'Recorded parent source unavailable; no ontology transition is claimed':row?'Partial source extraction':'Not reviewed'}</p><p><b>Agent’s intended edit:</b> ${html(point.intended_edit||'Not recorded')}</p><p><b>Agent’s mechanism claim:</b> ${html(point.mechanism||'Not recorded')}</p><p><b>Parent IDs:</b> ${html((point.parent_ids||[]).join(', ')||'Seed')}</p><details><summary>Fingerprint, training, inference and settings</summary><pre>${html(JSON.stringify(row||{},null,2))}</pre></details><div data-source>Loading recorded source…</div>`;if(point.is_seed){target.querySelector('[data-source]').textContent='Seed fingerprint is available above. Select a child to compare its source with the seed.';return;}const key=run.source_campaign_key||campaignDefs.find(([cid])=>cid===id)?.[2];try{const response=await fetch(`/api/ontology/source?${new URLSearchParams({campaign:key,run:run.source_run_id||run.run_id,proposal:point.proposal})}`);if(!response.ok)throw new Error('Source unavailable');const data=await response.json();if(request!==ontologyEvidenceRequest||!target.isConnected)return;target.querySelector('[data-source]').innerHTML=`<p class="note">${html(data.note)}</p>${data.parents.map(parent=>`<details open><summary>Diff from ${html(parent.candidate_id.slice(0,12))}</summary><pre>${html(!data.source_available||!parent.source_available?'Source missing; comparison is incomplete.':parent.diff||'No Python source difference.')}</pre></details>`).join('')}<details><summary>Candidate source</summary><pre>${html(Object.entries(data.source).map(([name,text])=>name+'\n'+text).join('\n\n'))}</pre></details>`;}catch(e){if(request===ontologyEvidenceRequest)target.querySelector('[data-source]').textContent=e.message;}}
const trajectoryDrawCharts=drawCharts;
drawCharts=function(id,payload){
  trajectoryDrawCharts(id,payload);drawOntology(id,payload);
  const chart=charts.get(`${id}-chart`),legend=document.getElementById(`${id}-ontology-family-key`);
  if(!chart)return;
  const individual=states[id].seriesMode==='runs'&&!interventionView(payload,states[id]);
  if(legend)legend.innerHTML=individual?'<b>Lines = runs / conditions · Marker color = ontology family</b><p>Family colors describe mechanisms, not performance or amount of change. Hover a proposal for its exact family and review status; click for source evidence.</p>':'<b>Summary / intervention view</b><p>Colors identify runs or conditions here. Switch to full trajectories with individual runs to see family-colored proposals.</p>';
  if(!individual)return;
  const families=new Set(),runFor=ds=>payload.runs.find(r=>ds.label===r.label||ds.label===r.label+' · raw outcomes');
  for(const ds of chart.data.datasets){
    const run=runFor(ds);if(!run)continue;
    for(const d of ds.data)if(d.point)families.add(ontologyFamily(ontologyReview(id,run,d.point)));
    ds.pointBorderColor=ctx=>ontologyFamilyColor(id,ontologyFamily(ontologyReview(id,run,ctx.raw?.point||{})));
    ds.pointBackgroundColor=ctx=>ctx.raw?.point?.valid===false?'#0d1117':ontologyFamilyColor(id,ontologyFamily(ontologyReview(id,run,ctx.raw?.point||{})));
    ds.pointBorderWidth=2;
    ds.pointStyle=ctx=>['changing','mixed'].includes(ontologyChanged(id,run,ctx.raw?.point||{}))?'triangle':pointStyle(ctx);
  }
  if(legend){
    legend.innerHTML+='<p>▲ changing / mixed edit · × invalid (unless marked ▲) · ● other proposals · gray = no assigned family. Retention and validity are shown on hover. Family IDs distinguish similar shades.</p><div class="ontology-family-swatches">'+[...families].sort().map(f=>{const style=ontologyFamilyStyle(id,f);return `<span><i style="background:${style.color}"></i><b>${html(style.tag)}</b> ${html(style.family)}</span>`;}).join('')+'</div>';
  }
  const callbacks=chart.options.plugins.tooltip.callbacks,previous=callbacks.afterLabel;
  callbacks.afterLabel=ctx=>{const run=runFor(ctx.dataset),point=ctx.raw?.point,row=run&&point?ontologyReview(id,run,point):null,style=ontologyFamilyStyle(id,ontologyFamily(row));return [...[].concat(previous?.(ctx)||[]),`Ontology family: ${style.tag} · ${style.family}`,`Ontology review: ${row?.classification||'unannotated'}`];};
  chart.options.onClick=(_event,elements)=>{const element=elements[0];if(!element)return;const ds=chart.data.datasets[element.datasetIndex],point=ds.data[element.index]?.point,run=runFor(ds);if(run&&point){ontologyView(id).run=run.run_id;ontologyView(id).selected=point.proposal;drawOntology(id,payload);document.getElementById(`${id}-ontology-evidence`).scrollIntoView({behavior:'smooth',block:'center'});}};
  chart.update('none');
};
const trajectoryRedraw=redraw;
redraw=function(id,payload){trajectoryRedraw(id,payload);if(!visibleSections.has(id))drawOntology(id,payload);};
const trajectoryDestroyCharts=destroySectionCharts;
destroySectionCharts=function(id){trajectoryDestroyCharts(id);for(const [key,chart] of charts)if(key.startsWith(`${id}-ontology-`)){chart.destroy();charts.delete(key);}};
function prepareOntologyLayout(id){
  const root=document.getElementById(`${id}-ontology`);
  if(!root)return;
  const trajectory=root.nextElementSibling;
  if(trajectory?.classList.contains('charts')){
    const notes=[];
    for(let node=trajectory.nextElementSibling;node?.classList.contains('note');node=node.nextElementSibling)notes.push(node);
    root.before(trajectory,...notes);
  }
  if(!document.getElementById(`${id}-ontology-family-key`)){
    const key=document.createElement('div');key.id=`${id}-ontology-family-key`;key.className='ontology-family-key';
    const chart=document.getElementById(`${id}-chart`);
    chart?.closest('.chart-card')?.append(key);
  }
  const view=ontologyView(id);
  view.panels??={};
  const trajectoryCard=document.getElementById(`${id}-chart`)?.closest('.chart-card');
  if(trajectoryCard&&trajectoryCard.tagName!=='DETAILS'){
    const panel=document.createElement('details');
    panel.className='chart-card ontology-disclosure';
    panel.open=view.panels.trajectory!==false;
    const summary=document.createElement('summary');
    const title=trajectoryCard.querySelector('.chart-head strong');
    summary.textContent=title?.textContent||'Trajectory chart';
    title?.remove();
    panel.append(summary);
    while(trajectoryCard.firstChild)panel.append(trajectoryCard.firstChild);
    trajectoryCard.replaceWith(panel);
    panel.addEventListener('toggle',()=>{
      view.panels.trajectory=panel.open;
      if(panel.open)requestAnimationFrame(()=>charts.get(`${id}-chart`)?.resize());
    });
  }
  root.querySelectorAll('.chart-card').forEach(card=>{
    const heading=card.querySelector('h3');
    if(!heading)return;
    const key=heading.textContent;
    const panel=document.createElement('details');
    panel.className='chart-card ontology-disclosure';
    panel.open=view.panels[key]===true;
    const summary=document.createElement('summary');
    summary.textContent=key;
    heading.remove();
    panel.append(summary);
    while(card.firstChild)panel.append(card.firstChild);
    card.replaceWith(panel);
    panel.addEventListener('toggle',()=>{
      view.panels[key]=panel.open;
      if(panel.open)requestAnimationFrame(()=>panel.querySelectorAll('canvas').forEach(canvas=>charts.get(canvas.id)?.resize()));
    });
  });
}
const trajectoryAttachSection=attachSection;
const ontologyAutoLoads=new Map();
const ontologyLocalOverrides=new Set();
async function loadSavedOntologyReviews(id,payload){
  const previous=ontologyAutoLoads.get(id);
  if(ontologyLocalOverrides.has(id)||previous?.pending||previous&&Date.now()-previous.checkedAt<10000)return;
  const campaign=campaignDefs.find(([key])=>key===id)?.[2];
  if(!campaign&&!payload.combined_all)return;
  const payloadRevision=payload.runs.map(run=>run.run_id+':'+run.points.map(point=>point.proposal+':'+point.candidate_id).join(',')).join('|');
  const knownRevision=previous?.payloadRevision===payloadRevision?previous.revision:'';
  const state={pending:true,checkedAt:Date.now(),revision:knownRevision||'',payloadRevision,sources:previous?.sources||new Map()};
  ontologyAutoLoads.set(id,state);
  try{
    if(payload.combined_all){
      const loaded=await Promise.allSettled(payload.source_campaigns.map(async source=>{
        const cached=state.sources.get(source.key);
        const response=await fetch(`/api/ontology/reviews?${new URLSearchParams({campaign:source.key,...(cached?.revision?{revision:cached.revision}:{})})}`,{cache:'no-store'});
        if(!response.ok)throw new Error(source.key);
        const result=await response.json();
        if(!result.unchanged)state.sources.set(source.key,result);
        return source.key;
      }));
      if(ontologyLocalOverrides.has(id))return;
      const candidates=new Map(payload.runs.flatMap(run=>run.points.map(p=>[`${run.run_id}:${p.proposal}`,p.candidate_id])));
      const rows=payload.source_campaigns.flatMap(source=>(state.sources.get(source.key)?.review?.rows||[]).map(row=>({...row,run_id:`${source.key}::${row.run_id}`,family:row.family&&row.family!=='unannotated'?`${source.key} · ${row.family}`:row.family,source_campaign_key:source.key}))).filter(row=>candidates.get(`${row.run_id}:${row.proposal}`)===row.candidate_id);
      const failed=loaded.filter(result=>result.status==='rejected').length;
      const doc=ontologyValidate({schema_version:'1.0',campaign:payload.campaign,rubric:`Source-campaign rubrics; family names remain campaign-specific${failed?`; ${failed} review sources unavailable`:''}`,rows},payload);
      ontologyReviews.set(id,doc);
      if(document.getElementById(`${id}-ontology`))redraw(id,payload);
      return;
    }
    const response=await fetch(`/api/ontology/reviews?${new URLSearchParams({campaign,...(knownRevision?{revision:knownRevision}:{})})}`,{cache:'no-store'});
    if(!response.ok)throw new Error('Saved review request failed');
    const {review,revision,unchanged}=await response.json();
    if(unchanged||ontologyLocalOverrides.has(id))return;
    if(!review)return;
    state.revision=revision||JSON.stringify([review.generated_at,review.review_pipeline_sha256,review.semantic_references,review.rows.length]);
    const candidates=new Map(payload.runs.flatMap(run=>run.points.map(p=>[`${run.run_id}:${p.proposal}`,p.candidate_id])));
    const rows=review.rows.filter(row=>candidates.get(`${row.run_id}:${row.proposal}`)===row.candidate_id);
    const doc=ontologyValidate({...review,rows},payload);
    ontologyReviews.set(id,doc);
    if(document.getElementById(`${id}-ontology`))redraw(id,payload);
  }catch(error){
    const status=document.getElementById(`${id}-ontology-status`);
    if(status)status.textContent='Saved review import failed: '+error.message;
  }finally{
    state.pending=false;state.checkedAt=Date.now();
  }
}
attachSection=function(id,payload){prepareOntologyLayout(id);trajectoryAttachSection(id,payload);const root=document.getElementById(`${id}-ontology`);if(!root)return;loadSavedOntologyReviews(id,payload);root.querySelector('[data-ontology="run"]').onchange=e=>{ontologyView(id).run=e.target.value;ontologyView(id).selected=null;drawOntology(id,payload);};root.querySelector('[data-ontology="mode"]').onchange=e=>{ontologyView(id).mode=e.target.value;drawOntology(id,payload);};root.querySelector('[data-ontology="cohort"]').onchange=e=>{ontologyView(id).cohort=e.target.value;drawOntology(id,payload);};root.querySelector('[data-ontology="template"]').onclick=()=>download('ontology-review-template.json','application/json',JSON.stringify(ontologyTemplate(payload),null,2));root.querySelector('[data-ontology="export"]').onclick=()=>download('ontology-reviews.json','application/json',JSON.stringify(ontologyReviews.get(id)||ontologyTemplate(payload),null,2));root.querySelector('[data-ontology="clear"]').onclick=()=>{ontologyLocalOverrides.add(id);ontologyReviews.delete(id);redraw(id,payload);};root.querySelector('[data-ontology="import"]').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const doc=ontologyValidate(JSON.parse(await file.text()),payload);ontologyLocalOverrides.add(id);ontologyReviews.set(id,doc);redraw(id,payload);}catch(error){document.getElementById(`${id}-ontology-status`).textContent='Review import failed: '+error.message;}};};
// Aggregate performance is normalized; ontology families keep their source-campaign namespace.
const trajectoryCampaignsFrom=campaignsFrom;
campaignsFrom=function(data){return trajectoryCampaignsFrom(data).filter(([_id,_title,p])=>!p?.combined_semantic||p.combined_all);};
