"""Closed exact B02-C3 crop-fusion lineage; no source-only admission."""
from __future__ import annotations
import difflib, json
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS

RUN="fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b02-c3"
SPECS={
"4202423a7bd0e0bfd2d0711f2f225a7df8afa6c78de6ae276126b0b5cc4b6212":("768592717b905a1ee2a561e5e52add4d289134693b1600b8831dcde2142237b0","1d750b7d47f6caececbfe18e8913d22039696140e8de445c5536bcb8c4998dc1","3a653f7becea7a2066ef729f8fac75d7c6f5e744a888e554289ab6d4035f4e6d",139,"expands fixed cardinal crop fusion to a fixed nine-view translated ensemble"),
"0d04a185c85e5e2907691c001093b701dd96d55477881fa1c40c5f814a558882":("a8776b0f165d101b3256e63cd0f114e835befc5ea0822e93201f2de1c4be11e6","4202423a7bd0e0bfd2d0711f2f225a7df8afa6c78de6ae276126b0b5cc4b6212","768592717b905a1ee2a561e5e52add4d289134693b1600b8831dcde2142237b0",142,"adds fixed diagonal views and fixed orientation-specific fusion weights"),
"06a37bc451d13d5529f5260d3454fab0822a1f8b0c0ce75f7e986a529a8d762f":("98f8cbf5953445d1b2750e679f710f361adbb53618c9b73184bd6a4629242c62","4202423a7bd0e0bfd2d0711f2f225a7df8afa6c78de6ae276126b0b5cc4b6212","768592717b905a1ee2a561e5e52add4d289134693b1600b8831dcde2142237b0",148,"replaces linear fixed crop probability fusion with fixed powered probability fusion"),}
BASE={"input_units":"grayscale image pixels","input_transform":"identity image tensor","embedding":"not applicable: pixels enter convolution directly","position":"implicit local image neighborhoods","mixing":"dense local Conv2d with depthwise-plus-pointwise refinement","routing":"fixed convolutional flow and fixed translated/flip inference views","state":"no persistent recurrent state","feedforward":"no transformer feedforward block","parameter_construction":"free learned convolutional and affine parameters","sharing":"modules reused only as explicitly called","normalization":"BatchNorm2d","connectivity":"fixed sequential convolution/pooling and affine head","aggregation":"flattened spatial features plus fixed multi-view probability fusion","output":"ten-class log probabilities","symmetry":"fixed translated and horizontal-flip view ensemble","conditional_compute":"fixed architecture","activation":"GELU and sigmoid gate","stochasticity":"Dropout during training","bottleneck":"no factorized bottleneck","iteration":"fixed feedforward stages","other":"complete exact queue-bound parent/child programs directly diffed","spatial_units":"image grid","spatial_operator":"dense and depthwise local Conv2d","scale_representation":"two MaxPool2d reductions","spatial_readout":"flattened spatial affine head","channel_interaction":"depthwise-pointwise refinement and sigmoid channel gate","spatial_downsampling":"two MaxPool2d(2) operations"}
def profiles():
 out=[]
 for sha,(cid,psha,pid,proposal,reason) in SPECS.items():
  child,parent=source(ROOT/RUN/'candidates'/cid),source(ROOT/RUN/'candidates'/pid)
  if digest(child)!=sha or digest(parent)!=psha: raise ValueError('source binding changed: '+cid)
  fp=dict(BASE); fp['routing']=BASE['routing']+'; '+reason
  if [k for k in CORE+TASK_KEYS['fashion'].split() if not fp.get(k)]: raise ValueError('incomplete fingerprint')
  diff=list(difflib.unified_diff(parent['train.py'].splitlines(),child['train.py'].splitlines()))
  family={'task':'fashion','mixing':'local gated convolution','routing':'fixed cardinal translated/flip probability fusion','readout':'flattened affine logits'}
  if proposal==139: family['routing']='fixed nine-view translated/flip probability fusion'; family['readout']='linear weighted nine-view probability logits'
  elif proposal==142: family['routing']='fixed nine-view orientation-weighted translated/flip probability fusion'; family['readout']='orientation-weighted nine-view probability logits'
  else: family['routing']='fixed powered translated/flip probability fusion'; family['readout']='inverse-power fused probability logits'
  out.append({'source_sha256':sha,'reference_run':RUN,'reference_proposal':proposal,'candidate_id':cid,'parent_candidate_id':pid,'parent_source_sha256':psha,'transition_classification':'changing','changed_components':['routing','aggregation','output'],'fingerprint':fp,'family_signature':family,'family_label':'gated CNN with fixed crop-fusion inference','training':{'optimizer':'AdamW (declared)'},'inference':{'procedure':'fixed translated and horizontal-flip probability fusion'},'notes':'Complete exact retained parent and child train.py sources were directly read and diffed. '+reason+'.','evidence':[{'kind':'complete exact-program queue-bound parent-child diff review','source_sha256':sha,'parent_source_sha256':psha,'candidate_id':cid,'retained_parent_candidate_id':pid,'directed_parent_source_sha256':psha,'classification':'changing','changed_components':['routing','aggregation','output'],'reason':reason,'changed_lines':{'added':sum(x.startswith('+') and not x.startswith('+++') for x in diff),'removed':sum(x.startswith('-') and not x.startswith('---') for x in diff)}}]})
 return out



