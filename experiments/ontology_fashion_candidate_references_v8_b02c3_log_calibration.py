"""Closed static exact P132 log-calibration precision profile."""
from __future__ import annotations
import difflib
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS
from experiments.ontology_fashion_candidate_references_v6_b02c3_crop_fusion import BASE
RUN='fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b02-c3'
SHA='02939a5e6eca3562d9eda71e0313fe081c12f887ca96d0bba2ba91957452505b'
CHILD='b7b236889ed94f4970b067691f906be8cd56bfd045ec63c625434c09afa0f143'
PARENT_SHA='f9d29eb967a68fecca60729b66035a1791408190b0109671f3790d8cc1f2b896'
PARENT='920a2c55a7a34c273f29da436b23e5ddc2b4c107858a46e819e661fc9f043a6b'
def profiles():
 child,parent=source(ROOT/RUN/'candidates'/CHILD),source(ROOT/RUN/'candidates'/PARENT)
 if digest(child)!=SHA or digest(parent)!=PARENT_SHA: raise ValueError('static source binding changed')
 fp=dict(BASE); fp['routing']='fixed translated and horizontal-flip views with fixed weighted powered-probability fusion'; fp['aggregation']='flattened spatial features plus fixed weighted powered crop probabilities'; fp['output']='ten-class log probabilities after fixed inverse-power calibration'
 if [k for k in CORE+TASK_KEYS['fashion'].split() if not fp.get(k)]: raise ValueError('incomplete fingerprint')
 diff=list(difflib.unified_diff(parent['train.py'].splitlines(),child['train.py'].splitlines()))
 return [{'source_sha256':SHA,'reference_run':RUN,'reference_proposal':132,'candidate_id':CHILD,'parent_candidate_id':PARENT,'parent_source_sha256':PARENT_SHA,'transition_classification':'preserving','changed_components':[],'fingerprint':fp,'family_signature':{'task':'fashion','mixing':'local gated convolution','routing':'fixed crop/flip fusion with float64 log-calibration precision','readout':'flattened affine logits with calibrated log conversion'},'family_label':'gated CNN with precise log-calibration inference','training':{'optimizer':'AdamW (declared)'},'inference':{'procedure':'fixed translated/flip fusion with float64 log-calibration conversion'},'notes':'Complete static exact retained parent and child train.py sources were directly read and diffed; changes only post-fusion log-calibration precision.','evidence':[{'kind':'complete static exact-program directed parent-child diff review','source_sha256':SHA,'parent_source_sha256':PARENT_SHA,'candidate_id':CHILD,'retained_parent_candidate_id':PARENT,'directed_parent_source_sha256':PARENT_SHA,'classification':'preserving','changed_components':[],'changed_lines':{'added':sum(x.startswith('+') and not x.startswith('+++') for x in diff),'removed':sum(x.startswith('-') and not x.startswith('---') for x in diff)}}]}]

