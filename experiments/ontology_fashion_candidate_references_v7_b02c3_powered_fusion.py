"""Closed static exact P84 powered crop-fusion profile."""
from __future__ import annotations
import difflib
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS
from experiments.ontology_fashion_candidate_references_v6_b02c3_crop_fusion import BASE
RUN='fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b02-c3'
SHA='d9b1fd37aaa8a584d386c6b7dc8beb8f1abc6e6d6211b3fca9ec5a639bf3a587'
CHILD='370f15cba4754adf4de64a301de6428a12a5d0daac8f9b62932e819dc0b602a4'
PARENT_SHA='332810a0c6ad58fac4c88f18d4320fb7382bb5eca2119e657741ec3be74370ec'
PARENT='64898e9b345fe342981c46b4f39144b14aa90705f2c7037058fc0e44811f705c'
def profiles():
 child,parent=source(ROOT/RUN/'candidates'/CHILD),source(ROOT/RUN/'candidates'/PARENT)
 if digest(child)!=SHA or digest(parent)!=PARENT_SHA: raise ValueError('static source binding changed')
 fp=dict(BASE); fp['routing']='fixed translated and horizontal-flip views with fixed weighted powered-probability fusion'; fp['aggregation']='flattened spatial features plus fixed weighted powered crop probabilities'; fp['output']='ten-class log probabilities after fixed inverse-power calibration'
 if [k for k in CORE+TASK_KEYS['fashion'].split() if not fp.get(k)]: raise ValueError('incomplete fingerprint')
 diff=list(difflib.unified_diff(parent['train.py'].splitlines(),child['train.py'].splitlines()))
 return [{'source_sha256':SHA,'reference_run':RUN,'reference_proposal':84,'candidate_id':CHILD,'parent_candidate_id':PARENT,'parent_source_sha256':PARENT_SHA,'transition_classification':'changing','changed_components':['routing','aggregation','output'],'fingerprint':fp,'family_signature':{'task':'fashion','mixing':'local gated convolution','routing':'fixed powered crop/flip probability fusion','readout':'flattened affine logits'},'family_label':'gated CNN with fixed powered crop-fusion inference','training':{'optimizer':'AdamW (declared)'},'inference':{'procedure':'fixed weighted translated/flip powered-probability fusion'},'notes':'Complete static exact retained parent and child train.py sources were directly read and diffed; adds fixed powered probability fusion.','evidence':[{'kind':'complete static exact-program directed parent-child diff review','source_sha256':SHA,'parent_source_sha256':PARENT_SHA,'candidate_id':CHILD,'retained_parent_candidate_id':PARENT,'directed_parent_source_sha256':PARENT_SHA,'classification':'changing','changed_components':['routing','aggregation','output'],'changed_lines':{'added':sum(x.startswith('+') and not x.startswith('+++') for x in diff),'removed':sum(x.startswith('-') and not x.startswith('---') for x in diff)}}]}]
