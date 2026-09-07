"""Closed static exact P138 orientation-weight precision profile."""
from __future__ import annotations
import difflib
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS
from experiments.ontology_fashion_candidate_references_v6_b02c3_crop_fusion import BASE
RUN='fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b02-c3'
SHA='36e5e0247f0a8108c211cf57560300a1235c37bd9c583bbc1eaeddb21da4630e'
CHILD='026fd4b787c7111f830d7bdb8fce522cb1986207f3369a82159c9f1fd7597993'
PARENT_SHA='62aba06a4f3f4ca9fcf6de2d158f479342350c9762da7ec8bd1524cdfa32a65f'
PARENT='1aa9a6a954b8eba0495539055cd77332a407f43264c45ac0060e219502196165'
def profiles():
 child,parent=source(ROOT/RUN/'candidates'/CHILD),source(ROOT/RUN/'candidates'/PARENT)
 if digest(child)!=SHA or digest(parent)!=PARENT_SHA: raise ValueError('static source binding changed')
 fp=dict(BASE); fp['routing']='fixed translated and horizontal-flip views with fixed weighted powered-probability fusion'; fp['aggregation']='flattened spatial features plus fixed weighted powered crop probabilities'; fp['output']='ten-class log probabilities after fixed inverse-power calibration'
 if [k for k in CORE+TASK_KEYS['fashion'].split() if not fp.get(k)]: raise ValueError('incomplete fingerprint')
 diff=list(difflib.unified_diff(parent['train.py'].splitlines(),child['train.py'].splitlines()))
 return [{'source_sha256':SHA,'reference_run':RUN,'reference_proposal':138,'candidate_id':CHILD,'parent_candidate_id':PARENT,'parent_source_sha256':PARENT_SHA,'transition_classification':'preserving','changed_components':[],'fingerprint':fp,'family_signature':{'task':'fashion','mixing':'local gated convolution','routing':'fixed crop/flip fusion with orientation-specific scalar weights','readout':'flattened affine logits with orientation-weighted probability conversion'},'family_label':'gated CNN with orientation-weighted fusion inference','training':{'optimizer':'AdamW (declared)'},'inference':{'procedure':'fixed translated/flip fusion with orientation-specific native/reflected weighted conversion'},'notes':'Complete static exact retained parent and child train.py sources were directly read and diffed; changes only fixed native/reflected orientation weights and log calibration precision.','evidence':[{'kind':'complete static exact-program directed parent-child diff review','source_sha256':SHA,'parent_source_sha256':PARENT_SHA,'candidate_id':CHILD,'retained_parent_candidate_id':PARENT,'directed_parent_source_sha256':PARENT_SHA,'classification':'preserving','changed_components':[],'changed_lines':{'added':sum(x.startswith('+') and not x.startswith('+++') for x in diff),'removed':sum(x.startswith('-') and not x.startswith('---') for x in diff)}}]}]


