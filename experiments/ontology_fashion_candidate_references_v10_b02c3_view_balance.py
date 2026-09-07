"""Closed static exact P141 native/reflected view balance profile."""
from __future__ import annotations
import difflib
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS
from experiments.ontology_fashion_candidate_references_v6_b02c3_crop_fusion import BASE
RUN='fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b02-c3'
SHA='83d76ef27519db98aaceb3651005ec3e1a10d59694a88f8489a414043d2ccd7a'
CHILD='c9c176e73fd1b40c02caaa1add2d54a9f636825638246d359e92cdcfa50dfd02'
PARENT_SHA='a8a8513c3cef44520ade85dcb0ffd80a98aca15077a342643610265125fd5745'
PARENT='3fdddf66e85a1ca12135be772ca641e60cd19df5880203e3698821aa8dd50ac5'
def profiles():
 child,parent=source(ROOT/RUN/'candidates'/CHILD),source(ROOT/RUN/'candidates'/PARENT)
 if digest(child)!=SHA or digest(parent)!=PARENT_SHA: raise ValueError('static source binding changed')
 fp=dict(BASE); fp['routing']='fixed translated and horizontal-flip views with fixed weighted powered-probability fusion'; fp['aggregation']='flattened spatial features plus fixed weighted powered crop probabilities'; fp['output']='ten-class log probabilities after fixed inverse-power calibration'
 if [k for k in CORE+TASK_KEYS['fashion'].split() if not fp.get(k)]: raise ValueError('incomplete fingerprint')
 diff=list(difflib.unified_diff(parent['train.py'].splitlines(),child['train.py'].splitlines()))
 return [{'source_sha256':SHA,'reference_run':RUN,'reference_proposal':141,'candidate_id':CHILD,'parent_candidate_id':PARENT,'parent_source_sha256':PARENT_SHA,'transition_classification':'preserving','changed_components':[],'fingerprint':fp,'family_signature':{'task':'fashion','mixing':'local gated convolution','routing':'fixed crop/flip fusion with native/reflected view-balance scalar weights','readout':'flattened affine logits with view-balance weighted probability conversion'},'family_label':'gated CNN with native/reflected view-balance inference','training':{'optimizer':'AdamW (declared)'},'inference':{'procedure':'fixed translated/flip fusion with fixed native/reflected weighted conversion'},'notes':'Complete static exact retained parent and child train.py sources were directly read and diffed; changes only fixed native/reflected view-balance weights only.','evidence':[{'kind':'complete static exact-program directed parent-child diff review','source_sha256':SHA,'parent_source_sha256':PARENT_SHA,'candidate_id':CHILD,'retained_parent_candidate_id':PARENT,'directed_parent_source_sha256':PARENT_SHA,'classification':'preserving','changed_components':[],'changed_lines':{'added':sum(x.startswith('+') and not x.startswith('+++') for x in diff),'removed':sum(x.startswith('-') and not x.startswith('---') for x in diff)}}]}]



