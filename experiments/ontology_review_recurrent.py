"""Task-aware, source-traced review of HAR and streaming keyword architectures.

The comparison is asymmetric: a witnessed mechanism difference can resolve a
transition without claiming that unrelated custom code has a complete rubric.
No fallback turns an unmatched edit into either a changing or preserving label.
Candidate source is parsed, never imported or executed.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from experiments.ontology_semantics import program, constant, UNKNOWN, PRIMITIVES, SKIP_METHODS
from experiments.ontology_transition_review import settings_signature, fixed_schedule
from experiments.ontology_affine_readout import affine_readout_signature
from experiments.review_ontology_sources import CORE, TASK_KEYS, dotted, analyze

VERSION = 'recurrent-semantic-v3'
CAMPAIGNS = {'uci_har_pareto_v21':'har', 'openevolve_v21_tiny_kws_rnn':'kws'}
RECURRENT = {'GRU','GRUCell','LSTM','LSTMCell','RNN','RNNCell'}
NORMALIZERS = {'LayerNorm','RMSNorm','BatchNorm1d','BatchNorm2d','GroupNorm','InstanceNorm1d','Identity'}
SHAPE = {'contiguous','transpose','permute','unsqueeze','squeeze','reshape','view','flatten','expand','expand_as','to','float','type_as','detach','clone'}
REDUCTIONS = {'mean','sum','amax','amin','max','min','std','var','norm','argmax','argmin'}
UNARY = {'square','sqrt','rsqrt','abs','sin','cos','exp','log','log1p','tanh','sigmoid','relu','gelu','silu','softmax','log_softmax','clamp','clamp_min','clamp_max','pow'}
EXTRA_PRIMITIVES={'AdaptiveMaxPool1d','AdaptiveMaxPool2d','BatchNorm2d','InstanceNorm1d','LeakyReLU','PReLU','ELU','Softmax'}
TENSOR_FUNCTIONS={'where','arange','linspace','pad','flip','roll','einsum','matmul','bmm','mm','avg_pool1d','max_pool1d','adaptive_avg_pool1d','adaptive_max_pool1d','avg_pool2d','interpolate','rfft','fft','irfft','rfft2','all','any','remainder','div','quantile','sort','topk','argsort','triu_indices','eye','tensor','as_tensor','dot','mv','one_hot','nonzero','split','chunk','cumsum','cummax','repeat_interleave'}

# Full-source reviews of the retained B01-C2 KWS branch.  These records use
# whole bundle hashes, so a nearby proposal with a similar diff cannot inherit
# the decision.  The parser supplies the complete candidate fingerprint at
# review time; the entries below only record the source-audited transition.
KWS_B01_C2_SOURCE_REVIEWS={
    '488f3a2641db35f200604e08fb5849c5e1b4f320ae85708cde61190051440f82':(
        'ad99f7684ee2e65dd504bb8caf2007675d7c919ea52e9d7e2a8fda28008a45b8','preserving',[],26,
        'The source combines reviewed normalization-coordinate removal, fixed reference-logit quotient construction and numeric frame schedule settings. It retains the same normalized GRU recurrence and mean/max/final-state readout mechanism.'),
    '451b739304daa31bd966800b22745c3dcb5ef5614c0691c4f60f932176b34d92':(
        '2d53f4c72b0c92d6507ec9b00c13af6b107af3a79edb287755d94da422e4313d','preserving',[],26,
        'The source uses the reviewed non-affine LayerNorm coordinate convention and fixed reference-logit quotient. Its changed frame positions are numeric schedule settings; recurrence and all observed history statistics remain the same.'),
    'e221a1ca61651969bee522c41d728aac5ddff7ac2d2c2365fe04150aa27eca87':(
        'e1da1eddaadddb3fa4d671ebbcfd6b6be9b676f96ebc79ac89c736c826dadbdd','preserving',[],26,
        'The source combines reviewed normalization-coordinate and reference-logit changes with a fixed numeric schedule. It does not add a recurrent state, acoustic transform or output-history observation.'),
    '21a602bb266d17ceb0e739b268047f977e2c0102a516733c2df3f23755109554':(
        '2d53f4c72b0c92d6507ec9b00c13af6b107af3a79edb287755d94da422e4313d','preserving',[],26,
        'The source combines the reviewed non-affine LayerNorm and fixed reference-logit construction. The altered literal frame schedule changes only numeric sampling positions under the retained KWS schedule rubric.'),
    '772b42af8929063fd44a8f2930e1ca87fc933d0f4ab32bb04fdaaf96948096ec':(
        '2d53f4c72b0c92d6507ec9b00c13af6b107af3a79edb287755d94da422e4313d','preserving',[],25,
        'The source combines a hidden-width setting with reviewed non-affine LayerNorm, fixed reference-logit construction and numeric schedule settings. The causal GRU state and mean/max/final-state readout are unchanged.'),
    '9d7be1ec01f0045043d9f2a05167ffce2038514de9ab21d14e918e00eac8e388':(
        '909d967ab02502b19e3583c6eecbaa5ee5d87cf234f7d3fd324ad780c781eb29','preserving',[],101,
        'The candidate replaces separate final-state and mean-history affine inputs with their sum before the same affine head. Both source programs observe the same mean, max and final GRU history; this is a readout coordinate reparameterization, not a new history statistic.'),
    '8fc95130c2db1d04054615a254802a4b3ff95e6326781f2adfcfb1168781d5e7':(
        'ee40480e428d317ef1b920f649f8d84eff6e63d7ed4764ad187ae06db423d36e','preserving',[],85,
        'The deployed GRU state, mean/max/final-state readout and exit observables are unchanged. The added private cache is written only while training and reuses that existing penultimate readout for an auxiliary loss; its scalar confidence threshold remains in the same reviewed exit family.'),
    '2f3ba788ea6a793ee6a87672bc38635ca40db98b247aa36e7d39685a0e34a56a':(
        'b27e80d5b63d4942f839ef698b734b2e3730765b6b4b0b3cbf41bd33439a6bc0','changing',['acoustic_representation'],62,
        'The last two normalized acoustic coordinates are replaced by their per-frame mean before the GRU. This changes the signal representation supplied to the recurrent input matrices.'),
    '0f2252d6f6fa7f0e2694178904d860b70ae137e02051c1200a1b7ac70f646482':(
        'db397a8bc39fd3a5003989cc12eaab8cb02e13bd6f95ae8c80eacf2e56a45a01','changing',['exit_policy','conditional_compute'],73,
        'The candidate adds an antepenultimate confidence exit and makes logits available at that earlier clock position. The private training cache is auxiliary, but the earlier inference termination is a distinct observed exit policy.'),
    'ef52049ac018cc4f43645c0af51ce189a2f2dd6c51741c6340005df49b042add':(
        '9cb5d1d26efd643f5f9969693546637ab30db821e780d75f613ea4e3eccf8c04','changing',['exit_policy','conditional_compute'],73,
        'The candidate adds an antepenultimate confidence exit and makes logits available at that earlier clock position. The auxiliary cache does not erase this changed inference termination mechanism.'),
    '1b2592fe050ff4eede0d3857b84ff9521a65d248aba43c8138bfbe7d2a7f45ff':(
        'db397a8bc39fd3a5003989cc12eaab8cb02e13bd6f95ae8c80eacf2e56a45a01','changing',['exit_policy','conditional_compute'],73,
        'The candidate introduces a confidence-gated antepenultimate exit. It changes when the streaming classifier may terminate, independently of the auxiliary training readout.'),
    '9b1d0293e0a0674dce273bdd765b71c05ed81ce5f415e26d1a1d9d3bc29adf32':(
        'db397a8bc39fd3a5003989cc12eaab8cb02e13bd6f95ae8c80eacf2e56a45a01','changing',['exit_policy','conditional_compute','output'],73,
        'The candidate adds a confidence-gated antepenultimate exit and scales the partial-horizon logits. Both are deployed inference changes, separate from its auxiliary cache.'),
    'dafc3baf2f0b4d5f5b6c411e045de060818d5827635eca869134ff4d6f9417e3':(
        'b27e80d5b63d4942f839ef698b734b2e3730765b6b4b0b3cbf41bd33439a6bc0','changing',['exit_policy','conditional_compute'],73,
        'The candidate adds a confidence-gated antepenultimate exit. This supplies a new earlier deployed termination path.'),
    'f3cddd37086fa47846c14f9d1a8d4fc8aebbb52c797f3f7fd4a247518f7ca978':(
        'db397a8bc39fd3a5003989cc12eaab8cb02e13bd6f95ae8c80eacf2e56a45a01','changing',['exit_policy','conditional_compute'],73,
        'The candidate adds a confidence-gated antepenultimate exit, changing the streaming termination policy even though its GRU and final readout remain familiar.'),
    '3cf3ede915f1651139c14207d8cae702ae07d09e6fd4836b0d46c7db409a2a13':(
        '9cb5d1d26efd643f5f9969693546637ab30db821e780d75f613ea4e3eccf8c04','preserving',[],28,
        'A free learned affine projection is inserted directly before the library GRU on both streaming and sequence paths. It is absorbed into the GRU input matrices and bias under the reviewed affine-input rule; no state update or history observation changes.'),
    'd3fdb79f12fd67c8167136b3525afe948590b1401dc4fa2646565eb60440edc8':(
        '7a751a6c5770a1961802dc3cdce065a806dd38a971db107eb90628c6adf32004','preserving',[],31,
        'Only the existing affine classifier bias and the scalar confidence threshold change. The recurrent update, history readout and exit observables are otherwise source-identical.'),
    '2bb2c3641880bc686521358826a9de37716f7d37d0266311c14a45f501454d60':(
        'b27e80d5b63d4942f839ef698b734b2e3730765b6b4b0b3cbf41bd33439a6bc0','preserving',[],31,
        'Only the existing affine classifier bias and scalar confidence threshold change. The source retains the same recurrent input representation, state update, readout and exit observables.'),
    '227ce9d48fef6e7c5ccd92d440afdadb37c199bceadba8b185d8b4397d131081':(
        'db397a8bc39fd3a5003989cc12eaab8cb02e13bd6f95ae8c80eacf2e56a45a01','preserving',[],31,
        'Only the existing affine classifier bias and scalar confidence threshold change. The source retains the same recurrent update and antepenultimate-cache-free deployed inference path.'),
    '95ae6eaf82b6dcd63375d1a341fa31d1c38c2e7cd2ede4c7635f2b62231aeebf':(
        '9cb5d1d26efd643f5f9969693546637ab30db821e780d75f613ea4e3eccf8c04','preserving',[],31,
        'Only the existing affine classifier bias and scalar confidence threshold change. The recurrent update, readout and deployed exit observables remain source-identical.'),
    'aa4fb4f4df4e6b5960c99e282d7c2289ed6f3a74740c4d44e48cd6ef14b5bcf1':(
        '7a751a6c5770a1961802dc3cdce065a806dd38a971db107eb90628c6adf32004','preserving',[],31,
        'Only the existing affine classifier bias and scalar confidence threshold change. The recurrent update, history readout and exit observables are otherwise source-identical.'),
    # The B01-C2 confidence-threshold lineage changes only the scalar at
    # ``exit_mask`` line 152.  Each entry deliberately binds both whole-source
    # hashes: a nearby confidence comparison, new observable, or different
    # parent must take the ordinary source-review path.
    '2d0809509cdf1364452af9254b124e1930a122057279998b8b8972b5586b30b8':(
        'aa4fb4f4df4e6b5960c99e282d7c2289ed6f3a74740c4d44e48cd6ef14b5bcf1','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '3527ca48d3a6517de6bc6feed38e82c14e3691c3ca137939965e44da9c74e679':(
        '95ae6eaf82b6dcd63375d1a341fa31d1c38c2e7cd2ede4c7635f2b62231aeebf','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '4208ab87ca9460450021f7c0a7713e2d6ffaa2ca917028430543b1974739e8a7':(
        '227ce9d48fef6e7c5ccd92d440afdadb37c199bceadba8b185d8b4397d131081','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'c3b1104a07658bd1cf48daafcb5866c83fcb9ba2a998ab7d841f947e72cff31b':(
        '4208ab87ca9460450021f7c0a7713e2d6ffaa2ca917028430543b1974739e8a7','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '0336a376ae2d5dec4552c9422154b4dff268e69b518b592c2635f0f8a82d32c6':(
        '3527ca48d3a6517de6bc6feed38e82c14e3691c3ca137939965e44da9c74e679','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '7968eaf92900889f75a7f0d183c5c97a89de5f6b358d347f789cf5e4c2a2bfaf':(
        '2d0809509cdf1364452af9254b124e1930a122057279998b8b8972b5586b30b8','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '38c016519c0e8657d8dbdcc24e73102358b0fabb09dc2559849c49bd24cb6444':(
        '7968eaf92900889f75a7f0d183c5c97a89de5f6b358d347f789cf5e4c2a2bfaf','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'a46165e36c6b19d95b4215207716b1a737f5475e67550ed1a9a0f06b764ab08a':(
        'c3b1104a07658bd1cf48daafcb5866c83fcb9ba2a998ab7d841f947e72cff31b','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'f67938866248e696de565d0264a77e2ce1c45a8399ae6974bd082fa643f09282':(
        '0336a376ae2d5dec4552c9422154b4dff268e69b518b592c2635f0f8a82d32c6','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '9c64a59865611c1eb9f8608de5e6b41e43affc9eba42d798025c93dab36bfbac':(
        '38c016519c0e8657d8dbdcc24e73102358b0fabb09dc2559849c49bd24cb6444','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'c5076bd877a6653c1f37c1386d96bc1c6939c2e6c8a1d747e782750067656d34':(
        'a46165e36c6b19d95b4215207716b1a737f5475e67550ed1a9a0f06b764ab08a','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '481a1670cbc9e17d49a2c23d653ccd071773aeda66e28dd7900c1d222296f93d':(
        'f67938866248e696de565d0264a77e2ce1c45a8399ae6974bd082fa643f09282','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '284687a1b657f3a773762e2dff8e4e0baaad9504e0eaca57c322d5270814f3eb':(
        '481a1670cbc9e17d49a2c23d653ccd071773aeda66e28dd7900c1d222296f93d','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670':(
        '38c016519c0e8657d8dbdcc24e73102358b0fabb09dc2559849c49bd24cb6444','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'ab413512e0f7b99cee9fe0768eaa29e3c2ae06ac1fd46f10007b49ae5d14f6da':(
        'a46165e36c6b19d95b4215207716b1a737f5475e67550ed1a9a0f06b764ab08a','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '723f2d0d537879c57748cdc4a54eec1556257ced6313a0ef87cc13b61f4456c3':(
        'ab413512e0f7b99cee9fe0768eaa29e3c2ae06ac1fd46f10007b49ae5d14f6da','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084':(
        '284687a1b657f3a773762e2dff8e4e0baaad9504e0eaca57c322d5270814f3eb','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '65116b765a0d12c8d0f2ae9dc6e3ff3c93cc92280343bbd1fe5b641c8e4696d7':(
        '449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '78e46556470466eef54f9ae709078c2a0e18aa5be8b51a217db3180ad909dac4':(
        'c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    'be0c4a7b368089c2a79124f040fa2d5a883393a6b320609dca4eb97039873fea':(
        '449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '7ca19048bbb39b3d194cf6850fbd44a00f2a8b2f51d80a5031527744347b8177':(
        'c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '6ac267c7dac30725f6d82cd0a583e2f4c36d358e71cf53de4ca73bc8e6325147':(
        '723f2d0d537879c57748cdc4a54eec1556257ced6313a0ef87cc13b61f4456c3','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '6e75e57d7537e0bb47cadf04e254bfbee6efa2acff13ba6163ce08e7906fab01':(
        '449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '0f561412026ca710fc887e44d1a4d3efee1283532fa95cbb4dee3143adbf4bc6':(
        'c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '17a4acec7cda13b10b9a125dc0457410a85b7cc1647a05099ee6f478acbcd4c3':(
        '723f2d0d537879c57748cdc4a54eec1556257ced6313a0ef87cc13b61f4456c3','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '7b4c10d699d9bb2b81399c8250c754ba9cd1c26dbf1ae110c83ec345f139c180':(
        '6e75e57d7537e0bb47cadf04e254bfbee6efa2acff13ba6163ce08e7906fab01','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '82ceddef8fb674f28fb13eec7c1503a328a1abbaf180cddc32582e75ef58d78a':(
        'c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','preserving',[],152,
        'Only the scalar confidence threshold of the existing final-frame posterior exit changes. The posterior observable, horizon test, recurrent state, readout and output are source-identical.'),
    '7d657a2e0075391827c410e99b04f9edc331af99ea365644d4a0a30417e29f2f':(
        '5d28ae1e2ce19f027328eeb0b4f97ed925de82fb7ddc2d8b1defedbe6aa1b87e','changing',['routing','conditional_compute','output'],132,
        'The candidate adds a copied specialist affine head and applies it only to an input-dependent narrow posterior region, where it can swap or zero selected output rows. This is a deployed conditional routing and output mechanism. Its dynamic indexed updates remain intentionally explicit in the residual fingerprint.'),
}

# The first member of the B01-C0 deployment-coordinate-pruning cohort is an
# independently read parent-child review.  Later members use the bounded
# parser contract above, but this record remains exact-pair-bound: knowing the
# child architecture never transfers this transition label to another parent.
KWS_B01_C0_COORDINATE_PRUNING_SOURCE_REVIEWS={
    'e102e8f38de7488298b05c918e3897f3aca8efcb03f0fa10d6e9d418c80ef977':(
        'c8b002b806387653e94ef82395f862f54ac7cd5d01fb1185ca22c9923ffd0d23',
        'preserving', [], 32,
        'The inference-only affine head is synchronized from the same seven reference-logit contrasts, then selects parameter-ranked existing readout coordinates. The source-checked pruning contract proves that no observation, recurrent state, or new readout statistic enters this deployment path.'),
}


def source_sha(sources):
    return hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()


def module_call_name(node,config):
    if isinstance(node,ast.Subscript):
        parent=dotted(node.value);index=constant(node.slice,config)
        if parent and type(index) is int:return parent+'['+str(index)+']'
    return dotted(node)


def _module_topology(spec):
    """A structural module record for family comparison, not a width setting.

    Module multiplicity and the temporal operator's literal geometry determine
    which channels and time locations can communicate.  They used to be
    collapsed into sets, which incorrectly joined distinct HAR architectures.
    Learned widths/biases deliberately remain outside this record.
    """
    kind=spec['kind']
    result=['module',kind]
    if kind.startswith('Conv'):
        result.extend(('communication',spec.get('communication','unknown')))
        for key in ('kernel_size','stride','dilation','padding'):
            result.extend((key,spec.get(key,'unspecified')))
    elif 'Pool' in kind:
        for key in ('kernel_size','stride','dilation','output_size'):
            if key in spec:result.extend((key,spec[key]))
    elif kind in RECURRENT:
        result.extend(('direction',spec.get('direction','forward')))
    return tuple(result)


def _forward_topology(method,modules,config):
    """Keep source order for active modules and nonlinearities in one method."""
    trace=[]
    for node in ast.walk(method):
        if not isinstance(node,ast.Call):continue
        name=module_call_name(node.func,config)
        if name in modules:
            trace.append(_module_topology(modules[name]));continue
        tail=dotted(node.func).split('.')[-1]
        if tail in UNARY:trace.append(('activation',tail))
    return tuple(trace)


def _walk(value):
    seen=set();pending=[value]
    while pending:
        item=pending.pop()
        if isinstance(item,tuple):
            if id(item) in seen:continue
            seen.add(id(item))
            if len(seen)>12000:raise Unsupported('tensor graph exceeds bounded review size')
            pending.extend(v for v in reversed(item[1:]) if isinstance(v,tuple))
        yield item


def _contains(value,label):
    return any(isinstance(n,tuple) and n and n[0]==label for n in _walk(value))


def _semantic_graph(value):
    """Canonical expression DAG, preserving labels/edges without expansion."""
    nodes=[];by_object={};by_content={}
    def encode(x):
        if isinstance(x,tuple):
            if id(x) in by_object:return {'node':by_object[id(x)]}
            if len(nodes)>12000:raise Unsupported('tensor graph exceeds bounded review size')
            content=[encode(v) for v in x]
            key=json.dumps(content,sort_keys=True,separators=(',',':'))
            if key not in by_content:by_content[key]=len(nodes);nodes.append(content)
            by_object[id(x)]=by_content[key]
            return {'node':by_content[key]}
        if isinstance(x,list):return [encode(v) for v in x]
        if isinstance(x,dict):return {k:encode(v) for k,v in sorted(x.items())}
        return x
    root=encode(value)
    return {'root':root,'equations':nodes} if nodes else root


def _json(value):return json.dumps(_semantic_graph(value),sort_keys=True,separators=(',',':'))


def _number(value):
    if type(value) in (int,float):return ('number',0 if value==0 else 1 if value==1 else -1 if value==-1 else 'positive' if value>0 else 'negative')
    if isinstance(value,(str,bool)) or value is None:return ('literal',value)
    return ('constant',type(value).__name__)


def _pretty(value):
    if not isinstance(value,tuple):return str(value)
    if not value:return 'empty tuple'
    graph=_semantic_graph(value)
    if len(graph['equations'])>10:
        def atom(x):return 'n'+str(x['node']) if isinstance(x,dict) and set(x)=={'node'} else str(x)
        return 'root='+atom(graph['root'])+'; '+'; '.join('n'+str(i)+'='+(str(n[0])+'('+', '.join(atom(x) for x in n[1:])+')' if n else 'empty tuple') for i,n in enumerate(graph['equations']))
    if len(value)==1:return str(value[0])
    return str(value[0])+'('+', '.join(_pretty(x) for x in value[1:])+')'


class Unsupported(ValueError):pass


class Equation:
    """Small tensor-expression interpreter retaining dataflow and state slots.

    It resolves straight-line equations. Unknown control flow is explicitly
    unsupported, rather than converted into a guessed architecture.
    """
    def __init__(self,modules,methods,config,stack=(),attributes=None):
        self.modules=modules;self.methods=methods;self.config=config;self.stack=stack
        self.attributes=attributes or {}
        self.values={};self.evidence=[];self.local_mutables=set();self.local_tensor_writes=set()

    def prepare_local_tensor_writes(self,method):
        """Functionalize writes only to fresh local tensors without aliases."""
        targets={t.value.id for n in ast.walk(method) if isinstance(n,ast.Assign) for t in n.targets
                 if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name)}
        parents={id(child):node for node in ast.walk(method) for child in ast.iter_child_nodes(node)}
        pure={'torch.cat','torch.concat','torch.stack','torch.where','torch.maximum','torch.minimum','torch.all','torch.any'}
        pure_methods={'sum','mean','all','any','square','abs','clone','numel','size'}
        for name in targets:
            assignments=[n for n in ast.walk(method) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
            if len(assignments)!=1 or len(assignments[0].targets)!=1 or not isinstance(assignments[0].value,ast.Call):raise Unsupported('tensor write without unique fresh allocation '+name)
            allocation=assignments[0].value;allocation_name=dotted(allocation.func)
            fresh=allocation_name in {'torch.zeros','torch.ones','torch.zeros_like','torch.ones_like'} or isinstance(allocation.func,ast.Attribute) and allocation.func.attr in {'clone','new_zeros'}
            if not fresh:raise Unsupported('tensor write may alias an existing value '+name)
            for n in ast.walk(method):
                if not isinstance(n,ast.Name) or n.id!=name or not isinstance(n.ctx,ast.Load):continue
                parent=parents[id(n)]
                if isinstance(parent,ast.Return) and parent.value is n:continue
                if isinstance(parent,ast.Subscript):
                    if parent.value is n and isinstance(parent.ctx,ast.Store):continue
                    if parent.slice is n:continue
                if isinstance(parent,ast.Attribute) and parent.value is n and parent.attr in pure_methods|{'shape','dtype','device'}:continue
                if isinstance(parent,ast.Call):
                    callee=module_call_name(parent.func,self.config)
                    if callee in pure or callee in self.modules and self.modules[callee]['kind']=='Linear':continue
                raise Unsupported('mutable tensor alias or unaccounted use '+name)
        self.local_tensor_writes=targets

    def prepare_local_lists(self,method):
        """Admit local list assembly only where no alias can observe mutation.

        Tensor indexes and sum/stack consume the current sequence immediately.
        Capturing the list in another value, passing it to a helper, returning it,
        or mutating a nonlocal list remains outside this bounded proof.
        """
        parents={id(child):node for node in ast.walk(method) for child in ast.iter_child_nodes(node)}
        targets={n.value.func.value.id for n in ast.walk(method)
                 if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                 and n.value.func.attr in {'append','extend'}}
        for name in targets:
            assignments=[n for n in ast.walk(method) if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
            if len(assignments)!=1 or len(assignments[0].targets)!=1 or not isinstance(assignments[0].value,(ast.List,ast.ListComp)):
                raise Unsupported('nonlocal or reassigned mutable collection '+name)
            for n in ast.walk(method):
                if not isinstance(n,ast.Name) or n.id!=name or not isinstance(n.ctx,ast.Load):continue
                parent=parents[id(n)]
                if isinstance(parent,ast.Subscript) and parent.value is n:continue
                if isinstance(parent,ast.Attribute) and parent.value is n and parent.attr in {'append','extend'}:continue
                if isinstance(parent,ast.Call) and dotted(parent.func) in {'sum','torch.stack','torch.cat','torch.concat','torch.concatenate'} and parent.args and parent.args[0] is n:continue
                raise Unsupported('mutable collection alias or escape '+name)
        self.local_mutables=targets

    def local_list_statement(self,n):
        if not isinstance(n,ast.Expr) or not isinstance(n.value,ast.Call):return False
        call=n.value
        if not isinstance(call.func,ast.Attribute) or not isinstance(call.func.value,ast.Name):return False
        name=call.func.value.id
        if name not in self.local_mutables or call.func.attr not in {'append','extend'}:return False
        if len(call.args)!=1 or call.keywords:raise Unsupported('unaccounted collection mutation arguments')
        current=self.values.get(name)
        if not current or current[0]!='tuple':raise Unsupported('unresolved mutable collection')
        value=self.expr(call.args[0])
        if call.func.attr=='append':items=(value,)
        elif value[0]=='tuple':items=value[1:]
        else:raise Unsupported('nonliteral collection extension')
        if len(current)+len(items)>513:raise Unsupported('oversized mutable collection')
        self.values[name]=(*current,*items)
        return True

    def tensor_index(self,n):
        """Keep fixed coordinates and the dataflow of runtime selectors."""
        if isinstance(n,ast.Slice):
            return ('slice-index',*(self.tensor_index(x) if x is not None else None for x in [n.lower,n.upper,n.step]))
        if isinstance(n,ast.Tuple):return ('axes-index',*(self.tensor_index(x) for x in n.elts))
        value=constant(n,self.config)
        if value is not UNKNOWN and isinstance(value,(int,float,str,bool,type(None))):return ('fixed-index',repr(value))
        return ('computed-index',self.expr(n))

    def order_statistic(self,call,receiver,argument_nodes):
        tail=dotted(call.func).split('.')[-1]
        names=['k','dim','largest','sorted'] if tail=='topk' else ['dim','descending','stable']
        defaults={'dim':ast.Constant(value=-1),'largest':ast.Constant(value=True),'sorted':ast.Constant(value=True),'descending':ast.Constant(value=False),'stable':ast.Constant(value=False)}
        if len(argument_nodes)>len(names):raise Unsupported('unaccounted order-statistic arguments')
        supplied=dict(zip(names,argument_nodes))
        for keyword in call.keywords:
            if keyword.arg not in names or keyword.arg in supplied:raise Unsupported('unaccounted order-statistic option')
            supplied[keyword.arg]=keyword.value
        values={**defaults,**supplied}
        if tail=='topk' and 'k' not in values:raise Unsupported('missing topk count')
        dim=constant(values['dim'],self.config)
        if dim is None:dim=-1
        if type(dim) is not int:raise Unsupported('dynamic order-statistic axis')
        flags=[]
        for name in names:
            if name in {'k','dim'}:continue
            flag=constant(values[name],self.config)
            if type(flag) is not bool:raise Unsupported('dynamic order-statistic option '+name)
            flags.append((name,flag))
        return (tail,receiver,*([self.expr(values['k'])] if tail=='topk' else []),('axis',dim),('options',*flags))

    def expr(self,n):
        if n is None:return ('none',)
        c=constant(n,self.config)
        if c is not UNKNOWN:return _number(c)
        if isinstance(n,ast.Name):
            if n.id in self.values:return self.values[n.id]
            if n.id in {'torch','F','nn','math'}:return ('namespace',n.id)
            raise Unsupported('unbound value '+n.id)
        if isinstance(n,ast.Attribute):
            name=dotted(n)
            if name.startswith('self.'):
                if name in self.modules:return ('module-reference',name)
                if name in self.attributes:return self.attributes[name]
                if n.attr in {'training'}:return ('training',)
                if n.attr.endswith(('weight','bias')):return ('learned-parameter',n.attr)
                raise Unsupported('unresolved model attribute '+name)
            return ('attribute',n.attr,self.expr(n.value))
        if isinstance(n,(ast.List,ast.Tuple)):
            values=[]
            for x in n.elts:
                if isinstance(x,ast.Starred):
                    expanded=self.expr(x.value)
                    if expanded[0]!='tuple':raise Unsupported('unresolved starred sequence length')
                    values.extend(expanded[1:])
                else:values.append(self.expr(x))
            return ('tuple',*values)
        if isinstance(n,(ast.ListComp,ast.GeneratorExp)):
            if len(n.generators)!=1 or n.generators[0].is_async:raise Unsupported('nested/dynamic comprehension')
            gen=n.generators[0]
            sequence=constant(gen.iter,self.config)
            if sequence is UNKNOWN and isinstance(gen.iter,ast.Call) and dotted(gen.iter.func)=='range':
                limits=[constant(x,self.config) for x in gen.iter.args]
                if limits and all(type(x) is int for x in limits):sequence=range(*limits)
            if not isinstance(sequence,(tuple,list,range)) or len(sequence)>512:raise Unsupported('nonliteral or oversized comprehension')
            if not isinstance(gen.target,ast.Name):raise Unsupported('comprehension destructuring')
            saved_values=self.values;saved_config=self.config;values=[]
            try:
                for value in sequence:
                    self.values=dict(saved_values);self.config=dict(saved_config)
                    self.config[gen.target.id]=value;self.values[gen.target.id]=_number(value)
                    filters=[constant(x,self.config) for x in gen.ifs]
                    if any(x is UNKNOWN for x in filters):raise Unsupported('input-dependent comprehension filter')
                    if all(filters):values.append(self.expr(n.elt))
            finally:self.values=saved_values;self.config=saved_config
            return ('tuple',*values)
        if isinstance(n,ast.Slice):
            return ('slice',*(ast.unparse(x) if x else None for x in [n.lower,n.upper,n.step]))
        if isinstance(n,ast.Subscript):
            module_name=module_call_name(n,self.config)
            if module_name in self.modules:return ('module-reference',module_name)
            value=self.expr(n.value)
            index=constant(n.slice,self.config)
            if value and value[0]=='tuple' and type(index) is int:
                return value[1:][index]
            if value and value[0]=='tuple' and isinstance(n.slice,ast.Slice):
                bounds=[None if v is None else constant(v,self.config) for v in [n.slice.lower,n.slice.upper,n.slice.step]]
                if any(v is UNKNOWN or v is not None and type(v) is not int for v in bounds):raise Unsupported('dynamic sequence slice')
                return ('tuple',*value[1:][slice(*bounds)])
            # Shape selections are dimensions, but signal selections remain.
            if value[0]=='attribute' and value[1]=='shape':
                if isinstance(n.slice,ast.Slice):return ('shape-slice',value,self.tensor_index(n.slice))
                return ('dimension',)
            if value==('state',) and type(index) is int:return ('select',value,str(index))
            return ('select',value,self.tensor_index(n.slice))
        if isinstance(n,ast.UnaryOp):
            return (type(n.op).__name__,self.expr(n.operand))
        if isinstance(n,ast.BinOp):
            left,right=self.expr(n.left),self.expr(n.right)
            op=type(n.op).__name__
            # Nonzero scalar weighting does not create a new signal feature.
            if op=='Mult':
                if left[0]=='number' and left[1]!=0:return right
                if right[0]=='number' and right[1]!=0:return left
            if op=='Div' and right[0]=='number' and right[1]!=0:return left
            if op in {'Add','Mult'}:
                items=sorted([left,right],key=_json)
                return (op,*items)
            return (op,left,right)
        if isinstance(n,ast.Compare):return ('comparison',*(type(x).__name__ for x in n.ops),self.expr(n.left),*(self.expr(x) for x in n.comparators))
        if isinstance(n,ast.BoolOp):return (type(n.op).__name__,*(self.expr(x) for x in n.values))
        if isinstance(n,ast.IfExp):return ('conditional',self.expr(n.test),self.expr(n.body),self.expr(n.orelse))
        if not isinstance(n,ast.Call):raise Unsupported(type(n).__name__)
        name=module_call_name(n.func,self.config);tail=name.split('.')[-1]
        args=[]
        for x in n.args:
            if not isinstance(x,ast.Starred):args.append(self.expr(x));continue
            spread=self.expr(x.value)
            if spread[0]=='tuple':args.extend(spread[1:])
            elif tail in {'zeros','ones','empty','full','new_zeros','new_empty'} and spread[0]=='attribute' and spread[1]=='shape':
                args.append(('allocation-shape',spread))
            elif tail in SHAPE|{'zeros','ones','empty','full','new_zeros','new_empty'} and spread[0]=='shape-slice':
                args.append(('shape-arguments',spread))
            else:raise Unsupported('unresolved starred call arguments')
        module_name=name
        if name in self.values and self.values[name][0]=='module-reference':module_name=self.values[name][1]
        if module_name in self.modules:
            spec=self.modules[module_name];kind=spec['kind']
            if kind=='Sequential':
                value=args[0]
                for child in spec['children']:
                    self.modules['__sequential_child__']=child
                    self.values['__sequential_input__']=value
                    value=self.expr(ast.Call(func=ast.Name(id='__sequential_child__'),args=[ast.Name(id='__sequential_input__')],keywords=[]))
                return value
            if kind in RECURRENT:
                # Standard library recurrence outputs a sequence and final state.
                state=('cell-state','LSTM',*args) if 'LSTM' in kind else ('recurrent-state',kind,*args)
                return ('tuple',('recurrent-output',kind,*args),('tuple',state,('cell-memory',*args)) if 'LSTM' in kind else state) if not kind.endswith('Cell') else state
            if kind in NORMALIZERS:return args[0]
            if kind=='Dropout':return args[0]
            if kind=='Linear':return ('affine',*args)
            if kind.startswith('Conv'):return ('convolution',kind,spec.get('communication','dense'),*args)
            if 'Pool' in kind:return ('pool',kind,*args)
            if kind in {'ReLU','GELU','SiLU','Tanh','Sigmoid'}:return (kind.lower(),*args)
            if kind=='Identity':return args[0]
            raise Unsupported('custom module '+kind)
        if name.startswith('self.') and tail in self.methods:
            if tail in self.stack:raise Unsupported('recursive helper '+tail)
            helper=Equation(self.modules,self.methods,self.config,self.stack+(tail,))
            return helper.run(self.methods[tail],args)
        if name in {'torch.cat','torch.concat','torch.concatenate','torch.stack'}:
            if not args or args[0][0]!='tuple':raise Unsupported('unresolved tensor collection')
            dim_nodes=list(n.args[1:])+[k.value for k in n.keywords if k.arg=='dim']
            if len(dim_nodes)>1 or any(type(constant(dim,self.config)) is not int for dim in dim_nodes):
                raise Unsupported('dynamic or invalid collection combine axis')
            if any(k.arg not in {'dim','out'} for k in n.keywords) or any(k.arg=='out' for k in n.keywords):
                raise Unsupported('unaccounted collection combine option')
            return ('combine',*args[0][1:])
        if name in {'torch.maximum','torch.minimum'}:return (tail,*args)
        if name in {'torch.topk','torch.sort','torch.argsort'}:
            if not n.args:raise Unsupported('missing order-statistic input')
            return self.order_statistic(n,args[0],n.args[1:])
        if name in {'F.linear','torch.nn.functional.linear'}:
            if not 1<=len(args)<=3 or any(k.arg not in {'weight','bias'} for k in n.keywords):raise Unsupported('unaccounted functional affine arguments')
            parameters=args[1:]+[self.expr(k.value) for k in n.keywords]
            if not parameters:raise Unsupported('missing functional affine weights')
            if any(isinstance(v,tuple) and v and v[0] in {'signal','component','state','recurrent-state','recurrent-output','cell-state','cell-memory','training'} for parameter in parameters for v in _walk(parameter)):
                raise Unsupported('signal-dependent functional affine parameters')
            return ('affine',args[0])
        if name in {'F.layer_norm','torch.nn.functional.layer_norm'}:
            if not 1<=len(args)<=5 or any(k.arg not in {'normalized_shape','weight','bias','eps'} for k in n.keywords):raise Unsupported('unaccounted functional normalization')
            parameters=args[2:]+[self.expr(k.value) for k in n.keywords if k.arg in {'weight','bias','eps'}]
            if any(isinstance(v,tuple) and v and v[0] in {'signal','component','state','recurrent-state','recurrent-output','cell-state','cell-memory','training'} for parameter in parameters for v in _walk(parameter)):
                raise Unsupported('signal-dependent normalization parameters')
            return args[0]
        if name.startswith(('torch.','F.','math.')) and tail in UNARY|REDUCTIONS|TENSOR_FUNCTIONS:
            axes=tuple((k.arg,ast.unparse(k.value)) for k in n.keywords if k.arg in {'dim','dims','axis','axes','keepdim'})
            return (tail,*args,('axes',*axes))
        if isinstance(n.func,ast.Attribute) and not name.startswith(('torch.','F.','math.','nn.')):
            receiver=self.expr(n.func.value)
            if tail in {'topk','sort','argsort'}:return self.order_statistic(n,receiver,n.args)
            if tail in {'any','all'}:
                if len(n.args)>2 or any(k.arg not in {'dim','keepdim'} for k in n.keywords):raise Unsupported('unaccounted Boolean reduction signature')
                options=dict(zip(['dim','keepdim'],n.args))
                for keyword in n.keywords:
                    if keyword.arg in options:raise Unsupported('duplicated Boolean reduction argument')
                    options[keyword.arg]=keyword.value
                return (tail,receiver,('axes',*((key,ast.unparse(value)) for key,value in sorted(options.items()))))
            if tail in SHAPE:return receiver
            if tail in {'add','sub','mul','div'} and len(n.args)==1:
                other=n.args[0]
                alpha=next((k.value for k in n.keywords if k.arg=='alpha'),None)
                if any(k.arg!='alpha' for k in n.keywords):raise Unsupported('unaccounted arithmetic keyword')
                if alpha is not None:
                    if tail not in {'add','sub'} or constant(alpha,self.config) is UNKNOWN:raise Unsupported('unaccounted arithmetic alpha')
                    other=ast.BinOp(left=other,op=ast.Mult(),right=alpha)
                return self.expr(ast.BinOp(left=n.func.value,op={'add':ast.Add,'sub':ast.Sub,'mul':ast.Mult,'div':ast.Div}[tail](),right=other))
            if tail in {'eq','ne','lt','le','gt','ge'} and len(args)==1 and not n.keywords:
                return ('comparison',{'eq':'Eq','ne':'NotEq','lt':'Lt','le':'LtE','gt':'Gt','ge':'GtE'}[tail],receiver,args[0])
            if tail in REDUCTIONS|UNARY:
                axes=tuple((k.arg,ast.unparse(k.value)) for k in n.keywords if k.arg in {'dim','dims','axis','axes','keepdim'})
                if tail in REDUCTIONS and n.args:axes+=(('dim',ast.unparse(n.args[0])),)
                if tail in {'clamp_min','clamp_max','clamp'}:return receiver
                return (tail,receiver,('axes',*axes))
            if tail in {'size','numel'}:return ('dimension',)
            if tail=='scatter':
                if len(args)!=3 or n.keywords:raise Unsupported('unaccounted scatter signature')
                return ('scatter',receiver,*args)
            if tail in {'index_select','gather','chunk','split','unbind','unfold','diff','cumsum','cummax','index_add','scatter_add','topk','sort','argsort','repeat_interleave','flip','roll'}:
                return (tail,receiver,*args,('axes',*(ast.unparse(k.value) for k in n.keywords if k.arg in {'dim','axis'})))
            if tail=='new_tensor' and len(args)==1:return ('tensor-data',args[0])
            if tail in {'item','tolist'}:return ('scalar-value',receiver)
        if tail in {'zeros','ones','full','empty','zeros_like','ones_like','full_like','new_zeros','new_empty'}:
            return ('initial-state',tail)
        if name in {'len','int','float','bool','min','max','round'}:return ('scalar',name,*args)
        if name in {'list','tuple'} and len(args)==1 and args[0][0]=='tuple':return args[0]
        if name=='sum' and 1<=len(args)<=2 and args[0][0]=='tuple' and not n.keywords:
            result=args[1] if len(args)==2 else ('number',0)
            for value in args[0][1:]:
                result=value if result==('number',0) else ('Add',result,value)
            return result
        raise Unsupported('call '+name)

    def bind(self,target,value):
        if isinstance(target,ast.Name):self.values[target.id]=value;return
        if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id in self.local_tensor_writes:
            name=target.value.id
            if name not in self.values:raise Unsupported('tensor write precedes allocation '+name)
            index=self.tensor_index(target.slice)
            # A uniform Boolean assignment consumes the selected set, not the
            # order of its indices. Keep score, k, axis and largest direction;
            # topk tie choice remains the same unspecified primitive.
            if value in {('literal',True),('literal',False)} and index[0]=='computed-index':
                selection=index[1]
                if selection[:2] in {('attribute','indices'),('component',1)}:
                    ranking=selection[2]
                    if isinstance(ranking,tuple) and ranking and ranking[0]=='topk':
                        options=tuple(x for x in ranking[-1][1:] if x[0]!='sorted')
                        index=('computed-index',('ranked-membership',*ranking[1:-1],('options',*options)))
            self.values[name]=('indexed-update',self.values[name],index,value)
            return
        if isinstance(target,(ast.Tuple,ast.List)):
            if value[0]=='tuple' and len(value)-1==len(target.elts):
                for t,v in zip(target.elts,value[1:]):self.bind(t,v)
            else:
                for i,t in enumerate(target.elts):self.bind(t,('component',i,value))
            return
        raise Unsupported('assignment '+ast.unparse(target))

    def run(self,method,args=None):
        self.prepare_local_lists(method)
        self.prepare_local_tensor_writes(method)
        arguments=[a.arg for a in method.args.args if a.arg!='self']
        for i,name in enumerate(arguments):
            self.values[name]=args[i] if args and i<len(args) else ('state',) if name=='state' else ('signal',i)
        for n in method.body:
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant):continue
            if self.local_list_statement(n):continue
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                value=('tuple',) if isinstance(n.value,ast.List) and not n.value.elts and any(isinstance(t,ast.Name) and t.id in self.local_mutables for t in (n.targets if isinstance(n,ast.Assign) else [n.target])) else self.expr(n.value)
                for t in n.targets if isinstance(n,ast.Assign) else [n.target]:self.bind(t,value)
            elif isinstance(n,ast.AugAssign):
                current=self.expr(n.target);value=self.expr(n.value)
                self.bind(n.target,(type(n.op).__name__,current,value))
            elif isinstance(n,ast.Return):return self.expr(n.value)
            elif isinstance(n,ast.Pass):continue
            elif isinstance(n,ast.Delete) and all(isinstance(x,ast.Name) for x in n.targets):
                for x in n.targets:self.values.pop(x.id,None)
            elif isinstance(n,ast.For):
                # A deliberately narrow, source-fixed statistic assembly form.
                # It admits only a literal/config-resolved finite lag tuple whose
                # body appends directly to a fresh local list.  Dynamic iterables,
                # conditional bodies, aliases and arbitrary statements remain
                # unsupported so a campaign cannot acquire a guessed statistic.
                if not isinstance(n.target,ast.Name) or n.orelse:
                    raise Unsupported('nonliteral lag-loop target or else')
                lags=constant(n.iter,self.config)
                if not isinstance(lags,(tuple,list)) or not lags or len(lags)>64 or any(not isinstance(x,(int,float)) for x in lags):
                    raise Unsupported('dynamic or invalid lag-loop iterable')
                direct=len(n.body)==1 and isinstance(n.body[0],ast.Expr)
                temporary=(len(n.body)==2 and isinstance(n.body[0],ast.Assign)
                           and len(n.body[0].targets)==1 and isinstance(n.body[0].targets[0],ast.Name)
                           and isinstance(n.body[1],ast.Expr))
                append_statement=n.body[-1] if direct or temporary else None
                call=append_statement.value if append_statement is not None else None
                if (not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute)
                    or not isinstance(call.func.value,ast.Name)
                    or call.func.value.id not in self.local_mutables or call.func.attr!='append'):
                    raise Unsupported('lag-loop does not directly append to a fresh local list')
                if temporary:
                    temporary_name=n.body[0].targets[0].id
                    temporary_uses=sum(1 for node in ast.walk(call.args[0]) if isinstance(node,ast.Name) and node.id==temporary_name) if len(call.args)==1 else 0
                    if len(call.args)!=1 or temporary_uses!=1 or call.keywords:
                        raise Unsupported('lag-loop temporary is not consumed exactly once')
                existing=self.values.get(call.func.value.id)
                if not existing:self.values[call.func.value.id]=('tuple',)
                previous=self.values.get(n.target.id)
                previous_temporary=self.values.get(temporary_name) if temporary else None
                try:
                    for lag in lags:
                        self.values[n.target.id]=('constant',repr(lag))
                        if temporary:
                            value=self.expr(n.body[0].value)
                            self.bind(n.body[0].targets[0],value)
                        if not self.local_list_statement(append_statement):
                            raise Unsupported('unaccounted lag-loop statement')
                finally:
                    if previous is None:self.values.pop(n.target.id,None)
                    else:self.values[n.target.id]=previous
                    if temporary:
                        if previous_temporary is None:self.values.pop(temporary_name,None)
                        else:self.values[temporary_name]=previous_temporary
            else:raise Unsupported('statement '+type(n).__name__+' at '+str(n.lineno))
        raise Unsupported('no returned tensor')


def _runtime(definitions,reached):
    for cls in sorted(reached):
        node=definitions[cls]
        if isinstance(node,ast.ClassDef):
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name!='__init__' and method.name not in SKIP_METHODS:
                    yield cls,method


def profile(sources,campaign_key):
    task=CAMPAIGNS[campaign_key]
    defs,files,classes,reached,roots,envs,globalenv,decisions,trees=program(sources)
    root=next(iter(roots));config=envs[root]
    methods={m.name:m for m in defs[root].body if isinstance(m,ast.FunctionDef)}
    modules={};constructors=[]
    for cls in reached&classes:
        ctor=next((n for n in defs[cls].body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
        if not ctor:continue
        for n in ast.walk(ctor):
            if isinstance(n,(ast.Assign,ast.AnnAssign)) and isinstance(n.value,ast.Call):
                target=dotted(n.targets[0] if isinstance(n,ast.Assign) else n.target)
                call=n.value;kind=dotted(call.func).split('.')[-1]
                if kind in PRIMITIVES|EXTRA_PRIMITIVES|{'Sequential','ModuleList','ModuleDict','Parameter'} or kind in classes:
                    spec={'kind':kind,'class':cls,'file':files[cls],'line':n.lineno,'code':ast.unparse(n)}
                    if kind.startswith('Conv'):
                        kw={k.arg:constant(k.value,envs[cls]) for k in call.keywords}
                        groups=kw.get('groups',1);cin=constant(call.args[0],envs[cls]) if call.args else kw.get('in_channels',UNKNOWN)
                        spec['communication']='dense' if groups==1 else 'depthwise' if groups is not UNKNOWN and groups==cin else 'grouped'
                        positional=('in_channels','out_channels','kernel_size','stride','padding','dilation')
                        for index,key in enumerate(positional[2:],2):
                            value=constant(call.args[index],envs[cls]) if len(call.args)>index else kw.get(key,UNKNOWN)
                            if value is not UNKNOWN:spec[key]=value
                    if 'Pool' in kind:
                        kw={k.arg:constant(k.value,envs[cls]) for k in call.keywords}
                        if kind.startswith('Adaptive'):
                            value=constant(call.args[0],envs[cls]) if call.args else kw.get('output_size',UNKNOWN)
                            if value is not UNKNOWN:spec['output_size']=value
                        else:
                            for index,key in enumerate(('kernel_size','stride','padding','dilation')):
                                value=constant(call.args[index],envs[cls]) if len(call.args)>index else kw.get(key,UNKNOWN)
                                if value is not UNKNOWN:spec[key]=value
                    if kind in RECURRENT:
                        spec['direction']='bidirectional' if any(k.arg=='bidirectional' and constant(k.value,envs[cls]) is True for k in call.keywords) else 'forward'
                    if kind in {'Sequential','ModuleList'}:
                        spec['children']=[]
                        children=call.args
                        if kind=='ModuleList':
                            children=call.args[0].elts if len(call.args)==1 and isinstance(call.args[0],(ast.List,ast.Tuple)) else []
                        for child in children:
                            if not isinstance(child,ast.Call):
                                spec['children'].append({'kind':'unresolved-module-element'});continue
                            item={'kind':dotted(child.func).split('.')[-1]}
                            if item['kind'].startswith('Conv'):
                                kw={k.arg:constant(k.value,envs[cls]) for k in child.keywords};groups=kw.get('groups',1);cin=constant(child.args[0],envs[cls]) if child.args else UNKNOWN
                                item['communication']='dense' if groups==1 else 'depthwise' if groups is not UNKNOWN and groups==cin else 'grouped'
                            if item['kind'] in RECURRENT:
                                item['direction']='bidirectional' if any(k.arg=='bidirectional' and constant(k.value,envs[cls]) is True for k in child.keywords) else 'forward'
                            spec['children'].append(item)
                    if cls==root:modules[target]=spec
                    # A literal Sequential has the same statically ordered
                    # children as a literal ModuleList.  Map only its known
                    # integer positions so `self.head[0](x)` retains the
                    # exact child primitive; a dynamic subscript still has no
                    # module entry and remains unsupported.
                    if cls==root and kind in {'ModuleList','Sequential'}:
                        for index,child in enumerate(spec['children']):modules[target+'['+str(index)+']']=child
                    constructors.append(spec)
    coordinate_pruning=None
    if task == 'kws':
        # This proof is deliberately separate from the generic interpreter:
        # it establishes that a nonpersistent inference buffer only selects
        # parameter-ranked coordinates of an otherwise unchanged affine head.
        from experiments.ontology_recurrent_readout import coordinate_pruning_contract
        coordinate_pruning=coordinate_pruning_contract(sources)
    attributes={name:('coordinate-indices',) for name in coordinate_pruning['buffers']} if coordinate_pruning else {}
    interpreter=Equation
    if coordinate_pruning:
        # Preserve both train/eval branches of the classifier.  This import is
        # delayed to avoid the module-level recurrent/branch dependency cycle.
        from experiments.ontology_recurrent_branches import BranchEquation
        interpreter=BranchEquation
    equations={};errors={};evidence={};operations=[];parameter_access=False;parameter_aliases=False;parameter_reads=[]
    for cls in reached&classes:
        for n in ast.walk(defs[cls]):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr in {'weight','bias'} for t in n.targets):parameter_aliases=True
    for cls,m in _runtime(defs,reached):
        ev={'file':files[cls],'line':m.lineno,'end_line':m.end_lineno,'scope':cls+'.'+m.name,'code':ast.unparse(m)}
        evidence[cls+'.'+m.name]=ev
        operations.extend(dotted(n.func) for n in ast.walk(m) if isinstance(n,ast.Call))
        deployment_hook=bool(coordinate_pruning and cls==root and m.name in coordinate_pruning['hooks'])
        if not deployment_hook:
            parameter_access |= any(isinstance(n,ast.Attribute) and n.attr in {'weight','bias'} for n in ast.walk(m))
            parameter_reads.extend((cls,m.name,n) for n in ast.walk(m) if isinstance(n,ast.Attribute) and n.attr in {'weight','bias'})
        if cls!=root:continue
        if deployment_hook:
            # The contract above accounts for this side-effecting deployment
            # synchronization method.  It is not an inference state update.
            continue
        if m.name=='frame_schedule' and fixed_schedule(m,config):equations[m.name]=('fixed-frame-count-schedule',);continue
        try:equations[m.name]=interpreter(modules,methods,config,attributes=attributes).run(m)
        except (Unsupported,IndexError,TypeError,ValueError) as e:errors[m.name]=str(e)
    used={module_call_name(n.func,envs[cls]) for cls,m in _runtime(defs,reached) for n in ast.walk(m) if isinstance(n,ast.Call)}
    active=[s for name,s in modules.items() if name in used]
    active=[child for s in active for child in (s.get('children') if s['kind']=='Sequential' else [s])]
    rec=sorted(set(s['kind'].replace('Cell','')+':'+s.get('direction','forward') for s in active if s['kind'] in RECURRENT))
    conv=sorted(set((s['kind'],s['communication']) for s in active if s['kind'].startswith('Conv')))
    main=equations.get('classify' if task=='kws' else 'forward')
    step=equations.get('recurrent_step')
    mechanism={'recurrent_operator':rec,'convolution_communication':conv,
               'operator_topology':tuple(_module_topology(s) for s in active),
               'execution_topology':_forward_topology(methods['classify' if task=='kws' else 'forward'],modules,config) if ('classify' if task=='kws' else 'forward') in methods else ()}
    if main:
        # The precise dependency equations establish which history statistics
        # reach the output, not just whether words like "mean" occur in code.
        mechanism['readout_features']=_readout_features(main)
    if coordinate_pruning and main:
        # The paired training/inference proof is stronger than a raw selector
        # graph: it retains the source readout statistics while abstracting only
        # parameter-derived affine coordinates.
        mechanism['readout_features']=coordinate_pruning['features']
    if step:
        mechanism['state_statistics']=_state_statistics(step)
        mechanism['input_representation']=_recurrent_inputs(step)
    elif task=='har' and main:
        mechanism['input_representation']=_recurrent_inputs(main)
    if 'should_exit' in methods or 'exit_mask' in methods or 'early_exit' in methods:
        hook=next(x for x in ['should_exit','exit_mask','early_exit'] if x in methods)
        mechanism['exit_policy']=('exit-observables',*_exit_observables(equations[hook])) if hook in equations else 'input/state-dependent hook present'
    elif task=='kws':mechanism['exit_policy']='fixed horizon; no exit hook'
    heads={dotted(n.value) for cls,method,n in parameter_reads}
    readonly_exit_parameters=bool(parameter_reads) and len(heads)==1 and all(
        cls==root and method in {'exit_mask','should_exit','early_exit'} and isinstance(n.ctx,ast.Load)
        and dotted(n.value) in modules and modules[dotted(n.value)]['kind']=='Linear'
        for cls,method,n in parameter_reads)
    if readonly_exit_parameters:
        classifier=methods.get('classify')
        readonly_exit_parameters=classifier is not None and any(isinstance(n,ast.Call) and dotted(n.func) in heads for n in ast.walk(classifier))
    return {'task':task,'equations':equations,'errors':errors,'mechanism':mechanism,'evidence':evidence,'constructors':constructors,'active':active,'modules':modules,'methods':methods,'operations':operations,'decisions':decisions,'root':root,'parameter_access':parameter_access,'readonly_exit_parameters':readonly_exit_parameters,'parameter_aliases':parameter_aliases,'coordinate_pruning':coordinate_pruning,'static_attributes':attributes}


def _readout_features(eq):
    """Features before independent affine heads; coordinate sums are settings."""
    memo={}
    def visit(n):
        if isinstance(n,tuple) and id(n) in memo:return memo[id(n)]
        result=convert(n)
        if isinstance(n,tuple):memo[id(n)]=result
        return result
    def convert(n):
        if not isinstance(n,tuple):return n
        if n[0]=='affine':return visit(n[1])
        if n[0] in {'recurrent-output','recurrent-state','cell-state','cell-memory'}:
            return (n[0],n[1]) if n[0]!='cell-memory' else ('cell-memory','LSTM')
        if n[0]=='convolution':return ('convolution-features',n[1],n[2])
        if n[0] in {'Add','Sub','combine'}:
            out=[]
            for item in n[1:]:
                v=visit(item)
                out.extend(v[1:] if isinstance(v,tuple) and v and v[0]=='features' else [v])
            unique={_json(x):x for x in out}
            return ('features',*(unique[k] for k in sorted(unique)))
        return (n[0],*(visit(x) if isinstance(x,tuple) else x for x in n[1:]))
    return visit(eq)


def _state_statistics(eq):
    if eq[0]!='tuple':return None
    # Preserve every persistent component's update; standard recurrence internals
    # are named primitives. Coordinate layouts do not enter this signature.
    return _readout_features(eq)


def _recurrent_inputs(eq):
    values=[_readout_features(n[2]) for n in _walk(eq) if isinstance(n,tuple) and len(n)>2 and n[0] in {'recurrent-output','recurrent-state','cell-state'}]
    unique={_json(x):x for x in values}
    return ('inputs',*(unique[k] for k in sorted(unique)))


def _positive_changes(before,after):
    a,b=before['mechanism'],after['mechanism'];changed=[]
    # Library recurrence and explicit tensor equations use different bases.
    # Absence of an nn.GRU/nn.LSTM declaration is not evidence that recurrence
    # disappeared: an exact handwritten/fused cell can implement the same map.
    comparable_recurrence=bool(a.get('recurrent_operator'))==bool(b.get('recurrent_operator'))
    a_kinds={s['kind'] for s in before['active'] if s['kind'] in RECURRENT}
    b_kinds={s['kind'] for s in after['active'] if s['kind'] in RECURRENT}
    if a_kinds!=b_kinds and {s.replace('Cell','') for s in a_kinds}=={s.replace('Cell','') for s in b_kinds}:
        comparable_recurrence=False
    if a.get('recurrent_operator')!=b.get('recurrent_operator') and any(op.endswith('.flip') for op in before['operations']+after['operations']):
        # A bidirectional library module can be expanded into two explicit
        # forward/reversed scans without introducing a new recurrence family.
        comparable_recurrence=False
    mapping={'recurrent_operator':'state_update','convolution_communication':'routing',
             'operator_topology':'temporal_operator', 'execution_topology':'connectivity',
             'readout_features':'readout_history' if after['task']=='kws' else 'temporal_readout','state_statistics':'state_structure','input_representation':'acoustic_representation' if after['task']=='kws' else 'sensor_fusion','exit_policy':'exit_policy'}
    for field,component in mapping.items():
        if field not in a or field not in b or _json(a[field])==_json(b[field]):continue
        if field in {'recurrent_operator','operator_topology','execution_topology'} and (before['errors'] or after['errors']):
            # A failed method profile leaves the active-call set incomplete.
            # Never use that partial set as a structural transition witness.
            continue
        if field in {'recurrent_operator','operator_topology','execution_topology'} and any(
            s['kind'] not in PRIMITIVES|EXTRA_PRIMITIVES|RECURRENT
            for p in (before,after) for s in p['active']
        ):
            # A custom/fused module has no reviewed body.  Its declaration
            # cannot become a positive witness merely through its presence.
            continue
        if not comparable_recurrence and field in {'recurrent_operator','readout_features','state_statistics','input_representation'}:continue
        if field in {'readout_features','state_statistics','input_representation'}:
            # Numerical/scalar shapes, affine coordinates, and variable renames
            # are normalized above. A remaining graph difference alone is not
            # enough: require a recognized mechanism feature witness below.
            witnesses=_feature_witnesses(a[field],b[field],field)
            if not witnesses:continue
        else:
            if field=='exit_policy' and ((isinstance(a[field],str) and 'hook present' in a[field]) or (isinstance(b[field],str) and 'hook present' in b[field])):continue
            witnesses=[field+' changed from '+_pretty(a[field])+' to '+_pretty(b[field])]
        changed.append({'component':component,'field':field,'before':_semantic_graph(a[field]),'after':_semantic_graph(b[field]),'witnesses':witnesses})
    return changed


def _feature_witnesses(a,b,field):
    def features(x):
        selected=set()
        for n in _walk(x):
            if not isinstance(n,tuple) or not n:continue
            if n[0]=='pool':
                # Existing pooling-layer counts are a setting under the handoff
                # rubric; only a different reduction family is a witness.
                selected.add(_json(('pool-family',n[1])));continue
            if n[0] in {'component','mean','sum','amax','amin','maximum','minimum','std','var','square','sqrt','rsqrt','abs','sin','cos','exp','log','softmax','sigmoid','tanh','relu','gelu','silu','convolution','recurrent-output','recurrent-state','cell-state','select','conditional','where','einsum','fft','rfft','bmm','matmul','diff','quantile','index_select','gather','unfold'}:
                selected.add(_json(n))
        return selected
    aa,bb=features(a),features(b)
    if aa==bb:return []
    return ['Source-traced '+field+' feature '+('added: '+v if v in bb else 'removed: '+v) for v in sorted(aa^bb)[:12]]


def _fingerprint(p):
    """Complete only for straight-line standard-module graphs we accounted for."""
    task=p['task'];required={'recurrent_step','classify'} if task=='kws' else {'forward'}
    residual=[x for x in required if x not in p['equations']]
    # Custom class and dynamic runtime code require further explicit review.
    residual.extend(k for k in p['errors'] if k not in {'initial_state','recurrent_sequence','frame_schedule'})
    residual.extend('module:'+s['kind'] for s in p['active'] if s['kind'] not in PRIMITIVES|EXTRA_PRIMITIVES)
    residual.extend('parameter_construction' for s in p['constructors'] if s['kind']=='Parameter')
    if p['parameter_access'] and not p.get('readonly_exit_parameters'):residual.append('parameter_construction')
    if p['parameter_aliases']:residual.append('sharing')
    if residual:return {},sorted(set(residual))
    # Each reported value is a direct description of the accounted graph. The
    # source trace covers all methods, including independent sequence fast paths.
    if p['errors']:return {},sorted(p['errors'])
    fp=dict.fromkeys(CORE+TASK_KEYS[task].split())
    kinds=sorted(set(s['kind'] for s in p['active']))
    rec=p['mechanism']['recurrent_operator'];conv=p['mechanism']['convolution_communication']
    graph=p['equations'].get('recurrent_step' if task=='kws' else 'forward')
    read=p['equations'].get('classify' if task=='kws' else 'forward')
    names=' + '.join(rec+[c[0]+' '+c[1] for c in conv]) or 'explicit affine/nonlinear tensor computation'
    features=p['mechanism'].get('input_representation',_recurrent_inputs(graph))
    sharing=Counter(op for op in p['operations'] if op.startswith('self.'))
    routed_readout=any(isinstance(n,tuple) and n and n[0]=='indexed-update' for n in _walk(read))
    fp.update(input_units='frozen acoustic feature frames' if task=='kws' else 'continuous sensor time frames',
      input_transform=_pretty(features),
      embedding='continuous features; no token embedding lookup in accounted graph',
      position='causal recurrent order' if task=='kws' else 'temporal order through convolution/recurrence',
      mixing=names,routing=('; '.join(c[0]+': '+c[1] for c in conv) or 'fixed dense recurrent communication')+('; indexed classifier routing: '+_pretty(read) if routed_readout else ''),
      state=_pretty(p['mechanism'].get('state_statistics',tuple(rec))),feedforward='explicit affine/nonlinear operators: '+', '.join(k for k in kinds if k not in RECURRENT),
      parameter_construction=('standard freely learned module parameters; a deployment-only source-checked hook derives fixed inference coordinate indices from affine-head weights' if p.get('coordinate_pruning') else 'standard freely learned module parameters; exit rule reads the same affine classifier weights and bias without generating or modifying weights' if p.get('readonly_exit_parameters') else 'standard freely learned module parameters'),sharing='module weights reused between streaming and sequence paths; no explicit cross-module weight aliases' if task=='kws' else 'no explicit cross-module weight aliases',
      normalization=', '.join([k for k in kinds if k in NORMALIZERS]+sorted(set(op for op in p['operations'] if op in {'F.layer_norm','torch.nn.functional.layer_norm'}))) or 'none in accounted execution',
      connectivity=_pretty(graph),aggregation=_pretty(p['mechanism'].get('readout_features',read)),
      output=_pretty(read),symmetry='no explicit group action/canonicalization in accounted graph',
      conditional_compute=('indexed classifier execution: '+_pretty(read)+'; ' if routed_readout else '')+_pretty(p['mechanism'].get('exit_policy','fixed accounted forward graph')),
      activation=', '.join(sorted(set(n[0] for n in _walk(graph) if isinstance(n,tuple) and n and n[0] in UNARY))) or 'standard recurrent gate nonlinearities',
      stochasticity='Dropout' if 'Dropout' in kinds else 'none in accounted runtime',
      bottleneck='affine dimension settings; no additional unaccounted factorization',iteration='fixed stages and recurrent scans',other='all runtime methods represented by source-traced equations')
    if task=='kws':
        exit_graph=next((p['equations'][name] for name in ['exit_mask','should_exit','early_exit'] if name in p['equations']),p['mechanism']['exit_policy'])
        fp.update(acoustic_representation=fp['input_transform'],state_update=names,state_structure=fp['state'],temporal_schedule=_pretty(p['equations'].get('frame_schedule',('all-frames',))),readout_history=fp['aggregation'],exit_policy=_pretty(exit_graph))
    else:fp.update(sensor_fusion='source-traced channel communication: '+_pretty(features),temporal_operator=names,directionality=', '.join(rec) or 'feedforward',state_update=', '.join(rec) or 'no recurrent state',frequency_representation='time-domain operators in accounted graph',temporal_readout=fp['aggregation'])
    return fp,[]


def _absorb_projection(before_sources,after_sources):
    """Fold a directly preceding free affine map into GRU input matrices.

    Restricted to newly declared Linear modules whose every invocation is the
    input of a library GRU/GRUCell, with only shape views in between. A norm or
    nonlinearity between projection and recurrence prevents this proof.
    """
    def declarations(sources):
        defs,_,_,_,roots,_,_,_,_=program(sources);root=next(iter(roots))
        node=defs[root];linear=set();gru=set()
        for n in ast.walk(node):
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
                kind=dotted(n.value.func).split('.')[-1];name=dotted(n.targets[0])
                if kind=='Linear':linear.add(name)
                if kind in {'GRU','GRUCell'}:gru.add(name)
        return node,linear,gru
    try:
        _,old,_=declarations(before_sources);node,new,gru=declarations(after_sources)
        candidates=new-old
        if not candidates:return False
        eligible=set()
        for name in candidates:
            uses=[n for n in ast.walk(node) if isinstance(n,ast.Call) and dotted(n.func)==name]
            absorbed=[]
            for call in ast.walk(node):
                if not isinstance(call,ast.Call) or dotted(call.func) not in gru or not call.args:continue
                x=call.args[0]
                while isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in SHAPE:x=x.func.value
                if isinstance(x,ast.Call) and dotted(x.func)==name:absorbed.append(x)
            if uses and len(uses)==len(absorbed):eligible.add(name)
        if not eligible:return False
        class Fold(ast.NodeTransformer):
            def visit_Assign(self,n):
                if isinstance(n.value,ast.Call) and dotted(n.value.func).split('.')[-1]=='Linear' and any(dotted(t) in eligible for t in n.targets):return None
                return self.generic_visit(n)
            def visit_Call(self,n):
                if dotted(n.func) in eligible and len(n.args)==1 and not n.keywords:return self.visit(n.args[0])
                return self.generic_visit(n)
        folded={f:ast.unparse(ast.fix_missing_locations(Fold().visit(ast.parse(src)))) for f,src in after_sources.items()}
        return settings_signature(before_sources)==settings_signature(folded)
    except (ValueError,TypeError,SyntaxError):return False


def _exit_observables(eq):
    def scalar_settings(n):
        if not isinstance(n,tuple):return n
        if n[0]=='number':return ('threshold-setting',)
        return (n[0],*(scalar_settings(x) if isinstance(x,tuple) else x for x in n[1:]))
    result=set()
    for n in _walk(eq):
        if not isinstance(n,tuple) or not n or n[0]!='comparison':continue
        for operand in n[2:]:
            if isinstance(operand,tuple) and operand[0] not in {'number','literal','constant'}:result.add(scalar_settings(operand))
    return tuple(sorted(result,key=_json))


def _same_exit_family(before_sources,after_sources,before,after):
    hook=next((x for x in ['should_exit','exit_mask','early_exit'] if x in before['equations'] and x in after['equations']),None)
    if not hook:return False
    a,b=_exit_observables(before['equations'][hook]),_exit_observables(after['equations'][hook])
    if not a or a!=b:return False
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,n):
            if n.name==hook:n.body=[ast.Return(value=ast.Constant(value='same scalar exit-observable threshold family'))]
            return self.generic_visit(n)
    def sig(sources):return settings_signature({f:ast.unparse(ast.fix_missing_locations(Strip().visit(ast.parse(s)))) for f,s in sources.items()})
    return sig(before_sources)==sig(after_sources)


def _automatic_pair(before_sources,after_sources,campaign_key):
    if campaign_key not in CAMPAIGNS or not before_sources or not after_sources:return None
    try:
        before=_cached_profile(_json(before_sources),campaign_key);after=_cached_profile(_json(after_sources),campaign_key)
    except (SyntaxError,ValueError,TypeError,RecursionError):return None
    changed=_positive_changes(before,after)
    # This is an exact reparameterization proof, stronger than the structural
    # topology witness above: the added Linear is consumed only by the GRU and
    # is folded into its input matrices.  Do not let its declaration count turn
    # that proven case into a family change.
    absorbed_projection=_absorb_projection(before_sources,after_sources) or _absorb_projection(after_sources,before_sources)
    if absorbed_projection:changed=[]
    from experiments.ontology_recurrent_readout import same_frequency_pool_extent,same_latent_split_widths
    same_band_pool=(campaign_key=='openevolve_v21_tiny_kws_rnn' and same_frequency_pool_extent(before_sources,after_sources))
    same_latent_widths=same_latent_split_widths(before_sources,after_sources)
    from experiments.ontology_recurrent_readout import same_coordinate_readout
    same_coordinates=same_coordinate_readout(before_sources,after_sources)
    if same_band_pool or same_latent_widths or same_coordinates:changed=[]
    # Exact bounded preserving proofs take priority over a coarser feature
    # witness. For example, dropping one coordinate of an existing mean readout
    # changes an index expression but does not add/remove a history mechanism.
    try:
        s1,a1=_cached_signatures(_json(before_sources));s2,a2=_cached_signatures(_json(after_sources))
        if s1 and s1==s2 or a1 and a1==a2:changed=[]
    except (SyntaxError,ValueError,TypeError,RecursionError):pass
    if changed and all(x['component'] in {'readout_history','temporal_readout'} for x in changed):
        from experiments.ontology_recurrent_readout import same_coordinate_readout
        if same_coordinate_readout(before_sources,after_sources):changed=[]
    classification=None;reason=None
    if changed:
        classification='changing';reason='Positive task-specific mechanism difference in source-traced dataflow.'
    else:
        try:
            if same_band_pool:
                classification='preserving';reason='The existing adjacent-frequency pair-average helper retains its normalization, axis, adjacency and untouched-prefix routing; only the pooled region/count and ordinary width settings changed, with the remaining source structure matched.'
            if same_latent_widths:
                classification='preserving';reason='Only coordinated widths of an existing learned latent split changed: upstream Linear output equals the split-size sum and every downstream Linear input equals its part width. The feature axis, gate equations, coefficients and every other source operation remain exact.'
            a,_=_cached_signatures(_json(before_sources));b,_=_cached_signatures(_json(after_sources))
            if a and a==b:classification='preserving';reason='Same configuration-resolved operations and state/readout dependencies; only numerical settings differ.'
            else:
                _,a=_cached_signatures(_json(before_sources));_,b=_cached_signatures(_json(after_sources))
                if a and a==b:classification='preserving';reason='Same core/state computation and readout features; independent affine readout coordinates differ.'
            if classification is None and absorbed_projection:
                classification='preserving';reason='A free affine projection directly before a library GRU is absorbed into its input matrices/biases; all other operation and dependency structure matches after numerical-setting normalization.'
            if classification is None and _same_exit_family(before_sources,after_sources,before,after):
                classification='preserving';reason='Only the threshold regions of the existing early-exit rule changed. The rule observes the same posterior/state/horizon quantities, and the remaining source computation matches.'
            if classification is None:
                if same_coordinates:
                    classification='preserving';reason='Same recurrent/core program and same readout feature equations; only parameter-derived affine head coordinates, selected columns, or reference-class contrasts changed.'
        except (SyntaxError,ValueError,TypeError,RecursionError):pass
    if classification is None:
        from experiments.ontology_recurrent_equivalence import review_pair as equation_review
        result=equation_review(before_sources,after_sources,campaign_key)
        if result:return result
        from experiments.ontology_recurrent_branches import review_pair as branch_review
        result=branch_review(before_sources,after_sources,campaign_key)
        if result:return result
        fp,residual=_fingerprint(after)
        if residual:return None
        return {'classification':'uncertain','fingerprint':fp,'fingerprint_complete':True,'residual_components':[],
          'notes':'Candidate runtime architecture is completely represented by source-traced equations; the parent-to-candidate ontology transition still needs an equivalence or positive-change review.',
          'changed_components':[],'evidence':list(after['evidence'].values()),'family_signature':_semantic_graph(after['mechanism']),
          'before_source_sha256':source_sha(before_sources),'source_sha256':source_sha(after_sources),'reviewer':VERSION}
    fp,residual=_fingerprint(after)
    if residual and after['errors']:
        from experiments.ontology_recurrent_branches import branch_profile
        try:
            expanded=branch_profile(after_sources,campaign_key)
            expanded_fp,expanded_residual=_fingerprint(expanded)
            if not expanded_residual:after,fp,residual=expanded,expanded_fp,[]
        except (ValueError,TypeError,SyntaxError,RecursionError):pass
    return {'classification':classification,'fingerprint':fp,'fingerprint_complete':not residual,'residual_components':residual,
      'notes':reason,'changed_components':sorted(set(x['component'] for x in changed)),
      'evidence':list(after['evidence'].values()),'transition_evidence':changed,
      'family_signature':_semantic_graph(after['mechanism']) if not residual else None,
      'before_source_sha256':source_sha(before_sources),'source_sha256':source_sha(after_sources),'reviewer':VERSION}


def _reviewed_fingerprint(reference,campaign_key):
    """A complete source audit must explicitly cover every task component."""
    fp=reference.get('fingerprint')
    required=set(CORE+TASK_KEYS[CAMPAIGNS[campaign_key]].split())
    if (reference.get('fingerprint_complete') is True and isinstance(fp,dict)
        and required<=set(fp) and all(isinstance(fp[k],str) and fp[k].strip() for k in required)
        and reference.get('evidence')):
        return copy.deepcopy(fp)
    return None


def _kws_b01_c2_source_review(before_sources, after_sources, campaign_key):
    """Return an exact retained-source decision with a fresh full fingerprint.

    The source hashes are the admission boundary.  Parsing the candidate again
    is intentional: it keeps every fingerprint component tied to the current
    bounded interpreter rather than copying a partial historical profile.
    """
    if campaign_key!='openevolve_v21_tiny_kws_rnn':return None
    candidate_sha=source_sha(after_sources);record=KWS_B01_C2_SOURCE_REVIEWS.get(candidate_sha)
    if record is None or source_sha(before_sources)!=record[0]:return None
    try:
        from experiments.ontology_recurrent_branches import branch_profile
        reviewed=branch_profile(after_sources,campaign_key);fp,residual=_fingerprint(reviewed)
    except (SyntaxError,ValueError,TypeError,RecursionError):
        return None
    source=after_sources.get('train.py','').splitlines();line=record[3]
    if not source or not 1<=line<=len(source):return None
    start=max(0,line-1);end=min(len(source),line+18)
    # The sparse specialist reranker has signal-dependent row updates.  Its
    # source proves a positive routing/output transition, but it deliberately
    # remains a fingerprint residual until that mutation form has a bounded
    # general interpreter.  This is not a source-unavailable terminal status.
    partial=bool(residual)
    return {'classification':record[1],'fingerprint':fp,'fingerprint_complete':not partial,'residual_components':residual,
      'notes':record[4],'changed_components':record[2],
      'evidence':[{'file':'train.py','line':line,'end_line':end,'kind':'exact retained B01-C2 KWS source review','code':'\n'.join(source[start:end])}],
      'family_signature':None if partial else _semantic_graph(reviewed['mechanism']),
      'before_source_sha256':source_sha(before_sources),'source_sha256':candidate_sha,
      'reviewer':'recurrent-kws-b01-c2-direct-source-v1'}


def _kws_b01_c0_coordinate_pruning_source_review(before_sources, after_sources, campaign_key):
    """Return the exact source-pair landmark for B01-C0 coordinate pruning."""
    if campaign_key!='openevolve_v21_tiny_kws_rnn':return None
    candidate_sha=source_sha(after_sources)
    record=KWS_B01_C0_COORDINATE_PRUNING_SOURCE_REVIEWS.get(candidate_sha)
    if record is None or source_sha(before_sources)!=record[0]:return None
    try:
        from experiments.ontology_recurrent_branches import branch_profile
        reviewed=branch_profile(after_sources,campaign_key)
        if not reviewed.get('coordinate_pruning'):return None
        fp,residual=_fingerprint(reviewed)
    except (SyntaxError,ValueError,TypeError,RecursionError):
        return None
    source=after_sources.get('train.py','').splitlines();line=record[3]
    if residual or not source or not 1<=line<=len(source):return None
    start=max(0,line-1);end=min(len(source),line+24)
    return {'classification':record[1],'fingerprint':fp,'fingerprint_complete':True,'residual_components':[],
      'notes':record[4],'changed_components':record[2],
      'evidence':[{'file':'train.py','line':line,'end_line':end,'kind':'exact retained B01-C0 KWS deployment-coordinate-pruning source review','code':'\n'.join(source[start:end])}],
      'family_signature':_semantic_graph(reviewed['mechanism']),
      'before_source_sha256':source_sha(before_sources),'source_sha256':candidate_sha,
      'reviewer':'recurrent-kws-b01-c0-coordinate-pruning-direct-source-v1'}


def reviewed_template_key(sources,slots):
    """Exact full-source AST with explicitly reviewed numeric slots only.

    Every other coefficient, unit, zero, selector, axis, loop bound and branch
    remains exact. This deliberately does not use NumericSettings.
    """
    trees={name:ast.parse(source) for name,source in sources.items()};values={}
    for slot in slots:
        tree=trees.get(slot['file'])
        if tree is None:return None
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==slot['function']]
        if len(functions)!=1:return None
        calls=[n for n in ast.walk(functions[0]) if isinstance(n,ast.Call) and dotted(n.func)==slot['callee']]
        if len(calls)!=1:return None
        keywords=[k for k in calls[0].keywords if k.arg==slot['keyword']]
        if len(keywords)!=1 or not isinstance(keywords[0].value,ast.Constant):return None
        value=keywords[0].value.value
        if slot['kind']=='width':
            if type(value) is not int or value<=1:return None
        elif slot['kind']=='dropout':
            if type(value) not in (int,float) or not 0<=value<1:return None
        else:return None
        label=slot['function']+'.'+slot['callee']+'.'+slot['keyword']
        values[label]=value;keywords[0].value=ast.Constant(value='reviewed numeric slot '+slot['kind']+' '+label)
    class StripDocs(ast.NodeTransformer):
        def visit_Expr(self,node):
            return None if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) else self.generic_visit(node)
    payload={name:ast.dump(StripDocs().visit(tree),include_attributes=False) for name,tree in trees.items()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),values


def review_pair(before_sources,after_sources,campaign_key):
    if campaign_key not in CAMPAIGNS or not before_sources or not after_sources:return None
    parent_sha=source_sha(before_sources);candidate_sha=source_sha(after_sources)
    retained=_kws_b01_c0_coordinate_pruning_source_review(before_sources,after_sources,campaign_key)
    if retained is not None:return retained
    retained=_kws_b01_c2_source_review(before_sources,after_sources,campaign_key)
    if retained is not None:return retained
    if campaign_key=='openevolve_v21_tiny_kws_rnn':
        from experiments.ontology_kws_progressive_deployment_compression import review_pair as progressive_deployment_compression_review
        retained=progressive_deployment_compression_review(before_sources,after_sources)
        if retained is not None:return retained
        from experiments.ontology_kws_progressive_deployment_compression_v2 import review_pair as progressive_deployment_compression_v2_review
        retained=progressive_deployment_compression_v2_review(before_sources,after_sources)
        if retained is not None:return retained
        from experiments.ontology_kws_calibrated_deployment_compression import review_pair as calibrated_deployment_compression_review
        retained=calibrated_deployment_compression_review(before_sources,after_sources)
        if retained is not None:return retained
        from experiments.ontology_kws_calibrated_selector_refinement import review_pair as calibrated_selector_refinement_review
        retained=calibrated_selector_refinement_review(before_sources,after_sources)
        if retained is not None:return retained
        from experiments.ontology_kws_b02_c0_exit_mask_structural import review_pair as b02_c0_exit_mask_structural_review
        retained=b02_c0_exit_mask_structural_review(before_sources,after_sources)
        if retained is not None:return retained
    candidate_reference=None;template_reference=None;template_settings=None;template_keys={}
    for reference in _direct_references():
        if reference['campaign']!=campaign_key:continue
        template=reference.get('reviewed_template')
        if template and _reviewed_fingerprint(reference,campaign_key) is not None:
            slot_key=json.dumps(template['numeric_slots'],sort_keys=True)
            if slot_key not in template_keys:
                try:template_keys[slot_key]=reviewed_template_key(after_sources,template['numeric_slots'])
                except (SyntaxError,ValueError,TypeError):template_keys[slot_key]=None
            matched=template_keys[slot_key]
            if matched and matched[0]==template['whole_source_ast_sha256']:
                template_reference=reference;template_settings=matched[1]
        if reference['source_sha256']!=candidate_sha:continue
        complete_fp=_reviewed_fingerprint(reference,campaign_key)
        if complete_fp is not None:candidate_reference=reference
        if reference['parent_source_sha256']!=parent_sha:continue
        if complete_fp is not None:fp,residual=complete_fp,[]
        elif 'fingerprint_complete' in reference:
            fp=copy.deepcopy(reference.get('fingerprint') or {})
            residual=list(reference.get('residual_components') or ['complete source architecture review'])
        else:
            try:fp,residual=_fingerprint(_cached_profile(_json(after_sources),campaign_key))
            except (SyntaxError,ValueError,TypeError,RecursionError):fp,residual={},['complete source architecture review']
        return {'classification':reference['classification'],'fingerprint':fp,'fingerprint_complete':not residual,
          'residual_components':residual,'notes':reference['notes'],'changed_components':reference.get('changed_components',[]),
          'evidence':copy.deepcopy(reference['evidence']),'family_signature':copy.deepcopy(reference.get('family_signature')) if not residual else None,
          'family_label':reference.get('family_label') if not residual else None,
          'before_source_sha256':parent_sha,'source_sha256':candidate_sha,'reviewer':'recurrent-direct-source-review-v2',
          **{k:copy.deepcopy(reference[k]) for k in ['implemented','executable','execution_diagnostic','settings','training','inference'] if k in reference}}
    template_preserving=False
    if template_reference is not None:
        template=template_reference['reviewed_template']
        try:parent_template=reviewed_template_key(before_sources,template['numeric_slots'])
        except (SyntaxError,ValueError,TypeError):parent_template=None
        template_preserving=bool(parent_template and parent_template[0]==template['whole_source_ast_sha256'])
    if template_preserving:
        result={'classification':'preserving','changed_components':[],
          'notes':'Both complete source programs match the same directly reviewed architecture template; only declared numerical slots, comments or documentation differ. Every other coefficient, selector, axis, branch and operation is identical.',
          'evidence':[],'before_source_sha256':parent_sha,'source_sha256':candidate_sha,'reviewer':'recurrent-whole-source-template-v1'}
    else:
        # A deliberately bounded HAR proof covers one whole-program skeleton
        # whose existing ranked inference-coordinate pruning changes only at
        # explicitly normalized source slots.  It runs before the generic
        # interpreter, which intentionally leaves that metaprogramming form
        # unresolved.
        result=None
        if campaign_key=='uci_har_pareto_v21':
            # This is an intentionally hash-bound retained source pair for
            # proposal 81. It is not an AST normalization rule for other
            # covariance or classifier changes.
            from experiments.ontology_har_lagged_covariance import review_pair as lagged_covariance_review
            result=lagged_covariance_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # This enumerates only eleven retained RawLag hidden-width source
            # pairs. It must not classify another width edit or proposal 160's
            # self-attention model switch.
            from experiments.ontology_har_raw_lag_width_sweep import review_pair as raw_lag_width_sweep_review
            result=raw_lag_width_sweep_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # This is a static 79-row B05-C2 source-hash ledger. It does not
            # infer equivalence from another ranking-loop or compression diff.
            from experiments.ontology_har_deployment_compression import review_pair as deployment_compression_review
            result=deployment_compression_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # This is a static 59-row B04-C2 source-hash ledger. Proposal 144
            # is deliberately absent because it changes temporal pooling.
            from experiments.ontology_har_inference_readout_pruning import review_pair as inference_readout_pruning_review
            result=inference_readout_pruning_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # Ten B03-C2 third-omission pairs only; the fourth-omission family
            # has distinct compact projection dimensions and is not included.
            from experiments.ontology_har_compact_reset_projection import review_pair as compact_reset_projection_review
            result=compact_reset_projection_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # Eight B04-C1 context-resolution pairs only; other pooling edits
            # that also alter filters, widths or training are excluded.
            from experiments.ontology_har_context_pooling import review_pair as context_pooling_review
            result=context_pooling_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # Five B03-C2 24-input fourth-omission pairs; this remains separate
            # from the 25-input third-omission reset-projection route.
            from experiments.ontology_har_fourth_reset_omission import review_pair as fourth_reset_omission_review
            result=fourth_reset_omission_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # Six B03-C2 second-row expansions only; nearby additions of another reset row or mixed compression are excluded.
            from experiments.ontology_har_secondary_reset_omission import review_pair as secondary_reset_omission_review
            result=secondary_reset_omission_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # Six B03-C2 third-row expansions only; nearby additions of another reset row or mixed compression are excluded.
            from experiments.ontology_har_tertiary_reset_omission import review_pair as tertiary_reset_omission_review
            result=tertiary_reset_omission_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            # Six B03-C2 fourth-row expansions only; nearby proposals that add a new row or another compact family are excluded.
            from experiments.ontology_har_quaternary_reset_omission import review_pair as quaternary_reset_omission_review
            result=quaternary_reset_omission_review(before_sources,after_sources)
        if result is None and campaign_key=='uci_har_pareto_v21':
            from experiments.ontology_har_microbi_pruning import review_pair as microbi_pruning_review
            result=microbi_pruning_review(before_sources,after_sources)
        if result is None:
            result=_automatic_pair(before_sources,after_sources,campaign_key)
    # Candidate architecture knowledge belongs to its source, independently of
    # which parent was selected for a particular transition. Never transfer the
    # transition label when only the candidate hash matches.
    if candidate_reference is not None:
        if result is None:
            result={'classification':'uncertain','changed_components':[],
              'notes':'The candidate architecture has a complete exact-source review; this different parent comparison still needs a transition proof.',
              'evidence':[],'before_source_sha256':parent_sha,'source_sha256':candidate_sha,'reviewer':VERSION}
        result.update(fingerprint=_reviewed_fingerprint(candidate_reference,campaign_key),fingerprint_complete=True,residual_components=[],
          family_signature=copy.deepcopy(candidate_reference.get('family_signature')),family_label=candidate_reference.get('family_label'))
        result['evidence']=result.get('evidence',[])+copy.deepcopy(candidate_reference['evidence'])
        result['fingerprint_reviewer']='recurrent-direct-source-review-v2'
        result.update({k:copy.deepcopy(candidate_reference[k]) for k in ['implemented','executable','execution_diagnostic','settings','training','inference'] if k in candidate_reference})
    elif template_reference is not None:
        if result is None:
            result={'classification':'uncertain','changed_components':[],
              'notes':'The candidate matches a fully source-audited whole-program template at explicitly reviewed numeric slots; its parent transition still needs a separate proof.',
              'evidence':[],'before_source_sha256':parent_sha,'source_sha256':candidate_sha,'reviewer':VERSION}
        result.update(fingerprint=_reviewed_fingerprint(template_reference,campaign_key),fingerprint_complete=True,residual_components=[],
          family_signature=copy.deepcopy(template_reference.get('family_signature')),family_label=template_reference.get('family_label'),
          settings={'reviewed_numeric_slots':template_settings},fingerprint_reviewer='recurrent-whole-source-template-v1')
        if template_reference.get('execution_diagnostic'):
            result['fingerprint']['other']='Reference source audit (its campaign execution outcome is separate from the current proposal): '+result['fingerprint']['other']
        result['evidence']=result.get('evidence',[])+[
            {'file':name,'line':1,'code':source,'kind':'complete current source matches reviewed whole-program template; only declared numeric slots vary'} for name,source in sorted(after_sources.items())]
    return result


@lru_cache(maxsize=256)
def _cached_profile(serialized,campaign_key):
    return profile(json.loads(serialized),campaign_key)


@lru_cache(maxsize=256)
def _cached_signatures(serialized):
    sources=json.loads(serialized)
    return settings_signature(sources),affine_readout_signature(sources)


@lru_cache(maxsize=1)
def _direct_references():
    path=Path(__file__).with_name('ontology_recurrent_references.json')
    return json.loads(path.read_text(encoding='utf-8'))['rows'] if path.exists() else []







