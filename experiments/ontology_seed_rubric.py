"""Human-readable reference fingerprints audited against the six campaign seeds.

Applied only to the exact archived seed source hash. Descendants may inherit
them only over a transition with a recorded preserving proof.
"""
from experiments.review_ontology_sources import CORE, TASK_KEYS, task_for

AUDITED_SEED_HASHES = {
    'openevolve_v21':'28cdaa0360af90929599419815a8736b70490f79d65e0d677a4757edd1aac416',
    'tiny_adderboard_v21':'916259af4c2a25425b912c9fd33cd33ad3ce7871829f270fdb97f7c797c4d4c3',
    'uci_har_pareto_v21':'15a6265a43216e097d62628a71a802de7bee2b0d4f9ba5c7da694220c3cfdb2f',
    'openevolve_v21_nanogpt':'ce4aa4d9203ed96262ae71b4c6c73e48f7e5d0cc9b6aa894e4f4e5daf378d80d',
    'openevolve_v21_fashion_mnist':'3293cc25cf3643aed4e86fa4604c544efe8c380016798df71a4c2b72c620e6bc',
    'openevolve_v21_tiny_kws_rnn':'8088807c590729ad90c68c9bc3a240a91392df17569de70f59c87861959f0f68',
}


def seed_fingerprint(key):
    task=task_for(key)
    common={
        'input_units':'token sequence', 'input_transform':'task encoding; no additional model-side feature transform',
        'embedding':'lookup table', 'position':'additive learned position embeddings',
        'mixing':'causal attention', 'routing':'causal softmax content QK',
        'state':'no explicit persistent recurrent state', 'feedforward':'ungated MLP',
        'parameter_construction':'free learned affine matrices and lookup tables',
        'sharing':'no additional explicit parameter tying', 'normalization':'LayerNorm',
        'connectivity':'sequential residual attention and MLP blocks',
        'aggregation':'per-position hidden features', 'output':'linear token-logit readout',
        'symmetry':'no explicit canonicalization or group-equivariant construction',
        'conditional_compute':'fixed architecture; no input-dependent early exit or experts',
        'activation':'GELU', 'stochasticity':'no model-side stochastic mechanism',
        'bottleneck':'no separate factorized bottleneck', 'iteration':'fixed block stack',
        'other':'no additional mechanism in the audited seed',
    }
    if task=='addition':
        common.update(input_units='operand-pair and output-digit tokens',
                      operand_encoding='ordered decimal digit-column pairs with delimiter tokens',
                      digit_order='least-significant column first',
                      carry_representation='implicit in learned hidden features; no explicit carry register',
                      attention_scores='causal softmax QK dot product',
                      projection_relations='free Q/K/V and output projections',
                      output_factorization='autoregressive digit-token logits')
        if 'tiny' in key:
            common.update(embedding='factorized lookup table',position='additive learned positions + learned relative-distance scores',
                          routing='causal position-only softmax scores',attention_scores='learned causal relative-distance table; no content QK',
                          projection_relations='independent value and output affine maps; no Q/K projections',
                          parameter_construction='factorized free lookup/readout matrices; free affine projections',
                          bottleneck='factorized token lookup and output projection',output='factorized linear token-logit readout')
        else:
            common.update(sharing='input embedding and output readout share weights',stochasticity='dropout declared; rate is a setting')
    elif task=='nanogpt':
        common.update(input_units='language tokens',position='rotary Q/K embedding',routing='causal content QK with configured local/global windows',
                      normalization='RMS normalization of hidden features and Q/K',activation='squared ReLU MLP; sigmoid value gate; tanh logit soft cap',
                      connectivity='attention/MLP residual blocks with learned residual and initial-embedding mixing',
                      context_topology='configured local/global causal windows',attention_scores='softmax QK dot product',
                      kv_memory='token value embeddings mixed into V with an input-dependent gate',
                      projection_relations='independent learned Q/K/V/output matrices',
                      vocabulary_representation='learned token lookup with an independent output vocabulary matrix',
                      block_composition='pre-normalized attention then squared-ReLU MLP',
                      output='linear token logits with tanh soft cap')
    else:
        common.update(embedding='not applicable: continuous-valued features enter affine/convolution operators directly',
                      feedforward='no separate transformer feedforward block',parameter_construction='free learned affine/convolution/recurrent matrices',
                      aggregation='feature pooling',output='linear class-logit readout',bottleneck='no separate factorized bottleneck')
        if task=='fashion':
            common.update(input_units='grayscale image pixels',input_transform='identity model input',
                          position='implicit spatial neighborhoods; no explicit position embedding',mixing='dense Conv2d',routing='fixed local convolution neighborhoods',
                          normalization='none',connectivity='sequential convolution/pooling stages and dense head',aggregation='flattened spatial features',
                          activation='GELU',iteration='fixed feedforward stages',spatial_units='image grid',spatial_operator='dense Conv2d',
                          scale_representation='successively downsampled feature maps',spatial_readout='flattened dense head',
                          channel_interaction='joint channel mixing in dense convolutions',spatial_downsampling='max pooling')
        elif task=='kws':
            common.update(input_units='acoustic feature frames',input_transform='LayerNorm of frozen acoustic features',
                          position='implicit causal order through recurrence',mixing='GRU',routing='fixed dense input/state communication',
                          state='GRU hidden state plus online sum and frame count',normalization='input LayerNorm',
                          connectivity='single recurrent stream and class head',aggregation='online mean of recurrent outputs',
                          activation='standard GRU sigmoid/tanh gates',iteration='causal recurrent frame scan',
                          acoustic_representation='frozen log-mel frontend; normalized continuous frames',state_update='GRU',
                          state_structure='hidden state plus sum/count accumulators',temporal_schedule='fixed frame-count-only sampling',
                          readout_history='online mean of recurrent outputs',exit_policy='fixed horizon; no exit hook')
        else:
            common.update(input_units='sensor time frames',input_transform='joint continuous sensor channels',
                          position='implicit convolution/recurrence order',mixing='Conv1d + bidirectional LSTM',
                          routing='local temporal convolutions and fixed recurrent transitions',state='bidirectional LSTM hidden and cell states',
                          normalization='BatchNorm1d',connectivity='two convolution/pooling stages then bidirectional recurrence and head',
                          aggregation='final recurrent output timestep',activation='ReLU convolution blocks; standard LSTM sigmoid/tanh gates',
                          stochasticity='dropout before classifier',iteration='fixed convolution stages and bidirectional recurrent scans',
                          sensor_fusion='early joint sensor-channel processing',temporal_operator='Conv1d + bidirectional LSTM',
                          directionality='bidirectional recurrence',state_update='LSTM',frequency_representation='time-domain sensor features',
                          temporal_readout='final recurrent output timestep')
    assert set(CORE+TASK_KEYS[task].split())<=common.keys()
    return common
