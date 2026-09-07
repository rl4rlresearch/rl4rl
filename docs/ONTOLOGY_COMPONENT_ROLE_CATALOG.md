# Campaign component role catalog

Generated from the current categorical registry. Every row has a source-review scope; all categories and role-scoped aliases are defined in `schemas/ontology-categorical-fingerprint-v1.json`. Provisional categories cannot be published. Update this catalog with schema revisions.

See [review rules](ONTOLOGY_COMPONENT_REVIEW_RULES.md) for equivalence, role ownership, and source coverage requirements.

## Ten digit addition

| Component | Scope | Excludes | Provisional values |
| --- | --- | --- | --- |
| operand_representation | Operand information at the input boundary. | Operand order, embedding implementation, output representation. | None |
| operand_order | The rule ordering or identifying operands before encoding. | Token feature generation and positional encoding. | None |
| token_embedding | The initial feature map from token identity. | Output projections, position terms, downstream feedforward transforms. | None |
| position_features | Position features applied to input representations. | Attention-score bias and visibility masks. | None |
| attention_topology | The source-to-target relation of attention. | Masks, score formulas, position features. | None |
| attention_mask | The constraint deciding which attention edges are allowed. | Weighting of allowed edges, numerical window size. | fixed_structured_mask |
| attention_score_features | The content-dependent formula forming attention compatibility scores. | Position bias, positional encoding, weight normalization, masks. | None |
| feedforward_mechanism | Connectivity of the main pointwise feedforward transform. | Scalar nonlinearity, gate law, fixed basis. | None |
| normalization | Normalization of main feature-stream activations. | Probability links, control networks, output calibration, optimizer statistics. Distinct normalization sites require a role split if their mechanisms differ. | explicit_centering_scaling |
| residual_connectivity | Skip paths reconnecting the input stream to transformed features. | Masks, ordinary serial composition, extra gate multiplication. | None |
| parameter_generation | Coefficient construction in core mixing and feedforward operators. | Token embedding and output head factorization, coefficient values, parameter ties. | algebraic_generated, nonlinear_generated |
| parameter_sharing | Tying between independently named subsystems. | Intra-attention Q/K/V relations, ordinary repeated invocation of one module. | None |
| carry_representation | The semantic representation of decimal carry. | Operand encoding and state lifetime. | explicit_arithmetic_state |
| output_representation | The semantic quantities predicted at the result boundary. | Readout implementation and decoding algorithm. | None |
| output_decoding | Conversion of predicted quantities into result symbols. | Whole-model invocation scheduling, probability links, losses. | None |
| conditional_compute | Data-dependent selection of executed branches, operators or update indices. | Dense gate multiplication with all branches executed; early exit has its own component. | conditional_branch |
| persistent_state | Lifetime of auxiliary state across inference invocations. | Update equation, state representation, attention-specific memory. | None |
| stochastic_runtime | Random operations active in evaluated training or inference paths. | Unused declarations, initialization randomness, numeric rates. Distinct active train/inference mechanisms require a role split. | sampling |
| inference_protocol | The algorithm scheduling whole-model calls to obtain a prediction. | Per-token decoding, cache-only implementation optimization, internal routing. | None |
| learning_objective | The base supervised error functional. | Optimizer, target transformation, additional auxiliary objectives. | None |
| optimization_rule | Parameter update equation and retained optimizer statistics. | Learning rates, schedules, model state and losses. Different update equations require distinct categories, even within one optimizer family. | adaptive_gradient |
| training_data_transform | Transformations of training examples before the model boundary. | Inference views, internal representation, numeric transform strengths. | None |
| feature_modulation | An extra multiplicative control path acting on the main feature stream. | Intrinsic SiLU product, recurrent gates, MLP gates, output transforms, branch skipping. | None |
| modulation_source | Information supplying an extra feature gate's control path. | Transfer function, hidden nonlinearity, summary dimension. | None |
| modulation_transfer | The link turning an extra feature gate score into a multiplier. | The same function used for backbone activation, probabilities, attention or intrinsic cell gates. | None |
| gate_hidden_activation | Hidden scalar nonlinearity inside an extra feature gate network. | Final gate transfer and backbone/readout nonlinearity. | None |
| readout_activation | Hidden scalar nonlinearity of the prediction head. | Backbone activation, final link, control networks. | None |
| output_link | Conversion of final scores to probabilities or log probabilities. | Attention softmax, gate sigmoid, calibration and loss internals. | None |
| auxiliary_objective | Additional objectives beyond the base task loss. | Base task error, scalar loss weights, optimizer. | None |
| loss_target_transform | Transformation of targets supplied to the base loss. | Input augmentation and numerical smoothing strength. | None |
| activation | Scalar nonlinearity in the main feature transform. | Gate hidden layers, gate transfer, recurrent-cell internals, readout hidden layers, output transforms. SiLU's intrinsic product is not an extra gate. | None |
| feedforward_gate | The gate law intrinsic to the main pointwise feedforward block. | Extra feature modulation, cell gates, ungated-block scalar activation. | None |
| feedforward_basis | Deterministic basis of the main pointwise feedforward block. | Token embeddings, learned projections, numerical degree/frequency. | None |
| attention_position_bias | Position-only additive contribution to attention scores. | Q/K encoding, content scores, masking. | None |
| attention_weighting | Conversion of allowed raw scores to attention mixing weights. | Output probabilities, masks, raw score computation. | None |

## nanoGPT

| Component | Scope | Excludes | Provisional values |
| --- | --- | --- | --- |
| token_representation | The semantic unit identified by an input token. | Embedding implementation, vocabulary size, sequence length. | augmented_tokens |
| token_embedding | The initial feature map from token identity. | Output projections, position terms, downstream feedforward transforms. | parametric_embedding |
| position_encoding | Position-dependent transforms or additions to input or Q/K vectors. | Additive score bias and visibility masks. Different sites/mechanisms require a role split. | None |
| attention_topology | The source-to-target relation of attention. | Masks, score formulas, position features. | None |
| context_connection | Arrangement of receptive contexts across sequence stages. | Numerical window size, score weighting inside a context. | None |
| attention_score_features | The content-dependent formula forming attention compatibility scores. | Position bias, positional encoding, weight normalization, masks. | None |
| key_value_parameterization | Relation of key/value projections and query heads within attention. | Across-subsystem parameter ties, scores, memory. | None |
| key_value_memory | Storage or injection of token-value information within attention. | External memory and cache-only execution optimization. | None |
| projection_topology | Factorization of main attention projections. | Token embeddings, output heads, parameter ties. | None |
| feedforward_mechanism | Connectivity of the main pointwise feedforward transform. | Scalar nonlinearity, gate law, fixed basis. | None |
| normalization | Normalization of main feature-stream activations. | Probability links, control networks, output calibration, optimizer statistics. Distinct normalization sites require a role split if their mechanisms differ. | None |
| residual_connectivity | Skip paths reconnecting the input stream to transformed features. | Masks, ordinary serial composition, extra gate multiplication. | None |
| parameter_sharing | Tying between independently named subsystems. | Intra-attention Q/K/V relations, ordinary repeated invocation of one module. | None |
| external_memory | Retrieval from a store outside ordinary attention and recurrence. | Token-value memory, attention cache, recurrent hidden state. | None |
| output_readout | The map from language features to vocabulary scores. | Output transforms/links and embedding parameter ties. | None |
| output_transform | Additional score calibration, constraints or nonlinear transforms. | Probability links, head connectivity, backbone nonlinearities. | conditional_logits, normalized_logits |
| conditional_compute | Data-dependent selection of executed branches, operators or update indices. | Dense gate multiplication with all branches executed; early exit has its own component. | conditional_branch |
| persistent_state | Lifetime of auxiliary state across inference invocations. | Update equation, state representation, attention-specific memory. | None |
| stochastic_runtime | Random operations active in evaluated training or inference paths. | Unused declarations, initialization randomness, numeric rates. Distinct active train/inference mechanisms require a role split. | sampling |
| inference_protocol | The algorithm scheduling whole-model calls to obtain a prediction. | Per-token decoding, cache-only implementation optimization, internal routing. | None |
| learning_objective | The base supervised error functional. | Optimizer, target transformation, additional auxiliary objectives. | None |
| optimization_rule | Parameter update equation and retained optimizer statistics. | Learning rates, schedules, model state and losses. Different update equations require distinct categories, even within one optimizer family. | adaptive_gradient |
| training_data_transform | Transformations of training examples before the model boundary. | Inference views, internal representation, numeric transform strengths. | token_augmentation |
| feature_modulation | An extra multiplicative control path acting on the main feature stream. | Intrinsic SiLU product, recurrent gates, MLP gates, output transforms, branch skipping. | None |
| modulation_source | Information supplying an extra feature gate's control path. | Transfer function, hidden nonlinearity, summary dimension. | None |
| modulation_transfer | The link turning an extra feature gate score into a multiplier. | The same function used for backbone activation, probabilities, attention or intrinsic cell gates. | None |
| gate_hidden_activation | Hidden scalar nonlinearity inside an extra feature gate network. | Final gate transfer and backbone/readout nonlinearity. | None |
| readout_activation | Hidden scalar nonlinearity of the prediction head. | Backbone activation, final link, control networks. | None |
| output_link | Conversion of final scores to probabilities or log probabilities. | Attention softmax, gate sigmoid, calibration and loss internals. | None |
| auxiliary_objective | Additional objectives beyond the base task loss. | Base task error, scalar loss weights, optimizer. | None |
| loss_target_transform | Transformation of targets supplied to the base loss. | Input augmentation and numerical smoothing strength. | None |
| activation | Scalar nonlinearity in the main feature transform. | Gate hidden layers, gate transfer, recurrent-cell internals, readout hidden layers, output transforms. SiLU's intrinsic product is not an extra gate. | None |
| feedforward_gate | The gate law intrinsic to the main pointwise feedforward block. | Extra feature modulation, cell gates, ungated-block scalar activation. | None |
| feedforward_basis | Deterministic basis of the main pointwise feedforward block. | Token embeddings, learned projections, numerical degree/frequency. | None |
| attention_position_bias | Position-only additive contribution to attention scores. | Q/K encoding, content scores, masking. | None |
| attention_weighting | Conversion of allowed raw scores to attention mixing weights. | Output probabilities, masks, raw score computation. | None |
| attention_mask | The constraint deciding which attention edges are allowed. | Weighting of allowed edges, numerical window size. | fixed_structured_mask |

## Fashion MNIST

| Component | Scope | Excludes | Provisional values |
| --- | --- | --- | --- |
| image_representation | The image-valued object at one model invocation boundary. | The multi-view invocation scheduler and preprocessing. | None |
| spatial_preprocessing | Transform of each image before learned spatial features. | Training-only augmentation, scheduling/fusion of inference views. | None |
| spatial_operator | The operation family mixing main-stream spatial positions. | Grouping, spatial support, control networks, pooling/readout. | None |
| spatial_connectivity | Channel connectivity of the main spatial operator. | Operator family, numeric group count, control projections. | None |
| spatial_context | Parallel or nonlocal context in the main spatial extractor. | Global summaries used only to control gates, numeric kernel sizes. | None |
| scale_handling | Representation and connections of feature maps at different resolutions. | Local downsampling operator and inference views. | None |
| pooling | The reduction operator downsampling main-stream spatial features. | Gate summaries, classifier readout and kernel sizes. | None |
| channel_interaction | Extra cross-channel mixing outside the base spatial operator. | Convolution grouping and multiplicative feature modulation. | None |
| normalization | Normalization of main feature-stream activations. | Probability links, control networks, output calibration, optimizer statistics. Distinct normalization sites require a role split if their mechanisms differ. | None |
| activation | Scalar nonlinearity in the main feature transform. | Gate hidden layers, gate transfer, recurrent-cell internals, readout hidden layers, output transforms. SiLU's intrinsic product is not an extra gate. | None |
| residual_connectivity | Skip paths reconnecting the input stream to transformed features. | Masks, ordinary serial composition, extra gate multiplication. | None |
| classifier_readout | The map from final image features to class scores. | Its scalar hidden nonlinearity, output links, gates, view fusion. | None |
| output_transform | Additional score calibration, constraints or nonlinear transforms. | Probability links, head connectivity, backbone nonlinearities. | calibrated_logits, normalized_logits |
| multi_view_inference | The rule selecting transformed inputs for repeated inference. | One-invocation tensor representation, fusion, weight values. | None |
| view_aggregation | The arithmetic operator combining view predictions. | Origin of weights, number of views, coefficient values. | None |
| view_weight_policy | How view weights are produced, learned or input dependent. | Whether fixed coefficients happen to be equal; fusion arithmetic. | None |
| fixed_feature_operator | Deterministic feature construction outside learned spatial operators. | Preprocessing, learned convolution, gate summaries. | None |
| conditional_compute | Data-dependent selection of executed branches, operators or update indices. | Dense gate multiplication with all branches executed; early exit has its own component. | None |
| persistent_state | Lifetime of auxiliary state across inference invocations. | Update equation, state representation, attention-specific memory. | None |
| stochastic_runtime | Random operations active in evaluated training or inference paths. | Unused declarations, initialization randomness, numeric rates. Distinct active train/inference mechanisms require a role split. | sampling |
| learning_objective | The base supervised error functional. | Optimizer, target transformation, additional auxiliary objectives. | None |
| optimization_rule | Parameter update equation and retained optimizer statistics. | Learning rates, schedules, model state and losses. Different update equations require distinct categories, even within one optimizer family. | adaptive_gradient |
| training_data_transform | Transformations of training examples before the model boundary. | Inference views, internal representation, numeric transform strengths. | spatial_augmentation |
| feature_modulation | An extra multiplicative control path acting on the main feature stream. | Intrinsic SiLU product, recurrent gates, MLP gates, output transforms, branch skipping. | None |
| modulation_source | Information supplying an extra feature gate's control path. | Transfer function, hidden nonlinearity, summary dimension. | None |
| modulation_transfer | The link turning an extra feature gate score into a multiplier. | The same function used for backbone activation, probabilities, attention or intrinsic cell gates. | None |
| gate_hidden_activation | Hidden scalar nonlinearity inside an extra feature gate network. | Final gate transfer and backbone/readout nonlinearity. | None |
| readout_activation | Hidden scalar nonlinearity of the prediction head. | Backbone activation, final link, control networks. | None |
| output_link | Conversion of final scores to probabilities or log probabilities. | Attention softmax, gate sigmoid, calibration and loss internals. | None |
| auxiliary_objective | Additional objectives beyond the base task loss. | Base task error, scalar loss weights, optimizer. | None |
| loss_target_transform | Transformation of targets supplied to the base loss. | Input augmentation and numerical smoothing strength. | None |

## Tiny keyword spotting RNN

| Component | Scope | Excludes | Provisional values |
| --- | --- | --- | --- |
| acoustic_representation | Acoustic features at the candidate input boundary. | Candidate preprocessing, protected frontend implementation, feature dimension. | None |
| acoustic_preprocessing | Candidate transforms before the main temporal operator. | Fixed input representation, internal recurrent normalization. | None |
| temporal_operator | Connectivity of the main sequence-processing mechanism. | Recurrent update law, frame sampling, classifier aggregation. | None |
| state_update | The recurrent update law, including intrinsic gates. | Extra feature gates, state layout, initialization, standard cell nonlinearities counted again as activation. | None |
| state_structure | Semantic constituents of temporal state. | Dimensions, update law, reset policy, lifetime. | multi_component_state |
| state_reset | Initialization/reset rule for temporal state. | Update equation, numeric initial values, horizon. | None |
| temporal_schedule | The policy selecting acoustic frames to process. | Numeric stride/horizon, recurrent law, exit decisions. | None |
| temporal_aggregation | The operator reducing temporal features for prediction. | History available to that operator, accumulator implementation. | history_features |
| frequency_operator | Extra computation across acoustic frequency coordinates. | Protected frontend, temporal mixing, frequency-bin counts. | None |
| feature_interaction | Extra interactions beyond the ordinary recurrent cell. | Cell gates, extra feature modulation, classifier readout. | None |
| readout | The map from temporal features to class scores. | History representation, temporal aggregation, output links, exit timing. | None |
| readout_history | Temporal information exposed to the classifier. | Aggregation operator and history length. | None |
| exit_policy | The rule deciding when inference returns. | Frame subsampling, threshold values, gates that do not skip computation. | None |
| normalization | Normalization of main feature-stream activations. | Probability links, control networks, output calibration, optimizer statistics. Distinct normalization sites require a role split if their mechanisms differ. | explicit_centering_scaling |
| activation | Scalar nonlinearity in the main feature transform. | Gate hidden layers, gate transfer, recurrent-cell internals, readout hidden layers, output transforms. SiLU's intrinsic product is not an extra gate. | None |
| conditional_compute | Data-dependent selection of executed branches, operators or update indices. | Dense gate multiplication with all branches executed; early exit has its own component. | None |
| persistent_state | Lifetime of auxiliary state across inference invocations. | Update equation, state representation, attention-specific memory. | None |
| stochastic_runtime | Random operations active in evaluated training or inference paths. | Unused declarations, initialization randomness, numeric rates. Distinct active train/inference mechanisms require a role split. | sampling |
| learning_objective | The base supervised error functional. | Optimizer, target transformation, additional auxiliary objectives. | None |
| optimization_rule | Parameter update equation and retained optimizer statistics. | Learning rates, schedules, model state and losses. Different update equations require distinct categories, even within one optimizer family. | adaptive_gradient |
| training_data_transform | Transformations of training examples before the model boundary. | Inference views, internal representation, numeric transform strengths. | audio_augmentation |
| feature_modulation | An extra multiplicative control path acting on the main feature stream. | Intrinsic SiLU product, recurrent gates, MLP gates, output transforms, branch skipping. | None |
| modulation_source | Information supplying an extra feature gate's control path. | Transfer function, hidden nonlinearity, summary dimension. | None |
| modulation_transfer | The link turning an extra feature gate score into a multiplier. | The same function used for backbone activation, probabilities, attention or intrinsic cell gates. | None |
| gate_hidden_activation | Hidden scalar nonlinearity inside an extra feature gate network. | Final gate transfer and backbone/readout nonlinearity. | None |
| readout_activation | Hidden scalar nonlinearity of the prediction head. | Backbone activation, final link, control networks. | None |
| output_link | Conversion of final scores to probabilities or log probabilities. | Attention softmax, gate sigmoid, calibration and loss internals. | None |
| auxiliary_objective | Additional objectives beyond the base task loss. | Base task error, scalar loss weights, optimizer. | None |
| loss_target_transform | Transformation of targets supplied to the base loss. | Input augmentation and numerical smoothing strength. | None |
