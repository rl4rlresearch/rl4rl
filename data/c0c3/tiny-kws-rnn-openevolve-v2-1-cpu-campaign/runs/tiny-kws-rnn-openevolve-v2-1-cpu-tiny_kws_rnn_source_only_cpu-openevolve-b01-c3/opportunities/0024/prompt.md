# Improve a recurrent keyword spotter

You are an autonomous ML engineer improving a learned causal recurrent model
that classifies one-second speech recordings into eight spoken commands.

## Goal

Produce a model with at least 85% accuracy on the fixed speaker-disjoint public
validation split, then minimize exact dense inference MACs. Among equal-MAC
models, fewer executed recurrent steps wins; among exact MAC-and-step ties,
fewer learned parameters wins. Every verification starts from fresh random
initialization and presents exactly 50,000 training clips drawn from a protected
training-speaker split.

The protected frontend supplies batches shaped `[batch, 32, 20]`: 32 causal
time frames with 20 normalized log-mel bands. `train.py` owns the model,
optimizer, loss, temporal augmentation, batch size, gradient handling, and
schedule. Keep its five top-level function interfaces intact.

The model interface is deliberately recurrent and evaluator-driven:

- `initial_state(batch_size, device, dtype)` returns batch-first tensor state,
  or a tuple/list of batch-first tensor states;
- `recurrent_step(frame, state)` updates that state from one `[batch, 20]`
  frame;
- `classify(state)` returns `[batch, 8]` logits;
- optional `recurrent_sequence(frames, state)` may run a standard causal
  sequence module efficiently, but must be numerically equivalent to repeated
  `recurrent_step` calls;
- optional `frame_schedule(available_frames)` returns 2–64 unique increasing
  input-frame indices, allowing causal striding;
- optional `exit_mask(state, logits, step, total_steps)` returns one boolean per
  active example after the mandatory first two recurrent steps.

All learned matrix operations must use `nn.Linear`, the standard
`nn.RNN`/`nn.GRU`/`nn.LSTM` modules, or their corresponding cell modules. Their
exact executed MACs are counted with protected runtime hooks over the complete
validation set. Bidirectional recurrence is rejected. Direct matmul, functional linear,
convolutions, and manually created Parameters are rejected because they could
bypass that counter. Dense matrices receive no credit for zero weights; only
structural reductions reduce cost. Elementwise gates, nonlinearities,
normalization, and recurrence logic remain flexible.

The verifier requires a state updated across at least two causal steps, material
dependence of the next state on the prior state, logits that materially depend
on recurrent output, learned recurrent-path weight changes, no complete-input
classifier bypass, and complete accounting of every executed recurrent step.
Layer C uses recordings from speakers absent from both search training and
public validation.

Public feedback includes accuracy, cross-entropy, the exact lexicographic
`inference_cost`, total and recurrent MACs, recurrent-step summaries, parameters,
peak hidden elements, training exposure, and training time.

## Work boundaries

Minimize inference_cost. Required result: validation_accuracy >= 0.85.
Editable source files: train.py.
Results reported after each verification: validation_accuracy, validation_cross_entropy, inference_cost, total_inference_macs, recurrent_macs, recurrent_steps, mean_recurrent_steps, median_recurrent_steps, p95_recurrent_steps, maximum_recurrent_steps, parameters, peak_hidden_elements, examples_processed, optimizer_steps, training_seconds, batch_size.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, external datasets, pretrained weights, or any
surrounding repository. Do not run training or validation yourself and do not
generate hidden alternatives. Return one patch for one implementation;
verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2588686809637269777, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 495611280, "recurrent_steps": 13040, "total_inference_macs": 496282840, "training_seconds": 75.18150233407505, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.46482491522478914}
prior_hypothesis: A 103-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 505,169,600 to approximately 496,282,840.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2497201760115921605, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 38125, "peak_hidden_elements": 103936, "recurrent_macs": 478085520, "recurrent_steps": 13040, "total_inference_macs": 478744040, "training_seconds": 85.76904862490483, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.45338834984902227}
prior_hypothesis: A 101-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 487,474,320 to approximately 478,744,040.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2407349157574880017, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 460872720, "recurrent_steps": 13040, "total_inference_macs": 461518200, "training_seconds": 64.19769616704434, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4651131003912241}
prior_hypothesis: A 99-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 470,092,000 to approximately 461,518,200.

## Recent verification evidence

RECENT RESULT
hypothesis: A 106-unit GRU with the proven batch-64 training regime and 16-frame even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.4% versus the qualified 108-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 116 to 106 units while preserving the validated training procedure and frame schedule.
mechanism: Boundary-seeking recurrent width reduction
evidence_used: The 108-unit design qualified at 85.399% accuracy and 541,499,040 total MACs after the 110-unit design qualified at 86.012%; this supports another two-unit boundary test, projected near 523,177,840 total MACs.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 107-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 541,499,040 to approximately 532,299,320.
change: Reduce the recurrent state, temporal summary, and classifier width from 108 to 107 units without changing training or temporal coverage.
mechanism: One-unit boundary width trim
evidence_used: The 108-unit design qualified at 85.399% accuracy, and the 110-unit design qualified at 86.012%; a one-unit reduction is the smallest informative capacity test below the current boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2776554249620885873, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 42313, "peak_hidden_elements": 110080, "recurrent_macs": 531601680, "recurrent_steps": 13040, "total_inference_macs": 532299320, "training_seconds": 108.52942700008862, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.45476566618936926}

RECENT RESULT
hypothesis: A 106-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 532,299,320 to approximately 523,177,840.
change: Reduce the recurrent state, temporal summary, and classifier width from 110 to 106 units while preserving the validated optimizer, augmentation, and even-index frame schedule.
mechanism: One-unit boundary width trim
evidence_used: The adjacent 107-unit design achieved 86.135% accuracy at 532,299,320 MACs; the prior 106-unit attempt timed out without producing contrary accuracy evidence, so retrying this smallest structural reduction directly tests the next unresolved cost boundary.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: A 105-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 532,299,320 to approximately 514,134,600.
change: Reduce recurrent-state, temporal-summary, and classifier width from the current 116 units to 105 while preserving training and temporal coverage.
mechanism: Nonduplicate boundary-width trim
evidence_used: The 107-unit design achieved 86.135% accuracy with 532,299,320 MACs; because the subsequent 106-unit submission reproduced an already verified implementation, 105 units is the next untested lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2681804306138924533, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 40893, "peak_hidden_elements": 108032, "recurrent_macs": 513450000, "recurrent_steps": 13040, "total_inference_macs": 514134600, "training_seconds": 113.32499116589315, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.47585878898761025}

RECENT RESULT
hypothesis: A 104-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 514,134,600 to approximately 505,169,600.
change: Reduce the recurrent state, temporal summary, and classifier width from 105 to 104 units while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 105-unit design achieved 85.644% accuracy at 514,134,600 MACs, leaving a 0.644-point margin and making a one-unit reduction the smallest informative test below the current cost boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2635041502015558832, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 504491520, "recurrent_steps": 13040, "total_inference_macs": 505169600, "training_seconds": 115.78899174998514, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.47056287519770895}

RECENT RESULT
hypothesis: A 103-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 505,169,600 to approximately 496,282,840.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 107 to 103 units while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 104-unit design achieved 86.626% accuracy at 505,169,600 MACs, leaving a 1.626-point margin and motivating the smallest untested structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2588686809637269777, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 495611280, "recurrent_steps": 13040, "total_inference_macs": 496282840, "training_seconds": 75.18150233407505, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.46482491522478914}

RECENT RESULT
hypothesis: A 102-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 496,282,840 to approximately 487,474,320.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from the current 110 units to 102 while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 103-unit design qualified at 85.644% accuracy and 496,282,840 MACs, leaving a 0.644-point margin; 102 units is the next untested lower-cost boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2542740229004057368, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 38808, "peak_hidden_elements": 104960, "recurrent_macs": 486809280, "recurrent_steps": 13040, "total_inference_macs": 487474320, "training_seconds": 79.22702487488277, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46595370988904333}

RECENT RESULT
hypothesis: A 101-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 487,474,320 to approximately 478,744,040.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 102 to 101 units while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 102-unit design achieved 85.276% accuracy at 487,474,320 MACs; 101 units is the next untested structural reduction, while the failed 15-frame result indicates temporal coverage should remain unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2497201760115921605, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 38125, "peak_hidden_elements": 103936, "recurrent_macs": 478085520, "recurrent_steps": 13040, "total_inference_macs": 478744040, "training_seconds": 85.76904862490483, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.45338834984902227}

RECENT RESULT
hypothesis: A 64-unit acoustic RNN feeding a 128-unit temporal RNN will retain at least 85% accuracy while reducing estimated total inference MACs from 478,744,040 to approximately 392,660,480.
change: Replace the single 103-unit GRU with two standard causal tanh RNNs, preserve all 16 scheduled frames, and classify from the mean upper-layer output plus final upper- and lower-layer states.
mechanism: Bottlenecked stacked tanh recurrence with multi-scale readout
evidence_used: The 101-unit GRU qualified at 85.276%, while reducing coverage to 15 frames collapsed accuracy to 80.25%. This tests whether gated full-width updates—not temporal coverage—are the costly assumption; standard sequence modules also avoid the execution problem encountered by the hierarchical GRU attempt.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2048176812595343648, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 33328, "peak_hidden_elements": 164352, "recurrent_macs": 390574080, "recurrent_steps": 13040, "total_inference_macs": 392660480, "training_seconds": 84.3896249581594, "validation_accuracy": 0.803680981595092, "validation_cross_entropy": 0.5804961783754313}

RECENT RESULT
hypothesis: A 100-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 478,744,040 to approximately 470,092,000.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 100 units while preserving the qualified batch-64 training procedure and 16-frame even-index schedule.
mechanism: One-unit recurrent width boundary trim
evidence_used: The adjacent 101-unit design qualified at 85.276% accuracy and 478,744,040 MACs; 100 units is the smallest untested structural reduction and isolates the next cost boundary without changing temporal coverage or optimization.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2452071402972862488, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 469440000, "recurrent_steps": 13040, "total_inference_macs": 470092000, "training_seconds": 63.406882541952655, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44861407133699194}

RECENT RESULT
hypothesis: A 99-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 470,092,000 to approximately 461,518,200.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 100 to 99 units while preserving the qualified training procedure and 16-frame schedule.
mechanism: One-unit recurrent width boundary trim
evidence_used: The adjacent 100-unit design achieved 85.521% accuracy at 470,092,000 MACs; its 0.521-point margin makes 99 units the smallest untested lower-cost boundary, while the failed 15-frame experiment supports keeping temporal coverage unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2407349157574880017, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 460872720, "recurrent_steps": 13040, "total_inference_macs": 461518200, "training_seconds": 64.19769616704434, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4651131003912241}

RECENT RESULT
hypothesis: A 98-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 461,518,200 to approximately 453,022,640.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 101 to 98 units while preserving the qualified batch-64 training procedure and 16-frame even-index schedule.
mechanism: One-unit recurrent width boundary trim
evidence_used: The adjacent 99-unit design achieved 85.521% accuracy at 461,518,200 MACs; 98 units is the next untested structural reduction, while the failed 15-frame design indicates temporal coverage should remain unchanged.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2363035023921974192, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 452383680, "recurrent_steps": 13040, "total_inference_macs": 453022640, "training_seconds": 57.88504658290185, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4804356393638564}



Use the available technical evidence to choose the most informative next
change. Treat unsuccessful or malformed work as evidence when a useful
subject-level reason is provided. Do not invent missing evidence.

## Response

Return these short metadata lines followed by one or more exact
`SEARCH`/`REPLACE` blocks that together produce one implementation:

`MECHANISM: <a concise free-form name for the computational idea>`

`HYPOTHESIS: <a falsifiable claim grounded in the evidence above>`

`INTENDED_EDIT: <what this patch changes>`

`EVIDENCE: <the most relevant prior result and why it motivates this patch>`

Start each block with `<<<<<<< SEARCH`, put the exact existing lines next, use a
line containing `=======` as the divider, put the replacement lines after it,
and finish the block with `>>>>>>> REPLACE`.

Every `SEARCH` section must be nonempty and match exactly once after earlier
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.
