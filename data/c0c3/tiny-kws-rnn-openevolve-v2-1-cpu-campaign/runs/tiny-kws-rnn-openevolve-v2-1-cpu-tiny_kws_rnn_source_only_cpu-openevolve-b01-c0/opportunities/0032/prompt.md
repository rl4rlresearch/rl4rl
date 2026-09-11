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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4910196462845663493, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 940581720, "recurrent_steps": 19560, "total_inference_macs": 941344560, "training_seconds": 73.7940290409606, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.42476536101358797}
prior_hypothesis: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 24 scheduled frames while reducing total inference MACs by approximately 4% versus the verified 25-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 124-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.48% versus the passing 125-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 125 to 124 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Successive additive-readout widths of 127, 126, and 125 units all passed; the 125-unit model achieved 85.031% accuracy, making a one-channel trim the smallest isolated structural test despite its narrow margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5697376000555829007, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 1091448000, "recurrent_steps": 20375, "total_inference_macs": 1092256480, "training_seconds": 115.06732637505047, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.39709994076219807}

RECENT RESULT
hypothesis: A 123-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 124-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 124 to 123 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 124-unit model achieved 86.012% accuracy, more than one percentage point above the requirement, and every successive additive-readout width from 128 through 124 has passed; a one-channel trim is therefore the smallest isolated structural cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5612212431919158097, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 54537, "peak_hidden_elements": 126464, "recurrent_macs": 1075127625, "recurrent_steps": 20375, "total_inference_macs": 1075929585, "training_seconds": 98.42873675003648, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4085447440118146}

RECENT RESULT
hypothesis: A 122-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 123-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 123 to 122 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 123 has passed; the 123-unit model achieved 85.276% accuracy, so another isolated one-channel reduction is the most direct test of the remaining width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5527686537884169443, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 53728, "peak_hidden_elements": 125440, "recurrent_macs": 1058929500, "recurrent_steps": 20375, "total_inference_macs": 1059724940, "training_seconds": 108.86680608382449, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.40788664086464727}

RECENT RESULT
hypothesis: A 121-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 122-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 122 to 121 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 122 has passed; the 122-unit model achieved 85.644% accuracy, so another isolated one-channel reduction is the most informative test of the width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5443798318450863045, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 52925, "peak_hidden_elements": 124416, "recurrent_macs": 1042853625, "recurrent_steps": 20375, "total_inference_macs": 1043642545, "training_seconds": 100.74323433404788, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40282296958876534}

RECENT RESULT
hypothesis: A 120-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 121-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 121 to 120 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 121 has passed, and the latest 121-unit model achieved 86.503% accuracy, providing a 1.503-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5360547773619238903, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1026900000, "recurrent_steps": 20375, "total_inference_macs": 1027682400, "training_seconds": 90.98146774992347, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4024813037708493}

RECENT RESULT
hypothesis: A 119-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.54% versus the passing 120-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 120 to 119 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 120 has passed, and the 120-unit model achieved 86.135% accuracy, providing a 1.135-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5277934903389297017, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 51337, "peak_hidden_elements": 122368, "recurrent_macs": 1011068625, "recurrent_steps": 20375, "total_inference_macs": 1011844505, "training_seconds": 79.4365180421155, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4005632459020322}

RECENT RESULT
hypothesis: A 118-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.55% versus the passing 119-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 119 to 118 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 119 has passed, and the latest 119-unit model achieved 86.135% accuracy, leaving a 1.135-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5195959707761037387, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 50552, "peak_hidden_elements": 121344, "recurrent_macs": 995359500, "recurrent_steps": 20375, "total_inference_macs": 996128860, "training_seconds": 50.26036816602573, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.41964874267578123}

RECENT RESULT
hypothesis: A 117-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.56% versus the passing 118-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 118 to 117 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 118 has passed; the 118-unit model achieved 85.153% accuracy, making the next isolated one-channel reduction the most informative test of the recurrent-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5114622186734460013, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 979772625, "recurrent_steps": 20375, "total_inference_macs": 980535465, "training_seconds": 70.04316212516278, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4050237082264906}

RECENT RESULT
hypothesis: A 116-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 117-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 117 to 116 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 117 has passed; the current 117-unit model achieved 85.767% accuracy, providing evidence for one more isolated channel reduction as the next test of the width boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5033922340309564895, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49000, "peak_hidden_elements": 119296, "recurrent_macs": 964308000, "recurrent_steps": 20375, "total_inference_macs": 965064320, "training_seconds": 73.46372124995105, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4151908453256806}

RECENT RESULT
hypothesis: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 24 scheduled frames while reducing total inference MACs by approximately 4% versus the verified 25-step model.
change: Remove one late intermediate frame from the schedule while preserving the first processed frame pattern, endpoint coverage, architecture, and training procedure.
mechanism: Single-step causal schedule trim
evidence_used: The 117-unit, 25-step model achieved 85.767% accuracy, while reducing width to 116 narrowly failed at 84.908%; testing a minimal temporal reduction at the passing width explores a different cost axis with a larger potential MAC improvement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4910196462845663493, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 940581720, "recurrent_steps": 19560, "total_inference_macs": 941344560, "training_seconds": 73.7940290409606, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.42476536101358797}

RECENT RESULT
hypothesis: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 23 scheduled frames while reducing total inference MACs by approximately 4.2% versus the verified 24-step model.
change: Remove one additional late intermediate frame while preserving the recurrent width, earlier frame pattern, final-frame coverage, and training procedure.
mechanism: Single-step causal schedule trim
evidence_used: The 117-unit model passed at both 25 steps (85.767%) and 24 steps (85.153%), whereas reducing width to 116 narrowly failed; this makes another isolated temporal-step reduction the most informative lower-cost test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4705770738956866973, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 901390815, "recurrent_steps": 18745, "total_inference_macs": 902153655, "training_seconds": 53.92610116698779, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4386127027265865}

RECENT RESULT
hypothesis: The 117-unit GRU will retain at least 85% validation accuracy with 23 recurrent steps when the second early frame is removed instead of a late intermediate frame, reducing total inference MACs by approximately 4.2%.
change: Preserve the passing 24-frame schedule’s first frame, late intermediate coverage, and final frame, but remove its second processed frame.
mechanism: Early-frame redundancy pruning
evidence_used: The 24-step model passed at 85.153%, while the prior 23-step schedule that removed a late intermediate frame fell to 84.294%; this directly motivates testing whether that failure was caused by losing informative late speech rather than by the step count itself.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4705770738956866973, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 901390815, "recurrent_steps": 18745, "total_inference_macs": 902153655, "training_seconds": 48.89489066693932, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4309426243319833}



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
