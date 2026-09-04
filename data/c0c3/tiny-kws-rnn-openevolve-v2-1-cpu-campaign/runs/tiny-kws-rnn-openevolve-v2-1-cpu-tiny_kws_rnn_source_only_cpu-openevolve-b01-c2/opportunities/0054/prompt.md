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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1316557537440635989, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.385276073619632, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 250977600, "recurrent_steps": 17429, "total_inference_macs": 252400140, "training_seconds": 70.64176125009544, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4910184052824243}
prior_hypothesis: Lowering the qualified penultimate exit threshold from 0.81 to 0.79 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,744,660.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1320723469530640300, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.447852760736197, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 251712000, "recurrent_steps": 17480, "total_inference_macs": 253198800, "training_seconds": 69.30112158297561, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.48908563157532114}
prior_hypothesis: Lowering the penultimate exit threshold from 0.85 to 0.83 will retain validation accuracy at or above 85% while reducing total inference MACs below 253,512,000.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1313290139722985549, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.336196319018406, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 250401600, "recurrent_steps": 17389, "total_inference_macs": 251773740, "training_seconds": 75.76942583289929, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4909987373586081}
prior_hypothesis: Lowering the qualified penultimate exit threshold from 0.77 to 0.75 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,165,240.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1318354606185343731, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.412269938650308, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 251294400, "recurrent_steps": 17451, "total_inference_macs": 252744660, "training_seconds": 63.470747916959226, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.49129469233787865}
prior_hypothesis: Lowering the penultimate exit threshold from the qualified 0.83 to 0.81 will retain validation accuracy at or above 85% while reducing total inference MACs below 253,198,800.

## Recent verification evidence

RECENT RESULT
hypothesis: The qualified 22-step schedule will retain at least 85% validation accuracy when one final-hidden coordinate is omitted from the classifier, reducing total inference MACs from 259,218,900 to 259,213,195 and parameters from 16,020 to 16,013.
change: Adopt the qualified frames 3–21, 24, 26, and 28 schedule and reduce the bias-free classifier input from 180 to 179 features by excluding one coordinate only from its final-hidden branch.
mechanism: Single-coordinate classifier-path pruning
evidence_used: The 22-step design achieved 85.40% accuracy at 259,218,900 MACs, while removing seven classifier biases preserved qualification. This equally small seven-weight reduction retains both complete mean and maximum summaries and 59 of 60 final-hidden coordinates, avoiding the aggressive recurrent-input and classifier-feature reductions that failed.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352095469051998338, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16013, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259213195, "training_seconds": 30.330491374945268, "validation_accuracy": 0.8269938650306748, "validation_cross_entropy": 0.502046325309145}

RECENT RESULT
hypothesis: Processing frames 3–21, 26, and 28 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total dense inference MACs to approximately 247,482,900.
change: Remove frame 24 from the qualified 22-step schedule, preserving uninterrupted early recurrence through frame 21 and the proven late frames 26 and 28.
mechanism: Late-bridge frame pruning
evidence_used: The 22-step frames 3–21, 24, 26, and 28 design achieved 85.40%, while removing frame 3 failed and prior evidence identifies frames 26 and 28 as important; frame 24 is therefore the most informative remaining single-frame pruning candidate.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1290908465357080035, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 246456000, "recurrent_steps": 17115, "total_inference_macs": 247482900, "training_seconds": 31.302681958070025, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.5169063638324386}

RECENT RESULT
hypothesis: A 59-unit GRU using the qualified 22-frame schedule will retain at least 85% validation accuracy while reducing total dense inference MACs from 259,218,900 to approximately 251,724,975.
change: Adopt the qualified non-affine normalization, bias-free seven-logit head, and frames 3–21, 24, 26, and 28 schedule, while reducing GRU width from 60 to 59 units.
mechanism: Minimal recurrent-width contraction
evidence_used: The 60-unit 22-step design achieved 85.40% accuracy, while tested 21-step schedules failed; preserving all qualified observations and making the smallest possible recurrent-width reduction targets MACs without discarding additional temporal or spectral information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1313035774116954481, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 15576, "peak_hidden_elements": 91136, "recurrent_macs": 250715190, "recurrent_steps": 17930, "total_inference_macs": 251724975, "training_seconds": 44.02463495801203, "validation_accuracy": 0.8294478527607362, "validation_cross_entropy": 0.5033503058497891}

RECENT RESULT
hypothesis: Processing frames 3–11, 13–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total dense inference MACs to approximately 247,482,900.
change: Remove only the central frame 12 from the qualified 22-step schedule, preserving both temporal boundaries of the dense early segment and all three proven late observations.
mechanism: Interior-frame temporal pruning
evidence_used: Removing frame 3 or frame 24 from the qualified schedule failed, indicating that the early boundary and late bridge are important; pruning one interior frame tests redundancy within the densely sampled 3–21 region while introducing only a single two-frame gap.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1290908465357080035, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 246456000, "recurrent_steps": 17115, "total_inference_macs": 247482900, "training_seconds": 37.79332912503742, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4972788804879218}

RECENT RESULT
hypothesis: On the qualified 22-frame schedule, exiting examples with at least 0.95 penultimate-step posterior confidence will preserve validation accuracy at or above 85% and reduce inference MACs if more than approximately 8.75% of examples skip the final GRU step.
change: Adopt the qualified frames 3–21, 24, 26, and 28 schedule; execute the learned classifier only for the last two steps; and exit high-confidence examples after step 21.
mechanism: Confidence-gated penultimate early exit
evidence_used: The 22-step schedule achieved 85.40% accuracy, while every tested unconditional 21-step schedule failed. Selective final-step omission therefore targets temporal savings without withholding the final observation from uncertain examples.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1337223828004775022, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.695705521472394, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 254620800, "recurrent_steps": 17682, "total_inference_macs": 256362120, "training_seconds": 77.47736016707495, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4873943773515385}

RECENT RESULT
hypothesis: Lowering the qualified penultimate exit threshold from 0.95 to 0.90 will preserve validation accuracy at or above 85% while increasing early exits and reducing total inference MACs below 256,362,120.
change: Adopt the qualified 22-frame schedule, avoid classifier MACs before the penultimate step, and exit examples with at least 0.90 posterior confidence after step 21.
mechanism: Relaxed confidence-gated penultimate exit
evidence_used: The 0.95 confidence-gated design achieved 85.52% accuracy while reducing mean execution to 21.696 steps; unlike every tested unconditional 21-step schedule, it demonstrates that confidence-selective final-frame omission preserves qualification, motivating a measured threshold relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1328075114395353790, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.558282208588956, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 253008000, "recurrent_steps": 17570, "total_inference_macs": 254608200, "training_seconds": 77.30697458283976, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4877863316448188}

RECENT RESULT
hypothesis: Lowering the qualified penultimate exit threshold from 0.90 to 0.85 will preserve validation accuracy at or above 85% while reducing total inference MACs below 254,608,200.
change: Adopt the qualified 22-frame schedule, non-affine normalization, and bias-free seven-logit head; avoid classifier MACs before the penultimate step; and exit examples with at least 0.85 posterior confidence after step 21.
mechanism: Relaxed confidence-gated penultimate exit
evidence_used: Lowering the threshold from 0.95 to 0.90 increased early exits, reduced total MACs from 256,362,120 to 254,608,200, and retained 85.40% accuracy; the small observed accuracy change motivates another measured relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1322357168389465520, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.47239263803681, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 252000000, "recurrent_steps": 17500, "total_inference_macs": 253512000, "training_seconds": 82.90616049990058, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.48912005278230447}

RECENT RESULT
hypothesis: Lowering the penultimate exit threshold from 0.85 to 0.83 will retain validation accuracy at or above 85% while reducing total inference MACs below 253,512,000.
change: Exit additional high-confidence examples after step 21 by changing only the posterior-confidence threshold to 0.83.
mechanism: Incrementally relaxed confidence-gated penultimate exit
evidence_used: Threshold reductions from 0.95 to 0.90 and then 0.85 consistently reduced MACs while remaining qualified; the 0.85 design achieved 85.15% accuracy and 253,512,000 MACs, supporting a smaller 0.02 relaxation given its narrow accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1320723469530640300, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.447852760736197, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 251712000, "recurrent_steps": 17480, "total_inference_macs": 253198800, "training_seconds": 69.30112158297561, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.48908563157532114}

RECENT RESULT
hypothesis: Lowering the penultimate exit threshold from the qualified 0.83 to 0.81 will retain validation accuracy at or above 85% while reducing total inference MACs below 253,198,800.
change: Adopt a 0.81 posterior-confidence threshold so more examples skip the final recurrent step.
mechanism: Incrementally relaxed confidence-gated penultimate exit
evidence_used: Reducing the threshold from 0.85 to 0.83 preserved 85.15% accuracy and reduced recurrent execution by 20 steps, supporting another cautious 0.02 relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1318354606185343731, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.412269938650308, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 251294400, "recurrent_steps": 17451, "total_inference_macs": 252744660, "training_seconds": 63.470747916959226, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.49129469233787865}

RECENT RESULT
hypothesis: Lowering the qualified penultimate exit threshold from 0.81 to 0.79 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,744,660.
change: Adopt a 0.79 posterior-confidence threshold so additional examples skip the final recurrent step.
mechanism: Incrementally relaxed confidence-gated penultimate exit
evidence_used: The 0.81 threshold achieved 85.28% accuracy and 252,744,660 MACs, while successive reductions from 0.95 through 0.81 all remained qualified; this supports testing one more measured 0.02 relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1316557537440635989, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.385276073619632, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 250977600, "recurrent_steps": 17429, "total_inference_macs": 252400140, "training_seconds": 70.64176125009544, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4910184052824243}

RECENT RESULT
hypothesis: Lowering the qualified penultimate exit threshold from 0.79 to 0.77 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,400,140.
change: Avoid classifier MACs before step 21 and exit examples whose penultimate-step posterior confidence is at least 0.77, while preserving the qualified model and 22-frame schedule.
mechanism: Incrementally relaxed confidence-gated penultimate exit
evidence_used: Thresholds from 0.95 through 0.79 all qualified; the 0.79 design achieved 85.28% accuracy and 252,400,140 MACs, leaving a modest accuracy margin for another measured 0.02 relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1315332263296517074, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.36687116564417, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 250761600, "recurrent_steps": 17414, "total_inference_macs": 252165240, "training_seconds": 69.82432112493552, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4906076162139331}

RECENT RESULT
hypothesis: Lowering the qualified penultimate exit threshold from 0.77 to 0.75 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,165,240.
change: Exit additional examples after recurrent step 21 by changing only the posterior-confidence threshold to 0.75.
mechanism: Incrementally relaxed confidence-gated penultimate exit
evidence_used: Every tested threshold from 0.95 through 0.77 qualified and progressively reduced MACs; the 0.77 design achieved 85.15% accuracy at 252,165,240 MACs, motivating one more cautious 0.02 relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1313290139722985549, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.336196319018406, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 250401600, "recurrent_steps": 17389, "total_inference_macs": 251773740, "training_seconds": 75.76942583289929, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4909987373586081}



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
