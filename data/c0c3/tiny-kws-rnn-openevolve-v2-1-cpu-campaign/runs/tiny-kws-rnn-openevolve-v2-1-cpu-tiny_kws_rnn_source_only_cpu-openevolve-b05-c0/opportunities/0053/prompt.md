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
verified_results: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215509820454169655, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14880, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233028060, "training_seconds": 32.01590104191564, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4756409065855061}
prior_hypothesis: Omitting a second mean-summary coordinate will retain at least 85% validation accuracy while reducing total inference by another 6,520 MACs and parameters by eight, with all 21 recurrent steps unchanged.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing frames 3, 5, 6, 8, 10–23, 27, and 29 will retain at least 85% validation accuracy while reducing recurrent execution from 21 to 20 steps.
change: Preserve frame 8 and remove frame 9 from the verified 21-step schedule.
mechanism: Progressive early-interior frame ablation
evidence_used: Removing frame 7 achieved 85.15%, while removing frame 8 fell to 84.05%; ablating frame 9 next tests whether frame 8 is specifically important rather than indicating a general 20-step limit.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1157881041036638316, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 522, "p95_recurrent_steps": 20, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 221223600, "recurrent_steps": 16300, "total_inference_macs": 221979920, "training_seconds": 35.85666554188356, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4764407198853288}

RECENT RESULT
hypothesis: Processing frames 3, 5, 6, 8, 9, 11–23, 27, and 29 will retain at least 85% validation accuracy while reducing recurrent execution from 21 to 20 steps.
change: Preserve frames 8 and 9 while removing frame 10 from the verified 21-step schedule.
mechanism: Progressive early-interior frame ablation
evidence_used: Removing frame 7 achieved 85.15%, while removing frame 8 scored 84.05% and removing frame 9 scored 84.42%; testing frame 10 is the next direct ablation that preserves every early frame currently implicated as important.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1157881041036638316, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 522, "p95_recurrent_steps": 20, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 221223600, "recurrent_steps": 16300, "total_inference_macs": 221979920, "training_seconds": 32.23693195800297, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4757479053333493}

RECENT RESULT
hypothesis: Disabling LayerNorm’s learned affine terms will preserve at least 85% accuracy and identical MACs and recurrent steps while reducing parameters from 14,896 to 14,856.
change: Retain input normalization but remove its 20 learned scales and 20 learned biases.
mechanism: Absorb redundant normalization affine into GRU input weights
evidence_used: The verified 21-step design achieved 85.15%, while three 20-step schedules and width-57 failed; preserving all recurrent computation is therefore lower risk. LayerNorm’s affine transform is representationally redundant before the GRU’s learned input weights and biases.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215577839078349071, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14856, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233041100, "training_seconds": 30.39917366602458, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.47644409928585124}

RECENT RESULT
hypothesis: Reducing the classifier input from 116 to 115 features by omitting one mean-summary coordinate will retain at least 85% validation accuracy while preserving all 21 recurrent steps and reducing validation inference by 6,520 MACs and eight parameters.
change: Keep the verified GRU and frame schedule unchanged, but classify from 57 mean-summary features plus all 58 final-hidden features.
mechanism: Single temporal-summary coordinate pruning
evidence_used: The 21-step width-58 model achieved 85.15%, whereas width-57 and several 20-step variants failed; this motivates a minimal classifier-only structural reduction that preserves recurrent capacity and every selected frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215543829766259383, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14888, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233034580, "training_seconds": 39.82155370898545, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4847530809648198}

RECENT RESULT
hypothesis: Omitting a second mean-summary coordinate will retain at least 85% validation accuracy while reducing total inference by another 6,520 MACs and parameters by eight, with all 21 recurrent steps unchanged.
change: Classify from 56 mean-summary features and all 58 final-hidden features, reducing the classifier input width from 115 to 114.
mechanism: Progressive temporal-summary coordinate pruning
evidence_used: Omitting one mean-summary coordinate retained 85.03% accuracy and reduced both MACs and parameters; pruning one adjacent coordinate is the smallest evidence-backed continuation that can improve the primary MAC objective without altering the verified recurrent path or schedule.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215509820454169655, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14880, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233028060, "training_seconds": 32.01590104191564, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4756409065855061}

RECENT RESULT
hypothesis: Omitting a third mean-summary coordinate will retain at least 85% validation accuracy while preserving all 21 recurrent steps and reducing total inference by another 6,520 MACs and parameters by eight.
change: Reduce the classifier input width from 114 to 113 by using 55 mean-summary features alongside all 58 final-hidden features.
mechanism: Progressive temporal-summary coordinate pruning
evidence_used: Omitting one and then two mean-summary coordinates each retained 85.03% accuracy while reducing MACs and parameters, making one additional coordinate the smallest evidence-backed structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215475811142079927, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14872, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233021540, "training_seconds": 33.41884083417244, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.46742825186325726}

RECENT RESULT
hypothesis: Pooling the final three mean-summary coordinates into one feature will retain at least 85% validation accuracy while reducing classifier input width from 114 to 113, saving 6,520 validation MACs and eight parameters.
change: Replace the two discarded summary coordinates with a single elementwise pooled feature and shrink the classifier accordingly, preserving information from all 58 summary channels without learned matrix cost.
mechanism: Zero-MAC pooled temporal-summary compression
evidence_used: Dropping two summary coordinates retained 85.03% accuracy, but dropping a third fell to 83.93%; pooling the next coordinate with the already omitted coordinates directly tests whether preserving their aggregate information enables the same structural reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compressing the final four mean-summary coordinates into one pooled feature will preserve at least 85% validation accuracy while reducing classifier width from 114 to 113, saving 6,520 validation MACs and eight parameters.
change: Retain 54 individual mean-summary features, add their final four coordinates’ mean as one feature, retain all 58 final-hidden features, and shrink the classifier input accordingly.
mechanism: Zero-MAC pooled temporal-summary compression
evidence_used: Dropping two summary coordinates achieved 85.03%, while dropping a third fell to 83.93%; preserving aggregate information from the compressed coordinates directly tests whether the failed third-coordinate reduction was caused by information loss rather than classifier width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215475811142079927, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14872, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233021540, "training_seconds": 36.21351662511006, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.46857631191885546}

RECENT RESULT
hypothesis: Replacing the eight-output classifier with seven learned relative logits plus one fixed zero reference logit will retain at least 85% accuracy while saving 114 MACs per validation example and 115 parameters, because softmax probabilities are invariant to subtracting one class logit from all logits.
change: Change the 114-to-8 classifier to 114-to-7 and append a constant eighth logit, preserving the full eight-class softmax function family without altering recurrence or temporal summaries.
mechanism: Reference-class logit parameterization
evidence_used: The verified 114-feature design achieved 85.03%, while further summary compression failed; this targets classifier redundancy instead of discarding recurrent information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215025187756891030, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14765, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232935150, "training_seconds": 35.53474549995735, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.5016462829215395}

RECENT RESULT
hypothesis: Omitting the first mean-summary coordinate instead of the third-from-last coordinate will retain at least 85% validation accuracy while reducing classifier width from 114 to 113, saving 6,520 validation MACs and eight parameters.
change: Keep the verified GRU, 21-step schedule, final-hidden features, and two previously pruned summary coordinates unchanged; additionally discard mean-summary coordinate 0 and shrink the classifier input to 113.
mechanism: Nonadjacent temporal-summary coordinate pruning
evidence_used: Dropping the final two summary coordinates achieved 85.03%, but dropping the adjacent third coordinate fell to 83.93%, suggesting coordinate-specific importance rather than a demonstrated minimum classifier width; testing a distant coordinate isolates that distinction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215475811142079927, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14872, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233021540, "training_seconds": 35.55434679193422, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.47215274740581864}

RECENT RESULT
hypothesis: A seven-output classifier mapped through fixed orthonormal Helmert contrasts will retain at least 85% accuracy while saving 114 MACs per example and 115 parameters.
change: Replace the 114-to-8 classifier with a 114-to-7 classifier and transform its outputs into eight zero-sum logits using elementwise operations.
mechanism: Orthonormal zero-sum logit parameterization
evidence_used: The fixed-reference seven-logit design failed at 83.93% despite preserving the full softmax family; an orthonormal, class-balanced contrast basis targets its asymmetric optimization geometry without restoring the redundant eighth learned output.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215025187756891030, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14765, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232935150, "training_seconds": 43.111012040870264, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.48560810674187593}

RECENT RESULT
hypothesis: Pooling the final two normalized mel bands before the GRU will retain at least 85% accuracy while preserving all 21 steps and reducing recurrent inference by exactly 2,978,010 MACs and parameters by 174.
change: Keep the verified temporal schedule, hidden width, summaries, and classifier, but compress the two highest mel-band features into one elementwise mean after LayerNorm and change the GRU input width from 20 to 19.
mechanism: Learned-affine spectral pair pooling
evidence_used: The 114-feature, 21-step design meets the threshold at 85.03%, while reducing hidden width, recurrent steps, or classifier information failed; adjacent-band pooling tests a new structural cost axis while retaining aggregate information from every input band.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1199976067157189871, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14706, "peak_hidden_elements": 59904, "recurrent_macs": 229306770, "recurrent_steps": 17115, "total_inference_macs": 230050050, "training_seconds": 42.71294216718525, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4677562783832199}



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
