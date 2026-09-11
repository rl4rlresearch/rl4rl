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
hypothesis: Reducing the GRU width from 58 to 57 while preserving the verified 23-frame schedule will retain at least 85% validation accuracy and reduce total inference MACs from 255,163,460 to approximately 247,558,695.
change: Shrink the GRU hidden state by one unit and adjust the temporal summary and classifier dimensions accordingly.
mechanism: One-unit recurrent width contraction
evidence_used: The current 23-step, width-58 model achieved 85.64%, while reducing temporal coverage to 22 steps failed at 84.66%; preserving all successful frame selections and testing the smallest recurrent-width reduction is the best-supported orthogonal cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1291303823773123109, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 522, "p95_recurrent_steps": 23, "parameters": 14469, "peak_hidden_elements": 58880, "recurrent_macs": 246815415, "recurrent_steps": 18745, "total_inference_macs": 247558695, "training_seconds": 45.6729015421588, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.45602725912456865}

RECENT RESULT
hypothesis: Processing frames 3, 5–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 23 to 22 recurrent steps and total inference MACs by approximately 4.3%.
change: Remove frame 4 while preserving frame 3 as the initial recurrent anchor, the remaining dense command-bearing prefix, and both informative late samples.
mechanism: Anchored-prefix interior-frame ablation
evidence_used: Removing frame 2 improved accuracy to 85.64%, whereas subsequently removing frame 3 fell to 84.66%; retaining frame 3 and ablating frame 4 directly tests whether the failure was specific to losing the earliest temporal anchor rather than requiring every early frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1273274637120059906, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 522, "p95_recurrent_steps": 22, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 243345960, "recurrent_steps": 17930, "total_inference_macs": 244102280, "training_seconds": 47.512227582978085, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4581193608009011}

RECENT RESULT
hypothesis: Processing frames 3, 6–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs by approximately 4.5%.
change: Remove frame 5 while retaining frame 3 as the initial recurrent anchor, the remaining dense command-bearing region, and both informative late samples.
mechanism: Anchored-prefix interior-frame ablation
evidence_used: Removing frame 3 failed at 84.66%, but retaining frame 3 and removing frame 4 achieved 85.03%; this motivates testing whether the next early interior frame is also redundant without sacrificing the demonstrated anchor or late temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215577839078349111, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233041100, "training_seconds": 39.8916966249235, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4791149864898869}

RECENT RESULT
hypothesis: Processing frames 3, 5, 7–23, 27, and 29 will achieve at least 85% validation accuracy with 21 recurrent steps because it preserves frame 5 and avoids the two-frame early gap created by the failed frame-5 ablation.
change: Remove frame 6 from the verified 22-step schedule, retaining frames 3 and 5 as early recurrent anchors plus the dense command-bearing region and both late samples.
mechanism: Preserve early anchors while ablating an adjacent redundant frame
evidence_used: The 22-step schedule using frames 3, 5–23, 27, and 29 achieved 85.03%, while removing frame 5 fell to 84.05%; testing frame 6 distinguishes whether that failure reflects frame 5’s specific importance or a general 21-step capacity limit.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215577839078349111, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233041100, "training_seconds": 38.09167179092765, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46601140308965205}

RECENT RESULT
hypothesis: Processing frames 3, 5, 6, 8–23, 27, and 29 will achieve at least 85% validation accuracy with 21 recurrent steps because it retains all early anchors implicated by prior failures while introducing only a single-frame gap before the dense command-bearing region.
change: Remove frame 7 from the verified 22-step schedule while retaining frames 3, 5, and 6 plus the dense later region and both late samples.
mechanism: Preserve early anchors while ablating the next adjacent frame
evidence_used: Removing frame 5 reduced accuracy to 84.05%, while removing frame 6 narrowly missed at 84.91%; ablating frame 7 is the next direct test of whether those outcomes reflect specific early-frame importance rather than a general 21-step limit.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215577839078349111, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233041100, "training_seconds": 37.18269337504171, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46355731238616754}

RECENT RESULT
hypothesis: Processing frames 3, 5, 6, 9–23, 27, and 29 will retain at least 85% validation accuracy while reducing recurrent execution from 21 to 20 steps.
change: Remove frame 8 from the verified 21-step schedule while preserving the established early anchors, dense later region, and both late samples.
mechanism: Progressive early-interior frame ablation
evidence_used: Removing frame 7 from the 22-step schedule achieved 85.15%, whereas removing frames 5 or 6 failed; frame 8 is therefore the next untested ablation that preserves all early frames shown to be important.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1157881041036638316, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 522, "p95_recurrent_steps": 20, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 221223600, "recurrent_steps": 16300, "total_inference_macs": 221979920, "training_seconds": 34.113308042054996, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4875313226430694}

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
