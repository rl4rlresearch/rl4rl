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
verified_results: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1214995429608812518, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14758, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232929445, "training_seconds": 35.75764554203488, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4756235508831001}
prior_hypothesis: Removing the classifier input with the smallest class-centered weight norm after conventional eight-logit training will retain at least 85% validation accuracy while saving 5,705 inference MACs and seven learned parameters, with all 21 recurrent steps unchanged.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Training the conventional eight-output classifier and algebraically converting it to seven relative logits only at evaluation will preserve at least 85% accuracy while reducing total inference by 92,910 MACs and parameters by 115.
change: Keep the verified training architecture unchanged, but on evaluation replace its 114-to-8 classifier with an equivalent 114-to-7 classifier whose weights and biases are differences from class eight, then append a fixed zero reference logit.
mechanism: Post-training reference-logit compression
evidence_used: The verified eight-output design achieved 85.03%, while training directly with fixed-reference or Helmert seven-logit parameterizations failed despite preserving the softmax function family; postponing exact compression until after conventional optimization isolates those failures as optimization effects.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215025187756891030, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14765, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232935150, "training_seconds": 39.62307258299552, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4756408972242859}

RECENT RESULT
hypothesis: Truncating only the weakest singular direction of the class-centered classifier will retain at least 85% accuracy while reducing inference by 53,790 MACs, with all recurrent computation and 114 classifier inputs unchanged.
change: Train the conventional eight-logit classifier, then at evaluation replace it with a class-balanced rank-six two-Linear SVD approximation.
mechanism: Post-training rank-six discriminative logit factorization
evidence_used: Exact post-training logit compression retained 85.03% while directly training compressed logits failed; meanwhile additional summary-coordinate pruning lost accuracy, motivating post-training compression of the weakest classifier direction without discarding recurrent features.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1214744610932150775, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14700, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232881360, "training_seconds": 40.53312229202129, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.5062236949710027}

RECENT RESULT
hypothesis: Removing the classifier input with the smallest class-centered weight norm after conventional eight-logit training will retain at least 85% validation accuracy while saving 5,705 inference MACs and seven learned parameters, with all 21 recurrent steps unchanged.
change: During evaluation, select and remove the least-used of the 114 classifier features before exact conversion to seven relative logits; preserve reversible train/eval transitions and index the retained features during classification.
mechanism: Post-training discriminative feature pruning
evidence_used: Exact post-training reference-logit compression retained 85.03% while directly constrained training failed, showing that compression after conventional optimization is safer; fixed-coordinate pruning failed, motivating a learned, weight-based coordinate choice and the smallest possible additional reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1214995429608812518, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14758, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232929445, "training_seconds": 35.75764554203488, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4756235508831001}

RECENT RESULT
hypothesis: Pruning the two classifier inputs with the smallest variance-weighted class-centered sensitivity after conventional training will retain at least 85% validation accuracy while saving another 5,705 inference MACs and seven parameters, with all 21 recurrent steps unchanged.
change: Accumulate classifier-feature moments during training, select two low-sensitivity inputs during evaluation, fold their empirical mean contributions into the classifier bias, and retain reversible seven-relative-logit compression.
mechanism: Mean-preserving activation-aware classifier pruning
evidence_used: Post-training pruning of one weight-selected feature preserved 85.03% accuracy, while fixed-coordinate pruning failed; this motivates a minimal second reduction chosen using both learned discriminative weights and observed feature variation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1214965671460734006, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14751, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 232923740, "training_seconds": 37.306912790983915, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4753252181538775}



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
