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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352125227200076850, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259218900, "training_seconds": 36.48290670802817, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4870919163241708}
prior_hypothesis: Processing frames 3–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 22 recurrent steps and approximately 259,218,900 total MACs.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1536540722251086223, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294573600, "training_seconds": 56.65610625012778, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4427040802189178}
prior_hypothesis: A 60-unit GRU processing frames 3–26 and frame 28 will achieve at least 85% validation accuracy while reducing total dense inference MACs from 306,309,600 to approximately 294,573,600.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067295, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 30.942437916994095, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4581207088166219}
prior_hypothesis: Disabling LayerNorm affine parameters and the seven-logit classifier bias will retain at least 85% validation accuracy at 294,426,900 MACs while reducing learned parameters from 16,067 to 16,020.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1413341989043073665, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 269928000, "recurrent_steps": 18745, "total_inference_macs": 270954900, "training_seconds": 31.087566582951695, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.460217977886551}
prior_hypothesis: Processing frames 3–22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing total dense inference MACs from 282,690,900 to approximately 270,954,900.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing one final-state coordinate from the classifier input will retain at least 85% validation accuracy while reducing total inference MACs by 5,705 and learned parameters by 7.
change: Keep the proven 60-unit GRU, 25-frame schedule, temporal summaries, and seven-logit head, but reduce the classifier input from 180 to 179 features by omitting one final-state coordinate.
mechanism: Single-coordinate classifier-feature pruning
evidence_used: The current design qualifies at 85.03%, while reducing recurrent width to 59 or input rank to 17 failed substantially; pruning one classifier coordinate is a smaller structural reduction, and that hidden coordinate remains represented through both mean and maximum summaries.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535745754580988790, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294421195, "training_seconds": 34.450375667074695, "validation_accuracy": 0.8233128834355828, "validation_cross_entropy": 0.4895681275911858}

RECENT RESULT
hypothesis: Disabling LayerNorm affine parameters and the seven-logit classifier bias will retain at least 85% validation accuracy at 294,426,900 MACs while reducing learned parameters from 16,067 to 16,020.
change: Make the input LayerNorm non-affine and the reference-class classifier bias-free, preserving the qualified 60-unit GRU, 25-frame schedule, and seven-logit formulation.
mechanism: Redundant affine-parameter pruning
evidence_used: The non-affine LayerNorm design already qualified at 85.03% with 16,027 parameters and identical MACs. Removing only seven classifier biases is a smaller additional restriction than the failed recurrent-width, input-rank, or classifier-feature pruning experiments.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067295, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 30.942437916994095, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4581207088166219}

RECENT RESULT
hypothesis: Removing the lowest-frequency mel band from the proven 60-unit, 25-step, seven-logit design will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 290,759,400.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, then reduce the GRU input from 20 to 19 features by omitting the lowest-frequency normalized mel band.
mechanism: Single edge-band recurrent-input pruning
evidence_used: The 20-band reference design achieved 85.52% accuracy at 294,426,900 MACs, while the more aggressive rank-17 bottleneck narrowly missed at 84.42%; pruning only one direct spectral input is the smallest structural probe between them.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1516645274678599615, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15840, "peak_hidden_elements": 92672, "recurrent_macs": 289732500, "recurrent_steps": 20375, "total_inference_macs": 290759400, "training_seconds": 36.92962120799348, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.47418165148401553}

RECENT RESULT
hypothesis: Omitting the highest-frequency mel band will retain at least 85% validation accuracy on the qualified 25-step design while reducing total inference MACs from 294,426,900 to approximately 290,759,400.
change: Adopt the qualified non-affine normalization, bias-free seven-logit head, and endpoint-preserving schedule, while reducing the GRU input to the lowest 19 mel bands.
mechanism: Upper-edge spectral input pruning
evidence_used: The full 20-band design achieved 85.52% accuracy, while removing the lowest-frequency band narrowly missed at 84.42%. Testing the opposite spectral edge determines whether that miss was band-specific rather than caused by 19-band capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1516645274678599615, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15840, "peak_hidden_elements": 92672, "recurrent_macs": 289732500, "recurrent_steps": 20375, "total_inference_macs": 290759400, "training_seconds": 51.304059291025624, "validation_accuracy": 0.8331288343558282, "validation_cross_entropy": 0.4896168527427627}

RECENT RESULT
hypothesis: Averaging the two highest-frequency mel bands into one GRU input will preserve at least 85% validation accuracy while reducing total inference MACs to approximately 290,759,400.
change: Keep all 20 bands in non-affine normalization, merge the adjacent upper-edge pair without learned operations, and reduce the GRU input width from 20 to 19.
mechanism: Fixed adjacent-band spectral pooling
evidence_used: Dropping either edge band reduced MACs to 290,759,400 but narrowly missed qualification; pooling preserves information from both bands while exploiting their expected local spectral redundancy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1516645274678599615, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15840, "peak_hidden_elements": 92672, "recurrent_macs": 289732500, "recurrent_steps": 20375, "total_inference_macs": 290759400, "training_seconds": 36.264635417144746, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.48189697265625}

RECENT RESULT
hypothesis: Removing the lowest mel band only from the GRU reset-gate input will retain at least 85% accuracy while reducing total inference MACs from 294,426,900 to approximately 293,204,400.
change: Replace the fused GRU with an equivalent Linear-based GRU whose update and candidate gates retain all 20 bands while its reset gate uses 19; also adopt the qualified bias-free seven-logit head.
mechanism: Gate-selective spectral pruning
evidence_used: Removing the lowest band from every GRU gate narrowly missed at 84.42%; retaining that band in the update and candidate paths tests a substantially smaller structural reduction, while the bias-free head already qualified at 85.52%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1529398766712244735, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15960, "peak_hidden_elements": 92672, "recurrent_macs": 292177500, "recurrent_steps": 20375, "total_inference_macs": 293204400, "training_seconds": 64.02440058393404, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.483909878408982}

RECENT RESULT
hypothesis: Processing frames 3–25 plus frame 28 will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 282,690,900.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, then remove frame 26 from the proven endpoint-preserving schedule for 24 recurrent steps.
mechanism: Iterative late-frame pruning with endpoint preservation
evidence_used: The bias-free 25-step reference achieved 85.52% accuracy, and removing frame 27 while preserving frame 28 previously qualified; pruning the next-latest contiguous frame is the most direct test of further temporal redundancy without disrupting the important endpoint or early continuity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1474558750886070480, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 281664000, "recurrent_steps": 19560, "total_inference_macs": 282690900, "training_seconds": 39.57340499991551, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4616131928800805}

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–24, 26, and 28 will achieve at least 85% validation accuracy while reducing total inference MACs to approximately 282,690,900.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, and use a 24-step schedule that retains informative frame 26 while distributing two-frame gaps across the late sequence.
mechanism: Distributed late-frame pruning with endpoint preservation
evidence_used: The qualified 25-step design reached 85.52%; the 24-step frames 3–25 plus 28 variant narrowly missed at 84.79%. Replacing frame 25 with later frame 26 preserves the proven endpoint and improves late-time coverage without adding MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1474558750886070480, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 281664000, "recurrent_steps": 19560, "total_inference_macs": 282690900, "training_seconds": 36.8413354170043, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4530906724052195}

RECENT RESULT
hypothesis: Processing frames 3–22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing total dense inference MACs from 282,690,900 to approximately 270,954,900.
change: Remove frame 23 from the qualified 24-step schedule, yielding 23 recurrent steps while preserving evenly spaced late observations and the important frame-28 endpoint.
mechanism: Distributed alternating late-frame pruning
evidence_used: The 24-step schedule retaining frames 26 and 28 qualified at 85.15%, whereas clustering the omitted late frames missed at 84.79%; this motivates extending the successful distributed-pruning pattern by retaining frames 24, 26, and 28.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1413341989043073665, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 269928000, "recurrent_steps": 18745, "total_inference_macs": 270954900, "training_seconds": 31.087566582951695, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.460217977886551}

RECENT RESULT
hypothesis: Processing frames 3–20, 22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing recurrent execution from 23 to 22 steps and total dense inference MACs from 270,954,900 to approximately 259,218,900.
change: Replace the current 25-step schedule with a 22-step schedule that preserves the continuous early context and samples every second late frame through the proven frame-28 endpoint.
mechanism: Extended alternating late-frame pruning
evidence_used: The 23-step frames 3–22, 24, 26, and 28 schedule achieved 85.52% accuracy; the prior successful transition from 24 to 23 steps showed that extending distributed late-frame pruning can remove a step without losing qualification.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352125227200076850, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259218900, "training_seconds": 31.1265710410662, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4731880047569977}

RECENT RESULT
hypothesis: Processing frames 3–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 22 recurrent steps and approximately 259,218,900 total MACs.
change: Remove frame 22 instead of frame 21 from the qualified 23-step schedule, preserve the proven late frames 24, 26, and 28, and remove the seven-logit classifier bias.
mechanism: Early-continuity-preserving temporal pruning
evidence_used: The 23-step schedule reached 85.52% accuracy, while the 22-step schedule that omitted frame 21 but retained frame 22 narrowly missed at 84.66%; preserving uninterrupted early recurrence through frame 21 tests whether continuity is more valuable than the final contiguous frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352125227200076850, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259218900, "training_seconds": 36.48290670802817, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4870919163241708}

RECENT RESULT
hypothesis: Processing frames 4–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 21 recurrent steps and approximately 247,482,900 total MACs.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, then remove only frame 3 from the qualified 22-step schedule while preserving continuous frames through 21 and the proven late frames 24, 26, and 28.
mechanism: Earliest-boundary temporal pruning
evidence_used: The 22-step frames 3–21, 24, 26, and 28 design achieved 85.40%; replacing frame 21 with frame 22 missed at 84.66%, and prior removal of frame 26 also missed, motivating preservation of the later contiguous context and all three late observations while pruning the earliest selected boundary frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1290908465357080035, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 246456000, "recurrent_steps": 17115, "total_inference_macs": 247482900, "training_seconds": 27.439611959038302, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.5186664604701878}



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
