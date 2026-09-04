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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1507292713609427258, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18928, "peak_hidden_elements": 164352, "recurrent_macs": 286880000, "recurrent_steps": 17930, "total_inference_macs": 288966400, "training_seconds": 84.59254504204728, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.44480722813518503}
prior_hypothesis: Removing frame 29 from the passing 23-step schedule will retain at least 85% validation accuracy while reducing execution to 22 steps and total inference MACs by approximately 4.3%, from 302.01M to 288.97M.

## Recent verification evidence

RECENT RESULT
hypothesis: An 84-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.2%, from 388.02M to 371.87M.
change: Reduce the recurrent, temporal-summary, and classifier width from 86 to 84 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 86-unit model achieved 87.24% accuracy, the strongest margin among the compressed single-gate designs, while every preceding two-unit reduction from 92 through 86 units remained above 86%; this supports continuing the established two-unit compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1939755126468310982, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 19872, "peak_hidden_elements": 129536, "recurrent_macs": 370231680, "recurrent_steps": 21190, "total_inference_macs": 371874720, "training_seconds": 86.15392604190856, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.42746424762749236}

RECENT RESULT
hypothesis: An 83-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 2.1%, from 371.87M to 363.93M.
change: Reduce the recurrent, temporal-summary, and classifier width from 84 to 83 units while preserving the recurrence, readout, frame schedule, and training procedure.
mechanism: Boundary-aware single-gate width compression
evidence_used: The 84-unit model passed at 85.64% accuracy, while every tested single-gate width from 86 through 96 also passed; because the latest result has narrower margin, a one-unit reduction is the most controlled test of the compression boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1898314779686986760, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 362306620, "recurrent_steps": 21190, "total_inference_macs": 363930100, "training_seconds": 82.48713024985045, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4196209772964197}

RECENT RESULT
hypothesis: Removing one additional densely spaced early frame from the 83-unit single-gate model will preserve at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and total inference MACs by approximately 3.8%.
change: Expand the existing schedule deletion from two to three early indices, preserving the recurrent architecture, temporal readout, training procedure, and full recording coverage.
mechanism: Redundant early-frame pruning
evidence_used: The current 83-unit single-gate model achieved 86.26% accuracy at 26 steps, leaving 1.26 percentage points of margin; its schedule already benefits from removing two adjacent early frames, motivating a controlled test that removes one more nearby frame while retaining the first and final frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1825628377341731875, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 348371750, "recurrent_steps": 20375, "total_inference_macs": 349995230, "training_seconds": 85.62386012496427, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.43377429985561256}

RECENT RESULT
hypothesis: Removing one additional densely spaced early frame from the verified 83-unit model will preserve at least 85% validation accuracy while reducing recurrent execution from 25 to 24 steps and total inference MACs by approximately 4%.
change: Expand the schedule deletion from three to four adjacent early indices while preserving the recurrence, temporal readout, training procedure, and first-to-final recording coverage.
mechanism: Incremental redundant early-frame pruning
evidence_used: The 83-unit model improved from 86.26% at 26 steps to 86.63% at 25 steps after pruning a third early frame, leaving 1.63 percentage points of accuracy margin and directly supporting one more controlled schedule reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1752941974996476990, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 334436880, "recurrent_steps": 19560, "total_inference_macs": 336060360, "training_seconds": 86.30709966714494, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.46379833572481305}

RECENT RESULT
hypothesis: Redistributing 24 recurrent steps across 27 uniformly sampled candidates while retaining frame 4 will achieve at least 85% validation accuracy at approximately 336.06M total inference MACs.
change: Use a 27-frame base schedule and remove the first three densely spaced nonzero frames, producing 24 steps with broader temporal coverage instead of creating the failed frame-0-to-frame-5 gap.
mechanism: Coverage-preserving 24-step temporal resampling
evidence_used: The 25-step schedule passed at 86.63%, whereas deleting a fourth adjacent early frame fell to 84.17%; this isolates the enlarged initial sampling gap as a plausible failure mode and motivates testing 24 steps through redistribution.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1752941974996476990, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 334436880, "recurrent_steps": 19560, "total_inference_macs": 336060360, "training_seconds": 81.05458683311008, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4259138704077598}

RECENT RESULT
hypothesis: A 128-channel input-conditioned diagonal memory using spectral deltas and explicit phase will retain at least 85% accuracy at roughly 213M dense MACs, despite removing the full-rank hidden-to-hidden transform.
change: Replace the 83-unit full-rank recurrence with 128 independent learned-timescale memories driven by current mel features, frame differences, and temporal phase; retain the verified 24-step schedule and mean-max-terminal readout.
mechanism: Phase-aware diagonal delta memory
evidence_used: The current 24-step full-rank model reaches 86.13% at 336.06M MACs. The failed 128-unit rank-32 bottleneck reached 82.70%, suggesting hidden information compression was harmful; this patch instead preserves 128 direct memory channels and supplies explicit local dynamics while eliminating the costly recurrent matrix entirely.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1110063948564494744, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 14384, "peak_hidden_elements": 207360, "recurrent_macs": 210309120, "recurrent_steps": 19560, "total_inference_macs": 212812800, "training_seconds": 92.54028333397582, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4794025444545629}

RECENT RESULT
hypothesis: An 82-unit single-gate model will retain at least 85% validation accuracy on the verified 24-step schedule while reducing total dense inference MACs by approximately 2.2%, from 336.06M to 328.80M.
change: Reduce the recurrent, temporal-summary, and classifier width from 83 to 82 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Boundary-aware recurrent width compression
evidence_used: The 83-unit model achieved 86.13% accuracy at 24 steps, and every tested single-gate width from 84 through 96 passed; the one-unit 84-to-83 compression also improved accuracy, supporting another controlled boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1715089610640618232, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 19072, "peak_hidden_elements": 126464, "recurrent_macs": 327199680, "recurrent_steps": 19560, "total_inference_macs": 328803600, "training_seconds": 109.09142383397557, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.43944508371177626}

RECENT RESULT
hypothesis: An 81-unit single-gate model will retain at least 85% validation accuracy on the verified 24-step schedule while reducing total dense inference MACs by approximately 2.2%, from 328.80M to 321.63M.
change: Reduce the recurrent state, temporal summaries, and classifier input width from 82 to 81 units while preserving the successful full-rank recurrence, readout, frame schedule, and training procedure.
mechanism: Boundary-aware full-rank width compression
evidence_used: The 82-unit model passed at 85.64% accuracy, the 83-unit model passed at 86.13% on the same 24-step schedule, and every tested full-rank width from 82 through 96 passed; this supports one more controlled one-unit compression while retaining the architecture whose full-rank hidden interaction outperformed the failed diagonal-memory alternative.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1677645358029836118, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 18678, "peak_hidden_elements": 124928, "recurrent_macs": 320040720, "recurrent_steps": 19560, "total_inference_macs": 321625080, "training_seconds": 96.27593166707084, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4246623969516871}

RECENT RESULT
hypothesis: An 80-unit single-gate model will retain at least 85% validation accuracy on the verified 24-step schedule while reducing total dense inference MACs by approximately 2.2%, from 321.63M to 314.52M.
change: Reduce the recurrent state, temporal summaries, and classifier input width from 81 to 80 units while preserving the successful recurrence, readout, schedule, and training procedure.
mechanism: Boundary-aware full-rank width compression
evidence_used: The 81-unit model achieved 85.77% accuracy at 321.63M MACs, while the 82- and 83-unit models also passed on the same 24-step schedule; this supports one more controlled one-unit compression to locate the accuracy boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1640609217164130648, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 18288, "peak_hidden_elements": 123392, "recurrent_macs": 312960000, "recurrent_steps": 19560, "total_inference_macs": 314524800, "training_seconds": 98.99160750000738, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4492342521807899}

RECENT RESULT
hypothesis: An 80-unit recurrence augmented with a negative-extrema summary will achieve at least 85% validation accuracy while using approximately 315.05M total inference MACs, about 2.0% fewer than the passing 81-unit model.
change: Reduce the recurrent width from 81 to 80 and add a parameter-free running trough alongside the existing mean, positive peak, and terminal-state views, expanding only the final classifier input.
mechanism: Signed-extrema temporal readout
evidence_used: The 80-unit model reached 84.91%—one correct validation prediction short of 85%—at 314.52M MACs, while 81 units passed at 85.77%. A signed trough restores complementary temporal information for only 0.52M additional classifier MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1643329962131308888, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 18928, "peak_hidden_elements": 164352, "recurrent_macs": 312960000, "recurrent_steps": 19560, "total_inference_macs": 315046400, "training_seconds": 98.1992635410279, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4129759338004457}

RECENT RESULT
hypothesis: Redistributing the 80-unit signed-extrema model across 23 recurrent steps will retain at least 85% validation accuracy while reducing total inference MACs by approximately 4.1%, from 315.05M to 302.01M.
change: Reduce the uniform base schedule from 27 to 26 candidates and retain the established deletion of the first three nonzero candidates, yielding 23 steps with frame 4 and the final frame preserved.
mechanism: Coverage-preserving 23-step temporal resampling
evidence_used: The signed-extrema 80-unit model achieved 86.99% accuracy at 24 steps, nearly two percentage points above threshold; prior coverage-preserving resampling successfully reduced the 83-unit model to 24 steps, while the failed schedule was specifically associated with enlarging the initial gap beyond frame 4.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1575311337870368073, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 18928, "peak_hidden_elements": 164352, "recurrent_macs": 299920000, "recurrent_steps": 18745, "total_inference_macs": 302006400, "training_seconds": 104.17156891711056, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.44049525348686736}

RECENT RESULT
hypothesis: Removing frame 29 from the passing 23-step schedule will retain at least 85% validation accuracy while reducing execution to 22 steps and total inference MACs by approximately 4.3%, from 302.01M to 288.97M.
change: Preserve the verified schedule, including frames 0, 4, and 31, while deleting one densely sampled late frame immediately before the final endpoint.
mechanism: Endpoint-preserving late-frame pruning
evidence_used: The 80-unit signed-extrema model achieved 86.38% accuracy at 23 steps; prior 24-step failure followed loss of early coverage, whereas coverage-preserving resampling passed, motivating pruning a redundant late frame without disturbing the successful onset pattern.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1507292713609427258, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18928, "peak_hidden_elements": 164352, "recurrent_macs": 286880000, "recurrent_steps": 17930, "total_inference_macs": 288966400, "training_seconds": 84.59254504204728, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.44480722813518503}



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
