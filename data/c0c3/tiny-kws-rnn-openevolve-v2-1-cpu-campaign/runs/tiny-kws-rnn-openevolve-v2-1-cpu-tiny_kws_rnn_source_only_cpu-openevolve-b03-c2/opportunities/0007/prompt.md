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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7010815633786821987, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1343224320, "recurrent_steps": 23635, "total_inference_macs": 1344058880, "training_seconds": 84.30788024980575, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4803711405561014}
prior_hypothesis: Processing 29 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.33% versus the qualified 30-step design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7252417786953693682, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1389542400, "recurrent_steps": 24450, "total_inference_macs": 1390376960, "training_seconds": 110.27748720813543, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4517616037942149}
prior_hypothesis: Processing 30 evenly distributed frames will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 6.25% versus the qualified 32-step design.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6431841104852922978, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1232280000, "recurrent_steps": 24450, "total_inference_macs": 1233062400, "training_seconds": 97.09306433307938, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4458835718821894}
prior_hypothesis: The 120-unit GRU using 30 uniformly distributed frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.3% versus the current 32-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: Training and inference on 24 uniformly distributed frames will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 25%.
change: Replace the full 32-frame schedule with a 24-step schedule spanning the complete recording, including the first and final frames.
mechanism: Uniform causal frame subsampling
evidence_used: The current 128-unit GRU achieves 87.24% accuracy but executes all 32 frames, with recurrence accounting for 1,482,178,560 of 1,483,013,120 MACs; removing eight temporally redundant steps directly targets the dominant cost with a 2.24-point accuracy margin.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 92.59417370893061, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.5135083157592024}

RECENT RESULT
hypothesis: Using 28 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 12.5% versus the qualified 32-step design.
change: Change the frame schedule from all 32 frames to 28 evenly spaced causal frames spanning the first through final input frames.
mechanism: Uniform 28-frame causal subsampling
evidence_used: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 83.93%; testing the midpoint of this bounded interval is the most informative next change and is expected to preserve the required accuracy with lower dominant recurrent cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 66.70559700019658, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4741032419029189}

RECENT RESULT
hypothesis: Processing 30 evenly distributed frames will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 6.25% versus the qualified 32-step design.
change: Replace the full-frame schedule with 30 unique, approximately uniform indices spanning the first and final input frames.
mechanism: Uniform 30-frame causal subsampling
evidence_used: The 32-step model reached 87.24% accuracy, while 28 steps fell to 83.93%; 30 steps is the midpoint of the bounded pass/fail interval and the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7252417786953693682, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1389542400, "recurrent_steps": 24450, "total_inference_macs": 1390376960, "training_seconds": 110.27748720813543, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4517616037942149}

RECENT RESULT
hypothesis: Processing 29 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.33% versus the qualified 30-step design.
change: Replace the full 32-frame schedule with 29 unique, approximately uniform indices spanning the first and final input frames.
mechanism: Uniform 29-frame causal subsampling
evidence_used: The 30-step design achieved 86.01% accuracy while the 28-step design achieved 83.93%; 29 steps directly tests the remaining pass/fail boundary and, if qualified, improves the dominant recurrent cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7010815633786821987, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1343224320, "recurrent_steps": 23635, "total_inference_macs": 1344058880, "training_seconds": 84.30788024980575, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4803711405561014}

RECENT RESULT
hypothesis: A 120-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 11.3% versus the current design and below the qualified 29-step, 128-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 120 while preserving the full causal frame schedule and training procedure.
mechanism: Moderate recurrent-width reduction
evidence_used: The current 128-unit, 32-step model achieves 87.24% accuracy with a 2.24-point margin, and recurrent computation accounts for virtually all MACs; a moderate width reduction targets that dominant cost while retaining all temporal evidence, unlike the failed 28-step schedule.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6860358437346396608, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1314432000, "recurrent_steps": 26080, "total_inference_macs": 1315214400, "training_seconds": 106.96294012502767, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4206474725454131}

RECENT RESULT
hypothesis: The 120-unit GRU using 30 uniformly distributed frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.3% versus the current 32-step model.
change: Change the frame schedule from all 32 frames to 30 approximately uniform causal frames spanning the complete recording.
mechanism: Joint moderate width reduction and uniform causal frame subsampling
evidence_used: The 120-unit model has 86.75% accuracy at 32 steps, while the 128-unit model retains 86.01% at 30 steps; these independent margins make their combination the most informative lower-cost candidate.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6431841104852922978, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1232280000, "recurrent_steps": 24450, "total_inference_macs": 1233062400, "training_seconds": 97.09306433307938, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4458835718821894}



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
