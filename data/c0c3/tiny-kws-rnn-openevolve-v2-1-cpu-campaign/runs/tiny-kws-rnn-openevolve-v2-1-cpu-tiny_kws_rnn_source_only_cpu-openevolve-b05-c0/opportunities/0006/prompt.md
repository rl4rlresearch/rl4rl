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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 69.82911758311093, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4651480739102042}
prior_hypothesis: Processing 28 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 12.5% versus the 32-step baseline.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 16 evenly spaced frames instead of all 32 will preserve validation accuracy at or above 85% while halving recurrent steps and recurrent MACs from the 87.24%-accurate baseline.
change: Select the final frame of each two-frame interval, retaining full-recording coverage and the most recent frame.
mechanism: Uniform 2× causal frame striding
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps, leaving a 2.24-point margin for temporal subsampling.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 67.76923475018702, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.5148614427063363}

RECENT RESULT
hypothesis: Processing 24 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 25% versus the 32-step baseline.
change: Replace the full 32-frame schedule with 24 monotonically increasing indices distributed across the complete recording, including the first and final frames.
mechanism: Uniform 4:3 causal frame subsampling
evidence_used: Sixteen-frame 2× striding reduced accuracy from 87.24% to 83.07%, narrowly missing the threshold; retaining half of the discarded temporal evidence is the most informative intermediate cost-accuracy test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 82.46124641690403, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4918780555023006}

RECENT RESULT
hypothesis: Processing 28 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 12.5% versus the 32-step baseline.
change: Use 28 unique, monotonically increasing frame indices spanning the complete recording, including the first and final frames.
mechanism: Uniform 7:8 causal frame subsampling
evidence_used: The 32-frame model achieved 87.24% accuracy, while 24 frames achieved 84.42%; retaining half of the four-frame gap is the most informative lower-cost test with a safer accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 69.82911758311093, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4651480739102042}

RECENT RESULT
hypothesis: Processing 26 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 7.1% versus the verified 28-frame design.
change: Reduce the frame schedule from 28 to 26 unique increasing indices spanning the first through final input frames.
mechanism: Uniform 13:16 causal frame subsampling
evidence_used: The 24-frame design narrowly missed the target at 84.42%, while 28 frames reached 86.13%; 26 frames directly tests the midpoint and is expected to preserve sufficient temporal evidence at lower cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6286009174286206902, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1204270080, "recurrent_steps": 21190, "total_inference_macs": 1205104640, "training_seconds": 112.81374025018886, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4942737602748754}

RECENT RESULT
hypothesis: Processing 27 uniformly distributed frames will achieve at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-frame design.
change: Reduce the causal frame schedule from 28 to 27 unique increasing indices spanning the complete recording.
mechanism: Boundary frame-count refinement
evidence_used: The 28-frame model achieved 86.13% accuracy while the 26-frame model missed at 83.93%; testing the intervening 27-frame schedule directly locates the lowest viable step count.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6527611327453078597, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1250588160, "recurrent_steps": 22005, "total_inference_macs": 1251422720, "training_seconds": 93.67891345894895, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4554509636814609}



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
