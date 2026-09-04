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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4208312280590011128, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 806784800, "training_seconds": 126.96434125001542, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4665855548133148}
prior_hypothesis: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing dense recurrent MACs by approximately 7.5% versus the verified 96-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 16 uniformly distributed frames instead of all 32 will retain at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps.
change: Replace the full-frame schedule with 16 causal indices distributed across the complete recording, including both endpoints.
mechanism: Uniform endpoint-preserving half-rate recurrence
evidence_used: The current 128-unit GRU reaches 87.24% accuracy at 32 steps, providing a 2.24-point margin for temporal subsampling while recurrent computation accounts for over 99.9% of its 1.483B inference MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 72.2863548330497, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4956934829431077}

RECENT RESULT
hypothesis: Increasing the uniform schedule from 16 to 20 frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 37.5% versus the 32-frame baseline.
change: Process 20 uniformly distributed causal frames spanning the complete recording.
mechanism: Moderate endpoint-preserving temporal subsampling
evidence_used: The 16-frame schedule narrowly missed the target at 84.54%, only 0.46 percentage points short; four additional distributed observations should restore sufficient temporal detail while preserving most of the computational savings.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4836396255284976732, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 926361600, "recurrent_steps": 16300, "total_inference_macs": 927196160, "training_seconds": 75.24583754199557, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.48686297832091163}

RECENT RESULT
hypothesis: Processing 24 evenly distributed frames will restore validation accuracy to at least 85% while reducing recurrent MACs and executed steps by 25% versus the verified 32-step baseline.
change: Replace the full 32-frame schedule with 24 unique increasing indices distributed across the complete recording, including both endpoints.
mechanism: Uniform endpoint-preserving 24-step causal recurrence
evidence_used: The 20-step result narrowly missed the threshold at 84.42%, while the 32-step baseline reached 87.24%; testing 24 steps is the most informative intermediate point for finding the lowest viable recurrent cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 90.53033870807849, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.47749484945659987}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 22% versus the verified 24-step, 128-unit design.
change: Reduce the GRU, temporal summary, and classifier width from 128 to 96, and restore the complete 32-frame causal schedule.
mechanism: Trade recurrent width for full temporal resolution
evidence_used: The 128-unit model achieved 87.24% at 32 steps, while temporal subsampling to 16 and 20 steps failed and 24 steps only reached 85.77%; this suggests preserving temporal detail while reducing excess recurrent capacity is the better cost tradeoff.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 106.57166195916943, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4620833508076112}

RECENT RESULT
hypothesis: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing dense recurrent MACs by approximately 7.5% versus the verified 96-unit model.
change: Reduce the GRU state, temporal summary, and classifier width from 96 to 92 while preserving the full causal frame schedule and training procedure.
mechanism: Incremental recurrent-width compression
evidence_used: The 96-unit, 32-step model achieved 85.40% accuracy at 871.9M MACs, whereas reducing temporal coverage to 16 or 20 steps failed; this motivates preserving all frames and probing a small width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4208312280590011128, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 806784800, "training_seconds": 126.96434125001542, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4665855548133148}



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
