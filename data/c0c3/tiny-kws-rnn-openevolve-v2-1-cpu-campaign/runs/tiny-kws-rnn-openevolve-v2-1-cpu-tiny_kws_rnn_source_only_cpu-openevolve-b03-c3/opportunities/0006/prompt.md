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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 122.09208366600797, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}
prior_hypothesis: A 104-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 6.7% versus the qualified 108-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 115.23716191691346, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42283919632800515}
prior_hypothesis: A 112-unit GRU processing all 32 consecutive frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit model.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5645409772253230232, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 1081589760, "recurrent_steps": 26080, "total_inference_macs": 1082293920, "training_seconds": 145.88244629185647, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4493955928123802}
prior_hypothesis: A 108-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.5% versus the qualified 112-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: Retraining the existing 128-unit GRU on 16 uniformly spaced frames will retain at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps.
change: Replace the full 32-frame schedule with 16 increasing indices spanning the complete recording, including both endpoints.
mechanism: Uniform 2× causal frame subsampling
evidence_used: The current model reaches 87.24% accuracy at 32 steps, providing a 2.24-point margin above the target; recurrence accounts for 1,482,178,560 of 1,483,013,120 MACs, so temporal subsampling targets nearly all inference cost without reducing model capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 47.52007208298892, "validation_accuracy": 0.8171779141104294, "validation_cross_entropy": 0.5379230335446223}

RECENT RESULT
hypothesis: Using 24 uniformly spaced frames will recover at least 85% validation accuracy while reducing recurrent MACs and executed steps by 25% versus the verified 32-step model.
change: Replace the full 32-frame schedule with 24 unique increasing indices spanning the complete recording, including both endpoints.
mechanism: Moderate uniform causal frame subsampling
evidence_used: The 32-step model achieved 87.24% accuracy, while the more aggressive 16-step schedule fell to 81.72%; testing the midpoint isolates whether moderate temporal redundancy can be removed while retaining the required accuracy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 88.85810908395797, "validation_accuracy": 0.8220858895705522, "validation_cross_entropy": 0.5078564134843511}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 consecutive frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit model.
change: Reduce the GRU hidden width and classifier input width from 128 to 112 while preserving the successful full-frame schedule and training procedure.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 128-unit, 32-step model achieved 87.24% accuracy, whereas temporal subsampling to either 24 or 16 frames fell below 83%; this indicates temporal resolution should be preserved and model width is the next structural cost dimension to test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 115.23716191691346, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42283919632800515}

RECENT RESULT
hypothesis: A 108-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.5% versus the qualified 112-unit design.
change: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 128 to 108 while preserving the successful full-frame schedule and training procedure.
mechanism: Conservative full-resolution recurrent width reduction
evidence_used: The 112-unit full-resolution GRU achieved 85.77% accuracy, whereas reducing temporal resolution caused both 24-step and 16-step models to fall below 83%; a modest further width reduction is therefore the lowest-risk structural cost reduction supported by prior results.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5645409772253230232, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 1081589760, "recurrent_steps": 26080, "total_inference_macs": 1082293920, "training_seconds": 145.88244629185647, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4493955928123802}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 6.7% versus the qualified 108-unit design.
change: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 128 to 104 while preserving the full-frame schedule and training procedure.
mechanism: Conservative full-resolution GRU width reduction
evidence_used: The 108-unit full-resolution GRU achieved 86.01% accuracy, while reducing temporal resolution to 24 or 16 steps failed below 83%; this supports preserving all frames and testing a small additional width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 122.09208366600797, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}



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
