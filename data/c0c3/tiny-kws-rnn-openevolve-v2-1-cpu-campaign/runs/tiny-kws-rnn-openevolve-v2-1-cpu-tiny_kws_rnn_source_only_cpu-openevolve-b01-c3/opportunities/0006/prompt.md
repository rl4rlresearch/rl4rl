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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 73.71617179107852, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.49484470109998085}
prior_hypothesis: Processing 16 uniformly distributed frames instead of all 32 will preserve at least 85% validation accuracy from the current 87.24% baseline while approximately halving recurrent MACs and executed recurrent steps.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3647838816047435592, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 698526720, "recurrent_steps": 13040, "total_inference_macs": 699335200, "training_seconds": 90.18403633311391, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4476142742882477}
prior_hypothesis: Halving the training batch size will roughly double optimizer updates over the same 50,000 examples, lifting the 124-unit, 16-step model’s prior 84.54% accuracy above 85% while retaining its 699,335,200-MAC inference cost—about 5.7% below the qualified 128-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 16 uniformly distributed frames instead of all 32 will preserve at least 85% validation accuracy from the current 87.24% baseline while approximately halving recurrent MACs and executed recurrent steps.
change: Change the causal frame schedule to select every second input frame.
mechanism: Uniform twofold causal temporal striding
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving accuracy margin for a structural reduction targeting temporal redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 73.71617179107852, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.49484470109998085}

RECENT RESULT
hypothesis: Sampling 15 frames uniformly across the complete 32-frame recording will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 6.25% versus the qualified 16-step design.
change: Replace full-frame processing with a 15-index schedule that includes both the first and final available frames.
mechanism: Endpoint-preserving 15-step uniform causal sampling
evidence_used: The 16-step uniform-striding design achieved 85.153% accuracy with 741,089,280 recurrent MACs, showing substantial temporal redundancy; distributing one fewer step across the full recording may preserve coverage better than simply truncating the even-index schedule.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3628385489450618257, "maximum_recurrent_steps": 15, "mean_recurrent_steps": 15.0, "median_recurrent_steps": 15, "optimizer_steps": 397, "p95_recurrent_steps": 15, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 694771200, "recurrent_steps": 12225, "total_inference_macs": 695605760, "training_seconds": 61.92250220803544, "validation_accuracy": 0.8024539877300614, "validation_cross_entropy": 0.5546215970092024}

RECENT RESULT
hypothesis: Retaining the qualified 16-frame even-index schedule while reducing GRU width from 128 to 124 will preserve at least 85% validation accuracy and reduce recurrent MACs by approximately 5.7% versus Reference Design 1.
change: Process every second frame and consistently reduce recurrent, summary, and classifier width to 124 units.
mechanism: Width-trimmed 16-step causal GRU
evidence_used: The 16-step 128-unit design achieved 85.153% accuracy, whereas reducing temporal coverage to 15 steps fell to 80.245%; this motivates preserving all 16 proven time samples and testing a modest orthogonal width reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3647838816047435592, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 698526720, "recurrent_steps": 13040, "total_inference_macs": 699335200, "training_seconds": 64.9078597498592, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.507219335637941}

RECENT RESULT
hypothesis: A 127-unit GRU using the proven 16-frame even-index schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.45% versus Reference Design 1.
change: Select every second frame and reduce recurrent, summary, and classifier width from 128 to 127 units.
mechanism: Boundary-width 16-step causal GRU
evidence_used: The 128-unit 16-step model reached 85.153% accuracy, while the 124-unit version narrowly missed at 84.540%; trimming only one unit tests the closest lower-cost capacity point with substantially less risk than the four-unit reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3813838268357361393, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 730331280, "recurrent_steps": 13040, "total_inference_macs": 731159320, "training_seconds": 79.36286050011404, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.49661150037145324}

RECENT RESULT
hypothesis: Halving the training batch size will roughly double optimizer updates over the same 50,000 examples, lifting the 124-unit, 16-step model’s prior 84.54% accuracy above 85% while retaining its 699,335,200-MAC inference cost—about 5.7% below the qualified 128-unit design.
change: Use the proven even-index 16-frame schedule, reduce GRU width to 124, and train with batches of 64 for more optimization steps.
mechanism: More-update training for a width-trimmed strided GRU
evidence_used: The 124-unit model missed qualification by only 0.46 percentage points with 397 optimizer steps, while the otherwise identical 128-unit 16-step model reached 85.153%; this suggests testing whether additional optimization can recover the small accuracy deficit without increasing inference cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3647838816047435592, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 698526720, "recurrent_steps": 13040, "total_inference_macs": 699335200, "training_seconds": 90.18403633311391, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4476142742882477}



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
