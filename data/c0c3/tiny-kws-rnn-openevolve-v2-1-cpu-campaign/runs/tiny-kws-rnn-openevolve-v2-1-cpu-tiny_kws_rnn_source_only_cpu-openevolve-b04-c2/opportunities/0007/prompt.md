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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 145.34480095887557, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4392715641325968}
prior_hypothesis: A 112-unit GRU processing all 32 frames will preserve at least 85% validation accuracy while reducing recurrent MACs by 22.0%, from 1,482,178,560 to 1,156,700,160.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 104.43626416590996, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44775579721649733}
prior_hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by 12.8% versus the qualified 112-unit model, from 1,156,700,160 to approximately 1,008,983,040.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 16 evenly distributed frames instead of all 32 will retain at least 85% validation accuracy while halving recurrent MACs from 1,482,178,560 to approximately 741,089,280 and halving executed recurrent steps.
change: Replace the full-frame schedule with a 16-step schedule spanning the complete recording, including its first and final frames.
mechanism: Uniform causal half-rate frame scheduling
evidence_used: The current 128-unit GRU reaches 87.24% accuracy, providing 2.24 percentage points of margin, while recurrent computation accounts for more than 99.9% of its 1,483,013,120 inference MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 73.73939166613854, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.5067759976065231}

RECENT RESULT
hypothesis: Processing 20 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent MACs by 37.5% versus the qualified 32-step model.
change: Replace the full 32-frame schedule with 20 unique, increasing indices spanning the first through final frame.
mechanism: Uniform causal 20-frame subsampling
evidence_used: The 32-step model reached 87.24% accuracy, while 16 steps reached 84.29%; 20 steps is the next cost-efficient test above the observed accuracy boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4836396255284976732, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 926361600, "recurrent_steps": 16300, "total_inference_macs": 927196160, "training_seconds": 78.55368520785123, "validation_accuracy": 0.8331288343558282, "validation_cross_entropy": 0.4976480121261503}

RECENT RESULT
hypothesis: Processing 24 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by 25% versus the qualified 32-step model.
change: Replace the full-frame schedule with 24 unique increasing indices spanning the first and final frames.
mechanism: Uniform causal 24-frame subsampling
evidence_used: The 32-step model achieved 87.24% accuracy, while 16- and 20-step schedules fell below 85%; 24 steps is the next cost-efficient test between the failed reduced schedules and the qualified baseline.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 101.34683662489988, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.4934807151373179}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will preserve at least 85% validation accuracy while reducing recurrent MACs by 22.0%, from 1,482,178,560 to 1,156,700,160.
change: Reduce the GRU state, temporal summary, and classifier input width from 128 to 112 while retaining complete temporal coverage.
mechanism: Full-resolution recurrent width bottleneck
evidence_used: The 128-unit, 32-step model achieved 87.24% accuracy, whereas every tested temporal subsampling schedule failed; this motivates preserving all frames and reducing recurrent width instead.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 145.34480095887557, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4392715641325968}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by 12.8% versus the qualified 112-unit model, from 1,156,700,160 to approximately 1,008,983,040.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 104 while preserving complete temporal coverage and the existing training procedure.
mechanism: Full-resolution recurrent width bottleneck
evidence_used: The full-resolution 112-unit GRU achieved 86.26% accuracy at substantially lower cost than the 128-unit baseline, while all tested temporal subsampling variants failed; this supports another conservative width reduction without dropping frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 104.43626416590996, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44775579721649733}

RECENT RESULT
hypothesis: A 102-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.5% versus the qualified 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 102 while preserving complete temporal coverage and the verified training procedure.
mechanism: Conservative full-resolution recurrent width reduction
evidence_used: The 104-unit full-resolution GRU achieved 85.03% accuracy at 1,008,983,040 recurrent MACs, while every temporal-subsampling design failed; this motivates a conservative two-unit width reduction without dropping frames.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5082011508174924488, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38808, "peak_hidden_elements": 104960, "recurrent_macs": 973618560, "recurrent_steps": 26080, "total_inference_macs": 974283600, "training_seconds": 94.50653929216787, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46548415107961083}



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
