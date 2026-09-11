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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4366000706850512297, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 786, "p95_recurrent_steps": 27, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 836344035, "recurrent_steps": 22005, "total_inference_macs": 837015595, "training_seconds": 84.215427959105, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3975456752660084}
prior_hypothesis: Halving the training batch to 64 will provide enough additional optimizer updates for the 103-unit GRU on frames 4–30 to reach at least 85% accuracy while reducing total inference MACs to approximately 837,015,595.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3719704744227515017, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 712441215, "recurrent_steps": 18745, "total_inference_macs": 713112775, "training_seconds": 63.0624906248413, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.47033460184108994}
prior_hypothesis: The 103-unit GRU trained at batch size 64 on frames 4–26 will retain at least 85% validation accuracy while reducing inference to 23 recurrent steps and approximately 713.1 million total MACs.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3881278734883264337, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 786, "p95_recurrent_steps": 24, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 743416920, "recurrent_steps": 19560, "total_inference_macs": 744088480, "training_seconds": 64.33718712511472, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4478612420017734}
prior_hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–27 will retain at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and total inference MACs to approximately 744,088,480.

## Recent verification evidence

RECENT RESULT
hypothesis: A 103-unit GRU processing frames 4–30 will achieve at least 85% validation accuracy while reducing execution to 27 recurrent steps and recurrent MACs to approximately 836,344,035.
change: Preserve the qualified architecture and training procedure, retain frame 4, and replace the final frame with an additional trailing-edge trim.
mechanism: Asymmetric edge trimming that preserves the apparent speech-onset boundary
evidence_used: Frames 4–31 qualified at 85.40%, whereas frames 5–31 fell sharply to 83.68%; this suggests frame 4 is important and makes dropping the opposite edge the most informative 27-step alternative.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4366000706850512297, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 836344035, "recurrent_steps": 22005, "total_inference_macs": 837015595, "training_seconds": 149.436953375116, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.47428562655770706}

RECENT RESULT
hypothesis: Examples reaching at least 0.97 class probability after 27 of 28 scheduled steps can skip the final recurrence while preserving validation accuracy of at least 85% and reducing total inference MACs below 867,991,300.
change: Add an `exit_mask` that stops only highly confident examples after the penultimate scheduled frame; uncertain examples retain all 28 steps.
mechanism: Confidence-gated penultimate-step early exit
evidence_used: Frames 4–31 qualified at 85.40%, while removing frame 31 for every example narrowly missed at 84.66%, motivating selective rather than universal omission of the final update.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Processing frames 4–29 and 31 will achieve at least 85% validation accuracy while reducing inference to 27 recurrent steps and approximately 836,344,035 recurrent MACs.
change: Replace the current 30-step schedule with a 27-step schedule that preserves the empirically important frame 4 and final frame while omitting the locally redundant penultimate frame.
mechanism: Penultimate-frame temporal skip
evidence_used: Frames 4–31 achieved 85.40% accuracy, while replacing frame 31 with frame 3 missed narrowly at 84.66%; retaining frame 31 and dropping adjacent frame 30 is the closest untested 27-step alternative.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4366000706850512297, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 836344035, "recurrent_steps": 22005, "total_inference_macs": 837015595, "training_seconds": 110.51716583315283, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4747970861891296}

RECENT RESULT
hypothesis: A learned classifier over both the temporal mean and final hidden state will recover the 0.34-point accuracy shortfall of the tested 27-step frames 4–29,31 schedule, reaching at least 85% while keeping recurrent MACs near 836,344,035.
change: Process frames 4–29 and 31, and classify the concatenated mean recurrent output and terminal GRU state with a 206-to-8 linear layer.
mechanism: Dual-view mean-and-terminal recurrent readout
evidence_used: The 28-step frames 4–31 model achieved 85.40%, while the otherwise similar 27-step frames 4–29,31 model reached 84.66%; retaining frame 31 but exposing its terminal recurrent representation separately targets that narrow loss.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4369503665995754281, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 40321, "peak_hidden_elements": 105984, "recurrent_macs": 836344035, "recurrent_steps": 22005, "total_inference_macs": 837687155, "training_seconds": 111.68456470803358, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.44383969921275884}

RECENT RESULT
hypothesis: A 105-unit GRU processing frames 4–30 will recover the 103-unit model’s 0.34-point accuracy shortfall at 27 steps, reaching at least 85% accuracy while reducing total inference MACs from 867,991,300 to approximately 867,131,475.
change: Drop the final scheduled frame and reinvest the saved recurrent computation in two additional GRU units, preserving the existing pooling and training procedure.
mechanism: Near-iso-MAC width-for-step exchange
evidence_used: The 103-unit frames 4–30 model narrowly missed at 84.66%, while the 103-unit frames 4–31 model reached 85.40%. Among tested 27-step schedules, frames 4–30 also had the lowest cross-entropy, making a small capacity increase the most targeted next test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4523089719392930373, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 40893, "peak_hidden_elements": 108032, "recurrent_macs": 866446875, "recurrent_steps": 22005, "total_inference_macs": 867131475, "training_seconds": 105.66111337509938, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.47450530631410565}

RECENT RESULT
hypothesis: A 102-unit GRU processing frames 4–31 will retain at least 85% validation accuracy while reducing recurrent MACs from 867,319,740 to approximately 851,916,240.
change: Use the best qualified 28-frame schedule and reduce the GRU state, temporal summary, and classifier width from 103 to 102 units.
mechanism: Boundary-seeking width reduction at the qualified 28-step frontier
evidence_used: The 103-unit frames 4–31 design achieved 85.40% accuracy, while the 102-unit full-resolution model missed by only 0.09 percentage points; testing 102 units on the current cost frontier is the smallest structural reduction supported by the evidence.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4447193688382207708, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 38808, "peak_hidden_elements": 104960, "recurrent_macs": 851916240, "recurrent_steps": 22820, "total_inference_macs": 852581280, "training_seconds": 106.41962700011209, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4797561926344421}

RECENT RESULT
hypothesis: A learned 20-to-18 projection feeding the 103-unit GRU over frames 4–31 will retain at least 85% accuracy while reducing total inference MACs from 867,991,300 to approximately 862,103,740.
change: Restore the qualified 28-frame schedule and factor the GRU input through the smallest shared spectral bottleneck that reduces exact MACs while preserving recurrent width.
mechanism: Shared low-rank spectral bottleneck at the qualified temporal frontier
evidence_used: The 103-unit, 28-step design achieved 85.40%, whereas reducing recurrent width to 102 missed at 84.54%; preserving the qualified memory width while compressing only the 20-band input is an orthogonal structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4496864288689244199, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 39239, "peak_hidden_elements": 105984, "recurrent_macs": 861432180, "recurrent_steps": 22820, "total_inference_macs": 862103740, "training_seconds": 106.17561037489213, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.4857652254631183}

RECENT RESULT
hypothesis: Halving the training batch to 64 will provide enough additional optimizer updates for the 103-unit GRU on frames 4–30 to reach at least 85% accuracy while reducing total inference MACs to approximately 837,015,595.
change: Use the previously tested 27-step contiguous schedule and halve the batch size, while preserving the architecture, optimizer, augmentation, and loss.
mechanism: Higher-update optimization for trailing-frame trimming
evidence_used: The 103-unit frames 4–30 model narrowly missed qualification at 84.66%, only 0.34 percentage points short; increasing width and changing the readout both performed worse, so increasing optimization updates without adding inference cost is the most direct untested intervention.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4366000706850512297, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 786, "p95_recurrent_steps": 27, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 836344035, "recurrent_steps": 22005, "total_inference_macs": 837015595, "training_seconds": 84.215427959105, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3975456752660084}

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–29 will retain at least 85% validation accuracy while reducing execution from the qualified 27-step design to 26 steps and total inference MACs to approximately 806 million.
change: Use the qualified batch-size-64 training procedure and remove frame 30 from its contiguous schedule, processing 26 frames per standard recording.
mechanism: Higher-update optimization with one-step trailing trim
evidence_used: The batch-size-64 model on frames 4–30 achieved 87.24% accuracy at 27 steps, providing 2.24 percentage points of margin; trimming one adjacent trailing frame is the smallest untested reduction below that qualified cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4204426716194762977, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 786, "p95_recurrent_steps": 26, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 805368330, "recurrent_steps": 21190, "total_inference_macs": 806039890, "training_seconds": 74.47286183293909, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.41319922698787387}

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–28 will retain at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and total inference MACs to approximately 775,064,185.
change: Halve the training batch size and schedule 25 contiguous frames for standard 32-frame recordings by omitting frames 0–3 and 29–31.
mechanism: Higher-update optimization with incremental trailing-frame trim
evidence_used: The batch-size-64, frames 4–29 design achieved 87.36% accuracy at 26 steps, leaving 2.36 percentage points of margin; removing the adjacent trailing frame is the smallest untested reduction below that qualified frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4042852725539013657, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 786, "p95_recurrent_steps": 25, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 774392625, "recurrent_steps": 20375, "total_inference_macs": 775064185, "training_seconds": 62.33617270900868, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.43089763430729966}

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–27 will retain at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and total inference MACs to approximately 744,088,480.
change: Remove frame 28 from the current contiguous schedule, processing 24 frames for standard 32-frame recordings.
mechanism: Incremental trailing-frame trim
evidence_used: The batch-size-64 frames 4–28 design achieved 86.26% accuracy at 25 steps; removing the adjacent trailing frame is the smallest untested reduction below the current qualified frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3881278734883264337, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 786, "p95_recurrent_steps": 24, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 743416920, "recurrent_steps": 19560, "total_inference_macs": 744088480, "training_seconds": 64.33718712511472, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4478612420017734}

RECENT RESULT
hypothesis: The 103-unit GRU trained at batch size 64 on frames 4–26 will retain at least 85% validation accuracy while reducing inference to 23 recurrent steps and approximately 713.1 million total MACs.
change: Schedule 23 contiguous frames for standard recordings by omitting frames 27–31, preserving the architecture and training procedure.
mechanism: Incremental trailing-frame trim at the qualified frontier
evidence_used: The qualified frames 4–27 design achieved 86.26% accuracy at 24 steps, leaving 1.26 percentage points of margin; removing the adjacent trailing frame is the smallest untested structural reduction below that frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3719704744227515017, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 712441215, "recurrent_steps": 18745, "total_inference_macs": 713112775, "training_seconds": 63.0624906248413, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.47033460184108994}



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
