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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4850722678817760257, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 929271150, "recurrent_steps": 24450, "total_inference_macs": 929942710, "training_seconds": 95.05833658296615, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.45136498761323335}
prior_hypothesis: A 103-unit GRU processing frames 2–31 will retain at least 85% validation accuracy while reducing recurrent execution from 31 to 30 steps and recurrent MACs from 960,246,855 to approximately 929,271,150.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4527574697506261617, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 867319740, "recurrent_steps": 22820, "total_inference_macs": 867991300, "training_seconds": 182.853933500126, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4648651684720092}
prior_hypothesis: A 103-unit GRU processing frames 4–31 will retain at least 85% validation accuracy while reducing execution from 29 to 28 recurrent steps and recurrent MACs from 898,295,445 to approximately 867,319,740.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5012296669473509577, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 960246855, "recurrent_steps": 25265, "total_inference_macs": 960918415, "training_seconds": 94.25584362493828, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4491593905021808}
prior_hypothesis: Skipping only the first frame will retain at least 85% validation accuracy while reducing recurrent execution from 32 to 31 steps and recurrent MACs from 991,222,560 to approximately 960,246,855.

## Recent verification evidence

RECENT RESULT
hypothesis: A 102-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.5% versus the qualified 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 102 while preserving complete temporal coverage and the verified training procedure.
mechanism: Conservative full-resolution recurrent width reduction
evidence_used: The 104-unit full-resolution GRU achieved 85.03% accuracy at 1,008,983,040 recurrent MACs, while every temporal-subsampling design failed; this motivates a conservative two-unit width reduction without dropping frames.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5082011508174924488, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38808, "peak_hidden_elements": 104960, "recurrent_macs": 973618560, "recurrent_steps": 26080, "total_inference_macs": 974283600, "training_seconds": 94.50653929216787, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46548415107961083}

RECENT RESULT
hypothesis: A 103-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs than the qualified 104-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 103 while preserving full temporal coverage and the verified training procedure.
mechanism: Boundary-seeking full-resolution GRU width reduction
evidence_used: The 104-unit GRU narrowly qualified at 85.03%, while 102 units narrowly missed at 84.91%; testing 103 units directly resolves the remaining width boundary without repeating failed temporal subsampling.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5173870660129258897, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 991222560, "recurrent_steps": 26080, "total_inference_macs": 991894120, "training_seconds": 105.33673537499271, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.45174431362035083}

RECENT RESULT
hypothesis: Skipping only the first frame will retain at least 85% validation accuracy while reducing recurrent execution from 32 to 31 steps and recurrent MACs from 991,222,560 to approximately 960,246,855.
change: Preserve the qualified 103-unit GRU and training procedure, but omit the earliest frame from recordings containing more than two frames.
mechanism: Single-edge-frame temporal trim
evidence_used: The 103-unit full-resolution model achieved 85.89% accuracy; unlike the failed 16–24-step schedules, this conservative change retains 31 of 32 frames and the complete trailing speech context.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5012296669473509577, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 960246855, "recurrent_steps": 25265, "total_inference_macs": 960918415, "training_seconds": 94.25584362493828, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4491593905021808}

RECENT RESULT
hypothesis: A 103-unit GRU processing frames 2–31 will retain at least 85% validation accuracy while reducing recurrent execution from 31 to 30 steps and recurrent MACs from 960,246,855 to approximately 929,271,150.
change: Use the qualified 103-unit recurrent width and omit the first two frames when more than two frames are available.
mechanism: Two-frame leading-edge temporal trim
evidence_used: The 103-unit GRU skipping only the first frame achieved 85.64% accuracy at 31 steps, leaving 0.64 percentage points of margin; trimming one additional adjacent leading frame is the smallest untested temporal-cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4850722678817760257, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 929271150, "recurrent_steps": 24450, "total_inference_macs": 929942710, "training_seconds": 95.05833658296615, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.45136498761323335}

RECENT RESULT
hypothesis: A 103-unit GRU processing frames 3–31 will retain at least 85% validation accuracy while reducing execution from 30 to 29 recurrent steps and recurrent MACs from 929,271,150 to approximately 898,295,445.
change: Reduce the GRU width from 112 to the qualified 103 units and omit the first three frames when more than three are available.
mechanism: Three-frame leading-edge temporal trim
evidence_used: The 103-unit GRU remained qualified when trimming one leading frame (85.64%) and two leading frames (85.15%); trimming one additional adjacent frame is the smallest untested reduction below the current 30-step frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4689148688162010937, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 898295445, "recurrent_steps": 23635, "total_inference_macs": 898967005, "training_seconds": 146.65369983389974, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44609670814560964}

RECENT RESULT
hypothesis: A 103-unit GRU processing frames 4–31 will retain at least 85% validation accuracy while reducing execution from 29 to 28 recurrent steps and recurrent MACs from 898,295,445 to approximately 867,319,740.
change: Preserve the qualified architecture and training procedure, but omit one additional leading frame when more than four frames are available.
mechanism: Four-frame leading-edge temporal trim
evidence_used: The 103-unit GRU remained qualified after progressively trimming one, two, and three leading frames, reaching 85.03% accuracy at 29 steps; trimming the adjacent fourth frame is the smallest untested reduction below the current cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4527574697506261617, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 867319740, "recurrent_steps": 22820, "total_inference_macs": 867991300, "training_seconds": 182.853933500126, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4648651684720092}

RECENT RESULT
hypothesis: A 103-unit GRU processing frames 5–31 will retain at least 85% validation accuracy while reducing execution from 28 to 27 recurrent steps and recurrent MACs from 867,319,740 to approximately 836,344,035.
change: Preserve the qualified architecture and training procedure, but omit the first five frames of standard 32-frame recordings.
mechanism: Five-frame leading-edge temporal trim
evidence_used: The 103-unit GRU qualified after progressively trimming one through four leading frames, achieving 85.40% accuracy at 28 steps; trimming the adjacent fifth frame is the smallest untested reduction below that cost frontier.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4366000706850512297, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 836344035, "recurrent_steps": 22005, "total_inference_macs": 837015595, "training_seconds": 188.8360079161357, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.4952002776912385}

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
