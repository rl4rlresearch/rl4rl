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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3199902167817717041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25881, "peak_hidden_elements": 121856, "recurrent_macs": 611915040, "recurrent_steps": 26080, "total_inference_macs": 613460280, "training_seconds": 128.8876136657782, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3818443438758148}
prior_hypothesis: A 79-unit full-resolution GRU using concatenated mean, final, and temporal-maximum outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.2% versus the qualified 80-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 77.72414637496695, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4091419711434768}
prior_hypothesis: An 80-unit full-resolution GRU classifying concatenated mean, final, and temporal-maximum recurrent outputs will recover at least 0.71 accuracy points over the failed 80-unit dual-readout model, reaching at least 85% while using approximately 2.1% fewer total inference MACs than the qualified 81-unit design.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3056042777678200325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24765, "peak_hidden_elements": 118784, "recurrent_macs": 584374560, "recurrent_steps": 26080, "total_inference_macs": 585880680, "training_seconds": 118.98261283291504, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4171936784053873}
prior_hypothesis: A 77-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 78-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: A 111-unit full-resolution GRU with a learned classifier over both mean and final recurrent states will retain at least 85% accuracy while reducing total MACs by about 1.6% versus the qualified 112-unit design.
change: Reduce the GRU width from 128 to 111, retain all 32 causal frames, and concatenate the temporal mean with the final hidden state for classification.
mechanism: Dual-timescale recurrent readout with slight width pruning
evidence_used: Full-frame 112-unit recurrence achieved 85.15%, whereas 24-frame variants failed even at 128 units; this motivates preserving every frame and recovering discriminative capacity through a richer low-cost readout rather than further temporal subsampling.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5941902955051412273, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 1137687840, "recurrent_steps": 26080, "total_inference_macs": 1139135280, "training_seconds": 155.94847820908763, "validation_accuracy": 0.8785276073619632, "validation_cross_entropy": 0.3702620991899923}

RECENT RESULT
hypothesis: A 104-unit GRU using all 32 frames and concatenated mean/final states will retain at least 85% accuracy while reducing total inference MACs by approximately 11.3% versus the qualified 111-unit design.
change: Reduce recurrent width from 128 to 104, preserve the full causal frame schedule, and classify from both the temporal mean and final recurrent output.
mechanism: Full-resolution GRU width pruning with dual-timescale readout
evidence_used: The 111-unit dual-readout design achieved 87.85% accuracy with substantial margin, while both 24-frame designs failed; this supports preserving temporal resolution and testing whether the richer readout permits further width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5270083004031078304, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 41024, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1010339200, "training_seconds": 111.99590533296578, "validation_accuracy": 0.8834355828220859, "validation_cross_entropy": 0.38310525344193347}

RECENT RESULT
hypothesis: A 96-unit GRU using all 32 frames and concatenated mean/final states will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13.6% versus the qualified 104-unit design.
change: Reduce the recurrent and summary width from 104 to 96 and resize the dual-readout classifier accordingly, while preserving the complete causal frame schedule and training procedure.
mechanism: Full-resolution GRU width pruning with dual-timescale readout
evidence_used: The 104-unit full-resolution dual-readout model achieved 88.34% accuracy, improving on the 111-unit model’s 87.85% despite lower width; meanwhile both 24-frame variants failed, motivating another moderate width reduction without temporal subsampling.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 103.49331941688433, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.3884638406016344}

RECENT RESULT
hypothesis: An 88-unit GRU processing all 32 frames and classifying concatenated mean/final states will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 14.7% versus the qualified 96-unit design.
change: Reduce the GRU hidden and summary width from 111 to 88 and resize the dual-readout classifier from 222 to 176 inputs, preserving the full causal schedule and training procedure.
mechanism: Full-resolution GRU width pruning with dual-timescale readout
evidence_used: The 96-unit full-resolution dual-readout model achieved 87.36% accuracy, leaving a 2.36-point margin; the preceding reduction from 104 to 96 units cost only 0.98 points while substantially reducing MACs, motivating another eight-unit reduction without the temporal subsampling that previously failed.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 92.61242508306168, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.40472788430430406}

RECENT RESULT
hypothesis: An 80-unit GRU processing all 32 frames and classifying concatenated mean/final states will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 15.8% versus the qualified 88-unit design.
change: Reduce the GRU and summary width from 112 to 80, resize the classifier for a 160-feature mean/final readout, and preserve the complete causal frame schedule and training procedure.
mechanism: Full-resolution GRU width pruning with dual-timescale readout
evidence_used: The 88-unit full-resolution dual-readout design achieved 86.87% accuracy, and the preceding reduction from 96 to 88 units cost only 0.49 points; this supports testing another eight-unit reduction while avoiding the temporal subsampling that repeatedly failed.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 58.791185040958226, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4433465594894316}

RECENT RESULT
hypothesis: An 84-unit full-resolution GRU will achieve at least 85% validation accuracy while reducing recurrent MACs by about 8% versus the qualified 88-unit design.
change: Reduce the GRU and summary width from 88 to 84 and resize the mean/final classifier input from 176 to 168, preserving all 32 causal frames and the established training procedure.
mechanism: Boundary-seeking GRU width pruning with dual-timescale readout
evidence_used: The 88-unit model achieved 86.87% accuracy, while 80 units achieved 84.29%; 84 units tests the midpoint of the observed feasibility boundary, where interpolation suggests roughly 85.58% accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3570977772028654184, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 684600000, "training_seconds": 56.57472599996254, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.40037742567939993}

RECENT RESULT
hypothesis: An 82-unit full-resolution GRU with concatenated mean/final states will retain at least 85% validation accuracy while reducing total inference MACs by approximately 4.3% versus the qualified 84-unit design.
change: Reduce the GRU and summary width from 96 to 82 and resize the classifier input from 192 to 164, while preserving all 32 causal frames and the established training procedure.
mechanism: Boundary-seeking GRU width pruning with dual-timescale readout
evidence_used: The 84-unit design achieved 87.73% accuracy, while 80 units achieved 84.29%; testing 82 units directly probes the remaining width boundary with substantially more accuracy margin than linear interpolation requires.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3419024165611784064, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26944, "peak_hidden_elements": 84480, "recurrent_macs": 654399360, "recurrent_steps": 26080, "total_inference_macs": 655468640, "training_seconds": 57.525018708081916, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223508308270226}

RECENT RESULT
hypothesis: An 81-unit full-resolution GRU with concatenated mean/final states will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 2.2% versus the qualified 82-unit design.
change: Reduce recurrent and summary width from 112 to 81, resize the classifier for a 162-feature mean/final readout, and retain all 32 causal frames and the established training procedure.
mechanism: Boundary-seeking GRU width pruning with dual-timescale readout
evidence_used: The 82-unit dual-readout model achieved 86.01% accuracy, while 80 units achieved 84.29%; testing the intervening 81-unit width directly resolves the remaining feasibility boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3344271697638578933, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 640081440, "recurrent_steps": 26080, "total_inference_macs": 641137680, "training_seconds": 70.14341566688381, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.43995596294754125}

RECENT RESULT
hypothesis: An 80-unit full-resolution GRU classifying concatenated mean, final, and temporal-maximum recurrent outputs will recover at least 0.71 accuracy points over the failed 80-unit dual-readout model, reaching at least 85% while using approximately 2.1% fewer total inference MACs than the qualified 81-unit design.
change: Reduce the GRU width from 81 to 80 and add an online elementwise maximum summary to the recurrent state and classifier, preserving all 32 causal steps and the established training procedure.
mechanism: Max-pooled temporal readout with boundary-width GRU
evidence_used: The 81-unit mean/final model achieved 85.64%, whereas the otherwise equivalent 80-unit model achieved 84.29%; adding a matrix-free temporal maximum supplies complementary transient-feature information at only 640 additional classifier MACs per example, while the width reduction saves substantially more recurrent compute.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 77.72414637496695, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4091419711434768}

RECENT RESULT
hypothesis: A 79-unit full-resolution GRU using concatenated mean, final, and temporal-maximum outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.2% versus the qualified 80-unit design.
change: Reduce the GRU width from 82 to 79 and add a matrix-free online temporal maximum to the recurrent state and classifier, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU with mean/final/maximum temporal readout
evidence_used: The 80-unit mean/final/maximum design achieved 86.87% accuracy, whereas the 80-unit mean/final design achieved only 84.29%; its 1.87-point margin motivates probing one unit narrower while retaining the beneficial maximum summary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3199902167817717041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25881, "peak_hidden_elements": 121856, "recurrent_macs": 611915040, "recurrent_steps": 26080, "total_inference_macs": 613460280, "training_seconds": 128.8876136657782, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3818443438758148}

RECENT RESULT
hypothesis: A 78-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 79-unit design.
change: Reduce the GRU width to 78, add an online elementwise maximum to the recurrent state, and classify the concatenated mean, final, and maximum outputs across all 32 frames.
mechanism: Boundary-width GRU with mean/final/maximum temporal readout
evidence_used: The 79- and 80-unit triple-readout designs both achieved 86.87% accuracy, while the 79-unit model had lower cross-entropy; this margin supports probing one unit narrower while retaining the matrix-free maximum summary that rescued the failed 80-unit dual-readout design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3127564361002882040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25320, "peak_hidden_elements": 120320, "recurrent_macs": 598066560, "recurrent_steps": 26080, "total_inference_macs": 599592240, "training_seconds": 154.36371741606854, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.398383457675302}

RECENT RESULT
hypothesis: A 77-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 78-unit design.
change: Reduce the GRU width from 78 to 77 and resize its recurrent state and triple-readout classifier accordingly, preserving all 32 causal frames and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 78-, 79-, and 80-unit triple-readout designs all achieved 86.87% validation accuracy, indicating a stable accuracy plateau with 1.87 points of margin above the requirement and motivating another one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3056042777678200325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24765, "peak_hidden_elements": 118784, "recurrent_macs": 584374560, "recurrent_steps": 26080, "total_inference_macs": 585880680, "training_seconds": 118.98261283291504, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4171936784053873}



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
