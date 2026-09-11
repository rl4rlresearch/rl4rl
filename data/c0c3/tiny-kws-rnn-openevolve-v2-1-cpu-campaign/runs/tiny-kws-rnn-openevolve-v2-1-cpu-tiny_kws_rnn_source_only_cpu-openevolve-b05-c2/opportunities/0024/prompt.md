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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2513254156726264901, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20541, "peak_hidden_elements": 106496, "recurrent_macs": 480471840, "recurrent_steps": 26080, "total_inference_macs": 481821480, "training_seconds": 60.18430820805952, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.433567388803681}
prior_hypothesis: A 69-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.52% versus the qualified 70-unit design.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2644053971023329041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21561, "peak_hidden_elements": 109568, "recurrent_macs": 505508640, "recurrent_steps": 26080, "total_inference_macs": 506897400, "training_seconds": 77.82525529200211, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.41365568008890913}
prior_hypothesis: A 71-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.46% versus the qualified 72-unit design.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2578245952129720328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21048, "peak_hidden_elements": 108032, "recurrent_macs": 492912000, "recurrent_steps": 26080, "total_inference_macs": 494281200, "training_seconds": 76.46355379093438, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40918661480301}
prior_hypothesis: A 70-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.49% versus the qualified 71-unit design.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 76-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 77-unit design.
change: Reduce the GRU and recurrent summary width from 79 to 76 and resize the triple-readout classifier from 237 to 228 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 77-unit triple-readout design achieved 86.26% validation accuracy with a 1.26-point margin, after 78–80 units all achieved 86.87%; this supports probing one unit below the smallest qualified width while retaining the successful matrix-free maximum summary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2985337417843671896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 570839040, "recurrent_steps": 26080, "total_inference_macs": 572325600, "training_seconds": 74.62129724980332, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4343214444587567}

RECENT RESULT
hypothesis: A 75-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 76-unit design.
change: Reduce the GRU and recurrent summaries from 80 to 75 units and resize the triple-readout classifier from 240 to 225 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 76-unit triple-readout design achieved 85.89% validation accuracy, and every tested triple-readout width from 76 through 80 qualified; probing 75 units directly identifies whether the stable pruning trend extends below the current lowest-MAC design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2915448281499296753, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23673, "peak_hidden_elements": 115712, "recurrent_macs": 557460000, "recurrent_steps": 26080, "total_inference_macs": 558927000, "training_seconds": 82.13960404112004, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.41049301989970766}

RECENT RESULT
hypothesis: A 74-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.37% versus the qualified 75-unit design.
change: Reduce the GRU and recurrent-summary width from 75 to 74 and resize the triple-readout classifier from 225 to 222 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 75-unit design achieved 86.75% validation accuracy, while every tested triple-readout width from 75 through 80 qualified; its 1.75-point margin motivates the next one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2846375368645074896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23136, "peak_hidden_elements": 114176, "recurrent_macs": 544237440, "recurrent_steps": 26080, "total_inference_macs": 545684880, "training_seconds": 80.2417831250932, "validation_accuracy": 0.8785276073619632, "validation_cross_entropy": 0.3959634605360909}

RECENT RESULT
hypothesis: A 73-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.4% versus the qualified 74-unit design.
change: Reduce the GRU and recurrent-summary width from 76 to 73 units and resize the triple-readout classifier from 228 to 219 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 74-unit design achieved 87.85% validation accuracy—the strongest result among the tested 74–80-unit triple-readout models—with a 2.85-point margin above the requirement, motivating a one-unit structural reduction to probe the next compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2778118679281006325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22605, "peak_hidden_elements": 112640, "recurrent_macs": 531171360, "recurrent_steps": 26080, "total_inference_macs": 532599240, "training_seconds": 113.48929166584276, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4272858789362059}

RECENT RESULT
hypothesis: A 72-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.43% versus the qualified 73-unit design.
change: Reduce the GRU and recurrent-summary width from 77 to 72 units and resize the triple-readout classifier from 231 to 216 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 73-unit triple-readout model qualified at 85.89% accuracy and 532,599,240 total MACs; testing the adjacent 72-unit width directly probes the next compute boundary without changing the successful temporal readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 61.63849383289926, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.425748027029213}

RECENT RESULT
hypothesis: A 71-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.46% versus the qualified 72-unit design.
change: Reduce the GRU and recurrent-summary width from 72 to 71 units and resize the triple-readout classifier from 216 to 213 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The current 72-unit design achieved 86.38% validation accuracy at 519,670,080 total MACs, leaving a 1.38-point margin above the requirement; every tested triple-readout width from 72 through 80 qualified, motivating the adjacent structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2644053971023329041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21561, "peak_hidden_elements": 109568, "recurrent_macs": 505508640, "recurrent_steps": 26080, "total_inference_macs": 506897400, "training_seconds": 77.82525529200211, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.41365568008890913}

RECENT RESULT
hypothesis: A 70-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.49% versus the qualified 71-unit design.
change: Reduce the GRU and recurrent-summary width from 73 to 70 units and resize the triple-readout classifier from 219 to 210 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 71-unit triple-readout model achieved 87.24% validation accuracy at 506,897,400 total MACs, leaving a 2.24-point margin after every tested width from 71 through 80 qualified; this motivates probing the adjacent lower width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2578245952129720328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21048, "peak_hidden_elements": 108032, "recurrent_macs": 492912000, "recurrent_steps": 26080, "total_inference_macs": 494281200, "training_seconds": 76.46355379093438, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40918661480301}

RECENT RESULT
hypothesis: A 69-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.52% versus the qualified 70-unit design.
change: Reduce the GRU and recurrent-summary width from 74 to 69 units and resize the triple-readout classifier from 222 to 207 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 70-unit triple-readout model achieved 86.38% validation accuracy at 494,281,200 total MACs, leaving a 1.38-point margin; every tested width from 70 through 80 qualified, so the adjacent 69-unit width is the most direct next compute-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2513254156726264901, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20541, "peak_hidden_elements": 106496, "recurrent_macs": 480471840, "recurrent_steps": 26080, "total_inference_macs": 481821480, "training_seconds": 60.18430820805952, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.433567388803681}



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
