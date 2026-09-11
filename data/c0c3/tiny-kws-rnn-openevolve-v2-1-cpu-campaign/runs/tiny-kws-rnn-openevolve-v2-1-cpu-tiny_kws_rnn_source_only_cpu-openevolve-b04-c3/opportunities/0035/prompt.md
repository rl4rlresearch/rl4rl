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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5410082336840918593, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 1035379260, "recurrent_steps": 22005, "total_inference_macs": 1037178780, "training_seconds": 31.742179125081748, "validation_accuracy": 0.8834355828220859, "validation_cross_entropy": 0.3520360934953748}
prior_hypothesis: A stacked 48-unit acoustic GRU and 90-unit contextual GRU, using all 27 qualified frames and both levels’ mean and terminal states, will retain at least 85% accuracy while reducing total inference MACs below the qualified 120-unit/26-step model’s 1.070B.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4810005029429351608, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 920337120, "recurrent_steps": 19560, "total_inference_macs": 922136640, "training_seconds": 43.2259170000907, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.39977568643956096}
prior_hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 960,484,020 to approximately 922,136,640.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4609979260292162613, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 881989740, "recurrent_steps": 18745, "total_inference_macs": 883789260, "training_seconds": 40.58797562494874, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43369420525486485}
prior_hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 922,136,640 to approximately 883,789,260.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5578887557316745014, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 55024, "peak_hidden_elements": 123392, "recurrent_macs": 1067976000, "recurrent_steps": 21190, "total_inference_macs": 1069540800, "training_seconds": 78.03866162500344, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.41509296440639376}
prior_hypothesis: The qualified 120-unit GRU will retain at least 85% validation accuracy after dropping the next-earliest scheduled frame, while reducing total inference MACs from 1.111B to approximately 1.070B.

## Recent verification evidence

RECENT RESULT
hypothesis: A 123-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 124-unit design.
change: Reduce the recurrent state and both readout widths from 124 to 123 units, preserving the schedule, dual readout, auxiliary losses, and all training settings.
mechanism: Deep-supervised recurrent-width boundary search
evidence_used: Training-only dual-view supervision raised the 124-unit design from 84.79% to 86.38% without adding inference MACs, leaving 1.38 percentage points of margin and motivating the adjacent structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6065037920228762065, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57505, "peak_hidden_elements": 126464, "recurrent_macs": 1161137835, "recurrent_steps": 22005, "total_inference_macs": 1162741755, "training_seconds": 102.38246341701597, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3980303430849789}

RECENT RESULT
hypothesis: A 122-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 123-unit design’s 1.163B.
change: Reduce recurrent and readout width to 122 units and add the qualified training-only auxiliary mean-state and terminal-state classifiers.
mechanism: Deep-supervised adjacent-width reduction
evidence_used: The otherwise identical deep-supervised 123-unit design achieved 85.77% accuracy at 1.163B MACs, leaving 0.77 percentage points of margin for an adjacent structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5973718666103851817, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 56672, "peak_hidden_elements": 125440, "recurrent_macs": 1143643860, "recurrent_steps": 22005, "total_inference_macs": 1145234740, "training_seconds": 102.06052891700529, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4081423332354774}

RECENT RESULT
hypothesis: A 121-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 122-unit design.
change: Reduce recurrent and readout width to 121 units and add auxiliary mean-state and terminal-state classifiers used only during training.
mechanism: Deep-supervised adjacent-width reduction
evidence_used: The otherwise identical deep-supervised 122-unit design achieved 85.77% accuracy at 1.145B MACs, matching the 123-unit design’s accuracy while using fewer MACs; this supports testing the adjacent lower-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5883088100548758405, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 55845, "peak_hidden_elements": 124416, "recurrent_macs": 1126281915, "recurrent_steps": 22005, "total_inference_macs": 1127859755, "training_seconds": 89.98986204201356, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.40845211473710696}

RECENT RESULT
hypothesis: A 120-unit GRU preserving the qualified 27-frame schedule, dual readout, and training-only auxiliary supervision will retain at least 85% validation accuracy while reducing exact inference MACs below the qualified 121-unit design.
change: Reduce the recurrent state and all associated readout dimensions from 121 to 120 units without changing training or temporal coverage.
mechanism: Deep-supervised adjacent-width reduction
evidence_used: The otherwise identical 121-unit design achieved 85.89% accuracy at 1.128B MACs, slightly improving upon both the qualified 122- and 123-unit designs; its 0.89-point qualification margin motivates testing the adjacent structural width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5793146223563481829, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 55024, "peak_hidden_elements": 123392, "recurrent_macs": 1109052000, "recurrent_steps": 22005, "total_inference_macs": 1110616800, "training_seconds": 53.50088058388792, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4027262213771329}

RECENT RESULT
hypothesis: The qualified 120-unit GRU will retain at least 85% validation accuracy after dropping the next-earliest scheduled frame, while reducing total inference MACs from 1.111B to approximately 1.070B.
change: Reconstruct the qualified 120-unit deep-supervised dual-readout model and reduce its schedule from 27 to 26 steps by removing the two earliest frames from the original 28-frame grid.
mechanism: Late-biased 26-step temporal thinning
evidence_used: The 120-unit 27-step design achieved 86.13% accuracy with 1.13 percentage points of margin, and prior schedule testing found that removing the earliest frame was the best-performing 27-frame thinning strategy, motivating one further early-frame removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5578887557316745014, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 55024, "peak_hidden_elements": 123392, "recurrent_macs": 1067976000, "recurrent_steps": 21190, "total_inference_macs": 1069540800, "training_seconds": 78.03866162500344, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.41509296440639376}

RECENT RESULT
hypothesis: A 119-unit GRU using the qualified 26-frame schedule and unchanged dual-view supervision will retain at least 85% validation accuracy while reducing total inference MACs from 1,069,540,800 to approximately 1,053,063,130.
change: Reconstruct the best 26-step design while reducing the GRU state and all associated readout dimensions from 120 to 119 units.
mechanism: Deep-supervised adjacent-width reduction at 26 recurrent steps
evidence_used: The 120-unit 26-step design qualified at 85.28%; at 27 steps, every adjacent reduction from 123 through 120 units qualified, with accuracy improving from 85.77% to 86.13%, supporting one adjacent width test on the lower-step design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5492937523337999329, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 54209, "peak_hidden_elements": 122368, "recurrent_macs": 1051511370, "recurrent_steps": 21190, "total_inference_macs": 1053063130, "training_seconds": 83.15202737506479, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4237176719618721}

RECENT RESULT
hypothesis: A 119-unit GRU whose independently supervised mean and terminal heads are also used for inference will recover the failed 119-unit design’s 0.34-point accuracy deficit, reaching at least 85% while retaining approximately 1.053B inference MACs and reducing learned parameters.
change: Reduce the recurrent width to 119, remove the redundant concatenated classifier, and average the separately supervised mean-state and terminal-state logits for the final prediction.
mechanism: Inference-head-tied dual-view supervision
evidence_used: The prior 119-unit design missed qualification narrowly at 84.66%, while training-only dual-view supervision raised the 124-unit design from 84.79% to 86.38%; tying those supervised views directly to inference targets the optimization gap without increasing inference MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5492937523337997417, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 52297, "peak_hidden_elements": 122368, "recurrent_macs": 1051511370, "recurrent_steps": 21190, "total_inference_macs": 1053063130, "training_seconds": 53.00572799984366, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.4435740933096482}

RECENT RESULT
hypothesis: A stacked 48-unit acoustic GRU and 90-unit contextual GRU, using all 27 qualified frames and both levels’ mean and terminal states, will retain at least 85% accuracy while reducing total inference MACs below the qualified 120-unit/26-step model’s 1.070B.
change: Replace the monolithic 120-unit recurrence with two narrower recurrent stages, normalize their interface and readouts, and classify from fine- and context-scale temporal summaries while retaining training-only dual-view supervision.
mechanism: Hierarchical recurrent factorization with multilevel temporal readout
evidence_used: The 120-unit model qualified at 86.13% with 27 steps, but reducing either temporal coverage or width approached the accuracy boundary. This suggests monolithic width and frame deletion are load-bearing assumptions. The factorized recurrence preserves 27-frame coverage and increases total recurrent state from 120 to 138 units while reducing structural recurrent cost from 50,400 to 47,052 MACs per frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5410082336840918593, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 1035379260, "recurrent_steps": 22005, "total_inference_macs": 1037178780, "training_seconds": 31.742179125081748, "validation_accuracy": 0.8834355828220859, "validation_cross_entropy": 0.3520360934953748}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% accuracy after dropping the next-earliest scheduled frame, while reducing total inference MACs from 1.037B to approximately 999M.
change: Replace the monolithic 123-unit GRU with the qualified acoustic/contextual hierarchy and reduce its schedule from 27 to 26 recurrent steps.
mechanism: Hierarchical recurrent factorization with late-biased temporal thinning
evidence_used: Reference Design 2 achieved 88.34% accuracy at 27 steps, leaving 3.34 points of margin; the monolithic 120-unit model retained 85.28% under the same adjacent 27-to-26-step early-frame thinning.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5210056567703729598, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 997031880, "recurrent_steps": 21190, "total_inference_macs": 998831400, "training_seconds": 47.61131420801394, "validation_accuracy": 0.8822085889570552, "validation_cross_entropy": 0.3578270192526601}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional early frame, while reducing total inference MACs from 998,831,400 to approximately 960,484,020.
change: Replace the monolithic GRU with the qualified hierarchical dual-readout model, retain auxiliary supervision, and reduce its schedule from 26 to 25 recurrent steps.
mechanism: Hierarchical recurrent factorization with 25-step late-biased temporal thinning
evidence_used: The identical 48/90 hierarchy achieved 88.22% accuracy at 26 steps—3.22 percentage points above the threshold—and previously lost only 0.12 points when reduced from 27 to 26 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5010030798566540603, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 958684500, "recurrent_steps": 20375, "total_inference_macs": 960484020, "training_seconds": 44.086129875155166, "validation_accuracy": 0.8846625766871166, "validation_cross_entropy": 0.35723108396939707}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 960,484,020 to approximately 922,136,640.
change: Reduce the current schedule from 25 to 24 recurrent steps by dropping the next-earliest frame from the original 28-frame grid.
mechanism: Hierarchical recurrence with 24-step late-biased temporal thinning
evidence_used: The identical hierarchy achieved 88.22% at 26 steps and 88.47% at 25 steps, so the latest thinning preserved accuracy and leaves 3.47 percentage points of qualification margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4810005029429351608, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 920337120, "recurrent_steps": 19560, "total_inference_macs": 922136640, "training_seconds": 43.2259170000907, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.39977568643956096}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 922,136,640 to approximately 883,789,260.
change: Reduce the schedule to 23 recurrent steps by dropping the five earliest frames from the original 28-frame grid, preserving architecture and training procedure.
mechanism: Hierarchical recurrence with 23-step late-biased temporal thinning
evidence_used: The identical hierarchy qualified at 86.26% with 24 steps and 922,136,640 MACs; testing the adjacent 23-step schedule directly locates the temporal-compression boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4609979260292162613, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 881989740, "recurrent_steps": 18745, "total_inference_macs": 883789260, "training_seconds": 40.58797562494874, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43369420525486485}



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
