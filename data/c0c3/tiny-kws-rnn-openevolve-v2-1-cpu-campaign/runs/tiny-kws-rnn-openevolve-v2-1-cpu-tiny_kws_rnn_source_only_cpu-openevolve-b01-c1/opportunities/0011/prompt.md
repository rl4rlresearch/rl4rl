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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4208312280590011128, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 806784800, "training_seconds": 118.667672208976, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.46424440723255367}
prior_hypothesis: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the verified 96-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 16 evenly spaced frames will preserve at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps.
change: Change the frame schedule from all 32 frames to every second frame, including the final frame.
mechanism: Twofold causal temporal subsampling
evidence_used: The 128-unit GRU achieves 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving margin to test temporal redundancy without changing learned capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 32.00494654220529, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.49316335104725845}

RECENT RESULT
hypothesis: Processing 24 evenly distributed frames, including both endpoints, will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 25% versus the 32-frame baseline.
change: Replace the full 32-frame schedule with 24 uniformly spaced, unique, increasing frame indices.
mechanism: Moderate uniform causal temporal subsampling
evidence_used: The 32-step model reached 87.24% accuracy, while aggressive 16-step subsampling reached 83.19%; testing the midpoint directly probes whether moderate temporal redundancy can be removed without crossing the 85% threshold.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 101.05816512485035, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.48110573423420727}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit baseline.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 128 to 112 while preserving the full causal frame schedule.
mechanism: Full-resolution recurrent width compression
evidence_used: Temporal subsampling reduced accuracy from 87.24% to 83.19% even at 24 steps, indicating that all 32 frames are valuable; compressing hidden width instead tests an independent structural cost reduction without discarding temporal evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 107.5079114169348, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.44323348530962425}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 12.8% versus the verified 112-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 112 to 104 while preserving the full causal frame schedule and training procedure.
mechanism: Full-resolution GRU width compression
evidence_used: The 112-unit full-resolution GRU achieved 85.89% accuracy, while temporal subsampling failed at both 16 and 24 steps; this motivates preserving all 32 frames and testing the next structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 113.093671125127, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.45115726213513707}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13.7% versus the verified 104-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 104 to 96 while preserving the full causal frame schedule and training procedure.
mechanism: Full-resolution GRU width compression
evidence_used: The 104-unit full-resolution GRU achieved 86.01% accuracy, slightly improving on the 112-unit model’s 85.89%, while both temporal-subsampling experiments failed; this supports another width reduction without discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 100.82147183292545, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4543663258932851}

RECENT RESULT
hypothesis: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the verified 96-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 96 to 92 while preserving the full causal frame schedule and training procedure.
mechanism: Boundary-seeking full-resolution GRU width compression
evidence_used: The 96-unit full-resolution GRU achieved 85.15% accuracy after the 104-unit model achieved 86.01%; testing the midpoint width of 92 is the most direct probe of the feasible compression boundary without discarding temporally valuable frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4208312280590011128, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 806784800, "training_seconds": 118.667672208976, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.46424440723255367}

RECENT RESULT
hypothesis: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.8% versus the verified 92-unit model.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 92 to 88 while preserving the full causal frame schedule and training procedure.
mechanism: Boundary-seeking full-resolution GRU width compression
evidence_used: The 92-unit full-resolution GRU achieved 85.89% accuracy with 806,184,960 recurrent MACs, leaving enough observed accuracy margin to test the next four-unit structural reduction without discarding valuable frames.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3881686847280337792, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29792, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744166720, "training_seconds": 103.23932075011544, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.47434615620806175}

RECENT RESULT
hypothesis: A 90-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.9% versus the verified 92-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 92 to 90 while retaining all 32 causal frames and the existing training procedure.
mechanism: Boundary-bisection GRU width compression
evidence_used: The 92-unit model achieved 85.89% accuracy, while the 88-unit model achieved 84.66%; width 90 directly probes the midpoint of the observed feasibility boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4043367116954867888, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31008, "peak_hidden_elements": 92672, "recurrent_macs": 774576000, "recurrent_steps": 26080, "total_inference_macs": 775162800, "training_seconds": 94.96799183310941, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46152874531190086}

RECENT RESULT
hypothesis: A 91-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs versus the verified 92-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 92 to 91 while preserving all 32 causal frames and the existing training procedure.
mechanism: Boundary-bisection GRU width compression
evidence_used: The 92-unit model achieved 85.89% accuracy, while widths 90 and 88 missed the threshold; width 91 is the only untested integer at the observed feasibility boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4125431587027362865, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31625, "peak_hidden_elements": 93696, "recurrent_macs": 790302240, "recurrent_steps": 26080, "total_inference_macs": 790895560, "training_seconds": 106.7818828751333, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.4779986048037289}

RECENT RESULT
hypothesis: Selective exits at steps 24 and 28 will preserve at least 85% accuracy while reducing mean recurrent steps and total MACs, because only high-confidence examples stop early while ambiguous examples retain all 32 frames.
change: Keep the verified 92-unit GRU, train its shared classifier on 24-, 28-, and 32-step summaries, cache logits only at those checkpoints, and exit at conservative confidence thresholds.
mechanism: Auxiliary-trained confidence-gated recurrent checkpoints
evidence_used: Uniform 16- and 24-frame execution failed at 83.19%, while the full 32-step 92-unit GRU reached 85.89%; this challenges the load-bearing assumption that every example needs the same temporal budget without again discarding later evidence for every recording.
result: the implementation could not be verified



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
