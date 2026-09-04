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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5783177243794182173, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 56173, "peak_hidden_elements": 128512, "recurrent_macs": 1107890625, "recurrent_steps": 20375, "total_inference_macs": 1108705625, "training_seconds": 111.40233691688627, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4153137955928873}
prior_hypothesis: A 125-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.47% versus the passing 126-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing only the earliest frame from the verified 28-frame schedule will retain at least 85% accuracy while reducing recurrent steps and MACs by approximately 3.6%.
change: Preserve 27 of the exact frame indices used by the passing design, dropping frame 0 instead of recomputing a phase-shifted uniform 27-frame schedule.
mechanism: Passing-schedule frame ablation
evidence_used: The uniform 28-frame schedule passed at 85.276%, whereas the uniform 27-frame schedule failed at 84.294%; because recomputing 27 uniform indices changes many sampled locations, ablating one likely low-information leading frame from the passing schedule is a more controlled cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6527611327453078597, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1250588160, "recurrent_steps": 22005, "total_inference_macs": 1251422720, "training_seconds": 111.21374308294617, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.44482486467419957}

RECENT RESULT
hypothesis: Removing the next-earliest frame from the passing 27-step schedule will retain at least 85% validation accuracy while reducing recurrent steps and MACs by approximately 3.7%.
change: Preserve 26 indices from the verified schedule by dropping its first two leading frames instead of recomputing uniformly spaced indices.
mechanism: Passing-schedule second leading-frame ablation
evidence_used: Dropping only frame 0 from the passing 28-frame schedule achieved 85.153% accuracy, whereas recomputing all indices for 27 frames achieved only 84.294%; this supports another controlled leading-frame ablation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6286009174286206902, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1204270080, "recurrent_steps": 21190, "total_inference_macs": 1205104640, "training_seconds": 98.09585466678254, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.44582651524456}

RECENT RESULT
hypothesis: Keeping frame 1 while removing frame 2 from the passing 27-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.7%.
change: Produce a 26-step schedule by preserving the passing schedule’s temporal endpoints and dropping the interior frame bracketed by adjacent frames 1 and 3.
mechanism: Interior redundant-frame ablation
evidence_used: Dropping frame 0 passed at 85.153%, while additionally dropping frame 1 narrowly failed at 84.908%; testing the neighboring but more redundant frame 2 isolates whether early coverage, rather than 26-step capacity, caused that miss.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6286009174286206902, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1204270080, "recurrent_steps": 21190, "total_inference_macs": 1205104640, "training_seconds": 128.42696470813826, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4429564306341066}

RECENT RESULT
hypothesis: Removing frame 30 while preserving the passing schedule’s early frames and final endpoint will retain at least 85% validation accuracy while reducing recurrent steps and MACs by approximately 3.7%.
change: Produce a 26-step schedule by dropping frame 30 from the verified 27-step schedule while retaining frame 31.
mechanism: Late redundant-frame ablation
evidence_used: The 27-step schedule passed at 85.153%, but 26-step variants removing frames 1 or 2 failed; ablating a frame from the dense 28–31 tail tests whether early temporal coverage, rather than a strict 27-step requirement, caused those failures.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6286009174286206902, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1204270080, "recurrent_steps": 21190, "total_inference_macs": 1205104640, "training_seconds": 145.79893083381467, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4555224740432084}

RECENT RESULT
hypothesis: Removing frame 29 while retaining frames 28 and 31 will preserve at least 85% validation accuracy and reduce recurrent steps and MACs by approximately 3.8% versus the passing 26-step design.
change: Produce a 25-step schedule by dropping frame 29 from the verified 26-step schedule while preserving all earlier sampled frames and the final endpoint.
mechanism: Consecutive late-frame ablation
evidence_used: Removing frame 30 from the passing 27-step schedule retained 85.153% accuracy, indicating redundancy in the densely sampled 28–31 tail; frame 29 is the next controlled ablation in that region.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6044407021119335207, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1157952000, "recurrent_steps": 20375, "total_inference_macs": 1158786560, "training_seconds": 200.00940612517297, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.46748758327741563}

RECENT RESULT
hypothesis: Removing frame 28 while preserving frames 29 and 31 will retain at least 85% validation accuracy and reduce recurrent steps and MACs by approximately 3.8% versus the passing 26-step design.
change: Produce a 25-step schedule by dropping frame 28 from the verified 26-step schedule while retaining frame 29 and the final endpoint.
mechanism: Alternative late-frame redundancy ablation
evidence_used: The 26-step schedule passed at 85.153%, whereas removing frame 29 fell to 84.540%; preserving frame 29 and ablating its neighboring frame 28 tests whether the failure was caused by losing that specific late observation rather than by the 25-step budget.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Concatenating the final GRU state with the temporal mean will recover the 0.46-point accuracy deficit of the previously tested 25-step schedule, reaching at least 85% accuracy while reducing total inference MACs by approximately 3.8% versus the passing 26-step model.
change: Remove frame 29 from the passing schedule and expand the classifier input to jointly use the mean recurrent output and final hidden state.
mechanism: Complementary mean-and-final recurrent readout
evidence_used: The same 25-step frame ablation reached 84.540%, narrowly below the target; its mean-only classifier discarded the complementary final recurrent representation, while the passing 26-step model establishes that the retained temporal coverage is near the accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6048760213066820391, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 59696, "peak_hidden_elements": 131584, "recurrent_macs": 1157952000, "recurrent_steps": 20375, "total_inference_macs": 1159621120, "training_seconds": 127.02226541703567, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40808657780746743}

RECENT RESULT
hypothesis: The 24-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs by 4% versus the passing 25-step model.
change: Remove frame 28 from the current schedule while preserving all earlier observations, the final frame, and the accuracy-improving concatenated readout.
mechanism: Mean-and-final readout with one fewer late observation
evidence_used: The current 25-step model achieved 86.380% accuracy, providing a 1.38-point margin; its mean-and-final readout improved the same schedule by 1.84 points over mean-only classification, motivating a controlled one-frame reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5807158059899948696, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 59696, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1113303040, "training_seconds": 159.22215054184198, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4324336794987778}

RECENT RESULT
hypothesis: Elementwise addition of the temporal mean and final GRU state will retain at least 85% validation accuracy while eliminating 834,560 classifier MACs and 1,024 learned parameters versus the passing concatenated-readout model.
change: Replace the 256-wide concatenated classifier input with a 128-wide additive fusion of the same two recurrent representations.
mechanism: Parameter-free mean–final state fusion
evidence_used: The 25-step concatenated readout achieved 86.380% accuracy versus 84.540% for mean-only classification, showing that the final state adds useful information; additive fusion preserves access to both representations without the extra dense classifier width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6044407021119335207, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1157952000, "recurrent_steps": 20375, "total_inference_macs": 1158786560, "training_seconds": 168.1866707089357, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4016205302045389}

RECENT RESULT
hypothesis: A 127-unit GRU with the passing 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.45%.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 127 while preserving the verified schedule and training procedure.
mechanism: Additive-readout recurrent-width trim
evidence_used: The current 128-unit additive-readout model reached 86.503% accuracy, 1.503 points above the threshold; additive fusion improved the same 25-step schedule from 84.540% with mean-only classification, providing evidence that the stronger readout may support the conservative one-unit width reduction that previously missed under the weaker architecture.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5956692754075935273, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1141142625, "recurrent_steps": 20375, "total_inference_macs": 1141970665, "training_seconds": 136.1219446659088, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40036090312559913}

RECENT RESULT
hypothesis: A 126-unit GRU with the passing 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.46% versus the verified 127-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 127 to 126 while preserving the successful schedule, additive readout, and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 127-unit additive-readout model achieved 86.380% accuracy, 1.380 percentage points above the requirement; this supports testing the smallest remaining structural width reduction without changing its verified temporal coverage or readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5869616161634217595, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 57000, "peak_hidden_elements": 129536, "recurrent_macs": 1124455500, "recurrent_steps": 20375, "total_inference_macs": 1125277020, "training_seconds": 135.7988055839669, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.40380626280614934}

RECENT RESULT
hypothesis: A 125-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.47% versus the passing 126-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 126 to 125 while preserving the successful frame schedule, additive readout, and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 126-unit additive-readout model achieved 85.890% accuracy after the 127-unit version achieved 86.380%; both passed with identical temporal coverage, supporting the next isolated one-channel reduction. The earlier 125-unit result used a weaker mean-only readout and missed by just 0.092 percentage points, so it does not rule out 125 units with the stronger verified additive readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5783177243794182173, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 56173, "peak_hidden_elements": 128512, "recurrent_macs": 1107890625, "recurrent_steps": 20375, "total_inference_macs": 1108705625, "training_seconds": 111.40233691688627, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4153137955928873}



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
