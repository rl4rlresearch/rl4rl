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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1859901261763147973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 354896640, "recurrent_steps": 22005, "total_inference_macs": 356565760, "training_seconds": 38.6987130001653, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.43020456700237253}
prior_hypothesis: A 64-unit, 27-step GRU retaining the temporal-deviation readout will achieve at least 85% validation accuracy while reducing dense inference MACs versus the verified 65-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 64-unit GRU augmented with a running-minimum summary will recover validation accuracy to at least 85% while retaining approximately 2.6% fewer dense inference MACs than the verified 65-unit model.
change: Reduce the GRU to 64 units and expand its temporal readout from mean/final/maximum to mean/final/maximum/minimum, producing a 256-feature classifier input while preserving all 32 causal steps and the training procedure.
mechanism: Complementary minimum-statistic temporal readout
evidence_used: The plain 64-unit model reached 84.54%, missing the threshold by only 0.46 points, while the 65-unit model reached 86.50%; adding a complementary extrema statistic costs only one additional 64-by-8 classifier slice and may recover the small accuracy deficit without restoring the substantially larger recurrent matrix.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: A 64-unit GRU with mean, final, maximum, and minimum summaries will recover validation accuracy to at least 85% while using approximately 2.6% fewer dense inference MACs than the verified 65-unit model.
change: Reduce the GRU width from 65 to 64 and add a running-minimum state and classifier feature, yielding a 256-feature readout across all 32 causal frames.
mechanism: Complementary minimum-statistic temporal readout
evidence_used: The plain 64-unit model missed the threshold by only 0.46 percentage points, while 65 units achieved 86.50%; the complementary minimum summary adds only a 64-by-8 classifier slice and preserves the lower-cost 64-unit recurrent matrix. The previous attempt was malformed before verification, so the idea remains untested.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2202715128035029648, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 422287360, "training_seconds": 45.80633666715585, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.44452852588489744}

RECENT RESULT
hypothesis: A 65-unit GRU processing 30 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 6.2% versus the verified 32-step model.
change: Preserve the verified 65-unit recurrent capacity and training procedure, but uniformly select 30 frames spanning the complete causal one-second input.
mechanism: Uniform causal temporal subsampling
evidence_used: The 65-unit, 32-step model achieved 86.50% accuracy, while both tested 64-unit variants failed; this motivates preserving recurrent width and probing modest temporal redundancy instead.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2120523122879197173, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 405258750, "recurrent_steps": 24450, "total_inference_macs": 406530150, "training_seconds": 31.898174749920145, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43595073676548124}

RECENT RESULT
hypothesis: A 65-unit GRU processing 29 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3.3% versus the verified 30-step model.
change: Reduce the uniform full-utterance frame schedule from 30 to 29 causal recurrent steps while preserving recurrent width, readout, and training procedure.
mechanism: One-step uniform temporal pruning
evidence_used: The 65-unit model remained above threshold at 30 steps with 85.77% accuracy after reducing the verified 32-step schedule; this motivates the smallest next probe of temporal redundancy while retaining the recurrent capacity that succeeded where 64-unit variants failed.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2050060079311807733, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 391750125, "recurrent_steps": 23635, "total_inference_macs": 393021525, "training_seconds": 32.77789683290757, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4446022361334116}

RECENT RESULT
hypothesis: A 65-unit GRU processing 28 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3.4% versus the verified 29-step model.
change: Reduce the uniform full-utterance frame schedule from 29 to 28 causal recurrent steps while preserving recurrent width, temporal readout, and training procedure.
mechanism: One-step uniform temporal pruning
evidence_used: The 65-unit model achieved 86.01% accuracy at 29 steps, improving on the qualifying 30-step result despite using one fewer frame; this supports testing the next single-step reduction while retaining the recurrent capacity that succeeded where 64-unit variants failed.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1979597035744418293, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 378241500, "recurrent_steps": 22820, "total_inference_macs": 379512900, "training_seconds": 57.021912374999374, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4522207061205905}

RECENT RESULT
hypothesis: A 65-unit GRU processing 27 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3.6% versus the verified 28-step model.
change: Reduce the uniform full-utterance frame schedule from 28 to 27 causal recurrent steps while preserving recurrent width, temporal readout, and training procedure.
mechanism: One-step uniform temporal pruning
evidence_used: The 65-unit model qualified at 28 steps with 85.77% accuracy after also qualifying at 29 and 30 steps; this supports the next minimal probe of temporal redundancy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1909133992177028853, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 364732875, "recurrent_steps": 22005, "total_inference_macs": 366004275, "training_seconds": 29.926977416034788, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.44354785380919287}

RECENT RESULT
hypothesis: A 66-unit GRU processing 27 uniformly spaced frames will recover validation accuracy to at least 85% while reducing total dense inference MACs by approximately 0.9% versus the verified 65-unit, 28-step model.
change: Increase recurrent width from 65 to 66 units, resize the classifier accordingly, and reduce the frame schedule from 28 to 27 steps.
mechanism: Width–time MAC rebalancing
evidence_used: The 65-unit model qualified at 28 steps with 85.77% accuracy but narrowly failed at 27 steps with 84.79%; the verified 66-unit model achieved 86.50% at 32 steps, suggesting one extra hidden unit may restore the small deficit while the removed recurrent step still yields a net MAC reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1961232007134469161, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 19056, "peak_hidden_elements": 101888, "recurrent_macs": 374701140, "recurrent_steps": 22005, "total_inference_macs": 375992100, "training_seconds": 51.84664774988778, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.433614831468079}

RECENT RESULT
hypothesis: A 67-unit GRU processing 26 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.1% versus the verified 66-unit, 27-step model.
change: Increase recurrent and summary width from 66 to 67 units, resize the classifier input from 198 to 201 features, and reduce the uniform full-utterance schedule from 27 to 26 steps.
mechanism: Width–time MAC rebalancing
evidence_used: The 66-unit, 27-step model achieved 85.77% accuracy and improved MACs over the qualifying 65-unit, 28-step model; continuing the same one-unit-for-one-step exchange tests whether modest added recurrent capacity can preserve accuracy at a lower net cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1939678605516108785, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 19545, "peak_hidden_elements": 103424, "recurrent_macs": 370549530, "recurrent_steps": 21190, "total_inference_macs": 371860050, "training_seconds": 48.22505724988878, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.46492427521688073}

RECENT RESULT
hypothesis: A 65-unit GRU using 27 frames obtained by removing only the second frame from the verified 28-frame schedule will achieve at least 85% accuracy while matching the 65-unit/27-step model’s 366,004,275 MAC cost.
change: Restore the 65-unit GRU and replace the rephased 27-frame schedule with a 27-frame subset of the successful 28-frame schedule that retains both utterance endpoints.
mechanism: Nested temporal pruning with boundary preservation
evidence_used: The 65-unit model achieved 85.77% at 28 steps but narrowly missed at 27 steps with 84.79%; the prior 27-step formula moved many sampling locations, so pruning one early, densely sampled frame from the successful schedule isolates frame count from schedule rephasing.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1909133992177028853, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 364732875, "recurrent_steps": 22005, "total_inference_macs": 366004275, "training_seconds": 40.939223709050566, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46081740490497985}

RECENT RESULT
hypothesis: A 65-unit, 27-step GRU augmented with per-unit temporal standard deviation will achieve at least 85% validation accuracy while using fewer dense inference MACs than the verified 66-unit, 27-step model.
change: Reduce the GRU width to 65 and extend the mean/final/maximum readout with a running second-moment statistic, producing a 260-feature classifier input without adding recurrent matrix cost.
mechanism: Variance-augmented temporal readout
evidence_used: The plain 65-unit, 27-step model reached 84.79%, only two validation examples short of qualification, while 65 units qualified at 28 steps. Temporal standard deviation supplies complementary activation-duration information at only one additional 65-by-8 classifier slice; unlike the unsuccessful minimum-statistic experiment, it preserves the stronger 65-unit recurrent core.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1911344597462861173, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 19093, "peak_hidden_elements": 133632, "recurrent_macs": 364732875, "recurrent_steps": 22005, "total_inference_macs": 366428075, "training_seconds": 44.28432758315466, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.41471749522203316}

RECENT RESULT
hypothesis: A 64-unit, 27-step GRU retaining the temporal-deviation readout will achieve at least 85% validation accuracy while reducing dense inference MACs versus the verified 65-unit model.
change: Reduce the GRU and all recurrent summary widths from 65 to 64 units, and resize the four-statistic classifier input from 260 to 256 features.
mechanism: Variance-assisted recurrent-width pruning
evidence_used: Temporal deviation raised the 65-unit, 27-step design from 84.79% to 86.50%; testing it at 64 units directly probes whether that accuracy gain permits the next structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1859901261763147973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 354896640, "recurrent_steps": 22005, "total_inference_macs": 356565760, "training_seconds": 38.6987130001653, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.43020456700237253}

RECENT RESULT
hypothesis: A 64-unit GRU with deviation readout processing 26 uniformly spaced frames will achieve at least 85% validation accuracy while reducing dense inference MACs below the verified 64-unit, 27-step model.
change: Reduce the uniform full-utterance frame schedule from 27 to 26 recurrent steps while preserving recurrent width, four-statistic readout, and training procedure.
mechanism: Variance-assisted temporal pruning
evidence_used: The deviation readout raised the 65-unit, 27-step design from 84.79% to 86.50% and enabled 64 units at 27 steps to qualify at 85.40%; this motivates testing whether its temporal information allows one additional step reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1791338488508771638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 341752320, "recurrent_steps": 21190, "total_inference_macs": 343421440, "training_seconds": 40.89927391707897, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.45492850461620493}



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
