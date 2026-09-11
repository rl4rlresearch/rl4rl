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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2708229542936630624, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 21504, "peak_hidden_elements": 74240, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519200640, "training_seconds": 89.2653078341391, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.38552489485477376}
prior_hypothesis: A 72-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 17.2% versus the verified 80-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 16 evenly spaced frames instead of all 32 will preserve validation accuracy at or above 85% while halving recurrent steps and recurrent MACs from the 87.24%-accurate baseline.
change: Select the final frame of each two-frame interval, retaining full-recording coverage and the most recent frame.
mechanism: Uniform 2× causal frame striding
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps, leaving a 2.24-point margin for temporal subsampling.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 67.76923475018702, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.5148614427063363}

RECENT RESULT
hypothesis: Processing 24 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 25% versus the 32-step baseline.
change: Replace the full 32-frame schedule with 24 monotonically increasing indices distributed across the complete recording, including the first and final frames.
mechanism: Uniform 4:3 causal frame subsampling
evidence_used: Sixteen-frame 2× striding reduced accuracy from 87.24% to 83.07%, narrowly missing the threshold; retaining half of the discarded temporal evidence is the most informative intermediate cost-accuracy test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 82.46124641690403, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4918780555023006}

RECENT RESULT
hypothesis: Processing 28 evenly distributed frames will recover validation accuracy to at least 85% while reducing recurrent steps and recurrent MACs by 12.5% versus the 32-step baseline.
change: Use 28 unique, monotonically increasing frame indices spanning the complete recording, including the first and final frames.
mechanism: Uniform 7:8 causal frame subsampling
evidence_used: The 32-frame model achieved 87.24% accuracy, while 24 frames achieved 84.42%; retaining half of the four-frame gap is the most informative lower-cost test with a safer accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 69.82911758311093, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4651480739102042}

RECENT RESULT
hypothesis: Processing 26 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 7.1% versus the verified 28-frame design.
change: Reduce the frame schedule from 28 to 26 unique increasing indices spanning the first through final input frames.
mechanism: Uniform 13:16 causal frame subsampling
evidence_used: The 24-frame design narrowly missed the target at 84.42%, while 28 frames reached 86.13%; 26 frames directly tests the midpoint and is expected to preserve sufficient temporal evidence at lower cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6286009174286206902, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1204270080, "recurrent_steps": 21190, "total_inference_macs": 1205104640, "training_seconds": 112.81374025018886, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4942737602748754}

RECENT RESULT
hypothesis: Processing 27 uniformly distributed frames will achieve at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-frame design.
change: Reduce the causal frame schedule from 28 to 27 unique increasing indices spanning the complete recording.
mechanism: Boundary frame-count refinement
evidence_used: The 28-frame model achieved 86.13% accuracy while the 26-frame model missed at 83.93%; testing the intervening 27-frame schedule directly locates the lowest viable step count.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6527611327453078597, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1250588160, "recurrent_steps": 22005, "total_inference_macs": 1251422720, "training_seconds": 93.67891345894895, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4554509636814609}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 10.8% versus the verified 128-unit, 28-frame model.
change: Reduce GRU and classifier width from 128 to 112, resize recurrent state tensors accordingly, and restore the full causal frame sequence.
mechanism: Rebalance recurrent width for complete temporal coverage
evidence_used: The 128-unit model reached 87.24% with all 32 frames, while reducing temporal coverage to 26–27 frames caused threshold failures; full coverage paired with a modest width reduction tests whether temporal evidence is more valuable than excess hidden capacity at lower MAC cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 134.72818629187532, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4385198019764906}

RECENT RESULT
hypothesis: A 104-unit GRU using both mean and final recurrent outputs, trained with batch size 64, will retain at least 85% accuracy while reducing total inference MACs by approximately 12.7% versus the verified 112-unit model.
change: Reduce GRU width to 104, concatenate mean-pooled and final hidden features for classification, and double the number of optimizer updates through a smaller training batch.
mechanism: Optimization-supported width reduction with dual temporal readout
evidence_used: Full 32-frame coverage with 112 units achieved 85.15%, whereas reducing temporal coverage repeatedly failed or approached the threshold; retaining all frames while trimming width is therefore the strongest cost-reduction direction, with the richer readout and additional optimizer steps intended to recover capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 5270083004031078304, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 41024, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1010339200, "training_seconds": 108.98324241722003, "validation_accuracy": 0.8834355828220859, "validation_cross_entropy": 0.3428506020387989}

RECENT RESULT
hypothesis: A 96-unit GRU retaining all 32 frames and the mean-plus-final readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 13.6% versus the verified 104-unit model.
change: Reduce the GRU hidden width from 104 to 96 and resize the classifier and recurrent summary state accordingly.
mechanism: Recurrent-width scaling with dual temporal readout
evidence_used: The 104-unit full-sequence model achieved 88.34% accuracy, providing a 3.34-point margin above the requirement; prior temporal subsampling was less reliable, so reducing width while preserving all temporal evidence is the best-supported next cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 90.31955345906317, "validation_accuracy": 0.8797546012269939, "validation_cross_entropy": 0.36061461278997314}

RECENT RESULT
hypothesis: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 14.6% versus the verified 96-unit model.
change: Reduce the GRU hidden width from 96 to 88 and resize the classifier and recurrent summary state accordingly.
mechanism: Recurrent-width scaling with dual temporal readout
evidence_used: The 96-unit full-sequence model achieved 87.98% accuracy, retaining a 2.98-point margin after the successful reduction from 104 units; width scaling has preserved accuracy more reliably than temporal subsampling.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 98.80110320891254, "validation_accuracy": 0.8822085889570552, "validation_cross_entropy": 0.36406354845667177}

RECENT RESULT
hypothesis: An 80-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 15.8% versus the verified 88-unit model.
change: Reduce the GRU hidden width from 88 to 80 and resize the classifier and recurrent summary state accordingly.
mechanism: Recurrent-width scaling with dual temporal readout
evidence_used: The 88-unit model achieved 88.22% accuracy after successful reductions from 104 and 96 units, leaving a 3.22-point margin and showing that width reduction with full temporal coverage is more reliable than frame subsampling.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 85.81770729180425, "validation_accuracy": 0.8822085889570552, "validation_cross_entropy": 0.36560476104174655}

RECENT RESULT
hypothesis: A 72-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 17.2% versus the verified 80-unit model.
change: Reduce the GRU hidden width from 80 to 72 and resize the classifier and recurrent state tensors accordingly.
mechanism: Recurrent-width scaling with dual temporal readout
evidence_used: The 80-unit model achieved 88.22% accuracy after successive eight-unit reductions from 104 to 96, 88, and 80 units all remained above 87.9%, supporting another eight-unit reduction while preserving full temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2708229542936630624, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 21504, "peak_hidden_elements": 74240, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519200640, "training_seconds": 89.2653078341391, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.38552489485477376}



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
