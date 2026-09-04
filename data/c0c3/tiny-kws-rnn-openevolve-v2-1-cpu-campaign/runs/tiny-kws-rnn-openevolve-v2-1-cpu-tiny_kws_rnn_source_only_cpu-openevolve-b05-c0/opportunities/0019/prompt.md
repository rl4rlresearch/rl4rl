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
verified_results: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1792545819495457061, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 522, "p95_recurrent_steps": 31, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 342896580, "recurrent_steps": 25265, "total_inference_macs": 343652900, "training_seconds": 78.78889991599135, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.427536357107338}
prior_hypothesis: The 58-unit GRU processing the final 31 frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1%.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 68-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9.6% versus the verified 72-unit model.
change: Reduce the GRU hidden width from 72 to 68 and resize the classifier and recurrent summary state accordingly.
mechanism: Fine-grained recurrent-width scaling with dual temporal readout
evidence_used: The 72-unit model achieved 86.13% accuracy after successive width reductions remained viable; testing the four-unit midpoint toward 64 is the most informative lower-risk continuation given its narrower 1.13-point margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2446765951590861256, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 19496, "peak_hidden_elements": 70144, "recurrent_macs": 468188160, "recurrent_steps": 26080, "total_inference_macs": 469074880, "training_seconds": 124.41145316697657, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3770450334607458}

RECENT RESULT
hypothesis: A 64-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9% versus the verified 68-unit model.
change: Reduce the GRU hidden width from 68 to 64 and resize the classifier and recurrent state tensors accordingly.
mechanism: Fine-grained recurrent-width scaling with full temporal coverage
evidence_used: The 68-unit full-sequence model achieved 86.87% accuracy, and every tested width reduction from 104 through 68 units remained viable; preserving all frames while testing the next four-unit reduction is the best-supported lower-cost change.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2198361936087544464, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 17584, "peak_hidden_elements": 66048, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 421452800, "training_seconds": 105.6367852089461, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4002592121896568}

RECENT RESULT
hypothesis: A 60-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9% versus the verified 64-unit model.
change: Reduce the GRU hidden width from 64 to 60 and resize the classifier and recurrent state tensors accordingly.
mechanism: Fine-grained recurrent-width scaling with full temporal coverage
evidence_used: The 64-unit model achieved 86.63% accuracy, and every tested width reduction from 104 through 64 units remained above the target while full temporal coverage proved more reliable than frame subsampling.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1963017496426680248, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 15768, "peak_hidden_elements": 61952, "recurrent_macs": 375552000, "recurrent_steps": 26080, "total_inference_macs": 376334400, "training_seconds": 183.3904404588975, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.39271601577477955}

RECENT RESULT
hypothesis: A 56-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9.7% versus the verified 60-unit model.
change: Reduce the GRU hidden width from 60 to 56 and resize the classifier and recurrent state tensors accordingly.
mechanism: Fine-grained recurrent-width scaling with full temporal coverage
evidence_used: The 60-unit full-sequence model achieved 86.50% accuracy, and every tested width reduction from 104 through 60 units remained above the target; another four-unit reduction preserves the proven temporal coverage and dual mean-plus-final readout.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 58-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.9% versus the verified 60-unit model.
change: Reduce the GRU hidden width from 60 to 58 and resize the classifier and recurrent state tensors accordingly.
mechanism: Two-unit recurrent-width refinement with full temporal coverage
evidence_used: The 60-unit full-sequence model achieved 86.50% accuracy, while the attempted 56-unit model produced no accuracy evidence because training timed out; testing the midpoint isolates a smaller, lower-risk structural reduction while retaining the successful full-frame dual readout.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1850242617537167856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 353957760, "recurrent_steps": 26080, "total_inference_macs": 354714080, "training_seconds": 207.61574637517333, "validation_accuracy": 0.8846625766871166, "validation_cross_entropy": 0.37396759782100747}

RECENT RESULT
hypothesis: A 58-unit GRU trained with batch size 96 will retain at least 85% validation accuracy, finish within the verification time limit, and reduce total inference MACs by approximately 5.7% versus the verified 60-unit model.
change: Reduce the GRU and temporal-summary width from 60 to 58, resize the classifier accordingly, and increase training batch size from 64 to 96 to reduce optimizer-step overhead.
mechanism: Throughput-assisted recurrent-width reduction
evidence_used: The 58-unit, batch-64 run reached 88.47% accuracy and 354,714,080 inference MACs, but its 207.6-second training time exceeded the limit; prior larger-batch runs completed substantially faster.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1850242617537167856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 522, "p95_recurrent_steps": 32, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 353957760, "recurrent_steps": 26080, "total_inference_macs": 354714080, "training_seconds": 108.16636183299124, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4278718913259682}

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 31 frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1%.
change: Omit only the earliest input frame while preserving the remaining contiguous causal sequence, model width, optimizer, and training procedure.
mechanism: Single-frame causal prefix trimming
evidence_used: The current 58-unit, batch-96 model achieved 85.77% accuracy over all 32 frames; prior 26–27-frame reductions were unreliable, motivating the smallest possible temporal reduction at the likely low-information recording boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1792545819495457061, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 522, "p95_recurrent_steps": 31, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 342896580, "recurrent_steps": 25265, "total_inference_macs": 343652900, "training_seconds": 78.78889991599135, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.427536357107338}



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
