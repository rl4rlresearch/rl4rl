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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2023962183202458518, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 20688, "peak_hidden_elements": 132608, "recurrent_macs": 386336080, "recurrent_steps": 21190, "total_inference_macs": 388018240, "training_seconds": 96.42121758288704, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.40881503988628737}
prior_hypothesis: An 86-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.1%, from 404.50M to 388.02M.

## Recent verification evidence

RECENT RESULT
hypothesis: An 86-unit GRU retaining the early frame lost by the failed 26-step design and instead pruning the penultimate frame will achieve at least 85% accuracy at 580.63M total inference MACs.
change: Increase the GRU/readout width from 85 to 86 and create a 26-frame schedule by removing the already-proven redundant second frame plus the penultimate frame from the successful 28-frame base schedule.
mechanism: Onset-preserving boundary frame pruning
evidence_used: The 85-unit/27-step schedule passed at 86.63%, while the 86-unit/26-step design that removed both earliest nonzero frames missed the threshold by only 0.09 points; preserving potential onset information while pruning a likely trailing-silence frame directly tests whether pruning location caused that failure.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3028631271644885638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 579504120, "recurrent_steps": 21190, "total_inference_macs": 580625560, "training_seconds": 77.47596350009553, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4552927496974454}

RECENT RESULT
hypothesis: Adding temporal max pooling to the closest-failing 86-unit, 26-step GRU will raise validation accuracy from 84.91% to at least 85% while keeping total inference MACs near 581.19M, below the current 590.29M design.
change: Use the previously tested early-pruned 26-frame schedule and concatenate mean, maximum, and terminal GRU outputs for classification.
mechanism: Transient-preserving max-augmented recurrent readout
evidence_used: The early-pruned 86-unit/26-step model missed the threshold by only 0.09 points. A max summary preserves transient command evidence that mean and terminal summaries can dilute after removing frames, at a cost of only 560,720 additional validation MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3031556072484602246, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29976, "peak_hidden_elements": 132608, "recurrent_macs": 579504120, "recurrent_steps": 21190, "total_inference_macs": 581186280, "training_seconds": 66.93220995785668, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4158541907562069}

RECENT RESULT
hypothesis: An 85-unit GRU with the successful 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.1%, from 581.19M to 569.02M.
change: Reduce the GRU and summary width from 86 to 85 and resize the concatenated classifier input from 258 to 255, preserving the verified frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 86-unit, 26-step model reached 86.01% after max pooling rescued the same schedule’s 84.91% mean-terminal result, while an 85-unit model previously reached 86.63% at 27 steps; this supports testing the smallest width reduction without removing the accuracy-restoring readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2968120203109251413, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29373, "peak_hidden_elements": 131072, "recurrent_macs": 567362250, "recurrent_steps": 21190, "total_inference_macs": 569024850, "training_seconds": 34.57787450007163, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41563148732565663}

RECENT RESULT
hypothesis: An 84-unit GRU with the successful 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.1%, from 569.02M to 556.99M.
change: Reduce the GRU and all summary-state widths from 85 to 84 and resize the concatenated classifier input from 255 to 252, preserving the verified schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 85-unit/26-step max-readout model achieved 86.26%, while an 84-unit model without max pooling passed at 85.64% over 28 steps; this supports testing one-unit compression while retaining the stronger readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2905347515319650126, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 555347520, "recurrent_steps": 21190, "total_inference_macs": 556990560, "training_seconds": 38.97008291701786, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4119058901546923}

RECENT RESULT
hypothesis: An 83-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.2% versus the successful 84-unit model.
change: Reduce the GRU and all recurrent summary widths from 84 to 83, and resize the classifier input from 252 to 249 while preserving the frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 84-unit/26-step model achieved 86.63% accuracy, providing substantial margin, and the prior 83-unit/29-step model achieved 86.01%; together these results support a one-unit width reduction while retaining the accuracy-restoring max readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2843238009115798385, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 28185, "peak_hidden_elements": 128000, "recurrent_macs": 543459930, "recurrent_steps": 21190, "total_inference_macs": 545083410, "training_seconds": 42.974731041816995, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4236403763659893}

RECENT RESULT
hypothesis: An 82-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.2% versus the successful 83-unit model.
change: Reduce the GRU and recurrent summary width from 83 to 82 and resize the classifier input from 249 to 246, preserving the frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 83-unit/26-step model achieved 86.87% accuracy, giving 1.87 percentage points of margin, while an 82-unit model previously achieved 85.52% at 30 steps; this supports testing the smallest structural width reduction with the stronger max-augmented readout intact.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2781791684497696190, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 27600, "peak_hidden_elements": 126464, "recurrent_macs": 531699480, "recurrent_steps": 21190, "total_inference_macs": 533303400, "training_seconds": 36.711444332962856, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4176021178075872}

RECENT RESULT
hypothesis: An 81-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 2.2%, from 533.30M to 521.65M.
change: Reduce the GRU and recurrent summary width from 82 to 81 and resize the concatenated classifier input from 246 to 243, preserving the frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 82-unit/26-step model achieved 86.50% accuracy, leaving 1.50 percentage points of margin, and every prior one-unit max-readout compression from 85 through 82 units passed; an earlier 81-unit model also passed at 31 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2721008541465343541, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 27021, "peak_hidden_elements": 124928, "recurrent_macs": 520066170, "recurrent_steps": 21190, "total_inference_macs": 521650530, "training_seconds": 31.502919625025243, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4182519362748035}

RECENT RESULT
hypothesis: A 96-unit full-rank single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame max-augmented schedule while reducing total dense inference MACs by about 9.2%, from 521.65M to 473.82M.
change: Replace the 81-unit three-gate GRU with a wider 96-unit recurrent update using one learned retention gate and one full-rank normalized proposal, while preserving mean-max-terminal classification and the successful frame schedule.
mechanism: Full-rank single-retention-gate recurrence
evidence_used: The current 81-unit GRU reaches 86.99%, providing accuracy margin, while the 128-unit rank-32 recurrent bottleneck failed at 82.70%. This motivates challenging the assumption that three gates are necessary without repeating the failed low-rank assumption: both retained state and proposal remain full-rank, and the wider state compensates for removing the reset gate.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2471524730303178038, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25008, "peak_hidden_elements": 147968, "recurrent_macs": 471943680, "recurrent_steps": 21190, "total_inference_macs": 473821440, "training_seconds": 116.72218300006352, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.3903332482086369}

RECENT RESULT
hypothesis: A 92-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 7.5%, from 473.82M to 438.48M.
change: Reduce the recurrent, temporal-summary, and classifier width from 96 to 92 units while preserving the successful recurrence, readout, schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 96-unit single-gate model achieved 87.73% accuracy, exceeding the requirement by 2.73 percentage points; this substantial margin motivates a moderate four-unit structural compression before altering the proven temporal schedule or gating mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2287194258776893862, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 23232, "peak_hidden_elements": 141824, "recurrent_macs": 436683520, "recurrent_steps": 21190, "total_inference_macs": 438483040, "training_seconds": 112.9495802919846, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40820705671251917}

RECENT RESULT
hypothesis: A 90-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 3.9%, from 438.48M to 421.32M.
change: Reduce the recurrent, temporal-summary, and classifier width from 92 to 90 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 92-unit single-gate model achieved 86.13% accuracy at 438.48M MACs, while the 96-unit version achieved 87.73%; this supports a measured two-unit compression that tests the lower-width boundary without changing the proven temporal or gating design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2197681749356749958, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 22368, "peak_hidden_elements": 138752, "recurrent_macs": 419562000, "recurrent_steps": 21190, "total_inference_macs": 421322400, "training_seconds": 92.76778687513433, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41911935630751534}

RECENT RESULT
hypothesis: An 88-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.0%, from 421.32M to 404.50M.
change: Reduce the recurrent, temporal-summary, and classifier width from 90 to 88 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 90-unit model achieved 86.26% accuracy, and the 92-unit model achieved 86.13%; both exceed the threshold by more than one percentage point, supporting another measured two-unit compression without changing the proven temporal or gating design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2109937724165271510, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 21520, "peak_hidden_elements": 135680, "recurrent_macs": 402779520, "recurrent_steps": 21190, "total_inference_macs": 404500800, "training_seconds": 102.02311599999666, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40685732788834833}

RECENT RESULT
hypothesis: An 86-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.1%, from 404.50M to 388.02M.
change: Reduce the recurrent, temporal-summary, and classifier width from 88 to 86 units while preserving the recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 88-unit model achieved 86.13% accuracy, while the 90- and 92-unit models also exceeded 86%; the stable results across successive two-unit reductions motivate continuing the same controlled compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2023962183202458518, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 20688, "peak_hidden_elements": 132608, "recurrent_macs": 386336080, "recurrent_steps": 21190, "total_inference_macs": 388018240, "training_seconds": 96.42121758288704, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.40881503988628737}



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
