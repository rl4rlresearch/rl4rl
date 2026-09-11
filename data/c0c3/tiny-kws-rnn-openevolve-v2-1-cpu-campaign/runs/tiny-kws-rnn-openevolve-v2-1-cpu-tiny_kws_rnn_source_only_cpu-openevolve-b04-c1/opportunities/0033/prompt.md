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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1506034369388107310, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 287590680, "recurrent_steps": 21190, "total_inference_macs": 288725160, "training_seconds": 34.169467833125964, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4451760321307036}
prior_hypothesis: Processing the 26 contiguous central frames (indices 3–28) will retain at least 85% validation accuracy while reducing recurrent steps by 3.7% and total dense inference MACs to approximately 288.7M.

## Recent verification evidence

RECENT RESULT
hypothesis: A 60-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.7% versus the verified 64-unit model.
change: Reduce the GRU hidden width from 64 to 60 and resize all recurrent state tensors and the three-way classifier input accordingly, preserving all 32 causal frames and the established training procedure.
mechanism: Continued recurrent-width reduction with salience-preserving temporal pooling
evidence_used: The 64-unit model achieved 86.01% accuracy, and every tested max-augmented width from 64 through 84 passed; its 1.01-point margin supports continuing the established four-unit structural width search.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1965058055152063928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 375552000, "recurrent_steps": 26080, "total_inference_macs": 376725600, "training_seconds": 85.66842616698705, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4337301523407544}

RECENT RESULT
hypothesis: A 58-unit GRU using mean, maximum, and final recurrent outputs will achieve at least 85% validation accuracy while reducing exact dense inference MACs by approximately 5.7% versus the verified 60-unit model.
change: Reduce the GRU hidden width from 60 to 58 and resize the classifier and recurrent state tensors, preserving all 32 causal frames and the established training procedure.
mechanism: Boundary-search recurrent width reduction
evidence_used: The 60-unit model achieved 85.64% accuracy at 376,725,600 MACs, while every tested max-augmented width from 60 through 84 passed; a two-unit reduction cautiously probes the remaining accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1852215157638372080, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 353957760, "recurrent_steps": 26080, "total_inference_macs": 355092240, "training_seconds": 56.11772225005552, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.44208198032496165}

RECENT RESULT
hypothesis: A 56-unit GRU with mean, maximum, and final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 5.9% versus the verified 58-unit model.
change: Reduce the GRU hidden width from 58 to 56 and resize the classifier and recurrent state tensors, preserving all 32 causal frames and the established training procedure.
mechanism: Boundary-search recurrent width reduction
evidence_used: The 58-unit model achieved 85.40% accuracy at 355,092,240 MACs after the 60-unit model achieved 85.64%; both passed, and their modest accuracy change supports a final two-unit probe of the width boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1742637154085293376, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 14496, "peak_hidden_elements": 86528, "recurrent_macs": 332989440, "recurrent_steps": 26080, "total_inference_macs": 334084800, "training_seconds": 58.89720487501472, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.453266073589676}

RECENT RESULT
hypothesis: A 57-unit GRU with mean, maximum, and final readout will achieve at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3% versus the verified 58-unit model.
change: Reduce the GRU hidden width from 58 to 57 and resize its classifier and recurrent state tensors, preserving the successful 32-frame training and readout design.
mechanism: One-unit recurrent-width boundary search
evidence_used: The 58-unit model passed at 85.40% accuracy while the 56-unit model narrowly failed at 84.54%; testing the only remaining integer width directly identifies the structural capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1797018044116756085, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 14925, "peak_hidden_elements": 88064, "recurrent_macs": 343395360, "recurrent_steps": 26080, "total_inference_macs": 344510280, "training_seconds": 59.055328166810796, "validation_accuracy": 0.8245398773006135, "validation_cross_entropy": 0.4741931189788631}

RECENT RESULT
hypothesis: The verified 58-unit GRU will retain at least 85% validation accuracy when processing 31 evenly distributed frames, while reducing total dense inference MACs below the failed 57-unit, 32-step model and reducing executed recurrent steps by 3.125%.
change: Change the causal frame schedule from 32 to 31 evenly spaced frames while preserving the verified recurrent width, pooling, and training procedure.
mechanism: Single-frame causal temporal subsampling
evidence_used: The 58-unit, 32-step model passed at 85.40% accuracy, whereas reducing width to 57 failed; removing one of 32 highly adjacent frames tests temporal redundancy while retaining the full verified recurrent capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1794518359596661285, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 342896580, "recurrent_steps": 25265, "total_inference_macs": 344031060, "training_seconds": 57.71596016618423, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.45309999618062213}

RECENT RESULT
hypothesis: The 58-unit GRU will exceed 85% accuracy using the final 31 contiguous frames, because this preserves the central frame omitted by the failed evenly spaced 31-frame schedule while retaining its lower MAC count and step count.
change: Process frames 1–31 instead of all 32 frames, dropping only the earliest boundary frame while preserving the verified model and training procedure.
mechanism: Boundary-frame temporal subsampling
evidence_used: The 58-unit, 32-step model achieved 85.40%, while the evenly spaced 31-step variant narrowly missed at 84.79%; with 32 available frames that schedule omits an interior frame, motivating a same-cost 31-step schedule that discards a likely less-informative boundary frame instead.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1794518359596661285, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 342896580, "recurrent_steps": 25265, "total_inference_macs": 344031060, "training_seconds": 55.61596116703004, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.442519102506111}

RECENT RESULT
hypothesis: Processing the central 30 contiguous frames will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs versus the verified 31-step model.
change: Change the schedule from frames 1–31 to frames 1–30, removing the remaining outer boundary frame while preserving the 58-unit GRU and training procedure.
mechanism: Symmetric boundary-frame pruning
evidence_used: The 58-unit model passed at 85.15% when dropping the earliest boundary frame, whereas omitting an interior frame fell to 84.79%; this indicates boundary frames are less valuable and motivates pruning the opposite boundary next.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1736821561554950490, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 331835400, "recurrent_steps": 24450, "total_inference_macs": 332969880, "training_seconds": 48.65552983316593, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.437953803875695}

RECENT RESULT
hypothesis: Processing 29 contiguous near-central frames (indices 2–30) will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs by roughly 3.3% versus the verified 30-step model.
change: Reduce the schedule from 30 to 29 steps and bias the odd-length window toward later frames, removing the earliest remaining boundary frame.
mechanism: Boundary-frame temporal pruning
evidence_used: The 58-unit GRU improved from 85.15% at 31 steps to 85.77% at 30 central steps, while dropping the original earliest boundary frame already passed; this supports testing one further boundary removal without reducing recurrent capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1679124763513239695, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 320774220, "recurrent_steps": 23635, "total_inference_macs": 321908700, "training_seconds": 50.2952406660188, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4395944794262845}

RECENT RESULT
hypothesis: Processing the 28 contiguous central frames (indices 2–29) will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs by approximately 3.4% versus the verified 29-step model.
change: Reduce the causal schedule from 29 to 28 steps, removing the latest remaining boundary frame while preserving the 58-unit GRU, temporal pooling, and training procedure.
mechanism: Symmetric boundary-frame temporal pruning
evidence_used: The central 30-step model achieved 85.77% accuracy and the 29-step model achieved 85.64%; both passed, indicating that boundary-frame pruning has preserved accuracy so far and motivating the complementary boundary removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1621427965471528900, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 309713040, "recurrent_steps": 22820, "total_inference_macs": 310847520, "training_seconds": 42.28234770800918, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.42989435488460986}

RECENT RESULT
hypothesis: A 42-unit fast GRU processing all 28 scheduled frames plus a 52-unit context GRU updated every fourth fast output will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 15.4% versus the verified 58-unit single-rate GRU.
change: Replace the homogeneous recurrent state with full-rank fast and slow GRUs, maintain separate mean/maximum/final summaries for both timescales, and update the command-scale state only seven times per recording.
mechanism: Hierarchical multi-rate fast/slow GRU
evidence_used: The 28-step 58-unit GRU reached 86.99% accuracy, leaving a 1.99-point margin. The failed MGU showed that weakening the gated update itself was harmful; this alternative retains standard GRUs while challenging the assumption that every unit must perform an equally expensive update at every frame.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Processing 27 contiguous near-central frames (indices 3–29) will retain at least 85% validation accuracy while reducing recurrent steps by 3.6% and total dense inference MACs to approximately 299.8M.
change: Reduce the causal schedule from 28 to 27 steps by removing the earliest remaining boundary frame, preserving the verified 58-unit GRU, pooling, and training procedure.
mechanism: Continued boundary-frame temporal pruning
evidence_used: The 28-step model achieved 86.99% accuracy, a 1.99-point margin above threshold, after every contiguous boundary-pruning result from 31 through 28 steps passed; this supports probing one additional boundary frame without weakening recurrent capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1563731167429818105, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 298651860, "recurrent_steps": 22005, "total_inference_macs": 299786340, "training_seconds": 40.470391666051, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4336063033964005}

RECENT RESULT
hypothesis: Processing the 26 contiguous central frames (indices 3–28) will retain at least 85% validation accuracy while reducing recurrent steps by 3.7% and total dense inference MACs to approximately 288.7M.
change: Reduce the causal schedule from 27 to 26 steps, removing the latest remaining boundary frame while preserving the verified 58-unit GRU, temporal pooling, and training procedure.
mechanism: Continued symmetric boundary-frame pruning
evidence_used: The contiguous 28-step and 27-step models achieved 86.99% and 86.38% accuracy respectively, and every boundary-pruning design from 31 through 27 steps passed; the current 1.38-point margin supports testing the complementary boundary removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1506034369388107310, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 287590680, "recurrent_steps": 21190, "total_inference_macs": 288725160, "training_seconds": 34.169467833125964, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4451760321307036}



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
