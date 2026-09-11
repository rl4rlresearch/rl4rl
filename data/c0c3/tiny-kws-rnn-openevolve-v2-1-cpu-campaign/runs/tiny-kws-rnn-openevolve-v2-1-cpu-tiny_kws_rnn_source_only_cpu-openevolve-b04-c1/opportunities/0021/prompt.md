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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2200538532061287056, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18096, "peak_hidden_elements": 98816, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 421870080, "training_seconds": 84.83112479210831, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.41598951655662864}
prior_hypothesis: A 64-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.7% versus the verified 68-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 92-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 7.5% versus the verified 96-unit model.
change: Reduce the GRU hidden width from 96 to 92 and resize the classifier and recurrent state tensors accordingly, while preserving all 32 frames and the successful dual-timescale readout.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 96-unit dual-readout model achieved 85.89% accuracy with 872,532,480 total MACs, improving on the 100-unit model’s accuracy despite its smaller width; this provides 0.89 percentage points of headroom for another four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4211441137302266104, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807384640, "training_seconds": 101.63187612500042, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.3870472633034174}

RECENT RESULT
hypothesis: A 128-dimensional residual state with an input-conditioned update gate and rank-64 learned recurrent transform will retain at least 85% accuracy while reducing dense MACs by roughly 30% versus the verified 92-unit GRU.
change: Replace the three-gate full-rank GRU with a residual recurrent update using one input projection and a 128→64→128 recurrent bottleneck, while preserving all 32 frames and the successful mean-plus-final readout.
mechanism: Input-gated low-rank residual recurrence
evidence_used: The 92-unit GRU achieved 87.12% accuracy, leaving 2.12 points of headroom, but spends nearly all 807.4M MACs on three full-rank gated transforms. Prior frame decimation harmed accuracy, so this patch preserves temporal resolution and instead challenges the assumption that every gate requires a full-rank recurrent matrix.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2934051375212373776, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23856, "peak_hidden_elements": 131584, "recurrent_macs": 560824320, "recurrent_steps": 26080, "total_inference_macs": 562493440, "training_seconds": 134.95084954216145, "validation_accuracy": 0.8233128834355828, "validation_cross_entropy": 0.49733295089628066}

RECENT RESULT
hypothesis: An 88-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 7.8% versus the verified 92-unit model.
change: Reduce the GRU hidden width from 92 to 88 and resize the classifier and recurrent state tensors accordingly, preserving all 32 frames and the successful dual-timescale readout.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 92-unit model achieved 87.12% accuracy with 807,384,640 MACs—its best observed accuracy among the width-reduction sequence and 2.12 percentage points above threshold—supporting another modest four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 120.34821412502788, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42309365301775786}

RECENT RESULT
hypothesis: An 84-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 8.1% versus the verified 88-unit model.
change: Reduce the GRU hidden width from 88 to 84 and resize the classifier and recurrent state tensors accordingly, while preserving the full 32-frame schedule and training procedure.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 88-unit dual-readout model achieved 85.77% accuracy with 744,740,480 MACs, and every tested dual-readout width from 92 through 100 exceeded the accuracy requirement; the next four-unit reduction directly continues the established width search.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3570977772028654184, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 684600000, "training_seconds": 27.641658791108057, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.42847105623022913}

RECENT RESULT
hypothesis: An 86-unit GRU with mean-plus-final readout will achieve at least 85% validation accuracy while reducing dense inference MACs by approximately 4% versus the verified 88-unit model.
change: Reduce the GRU hidden width from 88 to 86 and resize the classifier and recurrent state tensors, preserving all 32 frames and the established training procedure.
mechanism: Boundary-search recurrent width reduction
evidence_used: The 88-unit model achieved 85.77% accuracy, while 84 units narrowly missed at 84.79%; testing the midpoint directly resolves the remaining width boundary with a meaningful structural cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3726196272406137448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 713235840, "recurrent_steps": 26080, "total_inference_macs": 714357280, "training_seconds": 159.8941866250243, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4322061175949003}

RECENT RESULT
hypothesis: An 84-unit GRU classified from mean, maximum, and final recurrent outputs will recover the 0.22-point accuracy shortfall of the prior 84-unit model and reach at least 85% while retaining substantially fewer MACs than the verified 88-unit model.
change: Reduce the GRU width to 84 and add a running elementwise maximum of recurrent outputs to the state and classifier features, preserving all 32 causal steps and the existing training procedure.
mechanism: Salience-preserving max-augmented recurrent readout
evidence_used: The 84-unit mean-plus-final model narrowly missed at 84.79%, while the 88-unit version reached 85.77%; adding a parameter-light maximum readout exposes salient transient speech evidence that temporal averaging may suppress, while the 84-unit recurrent core preserves the prior model’s structural MAC reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3573834554244191336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 685147680, "training_seconds": 144.75289745815098, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.39589817538583205}

RECENT RESULT
hypothesis: An 80-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the verified 84-unit model.
change: Reduce the GRU hidden width from 84 to 80 and resize all recurrent state tensors and the three-way classifier input accordingly, while preserving all 32 causal frames and the successful max-augmented readout.
mechanism: Width reduction with salience-preserving temporal readout
evidence_used: The 84-unit mean-plus-final model missed the threshold at 84.79%, but adding the maximum recurrent output raised accuracy to 86.75% with the same recurrent core; that 1.75-point margin motivates testing a four-unit structural reduction without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 192.7032862920314, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.39644080788079944}

RECENT RESULT
hypothesis: A 76-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.8% versus the verified 80-unit model.
change: Reduce the GRU hidden width from 80 to 76, resize its recurrent state tensors, and reduce the three-way classifier input from 240 to 228 features while preserving all 32 causal frames and the training procedure.
mechanism: Continued recurrent-width reduction with salience-preserving temporal readout
evidence_used: Adding maximum-output pooling allowed the 84-unit model to reach 86.75%, and the subsequent 80-unit reduction still achieved 86.50%; the retained 1.50-point margin supports continuing the established four-unit structural width search.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2985337417843671896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 570839040, "recurrent_steps": 26080, "total_inference_macs": 572325600, "training_seconds": 98.62407408282161, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.38974172206012747}

RECENT RESULT
hypothesis: A 72-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 9.2% versus the verified 76-unit model.
change: Reduce the GRU hidden width from 76 to 72 and resize the classifier and recurrent state tensors, preserving all 32 causal frames and the established three-way readout.
mechanism: Continued width reduction with salience-preserving temporal pooling
evidence_used: The 84-, 80-, and 76-unit max-augmented models all exceeded 85% accuracy; the 76-unit result retained 1.38 percentage points of headroom while achieving the lowest verified inference cost, supporting the next four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 87.3660075000953, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3980412301841689}

RECENT RESULT
hypothesis: A 68-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 9.7% versus the verified 72-unit model.
change: Reduce the GRU hidden width from 72 to 68 and resize the classifier and recurrent state tensors accordingly, while preserving all 32 causal frames and the established training procedure.
mechanism: Continued recurrent-width reduction with salience-preserving temporal pooling
evidence_used: The 72-unit model achieved 87.24% accuracy—the strongest result in the 72–84-unit max-augmented sequence and 2.24 percentage points above threshold—supporting another four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2449078584812962760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20040, "peak_hidden_elements": 104960, "recurrent_macs": 468188160, "recurrent_steps": 26080, "total_inference_macs": 469518240, "training_seconds": 84.33418108313344, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.41111084171599405}

RECENT RESULT
hypothesis: A 64-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.7% versus the verified 68-unit model.
change: Reduce the GRU hidden width from 68 to 64 and resize the classifier and recurrent state tensors accordingly, preserving all 32 causal frames and the established training procedure.
mechanism: Continued recurrent-width reduction with salience-preserving temporal pooling
evidence_used: The 68-unit model achieved 87.24% accuracy—2.24 percentage points above threshold and equal to the 72-unit model—while every tested max-augmented width from 68 through 84 passed, motivating the next four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2200538532061287056, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18096, "peak_hidden_elements": 98816, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 421870080, "training_seconds": 84.83112479210831, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.41598951655662864}

RECENT RESULT
hypothesis: A 76-unit, two-transform MGU will preserve at least 85% validation accuracy while reducing total dense inference MACs by about 9.4% versus the verified 64-unit three-transform GRU.
change: Replace the GRU with a wider full-rank recurrent state using one state-conditioned update gate and one gated candidate transform, while retaining all 32 frames and mean/maximum/final pooling.
mechanism: Full-rank minimally gated recurrent update
evidence_used: The 64-unit GRU reached 86.01% accuracy at 421.87M MACs, while 72- and 76-unit GRUs reached 87.24% and 86.38%. This supports trading the GRU’s third dense transform for wider state capacity. Unlike the failed low-rank residual model, this update retains full-rank recurrence and makes its gate depend on prior state.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1992809653817275968, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 16768, "peak_hidden_elements": 117248, "recurrent_macs": 380559360, "recurrent_steps": 26080, "total_inference_macs": 382045920, "training_seconds": 200.53093170793727, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4609583474375719}



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
