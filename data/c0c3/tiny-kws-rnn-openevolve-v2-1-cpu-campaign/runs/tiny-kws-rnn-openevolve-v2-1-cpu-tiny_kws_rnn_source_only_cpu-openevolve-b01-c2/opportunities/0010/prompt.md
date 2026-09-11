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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 63.53902162518352, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.3980141294514475}
prior_hypothesis: An 80-unit GRU retaining mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 685,147,680 to approximately 627,484,800.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 60.07818637508899, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4197747072559193}
prior_hypothesis: A 72-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 572,325,600 to approximately 519,670,080.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3573834554244191336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 685147680, "training_seconds": 59.1400409580674, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.38266621221062597}
prior_hypothesis: An 84-unit GRU retaining mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total inference MACs from 745,314,240 to approximately 685,147,680.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing every second frame while retaining the full one-second temporal span will preserve at least 85% validation accuracy and approximately halve recurrent MACs and executed recurrent steps.
change: Change the frame schedule from all 32 frames to the 16 odd-indexed frames, including the final frame.
mechanism: Two-frame causal temporal striding
evidence_used: The current 128-unit GRU achieves 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving accuracy margin for a structural step reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 49.34897445794195, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.49316335104725845}

RECENT RESULT
hypothesis: Using 24 uniformly distributed frames across the full one-second span will retain at least 85% validation accuracy while reducing recurrent MACs and steps by 25% versus the 32-frame baseline.
change: Skip every fourth frame while preserving the final frame, yielding 24 recurrent steps.
mechanism: Moderate uniform causal frame thinning
evidence_used: The 32-step model reached 87.24% accuracy, while aggressive 16-step striding fell to 83.19%; 24 steps directly tests the intermediate compute–accuracy point with the same temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 86.32460291613825, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4494208400234854}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames, with mean and final-state features, will retain at least 85% accuracy while reducing total inference MACs from 1.483B to approximately 0.873B.
change: Reduce GRU width from 128 to 96, preserve the full frame schedule, and classify from concatenated mean-pooled and final recurrent states.
mechanism: Full-resolution narrow recurrence with dual temporal readout
evidence_used: Thinning to 24 frames reduced accuracy to 84.17% despite retaining the 128-unit GRU, suggesting temporal observations are more valuable than excess hidden width; width reduction preserves every frame while cutting the quadratic recurrent matrix cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 110.4936499169562, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4066338474765146}

RECENT RESULT
hypothesis: A 92-unit GRU using mean, peak, and final recurrent features will retain at least 85% accuracy while reducing total dense inference MACs from 872.5M to approximately 808.0M.
change: Preserve all 32 causal frames, reduce GRU width from 128 to 92, maintain an online elementwise maximum, and classify from concatenated mean, maximum, and final states.
mechanism: Peak-augmented narrow full-resolution GRU
evidence_used: The full-resolution 96-unit GRU reached 85.15%, whereas thinning even to 24 frames failed; this motivates preserving every observation and trading a small width reduction for a zero-matrix-cost temporal peak statistic.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4214569994014521080, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 33720, "peak_hidden_elements": 141824, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807984480, "training_seconds": 123.59691491583362, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.3681394612130943}

RECENT RESULT
hypothesis: An 88-unit GRU using mean, maximum, and final recurrent features will retain at least 85% validation accuracy while reducing total dense inference MACs from 807,984,480 to approximately 745,314,240.
change: Preserve all 32 causal frames, narrow the GRU from 128 to 88 units, maintain an online elementwise maximum, and classify from concatenated mean, maximum, and final states.
mechanism: Peak-augmented 88-unit full-resolution GRU
evidence_used: The 92-unit peak-augmented design achieved 87.61% accuracy at 807,984,480 MACs, providing a 2.61-point accuracy margin that motivates testing a modest four-unit structural reduction without discarding temporal observations.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3887672486208129920, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31200, "peak_hidden_elements": 135680, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 745314240, "training_seconds": 100.4527173328679, "validation_accuracy": 0.8846625766871166, "validation_cross_entropy": 0.37404556742475076}

RECENT RESULT
hypothesis: An 84-unit GRU retaining mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total inference MACs from 745,314,240 to approximately 685,147,680.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 88 to 84 hidden units.
mechanism: Peak-augmented 84-unit full-resolution GRU
evidence_used: The 88-unit design achieved 88.47% accuracy—3.47 points above the requirement—at 745,314,240 MACs, supporting another controlled four-unit structural reduction without discarding temporal observations.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3573834554244191336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 685147680, "training_seconds": 59.1400409580674, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.38266621221062597}

RECENT RESULT
hypothesis: An 80-unit GRU retaining mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 685,147,680 to approximately 627,484,800.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 92 to 80 hidden units.
mechanism: Peak-augmented 80-unit full-resolution GRU
evidence_used: The qualified 84-unit design achieved 87.73% accuracy at 685,147,680 MACs, leaving a 2.73-point margin and motivating another controlled four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 63.53902162518352, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.3980141294514475}

RECENT RESULT
hypothesis: A 76-unit GRU using mean, maximum, and final recurrent features will retain at least 85% validation accuracy while reducing total dense inference MACs from 627,484,800 to approximately 572,325,600.
change: Preserve all 32 causal frames, add the proven online maximum summary, and narrow the GRU from 96 to 76 hidden units.
mechanism: Peak-augmented 76-unit full-resolution GRU
evidence_used: The qualified 80-unit peak-augmented design achieved 86.63% accuracy at 627,484,800 MACs; its 1.63-point margin motivates the next controlled four-unit width reduction while preserving every temporal observation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2985337417843671896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 570839040, "recurrent_steps": 26080, "total_inference_macs": 572325600, "training_seconds": 64.62605424993671, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40984630935762556}

RECENT RESULT
hypothesis: A 72-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 572,325,600 to approximately 519,670,080.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 76 to 72 hidden units.
mechanism: Peak-augmented 72-unit full-resolution GRU
evidence_used: The qualified 76-unit design achieved 86.13% accuracy at 572,325,600 MACs, leaving a 1.13-point margin and motivating the next controlled four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 60.07818637508899, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4197747072559193}



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
