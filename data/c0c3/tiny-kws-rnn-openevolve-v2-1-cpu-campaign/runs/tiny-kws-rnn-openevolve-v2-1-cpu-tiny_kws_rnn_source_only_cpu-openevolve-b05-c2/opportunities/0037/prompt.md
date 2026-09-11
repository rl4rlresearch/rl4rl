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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1791338488508771638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 341752320, "recurrent_steps": 21190, "total_inference_macs": 343421440, "training_seconds": 46.085821042070165, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.41763563097620304}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 26 causal frames while reducing total inference MACs from 356,565,760 to approximately 343,421,440.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1859901261763147973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 354896640, "recurrent_steps": 22005, "total_inference_macs": 356565760, "training_seconds": 28.181677292101085, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4157459376048457}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 27 causal frames while reducing total inference MACs from 369,710,080 to approximately 356,565,760.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1997026808271900643, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 381185280, "recurrent_steps": 23635, "total_inference_macs": 382854400, "training_seconds": 32.3983185000252, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.41096569599549465}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 29 causal frames while reducing total inference MACs from 395,998,720 to approximately 382,854,400.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1928464035017524308, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 368040960, "recurrent_steps": 22820, "total_inference_macs": 369710080, "training_seconds": 37.94612995814532, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4123815056736484}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 28 causal frames while reducing total inference MACs from 382,854,400 to approximately 369,710,080.

## Recent verification evidence

RECENT RESULT
hypothesis: A 67-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the qualified 68-unit design.
change: Reduce the GRU and recurrent-summary width from 68 to 67 units and resize the triple-readout classifier from 204 to 201 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 68-unit design achieved 86.38% validation accuracy at 469,518,240 total MACs, and every tested triple-readout width from 68 through 80 qualified; its 1.38-point margin motivates probing the adjacent structural compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2385719236389813905, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19545, "peak_hidden_elements": 103424, "recurrent_macs": 456060960, "recurrent_steps": 26080, "total_inference_macs": 457371480, "training_seconds": 56.75140625005588, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.42030147248250577}

RECENT RESULT
hypothesis: A 66-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.62% versus the qualified 67-unit design.
change: Reduce the GRU and recurrent-summary width from 69 to 66 units and resize the triple-readout classifier from 207 to 198 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 67-unit design achieved 86.63% validation accuracy at 457,371,480 total MACs, while every tested triple-readout width from 67 through 80 qualified; its 1.63-point margin motivates testing the adjacent structural compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2323176111456818336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19056, "peak_hidden_elements": 101888, "recurrent_macs": 444090240, "recurrent_steps": 26080, "total_inference_macs": 445381200, "training_seconds": 63.44164304086007, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.433227473533958}

RECENT RESULT
hypothesis: A 65-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.66% versus the qualified 66-unit design.
change: Reduce the GRU and recurrent-summary width from 70 to 65 units and resize the triple-readout classifier from 210 to 195 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 66-unit design qualified at 85.28% accuracy and 445,381,200 total MACs, while every tested triple-readout width from 66 through 80 qualified; testing the adjacent 65-unit width directly identifies the next structural compute boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2261449210013976053, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 432276000, "recurrent_steps": 26080, "total_inference_macs": 433547400, "training_seconds": 62.24201879207976, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4447986064513037}

RECENT RESULT
hypothesis: A 74-unit GRU evaluated on 24 uniformly distributed causal frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 19% versus the current 71-unit, 32-step design.
change: Increase the GRU and triple-readout width from 71 to 74 units, resize the classifier from 213 to 222 inputs, and skip one frame in every four while retaining both temporal endpoints.
mechanism: Three-of-four causal frame scheduling with compensatory hidden width
evidence_used: The full-resolution 74-unit design achieved 87.85% accuracy, the strongest observed result and 2.85 points above threshold; its margin motivates exchanging modest width for a 25% recurrent-step reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2136669043304791416, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 23136, "peak_hidden_elements": 114176, "recurrent_macs": 408178080, "recurrent_steps": 19560, "total_inference_macs": 409625520, "training_seconds": 47.40071808407083, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4754275269303585}

RECENT RESULT
hypothesis: A 65-unit GRU augmented with temporal-minimum pooling will reach at least 85% validation accuracy while using approximately 433,971,200 total inference MACs, 2.56% fewer than the qualified 66-unit design.
change: Reduce the recurrent width to 65 and expand the classifier from mean/final/maximum to mean/final/maximum/minimum recurrent summaries.
mechanism: Signed temporal-extrema readout at the failed width boundary
evidence_used: The 65-unit triple-readout model missed qualification by only 0.092 percentage points at 433,547,400 MACs; adding the complementary negative-extrema summary costs only 423,800 classifier MACs while preserving the recurrent compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2263659815299808373, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19093, "peak_hidden_elements": 133632, "recurrent_macs": 432276000, "recurrent_steps": 26080, "total_inference_macs": 433971200, "training_seconds": 46.8842969161924, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4041227001354007}

RECENT RESULT
hypothesis: A 64-unit full-resolution GRU using mean, final, maximum, and minimum summaries will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.69% versus the qualified 65-unit design.
change: Reduce the recurrent width from 67 to 64, add temporal-minimum tracking, and resize the classifier to consume four 64-unit summaries while preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with signed temporal-extrema readout
evidence_used: The 65-unit signed-extrema design achieved 85.77% accuracy at 433,971,200 MACs, whereas the 65-unit triple-readout design failed; this isolates minimum pooling as useful at the width boundary and motivates testing the adjacent 64-unit signed-extrema model.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2202715128035029648, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 422287360, "training_seconds": 46.185714167077094, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.40811364109530773}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy when one central frame is omitted, while reducing total inference MACs from 422,287,360 to approximately 409,143,040 and recurrent steps from 32 to 31 per example.
change: Replace the 70-unit triple-readout GRU with the qualified 64-unit mean/final/maximum/minimum design and omit one interior frame while retaining both causal endpoints.
mechanism: Single-frame temporal pruning with signed-extrema readout
evidence_used: The 64-unit signed-extrema model achieved 86.63% accuracy at 422,287,360 MACs, providing a 1.63-point margin; the 24-step experiment indicates temporal pruning is computationally valuable but too aggressive, motivating the smallest possible one-frame schedule reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2134152354780653313, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 407473920, "recurrent_steps": 25265, "total_inference_macs": 409143040, "training_seconds": 39.63655120902695, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.40820536233164784}

RECENT RESULT
hypothesis: A 64-unit signed-extrema GRU using 30 causal frames will retain at least 85% validation accuracy while reducing total inference MACs from 409,143,040 to approximately 395,998,720.
change: Replace the 71-unit triple-readout model with the qualified 64-unit mean/final/maximum/minimum model and omit frames at one-quarter and one-half of the sequence, preserving both endpoints.
mechanism: Quarter-and-center temporal pruning with signed-extrema readout
evidence_used: The 64-unit signed-extrema model achieved 86.87% accuracy after one central frame was omitted, exceeding the threshold by 1.87 points; adding one separated interior omission is the smallest next temporal-compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2065589581526276978, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 394329600, "recurrent_steps": 24450, "total_inference_macs": 395998720, "training_seconds": 36.32589954207651, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4145544227646904}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 29 causal frames while reducing total inference MACs from 395,998,720 to approximately 382,854,400.
change: Add a third, evenly separated interior omission at three-quarters of the sequence, preserving both endpoints and reducing execution from 30 to 29 recurrent steps for 32-frame inputs.
mechanism: Three-point temporal pruning with signed-extrema readout
evidence_used: The current 30-step model achieved 86.63% accuracy, 1.63 points above threshold, after the preceding 31-step model achieved 86.87%; this supports testing the smallest next temporal reduction with omissions distributed across the sequence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1997026808271900643, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 381185280, "recurrent_steps": 23635, "total_inference_macs": 382854400, "training_seconds": 32.3983185000252, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.41096569599549465}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 28 causal frames while reducing total inference MACs from 382,854,400 to approximately 369,710,080.
change: Preserve the proven quarter, center, and three-quarter omissions and add one omission at one-eighth of the sequence, reducing 32-frame execution from 29 to 28 recurrent steps.
mechanism: Incremental early-frame temporal pruning with signed-extrema readout
evidence_used: The 29-step design achieved 86.63% accuracy, matching the 30-step result and retaining a 1.63-point margin; this supports the smallest next temporal reduction while preserving the three already-qualified omissions.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1928464035017524308, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 368040960, "recurrent_steps": 22820, "total_inference_macs": 369710080, "training_seconds": 37.94612995814532, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4123815056736484}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 27 causal frames while reducing total inference MACs from 369,710,080 to approximately 356,565,760.
change: Preserve the four proven omissions and additionally omit the frame at seven-eighths of the sequence, reducing 32-frame execution from 28 to 27 recurrent steps while retaining both endpoints.
mechanism: Symmetric late-frame temporal pruning
evidence_used: The 28-step design achieved 86.87% accuracy, 1.87 points above threshold, after every incremental reduction from 32 through 28 steps qualified; adding the late counterpart to the proven one-eighth omission is the smallest next compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1859901261763147973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 354896640, "recurrent_steps": 22005, "total_inference_macs": 356565760, "training_seconds": 28.181677292101085, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4157459376048457}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 26 causal frames while reducing total inference MACs from 356,565,760 to approximately 343,421,440.
change: Replace the 65-unit full-resolution model with the qualified 64-unit signed-extrema architecture and omit six distributed interior frames, adding a three-eighths omission to the proven 27-step schedule.
mechanism: Incremental mid-sequence temporal pruning with signed-extrema readout
evidence_used: The 27-step 64-unit design achieved 86.38% accuracy, 1.38 points above threshold, after every incremental reduction from 32 through 27 steps qualified; removing one additional interior frame is the smallest next structural compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1791338488508771638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 341752320, "recurrent_steps": 21190, "total_inference_macs": 343421440, "training_seconds": 46.085821042070165, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.41763563097620304}



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
