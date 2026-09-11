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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3969719951380080039, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 34264, "peak_hidden_elements": 96768, "recurrent_macs": 759817980, "recurrent_steps": 23635, "total_inference_macs": 761043740, "training_seconds": 95.7316139168106, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.4009463351197038}
prior_hypothesis: A 94-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 775,875,925 to approximately 761,043,740.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3742057365087481233, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 32353, "peak_hidden_elements": 93696, "recurrent_macs": 716211405, "recurrent_steps": 23635, "total_inference_macs": 717398045, "training_seconds": 73.04078183299862, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4208228924523102}
prior_hypothesis: A 91-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 731,804,800 to approximately 717,398,045.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4047086885220182473, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 34913, "peak_hidden_elements": 97792, "recurrent_macs": 774637125, "recurrent_steps": 23635, "total_inference_macs": 775875925, "training_seconds": 102.52321179187857, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4068840635334787}
prior_hypothesis: A 95-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 790,849,920 to approximately 775,875,925.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3667649241399184463, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 31728, "peak_hidden_elements": 92672, "recurrent_macs": 701959500, "recurrent_steps": 23635, "total_inference_macs": 703133100, "training_seconds": 70.69780524983071, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.41353488290236773}
prior_hypothesis: A 90-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 717,398,045 to approximately 703,133,100.

## Recent verification evidence

RECENT RESULT
hypothesis: A 98-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs to approximately 877,771,300 and recurrent steps from 26,080 to 25,265.
change: Increase recurrent width from 97 to 98, retain mean-plus-terminal classification, and omit only the earliest input frame.
mechanism: Single-edge-frame omission with capacity insurance
evidence_used: The 98-unit dual-view model achieved 87.12% accuracy across all 32 frames, providing 2.12 percentage points of margin; because larger reductions to 24 or 28 steps failed, skipping a single likely edge-context frame is the most conservative unresolved temporal-cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4578588665885341461, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 876493380, "recurrent_steps": 25265, "total_inference_macs": 877771300, "training_seconds": 133.824828249868, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40440371460709834}

RECENT RESULT
hypothesis: A 97-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the qualified 98-unit, 31-step design.
change: Reduce the GRU and dual-view pooling width from 98 to 97 and omit the earliest input frame, yielding 31 recurrent steps.
mechanism: Joint one-unit width refinement and single-edge-frame omission
evidence_used: The 97-unit dual-view model qualified at 86.01% using all frames, while the 98-unit 31-step model qualified at 86.50%; combining these individually successful, minimal reductions is the closest unresolved cost refinement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4493518622856917429, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36229, "peak_hidden_elements": 99840, "recurrent_macs": 860197455, "recurrent_steps": 25265, "total_inference_macs": 861462335, "training_seconds": 138.08664666698314, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4214622076303681}

RECENT RESULT
hypothesis: A 98-unit dual-view GRU processing frames 1–30 will retain at least 85% validation accuracy while reducing total inference MACs to approximately 849,497,320 and recurrent steps to 24,450.
change: Use a 98-unit GRU, classify concatenated mean and terminal recurrent states, and omit the first and final input frames.
mechanism: Dual-view pooling with two-edge-frame omission
evidence_used: The 98-unit dual-view model achieved 86.50% accuracy after omitting the first frame, leaving 1.50 percentage points of margin for testing one additional edge-frame omission; this should beat the qualified 97-unit, 31-step model’s 861,462,335 MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4431107283926769866, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 848219400, "recurrent_steps": 24450, "total_inference_macs": 849497320, "training_seconds": 140.15150250005536, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40198535568143695}

RECENT RESULT
hypothesis: The 98-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total inference MACs from 849,497,320 to approximately 821,223,340 and recurrent steps from 24,450 to 23,635.
change: Retain the qualified 98-unit architecture and training procedure, but omit one additional trailing input frame for a 29-step schedule.
mechanism: Progressive trailing-edge frame omission
evidence_used: Omitting the final frame from the qualified 31-step 98-unit model reduced MACs while accuracy changed only from 86.50% to 86.38%; its remaining 1.38-point margin motivates testing the next trailing-edge omission.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4283625901968198271, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 819945420, "recurrent_steps": 23635, "total_inference_macs": 821223340, "training_seconds": 93.3954102920834, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.40209003284664974}

RECENT RESULT
hypothesis: A 97-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total inference MACs from 821,223,340 to approximately 805,965,725.
change: Retain the current 97-unit architecture and training procedure, but omit the final two input frames in addition to the earliest frame, reducing execution from 31 to 29 recurrent steps.
mechanism: Dual-view width refinement at the qualified 29-step schedule
evidence_used: The 98-unit 29-step design achieved 86.99% accuracy with 821,223,340 MACs, providing substantially more margin than the current 97-unit 31-step design’s 85.03%; reducing width by one is the closest unresolved cost refinement, while prior 28-step reductions reportedly failed.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4204039860514241589, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36229, "peak_hidden_elements": 99840, "recurrent_macs": 804700845, "recurrent_steps": 23635, "total_inference_macs": 805965725, "training_seconds": 111.46466445783153, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4230663931442916}

RECENT RESULT
hypothesis: A 96-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total inference MACs from 805,965,725 to approximately 790,849,920.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 98 to 96, and use the qualified 29-frame schedule.
mechanism: Dual-view width refinement at the qualified 29-step schedule
evidence_used: The 97-unit 29-step design achieved 86.50% accuracy with 805,965,725 MACs; this 1.50-point margin supports testing the closest unresolved structural reduction, while the earlier 96-unit attempt timed out and supplied no negative accuracy evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4125193521598236323, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 789598080, "recurrent_steps": 23635, "total_inference_macs": 790849920, "training_seconds": 88.2217679170426, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.385877915832894}

RECENT RESULT
hypothesis: A 95-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 790,849,920 to approximately 775,875,925.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 96 to 95 while preserving the qualified schedule and training procedure.
mechanism: One-unit recurrent width refinement at the qualified 29-step schedule
evidence_used: The 96-unit 29-step model achieved 86.50% accuracy, matching the 97-unit model despite lower cost; its 1.50-point margin makes the adjacent 95-unit width the closest informative structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4047086885220182473, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 34913, "peak_hidden_elements": 97792, "recurrent_macs": 774637125, "recurrent_steps": 23635, "total_inference_macs": 775875925, "training_seconds": 102.52321179187857, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4068840635334787}

RECENT RESULT
hypothesis: A 94-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 775,875,925 to approximately 761,043,740.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from the current 97 units to 94 while preserving the qualified 29-frame schedule and training procedure.
mechanism: Adjacent recurrent-width refinement
evidence_used: The 95-unit, 29-step design achieved 86.87% accuracy with 775,875,925 MACs, improving slightly over the qualified 96- and 97-unit designs; its 1.87-point margin motivates testing the adjacent 94-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3969719951380080039, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 34264, "peak_hidden_elements": 96768, "recurrent_macs": 759817980, "recurrent_steps": 23635, "total_inference_macs": 761043740, "training_seconds": 95.7316139168106, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.4009463351197038}

RECENT RESULT
hypothesis: A 93-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 761,043,740 to approximately 746,353,365.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 98 to 93 while preserving the qualified 29-frame schedule and training procedure.
mechanism: Adjacent recurrent-width refinement
evidence_used: The 94-unit, 29-step design achieved 87.36% accuracy with 761,043,740 MACs, outperforming wider qualified designs and leaving 2.36 percentage points of margin for the adjacent structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3893092720077929021, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 33621, "peak_hidden_elements": 95744, "recurrent_macs": 745140645, "recurrent_steps": 23635, "total_inference_macs": 746353365, "training_seconds": 88.51285633305088, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4200117520759442}

RECENT RESULT
hypothesis: A 92-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 746,353,365 to approximately 731,804,800.
change: Replace the current 128-unit mean-pooled GRU with a 92-unit GRU using concatenated mean and terminal states, and execute the qualified 29-frame schedule.
mechanism: Adjacent recurrent-width refinement
evidence_used: The 93-unit, 29-step dual-view design achieved 86.13% accuracy with 746,353,365 MACs, leaving 1.13 percentage points of margin and making the adjacent 92-unit width the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3817205191313729419, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 730605120, "recurrent_steps": 23635, "total_inference_macs": 731804800, "training_seconds": 67.60597891709767, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4036617068425278}

RECENT RESULT
hypothesis: A 91-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 731,804,800 to approximately 717,398,045.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 92 to 91 while preserving the qualified 29-frame schedule and training procedure.
mechanism: Adjacent recurrent-width refinement
evidence_used: The 92-unit, 29-step design achieved 85.64% validation accuracy with 731,804,800 MACs; its positive margin makes the adjacent 91-unit width the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3742057365087481233, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 32353, "peak_hidden_elements": 93696, "recurrent_macs": 716211405, "recurrent_steps": 23635, "total_inference_macs": 717398045, "training_seconds": 73.04078183299862, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4208228924523102}

RECENT RESULT
hypothesis: A 90-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 717,398,045 to approximately 703,133,100.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 93 to 90 while preserving the qualified frame schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 91-unit, 29-step design achieved 86.75% validation accuracy with 717,398,045 MACs, leaving 1.75 percentage points of margin and making width 90 the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3667649241399184463, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 31728, "peak_hidden_elements": 92672, "recurrent_macs": 701959500, "recurrent_steps": 23635, "total_inference_macs": 703133100, "training_seconds": 70.69780524983071, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.41353488290236773}



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
