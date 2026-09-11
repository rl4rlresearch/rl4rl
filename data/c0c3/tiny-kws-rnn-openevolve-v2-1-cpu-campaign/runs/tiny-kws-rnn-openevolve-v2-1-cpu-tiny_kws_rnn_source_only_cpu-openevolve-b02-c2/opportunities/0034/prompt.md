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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3306704161026971853, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 632827125, "recurrent_steps": 23635, "total_inference_macs": 633935525, "training_seconds": 58.149112458806485, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4017870850358273}
prior_hypothesis: An 85-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 647,491,420 to approximately 633,935,525.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3236734252566383579, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 619426080, "recurrent_steps": 23635, "total_inference_macs": 620521440, "training_seconds": 49.875909709138796, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4067141667465491}
prior_hypothesis: An 84-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 633,935,525 to approximately 620,521,440.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3377413772025511543, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 646369980, "recurrent_steps": 23635, "total_inference_macs": 647491420, "training_seconds": 55.52645374997519, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.41081258387653374}
prior_hypothesis: An 86-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 689,009,965 to approximately 647,491,420.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3448863085562002649, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29889, "peak_hidden_elements": 89600, "recurrent_macs": 660054645, "recurrent_steps": 23635, "total_inference_macs": 661189125, "training_seconds": 51.28165933396667, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.41917986723542944}
prior_hypothesis: An 87-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 675,028,640 to approximately 661,189,125.

## Recent verification evidence

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

RECENT RESULT
hypothesis: An 89-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 703,133,100 to approximately 689,009,965.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 94 to 89 while preserving the qualified schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 90-unit, 29-step design achieved 86.87% validation accuracy with 703,133,100 MACs, leaving 1.87 percentage points of margin and motivating the adjacent untested width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3593980820248839109, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 31109, "peak_hidden_elements": 91648, "recurrent_macs": 687849405, "recurrent_steps": 23635, "total_inference_macs": 689009965, "training_seconds": 61.57348820916377, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40889182061505464}

RECENT RESULT
hypothesis: An 88-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 689,009,965 to approximately 675,028,640.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 95 to 88 while preserving the qualified frame schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 89-unit, 29-step design achieved 86.38% validation accuracy with 689,009,965 MACs, leaving 1.38 percentage points of margin and making width 88 the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3521052101636445171, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 673881120, "recurrent_steps": 23635, "total_inference_macs": 675028640, "training_seconds": 58.271423124941066, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4195692378319114}

RECENT RESULT
hypothesis: An 87-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 675,028,640 to approximately 661,189,125.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 88 to 87 while preserving the qualified frame schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 88-unit, 29-step model achieved 86.26% validation accuracy with 675,028,640 MACs, leaving 1.26 percentage points of margin and making width 87 the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3448863085562002649, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29889, "peak_hidden_elements": 89600, "recurrent_macs": 660054645, "recurrent_steps": 23635, "total_inference_macs": 661189125, "training_seconds": 51.28165933396667, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.41917986723542944}

RECENT RESULT
hypothesis: An 86-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 689,009,965 to approximately 647,491,420.
change: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 89 to 86 while preserving the qualified schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 87-unit, 29-step design achieved 86.01% validation accuracy with 661,189,125 MACs, leaving 1.01 percentage points of margin and making width 86 the closest unresolved structural reduction; reported prior 28-step reductions failed.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3377413772025511543, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 646369980, "recurrent_steps": 23635, "total_inference_macs": 647491420, "training_seconds": 55.52645374997519, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.41081258387653374}

RECENT RESULT
hypothesis: An 85-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 647,491,420 to approximately 633,935,525.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 90 to 85 while preserving the qualified schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 86-unit, 29-step design achieved 86.87% validation accuracy with 647,491,420 MACs, leaving 1.87 percentage points of margin and making width 85 the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3306704161026971853, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 632827125, "recurrent_steps": 23635, "total_inference_macs": 633935525, "training_seconds": 58.149112458806485, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4017870850358273}

RECENT RESULT
hypothesis: An 84-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 633,935,525 to approximately 620,521,440.
change: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 91 to 84 while preserving the qualified schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 85-unit design achieved 86.99% validation accuracy with 633,935,525 MACs, leaving a 1.99-point margin and making width 84 the closest unresolved structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3236734252566383579, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 619426080, "recurrent_steps": 23635, "total_inference_macs": 620521440, "training_seconds": 49.875909709138796, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4067141667465491}

RECENT RESULT
hypothesis: An 83-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 620,521,440 to approximately 607,249,165.
change: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 84 to 83 while preserving the qualified schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 84-unit design achieved 85.64% validation accuracy with 620,521,440 MACs; its positive 0.64-point margin makes width 83 the closest unresolved structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167504046643746721, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27521, "peak_hidden_elements": 85504, "recurrent_macs": 606166845, "recurrent_steps": 23635, "total_inference_macs": 607249165, "training_seconds": 52.139635833213106, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4415310350663823}



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
