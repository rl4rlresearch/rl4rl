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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1585650168745642633, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 302319360, "recurrent_steps": 18745, "total_inference_macs": 303988480, "training_seconds": 38.58691979106516, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41982194397347106}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 23 causal frames while reducing total inference MACs from 317,132,800 to approximately 303,988,480.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448524622236889963, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 276030720, "recurrent_steps": 17115, "total_inference_macs": 277699840, "training_seconds": 34.16089425003156, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4290467431940184}
prior_hypothesis: Omitting frame 6 instead of accuracy-critical frame 29 will retain at least 85% validation accuracy while executing 21 recurrent steps and approximately 277,699,840 total inference MACs.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1722775715254395303, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 328608000, "recurrent_steps": 20375, "total_inference_macs": 330277120, "training_seconds": 44.923914291895926, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4193570376905196}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 25 causal frames while reducing total inference MACs from 343,421,440 to approximately 330,277,120.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1517087395491266298, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 289175040, "recurrent_steps": 17930, "total_inference_macs": 290844160, "training_seconds": 39.453580874949694, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223669695707918}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 22 causal frames while reducing total inference MACs from 303,988,480 to approximately 290,844,160.

## Recent verification evidence

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

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 25 causal frames while reducing total inference MACs from 343,421,440 to approximately 330,277,120.
change: Add the missing five-eighths-frame omission, producing evenly spaced omissions at every fourth interior frame while retaining both endpoints.
mechanism: Uniform one-in-four temporal pruning with signed-extrema readout
evidence_used: The 64-unit design qualified at every incremental reduction from 32 through 26 steps; the current 26-step result achieved 85.77% accuracy, so the adjacent 25-step schedule is the most informative next temporal-compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1722775715254395303, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 328608000, "recurrent_steps": 20375, "total_inference_macs": 330277120, "training_seconds": 44.923914291895926, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4193570376905196}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 24 causal frames while reducing total inference MACs from 330,277,120 to approximately 317,132,800.
change: Restore the proven seven uniformly spaced omissions and additionally omit the frame at fifteen-sixteenths of the sequence, retaining both endpoints and reducing execution to 24 steps.
mechanism: Incremental late-edge temporal pruning
evidence_used: The 25-step design achieved 86.13% accuracy, and every incremental reduction from 32 through 25 steps qualified; its 1.13-point margin motivates testing the adjacent step boundary with a low-information edge-adjacent omission.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1654212942000018968, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 315463680, "recurrent_steps": 19560, "total_inference_macs": 317132800, "training_seconds": 46.19284929195419, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41971319467743484}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 23 causal frames while reducing total inference MACs from 317,132,800 to approximately 303,988,480.
change: Apply the qualified 24-step schedule and additionally omit the frame at one-sixteenth of the sequence, symmetrically complementing the proven fifteen-sixteenths omission while retaining both endpoints.
mechanism: Symmetric early-edge temporal pruning
evidence_used: The 24-step design achieved 86.26% accuracy with a 1.26-point margin after every adjacent reduction from 32 through 24 steps qualified; removing one edge-adjacent frame is the smallest and most informative next compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1585650168745642633, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 302319360, "recurrent_steps": 18745, "total_inference_macs": 303988480, "training_seconds": 38.58691979106516, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41982194397347106}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 22 causal frames while reducing total inference MACs from 303,988,480 to approximately 290,844,160.
change: Apply the qualified 23-step schedule and additionally omit the earliest interior frame, preserving both endpoints and reducing 32-frame execution to 22 recurrent steps.
mechanism: Incremental boundary-frame temporal pruning
evidence_used: The 23-step design achieved 86.26% accuracy with a 1.26-point margin after every adjacent reduction from 32 through 23 steps qualified; pruning one likely low-information boundary frame is the smallest next structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1517087395491266298, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 289175040, "recurrent_steps": 17930, "total_inference_macs": 290844160, "training_seconds": 39.453580874949694, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223669695707918}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 21 causal frames while reducing total inference MACs from 290,844,160 to approximately 277,699,840.
change: Omit frame 29, the latest remaining near-boundary interior frame, while preserving both causal endpoints and reducing execution from 22 to 21 recurrent steps.
mechanism: Symmetric boundary-frame temporal pruning
evidence_used: The 22-step design achieved 86.01% accuracy after every adjacent reduction from 32 through 22 steps qualified; complementing the omitted early frames 1 and 2 with late frame 29 is the smallest next structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448524622236889963, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 276030720, "recurrent_steps": 17115, "total_inference_macs": 277699840, "training_seconds": 38.7681831670925, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4345594441232506}

RECENT RESULT
hypothesis: Omitting frame 6 instead of the accuracy-critical frame 29 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 277,699,840.
change: Add a three-sixteenths omission to the qualified 22-step schedule, removing a frame surrounded by retained neighbors while preserving frame 29 and both causal endpoints.
mechanism: Dense-cluster temporal pruning
evidence_used: The 22-step schedule achieved 86.01% accuracy, whereas its 21-step variant fell to 84.91% after removing frame 29; this suggests testing the same compute reduction at a densely sampled interior location.
result: the patch search text did not match the source

RECENT RESULT
hypothesis: Omitting frame 6 instead of accuracy-critical frame 29 will retain at least 85% validation accuracy while executing 21 recurrent steps and approximately 277,699,840 total inference MACs.
change: Extend the qualified 22-step schedule by omitting frame 6, which is surrounded by retained frames, while preserving frame 29 and both causal endpoints.
mechanism: Dense-cluster temporal pruning
evidence_used: The qualified 22-step design achieved 86.01% accuracy; its 21-step variant reached only 84.91% after omitting frame 29, motivating relocation of the same compute reduction to densely sampled frame 6.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448524622236889963, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 276030720, "recurrent_steps": 17115, "total_inference_macs": 277699840, "training_seconds": 34.16089425003156, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4290467431940184}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy after additionally omitting frame 10, reducing execution from 21 to 20 recurrent steps and total inference MACs to approximately 264,555,520.
change: Apply the qualified 21-step schedule and omit frame 10, an interior frame surrounded by retained frames 9 and 11, while preserving frame 29 and both endpoints.
mechanism: Iterative dense-cluster temporal pruning
evidence_used: The 21-step schedule qualified at 85.40% after frame 6 was removed from a dense retained cluster, whereas removing frame 29 failed at 84.91%; this motivates testing the same local pruning strategy at another densely sampled interior location.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1379961848982513628, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 262886400, "recurrent_steps": 16300, "total_inference_macs": 264555520, "training_seconds": 33.20889158407226, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4528039850340299}

RECENT RESULT
hypothesis: Omitting frame 3 from the qualified 21-step schedule will retain at least 85% validation accuracy while reducing execution to 20 recurrent steps and total inference MACs to approximately 264,555,520.
change: Add frame 3 to the skipped set for 32-frame inputs while preserving frame 29, both endpoints, and the existing qualified omissions.
mechanism: Early-silence temporal pruning
evidence_used: Removing frame 6 produced a qualified 21-step model at 85.40%, while removing frame 10 produced only 84.29%; this suggests testing another earlier frame before the accuracy-sensitive speech region.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1379961848982513628, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 262886400, "recurrent_steps": 16300, "total_inference_macs": 264555520, "training_seconds": 33.223459207918495, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4746866050673409}

RECENT RESULT
hypothesis: Omitting frame 26 from the qualified 21-step schedule will retain at least 85% validation accuracy while reducing execution to 20 recurrent steps and total inference MACs to approximately 264,555,520.
change: Add frame 26 to the skipped set for 32-frame inputs, preserving frame 29 and both endpoints.
mechanism: Late dense-cluster temporal pruning
evidence_used: The 21-step schedule qualified at 85.40% by pruning the center of the dense 5–7 cluster; frames 3 and 10 failed as additional omissions, while frame 29 was already shown accuracy-critical, motivating the analogous late dense 25–27 cluster.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1379961848982513628, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 262886400, "recurrent_steps": 16300, "total_inference_macs": 264555520, "training_seconds": 37.323471709154546, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4343353552320984}



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
