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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1597757484094083038, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 305136000, "recurrent_steps": 21190, "total_inference_macs": 306309600, "training_seconds": 78.6076197498478, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4400964093354582}
prior_hypothesis: A 60-unit GRU processing frames 3–28 will retain at least 85% validation accuracy while reducing total inference MACs from 318,045,600 to approximately 306,309,600.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1658974245937079853, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 316872000, "recurrent_steps": 22005, "total_inference_macs": 318045600, "training_seconds": 72.58885912504047, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.43758639470199867}
prior_hypothesis: A 60-unit GRU processing frames 3–29 will retain at least 85% validation accuracy while reducing total inference MACs from 329,781,600 to approximately 318,045,600.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1720191007780076668, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 328608000, "recurrent_steps": 22820, "total_inference_macs": 329781600, "training_seconds": 76.99711412494071, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4377409110040021}
prior_hypothesis: The 60-unit GRU can process frames 3–30 and retain at least 85% validation accuracy while reducing total inference MACs from 341,517,600 to approximately 329,781,600.

## Recent verification evidence

RECENT RESULT
hypothesis: A 72-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 572,325,600 to approximately 519,670,080.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 76 to 72 hidden units.
mechanism: Peak-augmented 72-unit full-resolution GRU
evidence_used: The qualified 76-unit design achieved 86.13% accuracy at 572,325,600 MACs, leaving a 1.13-point margin and motivating the next controlled four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 60.07818637508899, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4197747072559193}

RECENT RESULT
hypothesis: A 68-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 519,670,080 to approximately 469,518,240.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 80 to 68 hidden units.
mechanism: Peak-augmented 68-unit full-resolution GRU
evidence_used: The qualified 72-unit design achieved 86.63% accuracy at 519,670,080 MACs, leaving a 1.63-point margin and motivating the next controlled four-unit width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2449078584812962760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20040, "peak_hidden_elements": 104960, "recurrent_macs": 468188160, "recurrent_steps": 26080, "total_inference_macs": 469518240, "training_seconds": 61.520951207960024, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.412482578769052}

RECENT RESULT
hypothesis: A 64-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 469,518,240 to approximately 421,870,080.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 84 to 64 hidden units.
mechanism: Peak-augmented 64-unit full-resolution GRU
evidence_used: The qualified 68-unit design achieved 86.87% accuracy at 469,518,240 MACs, leaving a 1.87-point margin and motivating the next controlled four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2200538532061287056, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18096, "peak_hidden_elements": 98816, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 421870080, "training_seconds": 86.37268924992532, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4109568590035468}

RECENT RESULT
hypothesis: A 60-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 421,870,080 to approximately 376,725,600.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU from 64 to 60 hidden units.
mechanism: Peak-augmented 60-unit full-resolution GRU
evidence_used: The qualified 64-unit design achieved 86.63% accuracy at 421,870,080 MACs, leaving a 1.63-point margin; every tested four-unit reduction from 92 through 64 units retained at least 85% accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1965058055152063928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 375552000, "recurrent_steps": 26080, "total_inference_macs": 376725600, "training_seconds": 68.21224204194732, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.41650208081204465}

RECENT RESULT
hypothesis: A 56-unit GRU retaining mean, maximum, and final recurrent features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 376,725,600 to approximately 334,084,800.
change: Preserve all 32 causal frames and existing temporal summaries while narrowing the GRU hidden width from 68 to 56 units.
mechanism: Peak-augmented 56-unit full-resolution GRU
evidence_used: The qualified 60-unit design achieved 87.12% accuracy at 376,725,600 MACs, leaving a 2.12-point margin; every tested four-unit reduction from 92 through 60 units retained at least 85% accuracy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1742637154085293376, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 14496, "peak_hidden_elements": 86528, "recurrent_macs": 332989440, "recurrent_steps": 26080, "total_inference_macs": 334084800, "training_seconds": 110.34265295788646, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.46787872314453127}

RECENT RESULT
hypothesis: A 58-unit GRU retaining all 32 frames plus mean, maximum, and final-state features will achieve at least 85% validation accuracy while reducing total dense inference MACs from 376,725,600 to approximately 355,092,240.
change: Narrow the recurrent hidden width from 72 to 58 units while preserving the proven full-resolution schedule, temporal summaries, and training procedure.
mechanism: Boundary-seeking peak-augmented 58-unit full-resolution GRU
evidence_used: The 60-unit design achieved 87.12% accuracy, while 56 units fell to 83.93%; testing the midpoint is the most informative probe of the observed capacity boundary and offers lower MACs if it qualifies.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1852215157638372080, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 353957760, "recurrent_steps": 26080, "total_inference_macs": 355092240, "training_seconds": 112.09787379181944, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4290311380398054}

RECENT RESULT
hypothesis: The 58-unit GRU can omit only the earliest frame while retaining at least 85% validation accuracy and reducing total inference MACs from 355,092,240 to approximately 344,031,060.
change: Change the causal frame schedule from all 32 frames to frames 1–31, preserving model capacity and temporal summaries while executing 31 recurrent steps.
mechanism: Single-frame leading-context pruning
evidence_used: The current 58-unit full-resolution model achieved 85.77% accuracy; since aggressive thinning to 24 frames failed, removing just one likely low-information boundary frame is a conservative probe that reduces more MACs than narrowing from 58 to 57 units.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1794518359596661285, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 342896580, "recurrent_steps": 25265, "total_inference_macs": 344031060, "training_seconds": 150.3387265841011, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4350045327028614}

RECENT RESULT
hypothesis: The stronger 60-unit GRU can process only frames 3–31 while retaining at least 85% validation accuracy and reducing total inference MACs from 344,031,060 to approximately 341,517,600.
change: Preserve the qualified 60-unit recurrent model and temporal summaries, but omit the first three likely low-information boundary frames, reducing execution from 32 to 29 recurrent steps.
mechanism: Capacity-for-time exchange via leading-context pruning
evidence_used: The 60-unit full-resolution model achieved 87.12% accuracy, while the narrower 58-unit model remained qualified after omitting the earliest frame at 85.03%; retaining two additional hidden units provides capacity to test two further leading-frame removals at lower MAC cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1781407769623073483, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 340344000, "recurrent_steps": 23635, "total_inference_macs": 341517600, "training_seconds": 143.85625949990936, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.43409743981858706}

RECENT RESULT
hypothesis: The 60-unit GRU can omit frame 3 and process frames 4–31 while retaining at least 85% validation accuracy, reducing total dense inference MACs from 341,517,600 to approximately 329,781,600.
change: Narrow the current GRU from 68 to 60 units and use the 28-step causal schedule covering frames 4–31, while preserving the proven mean, maximum, and final-state summaries and training procedure.
mechanism: Incremental leading-context pruning at the qualified capacity boundary
evidence_used: The qualified 60-unit model processing frames 3–31 achieved 85.40% accuracy at 341,517,600 MACs. Removing one additional leading boundary frame is the smallest direct probe beyond that result and saves one complete recurrent step per example.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1720191007780076668, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 328608000, "recurrent_steps": 22820, "total_inference_macs": 329781600, "training_seconds": 65.35321154189296, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4591733686763085}

RECENT RESULT
hypothesis: The 60-unit GRU can process frames 3–30 and retain at least 85% validation accuracy while reducing total inference MACs from 341,517,600 to approximately 329,781,600.
change: Keep 28 recurrent steps by removing the final boundary frame instead of the more informative frame 3.
mechanism: Trailing-boundary pruning
evidence_used: Processing frames 4–31 missed the accuracy requirement by only one validation example (84.91%), while frames 3–31 achieved 85.40%; exchanging the final frame for frame 3 tests whether the leading frame carries more discriminative speech context.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1720191007780076668, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 328608000, "recurrent_steps": 22820, "total_inference_macs": 329781600, "training_seconds": 76.99711412494071, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4377409110040021}

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–29 will retain at least 85% validation accuracy while reducing total inference MACs from 329,781,600 to approximately 318,045,600.
change: Widen the current GRU from 58 to 60 units and adopt a 27-step causal schedule that removes one additional trailing boundary frame from the qualified 60-unit design.
mechanism: Incremental trailing-boundary pruning
evidence_used: The 60-unit model processing frames 3–30 achieved 86.50% accuracy at 329,781,600 MACs, while excluding frame 3 was worse; its 1.50-point margin motivates preserving the informative leading boundary and pruning the next trailing frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1658974245937079853, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 316872000, "recurrent_steps": 22005, "total_inference_macs": 318045600, "training_seconds": 72.58885912504047, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.43758639470199867}

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–28 will retain at least 85% validation accuracy while reducing total inference MACs from 318,045,600 to approximately 306,309,600.
change: Use the qualified 60-unit architecture and remove one additional trailing frame, reducing recurrent execution from 27 to 26 steps.
mechanism: Incremental trailing-boundary pruning
evidence_used: The 60-unit model processing frames 3–29 achieved 86.26% accuracy at 318,045,600 MACs, only 0.25 points below the 28-step design; this margin and stability motivate one more trailing-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1597757484094083038, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 305136000, "recurrent_steps": 21190, "total_inference_macs": 306309600, "training_seconds": 78.6076197498478, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4400964093354582}



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
