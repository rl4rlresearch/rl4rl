# Improve fixed-exposure image classification

You are an autonomous ML engineer improving a learned classifier for 28×28
grayscale images in ten classes.

## Goal

Maximize `validation_score`. It ranks implementations first by the exact number
of correct predictions on the fixed 10,000-image validation set, then—only when
correct counts tie—by lower validation cross-entropy. Every verification starts
from a fresh initialization and presents exactly 100,000 examples from the
fixed 50,000-image training split.

You may change the model architecture, optimizer, loss, augmentation, batch
size, gradient handling, schedule, and other contents of `train.py`. The fixed
data split, normalization, example accounting, validation calculation,
250,000-learned-parameter ceiling, and device are not editable. The protected
loop calls the functions already defined in `train.py`; keep that interface
intact. The model must return one ten-class logit vector per image.

## Work boundaries

Maximize validation_score. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: validation_score, validation_correct, validation_accuracy, validation_cross_entropy, parameters, examples_processed, optimizer_steps, training_seconds, batch_size.

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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 65.67569866613485, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286347503662108, "validation_score": 9295.415674771391}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20574 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286348114013672.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 41.398907124996185, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286348114013672, "validation_score": 9295.415674769281}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20573 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028634925842285.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 54.3704225840047, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286345596313476, "validation_score": 9295.415674777982}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20576 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286346435546876.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 33.512020791880786, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028634925842285, "validation_score": 9295.415674765327}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20572 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286350631713868.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20565 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286357192993165.
change: Increase only the positive inference-time logit multiplier from 1.20562 to 1.20565.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20564 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20564 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 61.280698999995366, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028635612487793, "validation_score": 9295.415674741598}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20566 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028635612487793.
change: Increase only the positive inference-time logit multiplier from 1.20561 to 1.20566.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20565 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20565 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 68.93328458280303, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028635513305664, "validation_score": 9295.415674745025}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20567 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028635513305664.
change: Increase only the positive inference-time logit multiplier from 1.20560 to 1.20567.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20566 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale up to 1.20566; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 49.88121212506667, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028635425567627, "validation_score": 9295.415674748057}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20568 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028635425567627.
change: Increase only the positive inference-time logit multiplier from 1.20567 to 1.20568.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20567 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20567 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 38.95701791602187, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028635383605957, "validation_score": 9295.415674749507}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20569 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028635383605957.
change: Increase only the positive inference-time logit multiplier from 1.20566 to 1.20569.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20568 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20568 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 54.081613457994536, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286352462768556, "validation_score": 9295.415674754253}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20570 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286352462768556.
change: Increase only the positive inference-time logit multiplier from 1.20565 to 1.20570.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20569 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale up to 1.20569; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 52.461723416810855, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028635154724121, "validation_score": 9295.415674757418}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20571 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028635154724121.
change: Increase only the positive inference-time logit multiplier from 1.20564 to 1.20571.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20570 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale up to 1.20570; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 45.33169695804827, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286350631713868, "validation_score": 9295.415674760581}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20572 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286350631713868.
change: Increase only the positive inference-time logit multiplier from 1.20571 to 1.20572.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20571 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20571 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 33.512020791880786, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028634925842285, "validation_score": 9295.415674765327}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20573 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028634925842285.
change: Increase only the positive inference-time logit multiplier from 1.20570 to 1.20573.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20572 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale up to 1.20572; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 41.398907124996185, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286348114013672, "validation_score": 9295.415674769281}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20574 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286348114013672.
change: Increase only the positive inference-time logit multiplier from 1.20569 to 1.20574.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20573 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20573 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 65.67569866613485, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286347503662108, "validation_score": 9295.415674771391}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20575 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286347503662108.
change: Increase only the positive inference-time logit multiplier from 1.20568 to 1.20575.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20574 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale up to 1.20574; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 61.1668708329089, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286346435546876, "validation_score": 9295.415674775082}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20576 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286346435546876.
change: Increase only the positive inference-time logit multiplier from 1.20575 to 1.20576.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20575 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20575 achieved the lowest verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 54.3704225840047, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286345596313476, "validation_score": 9295.415674777982}



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
