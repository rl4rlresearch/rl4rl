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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 50.527536584064364, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028637351989746, "validation_score": 9295.415674681486}
prior_hypothesis: Increasing the evaluation-only logit multiplier to 1.20548 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286374397277832.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 50.9193975829985, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.202863671875, "validation_score": 9295.415674703368}
prior_hypothesis: Increasing the evaluation-only logit multiplier to 1.20554 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286368370056151.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 49.31954904203303, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286369438171387, "validation_score": 9295.41567469559}
prior_hypothesis: Increasing the evaluation-only logit multiplier to 1.20552 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286370239257812.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 52.47680624993518, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286368370056151, "validation_score": 9295.415674699283}
prior_hypothesis: Increasing the evaluation-only logit multiplier to 1.20553 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286369438171387.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20543 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028637908935547.
change: Increase only the positive inference-time logit multiplier from 1.20536 to 1.20543.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale from 1.20515 through 1.20542 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20542 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 44.404380791122094, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286378479003905, "validation_score": 9295.41567466435}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20544 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286378479003905.
change: Increase only the positive inference-time logit multiplier from 1.20543 to 1.20544.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale from 1.20515 through 1.20543 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20543 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 56.00442733312957, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286377601623534, "validation_score": 9295.415674667382}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20545 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286377601623534.
change: Increase only the positive inference-time logit multiplier from 1.20542 to 1.20545.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale from 1.20515 through 1.20544 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20544 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 39.96934100007638, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286376571655274, "validation_score": 9295.41567467094}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20546 will preserve exactly 9,295 correct predictions while reduce validation cross-entropy below 0.20286376571655274.
change: Increase only the positive inference-time logit multiplier from 1.20541 to 1.20546.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale from 1.20515 through 1.20545 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20545 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 56.53343808394857, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028637523651123, "validation_score": 9295.415674675554}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20547 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028637523651123.
change: Increase only the positive inference-time logit multiplier from 1.20540 to 1.20547.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20546 achieved the best verified loss, extending the monotonic improvement across every tested scale from 1.20515 through 1.20546; positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 40.640529334079474, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286374397277832, "validation_score": 9295.415674678454}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20548 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286374397277832.
change: Increase only the positive inference-time logit multiplier from 1.20547 to 1.20548.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale from 1.20515 through 1.20547 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20547 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 50.527536584064364, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028637351989746, "validation_score": 9295.415674681486}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20549 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028637351989746.
change: Increase only the positive inference-time logit multiplier from 1.20546 to 1.20549.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20548 achieved the best verified loss, extending monotonic cross-entropy improvement across every tested scale from 1.20515 through 1.20548; positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 44.98347125016153, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286372108459472, "validation_score": 9295.415674686363}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20550 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286372108459472.
change: Increase only the positive inference-time logit multiplier from 1.20545 to 1.20550.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20549 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20549 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 41.11590679199435, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286371383666993, "validation_score": 9295.415674688868}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20551 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286371383666993.
change: Increase only the positive inference-time logit multiplier from 1.20544 to 1.20551.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20550 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20550 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 58.128778957994655, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286370239257812, "validation_score": 9295.415674692824}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20552 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286370239257812.
change: Increase only the positive inference-time logit multiplier from 1.20551 to 1.20552.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20551 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20551 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 49.31954904203303, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286369438171387, "validation_score": 9295.41567469559}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20553 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286369438171387.
change: Increase only the positive inference-time logit multiplier from 1.20550 to 1.20553.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20552 achieved the best verified loss, extending monotonic cross-entropy improvement across every tested scale through 1.20552; positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 52.47680624993518, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286368370056151, "validation_score": 9295.415674699283}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20554 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286368370056151.
change: Increase only the positive inference-time logit multiplier from 1.20549 to 1.20554.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20553 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20553 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 50.9193975829985, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.202863671875, "validation_score": 9295.415674703368}



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
