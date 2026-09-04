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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 65.22391170891933, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286358070373536, "validation_score": 9295.415674734875}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20563 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286359062194825.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 58.09884516708553, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286359062194825, "validation_score": 9295.415674731448}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20562 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286360321044922.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 39.22900162497535, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286361045837403, "validation_score": 9295.415674724592}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20560 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028636215209961.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 45.63623299985193, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286360321044922, "validation_score": 9295.415674727097}
prior_hypothesis: Raising the evaluation-only logit multiplier to 1.20561 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286361045837403.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20555 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.202863671875.
change: Increase only the positive inference-time logit multiplier from 1.20548 to 1.20555.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20554 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20554 achieved the best verified loss, and positive scaling preserves finite logits’ argmax classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 59.72029208391905, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286366004943848, "validation_score": 9295.415674707456}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit multiplier to 1.20556 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286366004943848.
change: Increase only the positive inference-time logit multiplier from 1.20555 to 1.20556.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20555 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20555 achieved the best verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 56.75545358285308, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286365394592284, "validation_score": 9295.415674709564}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20557 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286365394592284.
change: Increase only the positive inference-time logit multiplier from 1.20554 to 1.20557.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20556 preserved all 9,295 correct predictions and monotonically reduced cross-entropy; 1.20556 achieved the best verified loss, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 38.99544941610657, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286364059448242, "validation_score": 9295.415674714179}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20558 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286364059448242.
change: Increase only the positive inference-time logit multiplier from 1.20553 to 1.20558.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20557 preserved all 9,295 correct predictions and monotonically reduced cross-entropy; 1.20557 achieved the best verified loss, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 57.87037716712803, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286362838745117, "validation_score": 9295.415674718397}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20559 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286362838745117.
change: Increase only the positive inference-time logit multiplier from 1.20552 to 1.20559.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20558 preserved all 9,295 correct predictions and monotonically reduced cross-entropy; 1.20558 achieved the best verified loss, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 62.680084417108446, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028636215209961, "validation_score": 9295.41567472077}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20560 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.2028636215209961.
change: Increase only the positive inference-time logit multiplier from 1.20559 to 1.20560.
mechanism: Argmax-preserving calibration continuation
evidence_used: Every tested scale through 1.20559 preserved all 9,295 correct predictions while monotonically reducing cross-entropy; 1.20559 achieved the best verified loss, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 39.22900162497535, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286361045837403, "validation_score": 9295.415674724592}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20561 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286361045837403.
change: Increase only the positive inference-time logit multiplier from 1.20558 to 1.20561.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20560 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 45.63623299985193, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286360321044922, "validation_score": 9295.415674727097}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20562 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286360321044922.
change: Increase only the positive inference-time logit multiplier from 1.20557 to 1.20562.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20561 produced the lowest verified cross-entropy after monotonic improvement through every tested scale from 1.20550 to 1.20561; positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 58.09884516708553, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286359062194825, "validation_score": 9295.415674731448}

RECENT RESULT
hypothesis: Raising the evaluation-only logit multiplier to 1.20563 will preserve exactly 9,295 correct predictions while reducing validation cross-entropy below 0.20286359062194825.
change: Increase only the positive inference-time logit multiplier from 1.20556 to 1.20563.
mechanism: Argmax-preserving calibration continuation
evidence_used: Scale 1.20562 achieved the lowest verified cross-entropy after monotonic improvement through every tested scale from 1.20550 to 1.20562; positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 65.22391170891933, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286358070373536, "validation_score": 9295.415674734875}



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
