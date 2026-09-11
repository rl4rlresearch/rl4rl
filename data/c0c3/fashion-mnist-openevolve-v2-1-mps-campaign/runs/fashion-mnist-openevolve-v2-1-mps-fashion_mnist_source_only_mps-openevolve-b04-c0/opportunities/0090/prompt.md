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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.87751333299093, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420372009276, "validation_score": 9359.421982163683}
prior_hypothesis: An evaluation-only scale of 1.226016 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.08 to 1.10 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.1881631706237793.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Successive scale increases from 1.02 through 1.08 preserved 9,359 correct predictions and monotonically reduced cross-entropy, most recently from 0.18926130714416503 to 0.1881631706237793.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.67124908301048, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18725096054077148, "validation_score": 9359.421140952181}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.10 to 1.12 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18725096054077148.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Every tested increase from 1.02 through 1.10 preserved 9,359 correct predictions and monotonically reduced cross-entropy, with 1.10 improving it to 0.18725096054077148.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 69.92816245788708, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18650871887207032, "validation_score": 9359.421404404407}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.12 to 1.14 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18650871887207032.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Every tested increase from 1.02 through 1.12 preserved 9,359 correct predictions and monotonically reduced cross-entropy; the latest increase from 1.10 to 1.12 improved it from 0.18725096054077148 to 0.18650871887207032.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 56.56203949986957, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18592204666137696, "validation_score": 9359.421612871949}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.14 to 1.16 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18592204666137696.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Every tested increase from 1.02 through 1.14 preserved 9,359 correct predictions and monotonically reduced cross-entropy; the latest increase from 1.12 to 1.14 improved it from 0.18650871887207032 to 0.18592204666137696.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 75.54348554089665, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1854779182434082, "validation_score": 9359.42177082534}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.16 to 1.18 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.1854779182434082.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Every tested increase from 1.02 through 1.16 preserved 9,359 correct predictions and monotonically reduced cross-entropy; the latest increase from 1.14 to 1.16 improved it from 0.18592204666137696 to 0.1854779182434082.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.49053220800124, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18516458778381348, "validation_score": 9359.421882331917}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.18 to 1.20 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18516458778381348.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Every tested increase from 1.02 through 1.18 preserved 9,359 correct predictions and monotonically reduced cross-entropy; the latest increase from 1.16 to 1.18 improved it from 0.1854779182434082 to 0.18516458778381348.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 50.687874959083274, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1849714141845703, "validation_score": 9359.421951107019}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.20 to 1.22 will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1849714141845703.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Every tested increase from 1.02 through 1.20 preserved 9,359 correct predictions and monotonically reduced cross-entropy; the latest increase from 1.18 to 1.20 improved it from 0.18516458778381348 to 0.1849714141845703.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 67.30079516698606, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488877143859864, "validation_score": 9359.421980536952}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.225 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18488877143859864.
change: Increase the positive flip-ensemble calibration factor from 1.22 to 1.225.
mechanism: Quadratic-refined evaluation logit calibration
evidence_used: Cross-entropy improvements shrank from 0.00019317 at 1.20 to 0.00008264 at 1.22; local quadratic interpolation places the minimum near 1.225.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 48.86259841709398, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488433990478514, "validation_score": 9359.421982115184}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.226 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488433990478514.
change: Increase only the positive flip-ensemble calibration factor from 1.225 to 1.226.
mechanism: Quadratic-refined evaluation logit calibration
evidence_used: Cross-entropy was 0.1849714141845703 at 1.20, 0.18488877143859864 at 1.22, and 0.18488433990478514 at 1.225; quadratic interpolation of these results places the local minimum near 1.226.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 74.10099791595712, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420639038086, "validation_score": 9359.421982162732}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.22603 will preserve all 9,359 argmax predictions and slightly reduce validation cross-entropy below 0.18488420639038086.
change: Refine only the positive flip-ensemble calibration factor from 1.226 to 1.22603.
mechanism: Fine-grained quadratic logit calibration
evidence_used: Cross-entropy improved from 0.18488877143859864 at 1.22 to 0.18488433990478514 at 1.225 and 0.18488420639038086 at 1.226; quadratic interpolation of these three measurements places the local minimum near 1.226032.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.97517937514931, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420486450197, "validation_score": 9359.421982163276}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.226032 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.
change: Refine only the positive flip-ensemble calibration factor from 1.22603 to 1.226032.
mechanism: Fine-grained quadratic logit calibration
evidence_used: Cross-entropy improved at every tested scale through 1.22603, while quadratic interpolation of the measurements at 1.22, 1.225, and 1.226 places the local minimum near 1.226032.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 77.51370766595937, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420715332032, "validation_score": 9359.421982162461}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.226016 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.
change: Refine only the positive flip-ensemble calibration factor from 1.22603 to 1.226016.
mechanism: Bracketed local logit-temperature refinement
evidence_used: Scale 1.22603 achieved the best observed cross-entropy, while 1.226032 regressed; a local quadratic fit using the measured results at 1.226, 1.22603, and 1.226032 places the minimum near 1.226016.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.87751333299093, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420372009276, "validation_score": 9359.421982163683}



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
