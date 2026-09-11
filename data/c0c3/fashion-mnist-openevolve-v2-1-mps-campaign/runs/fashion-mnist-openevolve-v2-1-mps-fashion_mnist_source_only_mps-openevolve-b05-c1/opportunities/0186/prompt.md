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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 54.84297208301723, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569574165344238, "validation_score": 9360.42169334209}
prior_hypothesis: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569591064453125.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.08 to 0.12 will preserve all 9,360 predictions because it applies a strictly positive per-image scalar, while lowering validation cross-entropy below 0.18578439636230468.
change: Increase only the consensus-conditioned confidence-scaling coefficient from 0.08 to 0.12.
mechanism: Stronger TTA-agreement-conditioned temperature scaling
evidence_used: Raising the coefficient from 0.04 to 0.08 preserved 9,360 correct predictions and reduced cross-entropy from 0.18581797180175783 to 0.18578439636230468, supporting another cautious step in the same direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.08 to 0.12 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18578439636230468.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.08 to 0.12.
mechanism: Stronger TTA-agreement-conditioned temperature scaling
evidence_used: Raising the coefficient from 0.04 to 0.08 preserved all 9,360 correct predictions and reduced cross-entropy from 0.18581797180175783 to 0.18578439636230468; the prior 0.12 attempt timed out without producing contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.18385050003417, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18575621643066406, "validation_score": 9360.4216718353}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.12 to 0.16 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18575621643066406.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.12 to 0.16.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Successive increases from 0.04 to 0.08 and then 0.12 preserved all 9,360 correct predictions while reducing cross-entropy from 0.18581797180175783 to 0.18578439636230468 and then 0.18575621643066406, supporting another equal-sized step.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.13296675006859, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18573341827392578, "validation_score": 9360.421679942805}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.16 to 0.20 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18573341827392578.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.16 to 0.20.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Each increase from 0.04 through 0.16 preserved exactly 9,360 correct predictions and progressively reduced cross-entropy; the latest 0.12→0.16 step improved it from 0.18575621643066406 to 0.18573341827392578.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.16 to 0.20 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18573341827392578.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.16 to 0.20.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Every verified increase from 0.04 through 0.16 preserved exactly 9,360 correct predictions and progressively reduced cross-entropy; the previous 0.20 attempt timed out without contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.97503604204394, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18571599617004395, "validation_score": 9360.421686138683}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.20 to 0.24 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18571599617004395.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.20 to 0.24.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Every verified increase from 0.04 through 0.20 preserved exactly 9,360 correct predictions and progressively lowered cross-entropy; the latest 0.16→0.20 step improved it from 0.18573341827392578 to 0.18571599617004395.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.57178612495773, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1857039375305176, "validation_score": 9360.421690427242}

RECENT RESULT
hypothesis: Bypassing the shared 30-unit positional bottleneck with normalized, class-specific 2×2 spatial evidence will exceed 9,360 correct predictions by distinguishing classes through coarse part layout.
change: Add a zero-initialized residual head over normalized quadrant-pooled feature maps; make the existing statistics BatchNorm affine-free to remain below the parameter ceiling.
mechanism: Class-conditional coarse spatial residual head
evidence_used: Agreement calibration repeatedly lowered cross-entropy without changing 9,360 decisions, while loss and dropout changes reduced accuracy; this challenges the assumption that confidence or regularization is limiting and instead tests new learned class-prediction capacity. The poorer diagonal-translation result also suggests preserving spatial layout is useful.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.24 to 0.28 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.1857039375305176.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.24 to 0.28.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Every verified increase from 0.04 through 0.24 preserved exactly 9,360 correct predictions and progressively lowered cross-entropy; the latest 0.20→0.24 step improved it from 0.18571599617004395 to 0.1857039375305176.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 79.32754804193974, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569724464416504, "validation_score": 9360.421692807551}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.28 to 0.32 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18569724464416504.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.28 to 0.32.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Every verified 0.04 coefficient increase through 0.28 preserved exactly 9,360 correct predictions and lowered cross-entropy; the latest 0.24→0.28 step improved it from 0.1857039375305176 to 0.18569724464416504, supporting one further equal-sized step despite diminishing gains.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 55.8633971249219, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569591064453125, "validation_score": 9360.421693281989}

RECENT RESULT
hypothesis: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18569591064453125.
change: Reduce only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.32 to 0.31.
mechanism: Evidence-fitted TTA-agreement temperature calibration
evidence_used: Cross-entropy improvements from successive 0.04 increases diminished consistently by about 5.36e-6 per step; fitting that trend places the minimum near 0.31, while positive scalar calibration cannot alter argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569591064453125.
change: Reduce only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.32 to 0.31.
mechanism: Evidence-fitted TTA-agreement temperature calibration
evidence_used: Successive coefficient increases produced diminishing cross-entropy gains, placing the estimated minimum near 0.31; the previous 0.31 verification timed out and supplied no contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting the agreement coefficient to 0.31 will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569591064453125.
change: Reduce only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.32 to the estimated optimum of 0.31.
mechanism: Evidence-fitted TTA-agreement temperature calibration
evidence_used: Successive 0.04 coefficient increases showed nearly constant diminishing gains, fitting a cross-entropy minimum near 0.31; two prior 0.31 runs timed out but produced no contrary validation evidence, and positive scalar scaling cannot change argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 54.84297208301723, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569574165344238, "validation_score": 9360.42169334209}



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
