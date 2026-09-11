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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 55.8633971249219, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569591064453125, "validation_score": 9360.421693281989}
prior_hypothesis: Increasing the agreement coefficient from 0.28 to 0.32 will preserve exactly 9,360 argmax predictions while lowering validation cross-entropy below 0.18569724464416504.

## Recent verification evidence

RECENT RESULT
hypothesis: Redistributing one eighth of vertical-view weight to horizontal translations will exceed 9,360 correct predictions by preserving class-bearing vertical alignment while retaining horizontal translation robustness.
change: Keep the existing ten views, center weight, total ensemble weight, logit-space aggregation, and calibration, but weight vertical shifts at 0.875 and horizontal shifts at 1.125.
mechanism: Anisotropic cardinal-view logit averaging
evidence_used: Diagonal translation augmentation reduced validation_correct from 9,360 to 9,340, indicating that translation invariance is not uniformly beneficial; probability-space TTA also underperformed, motivating a targeted weight redistribution within the verified logit-space ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.152112000156194, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.18580753479003906, "validation_score": 9357.421653586549}

RECENT RESULT
hypothesis: Centered per-image temperature adjustments based on ten-view prediction agreement will retain all 9,360 predictions while lowering validation cross-entropy below 0.18585695190429688.
change: Preserve the verified weighted logit ensemble, then slightly sharpen high-consensus examples and soften low-consensus examples with a strictly positive scale that cannot change argmax predictions.
mechanism: TTA-consensus-conditioned logit calibration
evidence_used: Global scaling preserved 9,360 correct and improved cross-entropy, while anisotropic TTA weighting lost three correct; this motivates richer calibration without altering ensemble decisions.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 76.05017645796761, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18581797180175783, "validation_score": 9360.421649875352}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.04 to 0.08 will retain all 9,360 predictions while further lowering validation cross-entropy below 0.18581797180175783.
change: Double only the strength of the strictly positive per-image confidence scale, preserving training, TTA logits, global calibration, and argmax predictions.
mechanism: Stronger TTA-consensus-conditioned temperature scaling
evidence_used: Adding the 0.04 agreement-conditioned scale preserved 9,360 correct predictions and improved cross-entropy from the global-scaling result of 0.18585695190429688 to 0.18581797180175783, indicating that greater consensus is positively associated with correctness and motivating a cautious step farther in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 51.011631625005975, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18578439636230468, "validation_score": 9360.421661814351}

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
