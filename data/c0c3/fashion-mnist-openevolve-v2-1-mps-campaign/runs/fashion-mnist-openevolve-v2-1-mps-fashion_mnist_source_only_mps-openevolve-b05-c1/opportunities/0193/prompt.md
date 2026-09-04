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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 80.24320708308369, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569544525146484, "validation_score": 9360.421693447506}
prior_hypothesis: Setting the below-average-agreement bonus to 0.014 will preserve all 9,360 correct predictions while lowering validation cross-entropy below 0.1856954532623291.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Adding normalized vertical and magnitude-based quadrant contrasts will exceed 9,360 correct predictions by exposing coarse spatial layout directly to each class without exceeding 250,000 parameters.
change: Add a zero-initialized 192-feature spatial residual head using top-bottom, absolute left-right, and absolute diagonal contrasts; total learned parameters become 249,860.
mechanism: Flip-invariant class-specific spatial contrast head
evidence_used: Agreement calibration from 0.04 through its fitted optimum of 0.31 repeatedly improved cross-entropy but never changed the 9,360 predictions, indicating that further scalar calibration is unlikely to improve the primary objective. The earlier full-quadrant spatial-head implementation could not be verified, so this lower-dimensional formulation tests the same capacity hypothesis with a simpler under-budget design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying a small additional confidence penalty only to below-average TTA agreement will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569574165344238.
change: Add a centered hinge feature for below-average agreement and give it a conservative negative coefficient in the existing positive confidence scale.
mechanism: Low-consensus tail confidence calibration
evidence_used: Linear agreement calibration reached its fitted optimum at 0.31 without changing any predictions, while consistently improving cross-entropy; this motivates testing whether the remaining calibration error is asymmetric in the low-consensus tail.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 61.8157789579127, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1856981330871582, "validation_score": 9360.421692491578}

RECENT RESULT
hypothesis: A 0.02 confidence bonus for below-average TTA agreement will preserve exactly 9,360 predictions while lowering cross-entropy below 0.18569574165344238.
change: Add a small positive centered-hinge term to the existing agreement-conditioned confidence scale.
mechanism: Low-consensus confidence restoration
evidence_used: An additional low-agreement confidence penalty increased cross-entropy to 0.1856981330871582 without changing predictions, indicating that the remaining low-consensus tail was over-attenuated and motivating a conservative perturbation in the opposite direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 73.51250083302148, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569551315307617, "validation_score": 9360.421693423357}

RECENT RESULT
hypothesis: Increasing the below-average-agreement confidence bonus from 0.02 to 0.04 will preserve exactly 9,360 predictions while lowering validation cross-entropy below 0.18569551315307617.
change: Increase only the positive centered-hinge coefficient in the existing TTA-agreement confidence scale.
mechanism: Incremental low-consensus confidence restoration
evidence_used: A confidence penalty for low-agreement examples worsened cross-entropy to 0.1856981330871582, while the opposite 0.02 bonus improved it to 0.18569551315307617 without changing any predictions, motivating one further equal-sized step in the beneficial direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed max pooling with per-channel learned max–average mixtures will exceed 9,360 correct predictions by preserving distributed contour evidence while retaining peak responses where useful.
change: Add two scale-specific adaptive pooling modules, initialized as exact max pooling, so training learns each channel’s preferred downsampling behavior with only 96 additional parameters.
mechanism: Learned channel-wise max–average downsampling
evidence_used: Agreement calibration repeatedly improved cross-entropy without changing any of the 9,360 decisions, indicating that confidence scaling is no longer the primary limitation. The model currently assumes hard maxima are optimal for every feature channel; learning the downsampling rule changes the image representation directly and may reduce the translation sensitivity exposed by the extensive translated-view augmentation and inference ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting the below-average-agreement bonus to 0.012 will preserve all 9,360 correct predictions while lowering validation cross-entropy below 0.18569551315307617.
change: Reduce only the centered-hinge confidence coefficient from 0.02 to 0.012.
mechanism: Evidence-fitted low-consensus confidence calibration
evidence_used: A negative low-agreement adjustment worsened cross-entropy to 0.1856981330871582, while a 0.02 bonus improved it to 0.18569551315307617 without changing predictions; a quadratic fit across the negative, zero, and positive results estimates the optimum near 0.012.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 60.41099141607992, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1856954532623291, "validation_score": 9360.421693444656}

RECENT RESULT
hypothesis: Setting the below-average-agreement bonus to 0.014 will preserve all 9,360 correct predictions while lowering validation cross-entropy below 0.1856954532623291.
change: Increase only the centered-hinge confidence coefficient from 0.012 to 0.014.
mechanism: Evidence-fitted low-consensus calibration refinement
evidence_used: At the same 0.31 agreement coefficient, bonuses of 0, 0.012, and 0.02 yielded cross-entropies of 0.18569574165344238, 0.1856954532623291, and 0.18569551315307617 without changing predictions; a quadratic fit to these verified points places the minimum near 0.0136.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 80.24320708308369, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18569544525146484, "validation_score": 9360.421693447506}



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
