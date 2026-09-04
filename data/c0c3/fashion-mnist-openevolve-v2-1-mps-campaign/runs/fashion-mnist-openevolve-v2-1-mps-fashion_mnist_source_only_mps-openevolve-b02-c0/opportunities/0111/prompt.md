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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.89293937501498, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259567527770995, "validation_score": 9253.412338597434}
prior_hypothesis: Increasing the positive ensemble scale from 1.3605 to 1.3606 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125958950.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble scores from 1.30 to 1.34 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21311241.
change: Increase only the positive inference-time calibration multiplier from 1.30 to 1.34.
mechanism: Evidence-fitted post-ensemble temperature calibration
evidence_used: Every verified scale increase through 1.30 preserved 9,253 correct predictions and reduced cross-entropy; finite-difference evidence estimates the optimum near 1.36, while higher-scale attempts timed out without contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 57.0442100840155, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21268247032165527, "validation_score": 9253.412309085219}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.34 to 1.36 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21268247.
change: Change only the inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.
mechanism: Evidence-fitted post-ensemble temperature calibration
evidence_used: Scaling from 1.30 to 1.34 preserved all 9,253 argmax predictions and reduced cross-entropy from 0.21311241 to 0.21268247; prior finite-difference evidence places the calibration optimum near 1.36, while earlier 1.36 attempts timed out without producing contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 74.53653483302332, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259699668884277, "validation_score": 9253.412338148095}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.36 to 1.37 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21259700.
change: Change only the inference-time ensemble calibration multiplier from 1.36 to 1.37.
mechanism: Quadratic-fit post-ensemble temperature calibration
evidence_used: Scaling from 1.30 to 1.34 reduced cross-entropy by 0.00042994, and scaling from 1.34 to 1.36 reduced it by another 0.00008547 while preserving all 9,253 predictions. These finite differences place the local convex optimum near 1.370; positive scaling preserves argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores from 1.36 to 1.3698 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21259700.
change: Change only the positive inference-time ensemble calibration multiplier from 1.36 to 1.3698.
mechanism: Quadratic-fit post-ensemble temperature calibration
evidence_used: Cross-entropy fell from 0.21311241 at 1.30 to 0.21268247 at 1.34 and 0.21259700 at 1.36, with all 9,253 argmax predictions preserved. A quadratic fit to these measurements places the minimum near 1.3698; the timed-out 1.37 verification supplied no contradictory score evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3599 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21259700.
change: Decrease only the positive inference-time calibration multiplier from 1.36 to 1.3599.
mechanism: Local quadratic post-ensemble temperature calibration
evidence_used: Cross-entropy slopes over 1.30–1.34 and 1.34–1.36 were approximately -0.010748 and -0.004274; a quadratic fit to those verified measurements places the local minimum near 1.3599. Any positive scale preserves argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 59.09892145800404, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2125972194671631, "validation_score": 9253.41233807234}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.36 to 1.3601 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2125969967.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: Decreasing the scale from 1.36 to 1.3599 preserved 9,253 correct predictions but worsened cross-entropy from 0.2125969967 to 0.2125972195, providing local evidence that the improving direction at 1.36 is upward.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 70.46746816718951, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259677238464356, "validation_score": 9253.412338224369}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.3601 to 1.3602 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2125967724.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: Moving from 1.36 to 1.3601 reduced cross-entropy from 0.2125969967 to 0.2125967724, while moving downward to 1.3599 worsened it to 0.2125972195. This brackets 1.36 with consistent evidence that the locally improving direction remains upward; positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 78.10502237500623, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259654693603516, "validation_score": 9253.412338301032}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.3602 to 1.3603 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125965469.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: The consecutive increases from 1.36 to 1.3601 and then 1.3602 preserved all 9,253 argmax predictions while reducing cross-entropy from 0.2125969967 to 0.2125967724 and then 0.2125965469; positive scaling preserves argmax decisions, and the locally improving direction remains upward.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 78.52325112512335, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259632873535156, "validation_score": 9253.41233837523}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.3603 to 1.3604 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125963287.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: Each consecutive 0.0001 increase from 1.3600 through 1.3603 preserved all 9,253 argmax predictions and reduced cross-entropy; positive scaling cannot change those predictions, and the latest step improved cross-entropy from 0.2125965469 to 0.2125963287.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 64.94315858301707, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2125961082458496, "validation_score": 9253.412338450205}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.3604 to 1.3605 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125961082.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: Every consecutive 0.0001 increase from 1.3600 through 1.3604 preserved all 9,253 argmax predictions and reduced cross-entropy; the latest increase improved it from 0.2125963287 to 0.2125961082, so the locally improving direction remains upward.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 70.48626191588119, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259589500427245, "validation_score": 9253.412338522718}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.3605 to 1.3606 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125958950.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: Every consecutive 0.0001 increase from 1.3600 through 1.3605 preserved all 9,253 argmax predictions and reduced cross-entropy; the latest step improved it from 0.2125961082 to 0.2125958950, so the locally improving direction remains upward.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.89293937501498, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21259567527770995, "validation_score": 9253.412338597434}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.3606 to 1.3607 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125956753.
change: Increase only the inference-time ensemble calibration multiplier by 0.0001.
mechanism: One-sided local temperature calibration
evidence_used: Every verified 0.0001 increase from 1.3600 through 1.3606 preserved all 9,253 argmax predictions and reduced cross-entropy; the latest step improved it from 0.2125958950 to 0.2125956753, so the locally improving direction remains upward.
result: training did not finish within the verification time limit



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
