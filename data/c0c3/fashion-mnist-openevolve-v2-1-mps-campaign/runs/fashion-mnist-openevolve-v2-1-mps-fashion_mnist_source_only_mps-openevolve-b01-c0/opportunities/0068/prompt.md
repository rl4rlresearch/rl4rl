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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 59.543149082921445, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249243392944336, "validation_score": 9290.415803032012}
prior_hypothesis: A temperature of 0.71715 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924347.

## Recent verification evidence

RECENT RESULT
hypothesis: An ensemble temperature of 0.70 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.70.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while lowering cross-entropy to 0.2174928, 0.2119528, 0.2076125, and 0.2045404; the diminishing improvements extrapolate to a calibration minimum near 0.70, while positive logit scaling preserves argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.70 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2045404.
change: Scale the final arithmetic-ensemble log-probabilities by 1/0.70 instead of 1/0.80.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while cross-entropy decreased with diminishing gains; the measured curve estimates its minimum near 0.70, and the prior 0.70 run timed out without contradicting that hypothesis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.76 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to the untested near-optimal value 0.76.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy with diminishing gains, placing the apparent calibration minimum near 0.75–0.76; previous nearby attempts timed out without producing contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 69.88909733411856, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20305419273376465, "validation_score": 9290.415608875328}

RECENT RESULT
hypothesis: An ensemble temperature of 0.74 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2030542.
change: Reduce the final arithmetic-ensemble temperature from 0.76 to 0.74.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Temperatures from 0.95 through 0.76 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; the diminishing improvements place the calibration minimum near 0.73–0.74, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 78.94788929098286, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20265436897277833, "validation_score": 9290.415747044954}

RECENT RESULT
hypothesis: An ensemble temperature of 0.72 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2026544.
change: Reduce the final arithmetic-ensemble temperature from 0.74 to 0.72.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Temperature reductions through 0.74 preserved exactly 9,290 correct predictions and monotonically reduced cross-entropy; the smaller improvement from 0.76 to 0.74 suggests the calibration minimum is nearby, motivating a conservative 0.02 step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.73 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2026544.
change: Reduce the final arithmetic-ensemble temperature from 0.74 to 0.73.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Reductions through 0.74 preserved exactly 9,290 correct predictions and monotonically lowered cross-entropy; improvements are diminishing toward an estimated minimum near 0.72, while the 0.72 timeout supplied no contrary accuracy or calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.735 will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.2026544.
change: Reduce the final arithmetic-ensemble temperature from 0.74 to the untested midpoint 0.735.
mechanism: Fine-grained decision-preserving temperature calibration
evidence_used: Moving from 0.76 to 0.74 preserved exactly 9,290 correct predictions and lowered cross-entropy from 0.2030542 to 0.2026544; 0.73 timed out without contrary performance evidence, so its midpoint with the best verified temperature is the most conservative informative refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A temperature of 0.723 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2026544.
change: Reduce the final arithmetic-ensemble temperature from 0.74 to the evidence-fitted estimate 0.723.
mechanism: Local-quadratic ensemble temperature calibration
evidence_used: Cross-entropy fell from 0.2045404 at 0.80 to 0.2030542 at 0.76 and 0.2026544 at 0.74; fitting the diminishing local improvements estimates the calibration minimum near 0.723, while positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.94347249995917, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20250318489074706, "validation_score": 9290.415799314533}

RECENT RESULT
hypothesis: A temperature of 0.717 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2025032.
change: Reduce the final arithmetic-ensemble temperature from 0.723 to 0.717.
mechanism: Local-quadratic ensemble temperature calibration
evidence_used: Cross-entropy improved from 0.2030542 at 0.76 to 0.2026544 at 0.74 and 0.2025032 at 0.723; the declining finite-difference slope places the local quadratic minimum near 0.717, while positive logit scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.663398292148486, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249244232177735, "validation_score": 9290.415803029111}

RECENT RESULT
hypothesis: A temperature of 0.7171 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2024924423.
change: Increase the final arithmetic-ensemble temperature from 0.717 to 0.7171.
mechanism: Sub-millistep ensemble temperature calibration
evidence_used: Cross-entropy values at temperatures 0.740, 0.723, and 0.717 fit a local quadratic whose estimated minimum is approximately 0.7171; positive temperature scaling preserves every argmax prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 67.66197170806117, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249243469238282, "validation_score": 9290.415803031749}

RECENT RESULT
hypothesis: A temperature of 0.7172 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924347.
change: Increase the final arithmetic-ensemble temperature from 0.7171 to 0.7172.
mechanism: One-sided micro-bracketing temperature calibration
evidence_used: Increasing temperature from 0.7170 to 0.7171 preserved all predictions and reduced cross-entropy by 7.63e-9; combined with the broader local curvature, this places the estimated minimum near 0.71717 and motivates testing the adjacent upper point.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A temperature of 0.71715 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924347.
change: Increase the final arithmetic-ensemble temperature from 0.7171 to 0.71715.
mechanism: Midpoint micro-bracketing temperature calibration
evidence_used: Raising the temperature from 0.7170 to 0.7171 reduced cross-entropy by 7.63e-9 without changing any prediction, while the fitted minimum is approximately 0.71717; 0.71715 is a conservative midpoint below the timed-out 0.7172 trial.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 59.543149082921445, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249243392944336, "validation_score": 9290.415803032012}



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
