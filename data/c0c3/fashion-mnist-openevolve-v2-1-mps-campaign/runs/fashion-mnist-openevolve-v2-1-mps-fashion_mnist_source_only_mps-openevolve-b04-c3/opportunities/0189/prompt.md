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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.2084908329416, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031356811523, "validation_score": 9251.414404599938}
prior_hypothesis: A 75.2265625% EMA / 24.7734375% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.36513733305037, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.2065504592895508, "validation_score": 9251.414404549889}
prior_hypothesis: A 75.2/24.8 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065509983062744.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 73.41271320800297, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.2065509983062744, "validation_score": 9251.414404364758}
prior_hypothesis: A 75.1/24.9 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.20655155067443848.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.50678191613406, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655032272338866, "validation_score": 9251.414404596793}
prior_hypothesis: A 75.225% EMA / 24.775% live blend will preserve 9,251 correct predictions while lowering cross-entropy below 0.2065504592895508.

## Recent verification evidence

RECENT RESULT
hypothesis: A 75.1/24.9 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.20655155067443848.
change: Reduce the live-model validation contribution from 25% to 24.9%, leaving training, runtime, parameters, flip ensembling, and logit scaling unchanged.
mechanism: Ultra-local EMA–endpoint blend calibration
evidence_used: The 75/25 blend achieved 9,251 correct, while 77.5/22.5 lowered cross-entropy to 0.20653982543945312 but lost two correct predictions; this small lower-side step tests for tie-break improvement while minimizing the risk of crossing a prediction boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 73.41271320800297, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.2065509983062744, "validation_score": 9251.414404364758}

RECENT RESULT
hypothesis: A 75.2/24.8 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065509983062744.
change: Increase the EMA validation contribution from 75% to 75.2%, reduce the live contribution to 24.8%, and use the calibration verified with the improved 75.1/24.9 blend.
mechanism: Ultra-local EMA–endpoint blend refinement
evidence_used: The 75.1/24.9 blend preserved 9,251 correct while improving cross-entropy to 0.2065509983062744; 77.5/22.5 further lowered cross-entropy but lost two correct predictions, motivating a small continuation toward EMA rather than a large step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.2/24.8 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065509983062744.
change: Increase the EMA validation contribution from 75.0% to 75.2%, reduce the live contribution to 24.8%, and use the 1.04953 calibration associated with the best verified blend.
mechanism: Ultra-local EMA–endpoint blend continuation
evidence_used: Moving from 75/25 to 75.1/24.9 preserved 9,251 correct and reduced cross-entropy from 0.20655155067443848 to 0.2065509983062744; the prior 75.2/24.8 attempt timed out without contradictory validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.36513733305037, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.2065504592895508, "validation_score": 9251.414404549889}

RECENT RESULT
hypothesis: Replacing the shared 38-dimensional prediction bottleneck with a 29-dimensional nonlinear context path plus a direct class-specific spatial path will exceed 9,251 correct predictions, or tie while lowering cross-entropy, under the verified EMA ensemble.
change: Reallocate the classifier parameters into complementary contextual and direct spatial-logit paths, zero-initialize the direct residual for stable optimization, and retain the verified 75.2/24.8 EMA/live flip ensemble and calibration.
mechanism: Residual class-specific spatial template head
evidence_used: Existing designs devote about 73% of learned parameters to a single shared bottleneck, while blend and temperature refinements plateaued at 9,251 correct. The hybrid head tests a different prediction mechanism without the runtime cost that caused added high-resolution blocks and attention to time out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.3/24.7 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065504592895508.
change: Increase the EMA validation contribution from 75.2% to 75.3% and reduce the live contribution from 24.8% to 24.7%, leaving training and calibration unchanged.
mechanism: Ultra-local EMA–endpoint blend continuation
evidence_used: Successive moves from 75.0/25.0 to 75.1/24.9 and 75.2/24.8 each preserved 9,251 correct while lowering cross-entropy; the next equally sized step directly tests whether that improvement continues before the known accuracy loss at 77.5/22.5.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 74.78508037491702, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20654992179870604, "validation_score": 9250.414404734496}

RECENT RESULT
hypothesis: A 75.25/24.75 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065504592895508.
change: Increase the EMA validation contribution from 75.1% to 75.25% and reduce the live contribution from 24.9% to 24.75%, leaving training and calibration unchanged.
mechanism: Boundary-seeking EMA–endpoint blend refinement
evidence_used: The 75.2/24.8 blend retained 9,251 correct with lower cross-entropy than 75.1/24.9, while 75.3/24.7 lost one correct prediction; their midpoint probes the accuracy boundary for further tie-break improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 65.65688662510365, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655018920898438, "validation_score": 9250.414404642652}

RECENT RESULT
hypothesis: A 75.225% EMA / 24.775% live blend will preserve 9,251 correct predictions while lowering cross-entropy below 0.2065504592895508.
change: Move the validation ensemble halfway from the best verified 75.2/24.8 blend toward the accuracy-losing 75.25/24.75 blend, and restore its verified 1.04953 calibration.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.2/24.8 blend achieved 9,251 correct at 0.2065504592895508 cross-entropy, while 75.25/24.75 lowered cross-entropy but lost one correct prediction; their midpoint is the most informative probe of the prediction boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.50678191613406, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655032272338866, "validation_score": 9251.414404596793}

RECENT RESULT
hypothesis: A 75.2375% EMA / 24.7625% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.
change: Add the verified 0.99 parameter-and-buffer EMA procedure and flip-ensembled EMA/live probability blend, then test the midpoint between the successful 75.225% and accuracy-losing 75.25% EMA weights with 1.04953 calibration.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.225/24.775 blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.25/24.75 lost one correct prediction despite lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.06137579097413, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.2065502586364746, "validation_score": 9250.414404618805}

RECENT RESULT
hypothesis: A 75.23125% EMA / 24.76875% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.
change: Move the validation probability ensemble to the midpoint between the successful 75.225% EMA weight and the accuracy-losing 75.2375% weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.225% EMA blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.2375% lowered cross-entropy to 0.2065502586364746 but lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.23125% EMA / 24.76875% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.
change: Move the validation probability ensemble to the midpoint between the successful 75.225% EMA blend and the accuracy-losing 75.2375% blend, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.225% blend retained 9,251 correct at 0.20655032272338866 cross-entropy, while 75.2375% reduced cross-entropy but lost one correct prediction; the prior midpoint attempt timed out and supplied no contradictory validation result.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.7268764169421, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655028762817382, "validation_score": 9250.414404608848}

RECENT RESULT
hypothesis: A 75.228125% EMA / 24.771875% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.
change: Move the validation probability blend to the midpoint between the successful 75.225% EMA weight and the accuracy-losing 75.23125% weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.225% blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.23125% reduced cross-entropy but lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 67.90749216708355, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655030632019042, "validation_score": 9250.414404602428}

RECENT RESULT
hypothesis: A 75.2265625% EMA / 24.7734375% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.
change: Move the validation probability blend halfway between the successful 75.225% EMA weight and the accuracy-losing 75.228125% weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.225% blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.228125% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.2084908329416, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031356811523, "validation_score": 9251.414404599938}



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
