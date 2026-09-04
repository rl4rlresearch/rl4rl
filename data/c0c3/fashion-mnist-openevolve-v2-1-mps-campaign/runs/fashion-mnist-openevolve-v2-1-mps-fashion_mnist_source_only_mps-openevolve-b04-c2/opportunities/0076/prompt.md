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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.86494216718711, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177375793457, "validation_score": 9279.413695732284}
prior_hypothesis: Momentum 0.00623360271453857421875 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 46.73893504100852, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177230834961, "validation_score": 9279.413695737247}
prior_hypothesis: With the best verified batch-normalization momentum restored, evaluation scale 1.17816 will preserve all 9,279 argmax predictions and reduce validation cross-entropy below 0.20861771774291993.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 83.27274833293632, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861771774291993, "validation_score": 9279.413695739075}
prior_hypothesis: Momentum 0.00623359375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617760.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 59.43554624984972, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861771774291993, "validation_score": 9279.413695739075}
prior_hypothesis: Restoring evaluation scale 1.17817 will preserve all 9,279 argmax predictions and reduce validation cross-entropy from 0.20861771850585936 to the best verified 0.20861771774291993.

## Recent verification evidence

RECENT RESULT
hypothesis: Momentum 0.00623360271453857421875 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the midpoint between the highest verified successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.0062336025238037109375 retained 9,279 correct, while 0.0062336029052734375 produced 9,278 with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.86494216718711, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177375793457, "validation_score": 9279.413695732284}

RECENT RESULT
hypothesis: Momentum 0.006233602809906005859375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the untested midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623360271453857421875 retained 9,279 correct, while 0.0062336029052734375 produced 9,278 with substantially lower cross-entropy; bisecting that remaining boundary is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.26717204088345, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861754760742188, "validation_score": 9278.41369579731}

RECENT RESULT
hypothesis: Momentum 0.0062336027622222900390625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the untested midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623360271453857421875 retained 9,279 correct, while 0.006233602809906005859375 produced 9,278 with lower cross-entropy; bisecting this remaining boundary is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.100975041044876, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177375793457, "validation_score": 9279.413695732284}

RECENT RESULT
hypothesis: Momentum 0.00623360278606414794921875 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.2086177375793457.
change: Set all four batch-normalization momenta to the untested midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.0062336027622222900390625 retained 9,279 correct, while 0.006233602809906005859375 produced 9,278 with lower cross-entropy; bisecting this remaining boundary is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 70.7453582091257, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861754760742188, "validation_score": 9278.41369579731}

RECENT RESULT
hypothesis: Momentum 0.006233602774143218994140625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the untested midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.0062336027622222900390625 retained 9,279 correct, while 0.00623360278606414794921875 produced 9,278 with lower cross-entropy; bisecting this remaining boundary is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.087944832863286, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861754760742188, "validation_score": 9278.41369579731}

RECENT RESULT
hypothesis: Increasing the positive evaluation logit scale from 1.17817 to 1.17818 will preserve all 9,279 argmax predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Apply a minimal upward calibration probe only during evaluation; training remains unchanged.
mechanism: Prediction-invariant logit calibration
evidence_used: Batch-normalization boundary probes either retained 9,279 predictions with worse cross-entropy or lost a prediction. Positive logit scaling cannot change argmax classes, so it targets the tie-breaking cross-entropy without risking validation_correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 73.12416499992833, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861771850585936, "validation_score": 9279.413695738813}

RECENT RESULT
hypothesis: Increasing the evaluation logit scale to 1.17819 will retain all 9,279 argmax predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Increase only the evaluation-time logit multiplier from 1.17817 to 1.17819; training remains unchanged.
mechanism: Prediction-invariant logit calibration
evidence_used: The 1.17818 probe preserved 9,279 correct and reduced cross-entropy from 0.2086177345275879 to 0.20861771850585936, narrowly missing the best reference by 7.63e-10; another equal increment follows the observed improving direction without changing argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 68.73369229212403, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861773071289064, "validation_score": 9279.413695734635}

RECENT RESULT
hypothesis: Restoring momentum 0.00623359375 will retain the verified 9,279 correct predictions, while evaluation scaling at 1.17818 will preserve their argmaxes and lower cross-entropy below 0.20861771774291993.
change: Use the best verified batch-normalization momentum in all four layers and increase the evaluation-only logit multiplier from 1.17817 to 1.17818.
mechanism: Calibrated best-known batch-normalization trajectory
evidence_used: Momentum 0.00623359375 achieved the best reported 9,279-correct cross-entropy of 0.20861771774291993; separately, increasing the evaluation scale from 1.17817 to 1.17818 preserved all predictions and improved cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring momentum 0.00623359375 will reproduce 9,279 correct predictions, while evaluation scaling at 1.17818 preserves every argmax and lowers cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization layers to the best verified momentum and apply a minimally higher evaluation-only logit scale.
mechanism: Calibrated best-known batch-normalization trajectory
evidence_used: Momentum 0.00623359375 achieved the best verified 9,279-correct cross-entropy; separately, scale 1.17818 preserved 9,279 correct and improved cross-entropy. The previous combined implementation was unverifiable, so a clean exact patch is the most informative next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 68.88874012511224, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861771850585936, "validation_score": 9279.413695738813}

RECENT RESULT
hypothesis: Evaluation scale 1.1781743 will preserve all 9,279 argmax predictions and reduce validation cross-entropy below 0.20861771774291993.
change: Increase only the evaluation-time logit multiplier from 1.17817 to the estimated local cross-entropy minimum.
mechanism: Prediction-invariant quadratic logit calibration
evidence_used: With identical momentum, scales 1.17817, 1.17818, and 1.17819 produced cross-entropies 0.20861771774291993, 0.20861771850585936, and 0.20861773071289064; quadratic interpolation places the minimum near 1.1781743, while positive scaling cannot alter argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.33845341601409, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861772079467775, "validation_score": 9279.41369573803}

RECENT RESULT
hypothesis: Restoring evaluation scale 1.17817 will preserve all 9,279 argmax predictions and reduce validation cross-entropy from 0.20861771850585936 to the best verified 0.20861771774291993.
change: Decrease only the evaluation-time logit multiplier from 1.17818 to 1.17817; training remains unchanged.
mechanism: Best-known prediction-invariant logit calibration
evidence_used: With the same verified momentum 0.00623359375, scale 1.17817 achieved 9,279 correct and cross-entropy 0.20861771774291993, while 1.17818 preserved accuracy but produced the slightly worse 0.20861771850585936; the interpolated 1.1781743 probe was also worse.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 59.43554624984972, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861771774291993, "validation_score": 9279.413695739075}

RECENT RESULT
hypothesis: With the best verified batch-normalization momentum restored, evaluation scale 1.17816 will preserve all 9,279 argmax predictions and reduce validation cross-entropy below 0.20861771774291993.
change: Restore all four batch-normalization momenta to 0.00623359375 and lower the evaluation-only logit multiplier to the first untested scale below the best-known 1.17817.
mechanism: Prediction-invariant downward calibration probe
evidence_used: Momentum 0.00623359375 with scale 1.17817 achieved the best verified 9,279-correct cross-entropy. Every tested higher scale was worse, while positive scaling cannot change argmax predictions; probing 1.17816 brackets the unexplored side of the calibration minimum without risking validation_correct.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 46.73893504100852, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177230834961, "validation_score": 9279.413695737247}



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
