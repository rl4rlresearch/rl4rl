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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 45.43381454190239, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2067180076599121, "validation_score": 9281.414347011336}
prior_hypothesis: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while evaluation scale 1.20512 will preserve every argmax and reduce validation cross-entropy below 0.20671802101135253.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 49.8065586250741, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20682972106933595, "validation_score": 9281.414308656202}
prior_hypothesis: Label smoothing of 0.023 will exceed 9,279 correct predictions by probing near the empirical accuracy maximum implied by the verified 0.01, 0.02, and 0.03 results.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 43.95526629104279, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20675255126953124, "validation_score": 9281.414335150545}
prior_hypothesis: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while increasing the positive evaluation scale to 1.19 will retain those argmax predictions and lower cross-entropy by compensating for the softer targets.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 53.020769792143255, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20671802101135253, "validation_score": 9281.414347006752}
prior_hypothesis: Evaluation scale 1.20485 will preserve all 9,281 argmax predictions while reducing validation cross-entropy below 0.20671812667846678.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding an identity skip around the existing grouped 5×5 refinement will improve gradient flow and preserve useful third-stage features, raising validation_correct above 9,279 within the fixed two-pass exposure budget.
change: Wrap the final grouped convolution, batch normalization, and GELU in a parameter-neutral residual module; leave optimization, calibration, and example accounting unchanged.
mechanism: Residual pairwise feature refinement
evidence_used: Calibration has plateaued at 9,279 correct, while translation TTA and augmentation reduced accuracy, motivating a new prediction-changing architectural probe that retains the best-known training procedure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.089021666906774, "validation_accuracy": 0.9269, "validation_correct": 9269, "validation_cross_entropy": 0.2097046287536621, "validation_score": 9269.413324036394}

RECENT RESULT
hypothesis: Halving label smoothing from 0.02 to 0.01 will sharpen class-boundary learning within the fixed two-pass budget and increase validation_correct above 9,279.
change: Reduce only training-time cross-entropy label smoothing; preserve the architecture, optimizer, schedule, and best-known evaluation calibration.
mechanism: Reduced target smoothing
evidence_used: Evaluation calibration has plateaued at 9,279 correct, while augmentation and residual refinement reduced accuracy; the verified scale above 1.0 also indicates softened logits, motivating a conservative prediction-changing reduction in target smoothing.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 71.29811716708355, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.20767723274230956, "validation_score": 9258.414017906809}

RECENT RESULT
hypothesis: Raising label smoothing from 0.02 to 0.03 will increase validation_correct above 9,279 by extending the accuracy-improving direction observed between 0.01 and 0.02.
change: Increase training label smoothing to 0.03 and restore the best-verified evaluation scale of 1.17817.
mechanism: Increased target smoothing
evidence_used: Reducing smoothing to 0.01 lowered validation_correct from 9,279 to 9,258, making a symmetric higher-side probe the most informative next test; scale 1.17817 has the best verified baseline cross-entropy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 64.13546683313325, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.20804154930114746, "validation_score": 9274.413893048868}

RECENT RESULT
hypothesis: Label smoothing of 0.023 will exceed 9,279 correct predictions by probing near the empirical accuracy maximum implied by the verified 0.01, 0.02, and 0.03 results.
change: Increase label smoothing from 0.02 to 0.023 and restore the best-verified evaluation scale of 1.17817.
mechanism: Interpolated target smoothing
evidence_used: Smoothing 0.01 produced 9,258 correct, 0.02 produced 9,279, and 0.03 produced 9,274; quadratic interpolation of those results places the estimated maximum near 0.023, while scale 1.17817 has the best verified baseline cross-entropy.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 49.8065586250741, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20682972106933595, "validation_score": 9281.414308656202}

RECENT RESULT
hypothesis: Label smoothing of 0.0235 will exceed 9,281 correct predictions by probing the local accuracy maximum implied by the verified 0.02, 0.023, and 0.03 results.
change: Increase only training-time label smoothing from 0.023 to 0.0235, preserving the best-known architecture, optimizer, schedule, and evaluation calibration.
mechanism: Local target-smoothing refinement
evidence_used: Smoothing 0.02 produced 9,279 correct, 0.023 improved this to 9,281, and 0.03 declined to 9,274; local quadratic interpolation places the estimated maximum near 0.0235.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.36784795788117, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2090172637939453, "validation_score": 9252.413559024319}

RECENT RESULT
hypothesis: Label smoothing of 0.0225 will exceed 9,281 correct predictions by probing between the strong 0.023 result and the weaker 0.02 baseline, while avoiding the sharp degradation observed at 0.0235.
change: Increase only training-time label smoothing from 0.02 to 0.0225; preserve the architecture, optimizer, schedule, and evaluation calibration.
mechanism: Lower-side target-smoothing refinement
evidence_used: Smoothing 0.023 achieved the best result with 9,281 correct, versus 9,279 at 0.02, while 0.0235 fell to 9,252; this brackets the promising region on the lower side of 0.023.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 77.03257045778446, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.2085634147644043, "validation_score": 9274.413714327186}

RECENT RESULT
hypothesis: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while increasing the positive evaluation scale to 1.19 will retain those argmax predictions and lower cross-entropy by compensating for the softer targets.
change: Use the best-verified 0.023 training label smoothing and replace the calibration inherited from the 0.02-smoothed model with a conservative higher evaluation-only logit scale.
mechanism: Smoothing-specific logit recalibration
evidence_used: Label smoothing 0.023 achieved the best verified accuracy at 9,281 correct using scale 1.17817; because that scale was optimized around smoothing 0.02 and positive scaling cannot change predicted classes, a higher smoothing-specific calibration probe preserves the primary result while testing the tie-breaker.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 43.95526629104279, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20675255126953124, "validation_score": 9281.414335150545}

RECENT RESULT
hypothesis: Label smoothing 0.023 will reproduce 9,281 correct predictions, and evaluation scale 1.206 will lower cross-entropy below 0.20675255126953124 without changing argmax predictions.
change: Restore the best-verified label smoothing and raise its evaluation-only logit scale from 1.19 to a theoretically motivated calibration probe.
mechanism: Smoothing-adjusted logit recalibration
evidence_used: Smoothing 0.023 achieved 9,281 correct, while raising its scale from 1.17817 to 1.19 reduced cross-entropy; compensating for the 0.02-to-0.023 target-softening ratio suggests an optimum near 1.206.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.361893416848034, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20671812667846678, "validation_score": 9281.41434697047}

RECENT RESULT
hypothesis: Evaluation scale 1.20485 will preserve all 9,281 argmax predictions while reducing validation cross-entropy below 0.20671812667846678.
change: Change only the evaluation-time logit multiplier from 1.206 to the estimated calibration optimum 1.20485.
mechanism: Three-point quadratic logit recalibration
evidence_used: With label smoothing 0.023, scales 1.17817, 1.19, and 1.206 produced cross-entropies 0.2068297211, 0.2067525513, and 0.2067181267 respectively; quadratic interpolation estimates the minimum near 1.20485, and positive scaling cannot alter predicted classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 53.020769792143255, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20671802101135253, "validation_score": 9281.414347006752}

RECENT RESULT
hypothesis: Expanding each refinement group from two to four channels while slightly narrowing the dense layer will raise validation_correct above 9,281 by improving spatial feature interaction within the parameter ceiling.
change: Change the refinement convolution from 64 to 32 groups, reduce the classifier width from 128 to 123, and use the best-verified evaluation scale; the resulting model has 249,011 learned parameters.
mechanism: Broader grouped-convolution channel mixing
evidence_used: The parameter-neutral residual refinement reduced accuracy to 9,269, while smoothing and calibration searches plateaued at 9,281; reallocating capacity from the large dense layer into the existing spatial refinement tests stronger channel mixing without the unsuccessful skip connection.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249011, "training_seconds": 57.89587458292954, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.20779674530029296, "validation_score": 9268.41397693937}

RECENT RESULT
hypothesis: Global channel gating will raise validation_correct above 9,281 by adding input-dependent cross-channel interaction without disturbing the successful pairwise spatial refinement.
change: Add a parameter-efficient channel gate after pairwise refinement, narrow the dense classifier to 124 units to remain below 250,000 parameters, and use the best-verified evaluation scale of 1.20485.
mechanism: Identity-centered squeeze-and-excitation channel gating
evidence_used: Expanding the grouped 5×5 convolution reduced validation_correct to 9,268, suggesting additional local spatial mixing is unhelpful; global channel recalibration tests a distinct interaction mechanism while preserving the best 0.023-smoothed training procedure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248014, "training_seconds": 64.08124987501651, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.20357024993896483, "validation_score": 9256.415430673886}

RECENT RESULT
hypothesis: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while evaluation scale 1.20512 will preserve every argmax and reduce validation cross-entropy below 0.20671802101135253.
change: Restore the best-verified training smoothing and set the evaluation-only multiplier to the quadratic optimum estimated from the three closest calibration results.
mechanism: Local convex logit recalibration
evidence_used: Scales 1.19, 1.20485, and 1.206 produced cross-entropies 0.2067525513, 0.2067180210, and 0.2067181267 with the same 9,281 correct predictions; local quadratic interpolation places the minimum near 1.20512.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 45.43381454190239, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2067180076599121, "validation_score": 9281.414347011336}



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
