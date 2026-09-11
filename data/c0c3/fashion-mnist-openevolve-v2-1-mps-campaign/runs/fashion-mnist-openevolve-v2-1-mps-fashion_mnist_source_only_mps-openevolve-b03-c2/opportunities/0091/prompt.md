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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 36.167439040960744, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21452123413085938, "validation_score": 9256.411684856508}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.33 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2147829.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.383009457960725, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2147829204559326, "validation_score": 9256.411596172107}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.30 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2152603.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.09292483399622, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144522300720215, "validation_score": 9256.411708248064}
prior_hypothesis: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.661927457898855, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445223388671875, "validation_score": 9256.411708246771}
prior_hypothesis: Scaling the ensemble logits by 1.356 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.214452343.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.35 will retain exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2145212.
change: Increase the post-ensemble logit multiplier from 1.21 to 1.35 while leaving training and ensemble weighting unchanged.
mechanism: Near-optimal argmax-preserving temperature sharpening
evidence_used: Scaling from 1.30 to 1.33 preserved all 9,256 correct predictions and lowered cross-entropy from 0.2147829 to 0.2145212; the sharply diminishing gains place the likely calibration optimum near 1.35.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 75.47665804205462, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445588340759278, "validation_score": 9256.41170700956}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.355 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2144559.
change: Replace the current 1.18 post-ensemble multiplier with the estimated calibration optimum of 1.355.
mechanism: Quadratic-fit ensemble temperature calibration
evidence_used: Cross-entropy improved from 0.2147829 at 1.30 to 0.2145212 at 1.33 and 0.2144559 at 1.35; the diminishing gains imply a local quadratic minimum near 1.355, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 65.28117383294739, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445234298706053, "validation_score": 9256.411708209786}

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.356 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.214452343.
change: Increase the post-ensemble logit multiplier from 1.355 to 1.356.
mechanism: Fine-grained convex temperature calibration
evidence_used: Cross-entropy fell from 0.214455883 at 1.35 to 0.214452343 at 1.355; local finite-difference curvature estimates place the minimum near 1.356, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.661927457898855, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445223388671875, "validation_score": 9256.411708246771}

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.
change: Increase the post-ensemble logit multiplier from 1.35 to 1.35605 while leaving training and ensemble weighting unchanged.
mechanism: Fine-grained quadratic temperature calibration
evidence_used: Cross-entropy values at multipliers 1.35, 1.355, and 1.356 imply a local quadratic minimum near 1.356046; all tested positive scalings preserved exactly 9,256 correct predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.09292483399622, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144522300720215, "validation_score": 9256.411708248064}

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy from 0.2145212341 to approximately 0.2144522301.
change: Replace the current 1.33 post-ensemble multiplier with the best verified multiplier, leaving training and ensemble weighting unchanged.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 directly verified 1.35605 with 9,256 correct predictions and the best available cross-entropy, 0.2144522300720215.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing the current 1.30 multiplier with 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy from 0.2147829205 to approximately 0.2144522301.
change: Change only the post-ensemble logit multiplier from 1.30 to the best verified value, 1.35605.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 directly verified 1.35605 with 9,256 correct predictions and the lowest available cross-entropy, 0.2144522300720215; the earlier unverifiable attempt does not contradict that successful result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the central-view weight from 3.25 to 3.5 will improve the net validation-correct count above 9,256 by emphasizing the unshifted view, which dominates training exposure.
change: Increase the center-to-shift ensemble ratio to 3.5:1 and update its normalization constant, preserving the best verified temperature.
mechanism: Center-weighted test-time ensemble refinement
evidence_used: Temperature scaling from 1.03 through 1.35605 never changed the 9,256 predictions and has reached its cross-entropy optimum; changing relative ensemble weights is the smallest available edit capable of improving the primary correct-count objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 63.054180917097256, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2143974189758301, "validation_score": 9253.411726830267}

RECENT RESULT
hypothesis: Increasing the multiplier from 1.356 to 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.
change: Replace only the post-ensemble logit multiplier with the best verified value.
mechanism: Verified fine-grained ensemble temperature calibration
evidence_used: Reference Design 3 verified multiplier 1.35605 at 9,256 correct and cross-entropy 0.2144522300720215, a strict improvement over the current 1.356 result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the center-view weight from 3.25 to 3.0 will increase validation-correct above 9,256 by correcting the directional harm observed when it was raised to 3.5; the verified 1.35605 temperature will improve calibration without changing argmax predictions.
change: Set the center-to-shift ensemble ratio to 3.0:1, update normalization from 14.5 to 14.0, and use the best verified post-ensemble multiplier of 1.35605.
mechanism: Counter-directional center-weight ensemble refinement
evidence_used: Raising the center weight to 3.5 reduced validation-correct from 9,256 to 9,253, motivating exploration in the opposite direction; independently, multiplier 1.35605 achieved the best verified cross-entropy at the baseline 3.25 weighting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the post-ensemble multiplier to 1.35605 will preserve all 9,256 correct predictions and reduce validation cross-entropy from 0.2147829205 to approximately 0.2144522301.
change: Replace only the evaluation-time ensemble logit multiplier, leaving training and ensemble weighting unchanged.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 directly verified 1.35605 with 9,256 correct predictions and the best available cross-entropy, 0.2144522300720215; prior positive scaling preserved argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the center-view weight from 3.25 to 3.125 will correct at least one ensemble error and raise validation-correct above 9,256 while retaining most of the center view’s stabilizing influence.
change: Set the center-to-shift ensemble ratio to 3.125:1 and update its normalization constant, preserving the best verified temperature.
mechanism: Counter-directional center-weight refinement
evidence_used: Raising the center weight to 3.5 reduced validation-correct from 9,256 to 9,253; testing a smaller counter-directional step is the most direct probe of whether lower center weighting improves the primary objective, while avoiding the unverified larger move to 3.0.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 56.139978625113145, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21448392181396483, "validation_score": 9254.41169750461}

RECENT RESULT
hypothesis: Increasing the multiplier to 1.35605 will preserve all 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.
change: Change only the evaluation-time ensemble logit multiplier from 1.356 to 1.35605.
mechanism: Verified fine-grained ensemble temperature calibration
evidence_used: Reference Design 3 directly verified 1.35605 with 9,256 correct predictions and cross-entropy 0.2144522300720215, a strict improvement over the current 1.356 result.
result: the implementation could not be verified



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
