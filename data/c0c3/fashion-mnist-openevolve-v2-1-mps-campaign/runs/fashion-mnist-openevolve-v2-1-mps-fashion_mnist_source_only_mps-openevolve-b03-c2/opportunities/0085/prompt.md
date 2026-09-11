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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.09292483399622, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144522300720215, "validation_score": 9256.411708248064}
prior_hypothesis: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 36.167439040960744, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21452123413085938, "validation_score": 9256.411684856508}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.33 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2147829.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.383009457960725, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2147829204559326, "validation_score": 9256.411596172107}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.30 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2152603.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.661927457898855, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445223388671875, "validation_score": 9256.411708246771}
prior_hypothesis: Scaling the ensemble logits by 1.356 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.214452343.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.18 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2198722.
change: Increase the post-ensemble logit multiplier from 1.15 to 1.18, leaving training and ensemble weighting unchanged.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Multipliers from 1.03 through 1.15 consistently preserved 9,256 correct predictions while progressively lowering cross-entropy; the latest 1.15 result reached 0.2198722, supporting another equal sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.88835083297454, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21824926071166992, "validation_score": 9256.41042503872}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.21 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2182493.
change: Increase the post-ensemble logit multiplier from 1.12 to 1.21, leaving training and ensemble weighting unchanged.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Multipliers from 1.03 through 1.18 preserved exactly 9,256 correct predictions while monotonically lowering cross-entropy; 1.18 achieved the best observed 0.2182493, and its positive marginal improvement supports one more 0.03 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 46.40492158406414, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21696340217590332, "validation_score": 9256.410858698879}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.24 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2169634.
change: Increase the post-ensemble logit multiplier from 1.09 to 1.24 while leaving training and ensemble weighting unchanged.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Multipliers from 1.03 through 1.21 preserved exactly 9,256 correct predictions and monotonically reduced cross-entropy; the latest 1.21 result achieved 0.2169634, with a positive marginal improvement supporting another 0.03 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 51.92700920905918, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21597776107788086, "validation_score": 9256.411191730642}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.27 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2159778.
change: Increase the post-ensemble logit multiplier from 1.06 to 1.27 while leaving training and ensemble weighting unchanged.
mechanism: Continued argmax-preserving temperature sharpening
evidence_used: Multipliers through 1.24 preserved exactly 9,256 correct predictions and monotonically reduced cross-entropy; 1.24 achieved the best observed 0.2159778, and its positive marginal gain motivates the next 0.03 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 64.34292887500487, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2152602668762207, "validation_score": 9256.411434499776}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.30 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2152603.
change: Increase the post-ensemble logit multiplier from 1.27 to 1.30 while leaving training and ensemble weighting unchanged.
mechanism: Continued argmax-preserving temperature sharpening
evidence_used: Multipliers from 1.03 through 1.27 preserved exactly 9,256 correct predictions and monotonically lowered cross-entropy; the latest 1.27 result reached 0.2152603, and its positive marginal improvement supports another 0.03 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.383009457960725, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2147829204559326, "validation_score": 9256.411596172107}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.33 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2147829.
change: Increase the post-ensemble logit multiplier from 1.24 to 1.33 while leaving training and ensemble weighting unchanged.
mechanism: Continued argmax-preserving temperature sharpening
evidence_used: Multipliers from 1.03 through 1.30 preserved exactly 9,256 correct predictions and monotonically lowered cross-entropy; the verified 1.30 design achieved the best result at 0.2147829, and its remaining positive marginal improvement motivates the next 0.03 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 36.167439040960744, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21452123413085938, "validation_score": 9256.411684856508}

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
