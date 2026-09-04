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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 72.222799875075, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20007479591369629, "validation_score": 9319.416640697482}
prior_hypothesis: Dividing evaluation logits by 0.7382 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.2000748046875.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing label smoothing from 0.04 to 0.03 will exceed 9,319 correct predictions by strengthening decision margins while retaining most of the baseline’s regularization.
change: Use fixed 0.03 label smoothing while preserving the verified architecture, optimizer, schedule, and 0.80 evaluation temperature.
mechanism: Moderately reduced label smoothing
evidence_used: Evaluation temperatures from 0.95 through 0.80 preserved exactly 9,319 predictions while progressively lowering cross-entropy, indicating underconfident logits; the 0.02 smoothing attempt timed out without contrary accuracy evidence, motivating a more conservative intermediate reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging calibrated probabilities for each validation image and its horizontal reflection will exceed 9,319 correct predictions by adding reflection invariance without the padding artifacts that caused the translation ensemble to lose 19 correct predictions.
change: During evaluation only, combine the original and horizontally reflected predictions in probability space while retaining the verified 0.80 temperature; training remains unchanged.
mechanism: Two-view horizontal-reflection probability ensemble
evidence_used: The three-view translation ensemble fell from 9,319 to 9,300 correct, implicating shifted-boundary artifacts, while the reflection-training attempt timed out without testing its accuracy; an evaluation-only reflection ensemble isolates the proposed semantic invariance without altering training time or learned parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.775 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20153608.
change: Reduce the evaluation-only temperature from 0.80 to 0.775 without changing training, runtime, or learned parameters.
mechanism: Evaluation-only temperature calibration
evidence_used: Temperatures from 0.95 through 0.80 preserved exactly 9,319 correct predictions while progressively lowering cross-entropy; lower-temperature attempts timed out without contrary calibration evidence, so 0.775 tests the midpoint of the most promising unverified interval.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 75.71691079088487, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20059862747192383, "validation_score": 9319.416458913545}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.77 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20059863.
change: Reduce the evaluation-only temperature from 0.775 to 0.77 without changing training, parameters, or argmax predictions.
mechanism: Evaluation-only temperature calibration
evidence_used: Sharpening from 0.95 through 0.775 consistently preserved 9,319 correct predictions while lowering cross-entropy; the prior 0.77 verification timed out and supplied no contrary calibration evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 65.27861908287741, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20046691131591796, "validation_score": 9319.416504607738}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.765 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20046691.
change: Reduce the evaluation-only logit temperature from 0.77 to 0.765 without changing training, parameters, runtime, or argmax predictions.
mechanism: Evaluation-only temperature calibration
evidence_used: Reducing temperature from 0.775 to 0.77 preserved all 9,319 correct predictions and lowered cross-entropy from 0.20059863 to 0.20046691, supporting another conservative sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.06888399994932, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.2003540313720703, "validation_score": 9319.416543775364}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.74 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20035403.
change: Reduce the evaluation-only logit temperature from 0.765 to 0.74 without changing training, parameters, or argmax predictions.
mechanism: Quadratic-guided evaluation temperature calibration
evidence_used: Temperatures 0.775, 0.77, and 0.765 preserved all 9,319 correct predictions while lowering cross-entropy to 0.20059863, 0.20046691, and 0.20035403; their diminishing improvements estimate the local cross-entropy minimum near 0.738.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.62205670913681, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.200076131439209, "validation_score": 9319.416640233816}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.738 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20007613.
change: Reduce the evaluation-only logit temperature from 0.74 to 0.738 without changing training, parameters, or argmax predictions.
mechanism: Quadratic-refined evaluation temperature calibration
evidence_used: Temperatures from 0.775 through 0.74 preserved all 9,319 correct predictions while progressively lowering cross-entropy; the measured local trend estimated its minimum near 0.738.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 83.52612749999389, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.2000748046875, "validation_score": 9319.416640694435}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.7382 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.2000748046875.
change: Reduce the evaluation-only temperature from 0.74 to 0.7382 without changing training, parameters, or runtime.
mechanism: Quadratic-refined evaluation temperature calibration
evidence_used: Temperature 0.738 preserved all 9,319 predictions and reduced cross-entropy to 0.2000748046875 despite timing out; together with the 0.74 result, the local calibration trend places the minimum near 0.7382.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 72.222799875075, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20007479591369629, "validation_score": 9319.416640697482}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.73816 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20007479591369629.
change: Refine the evaluation-only temperature from 0.7382 to 0.73816 without changing training, parameters, runtime, or predicted classes.
mechanism: Local quadratic evaluation-temperature refinement
evidence_used: Temperatures 0.738, 0.7382, and 0.74 produced cross-entropies 0.2000748046875, 0.20007479591369629, and 0.200076131439209; a local quadratic fit places the minimum near 0.73816, and temperature scaling preserves argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.738156 will preserve all 9,319 argmax predictions while reducing validation cross-entropy below 0.20007479591369629.
change: Refine the evaluation-only temperature from 0.7382 to 0.738156 without changing training, parameters, or computational cost.
mechanism: Quadratic-optimal evaluation temperature calibration
evidence_used: Temperatures 0.738, 0.7382, and 0.74 yielded cross-entropies 0.2000748046875, 0.20007479591369629, and 0.200076131439209; their quadratic fit places the minimum near 0.738156. The prior 0.73816 run timed out but supplied no contrary accuracy or calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.738156 will preserve all 9,319 predictions while reducing validation cross-entropy below 0.20007479591369629.
change: Refine the evaluation-only temperature from 0.7382 to 0.738156 without affecting training, parameters, or computational cost.
mechanism: Quadratic-optimal evaluation-temperature calibration
evidence_used: Temperatures 0.738, 0.7382, and 0.74 yielded cross-entropies 0.2000748046875, 0.20007479591369629, and 0.200076131439209; their quadratic fit places the minimum near 0.738156. Earlier verification attempts at this refinement timed out but provided no contrary accuracy or calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decaying label smoothing from 0.04 to zero will exceed 9,319 correct predictions by retaining early regularization while strengthening class margins late in training.
change: Replace fixed 0.04 label smoothing with a linear decay over the fixed training exposure; preserve the architecture, optimizer, runtime profile, and calibrated evaluation temperature.
mechanism: Linearly annealed label smoothing
evidence_used: The model required sharpening to temperature 0.7382 while preserving every argmax, indicating underconfident logits. Fixed smoothing reductions timed out without contrary accuracy evidence, motivating a no-overhead schedule that reduces smoothing conservatively.
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
