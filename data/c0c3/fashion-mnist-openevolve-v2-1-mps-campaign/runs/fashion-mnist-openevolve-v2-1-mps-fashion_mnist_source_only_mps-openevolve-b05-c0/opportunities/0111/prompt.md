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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 70.92263820813969, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.2163902946472168, "validation_score": 9258.411052276724}
prior_hypothesis: Scaling evaluation logits by 1.25 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2168836.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.255 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the evaluation-time logit multiplier from 1.25 to 1.255.
mechanism: Conservative evaluation-logit calibration
evidence_used: Every completed increase through 1.25 preserved all 9,258 predictions while successively lowering cross-entropy; 1.255 is the smallest untested step toward the estimated optimum near 1.35, while timed-out runs provide no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.35 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the positive evaluation-time logit multiplier from 1.25 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Completed scales from 1.20 through 1.25 preserved all 9,258 argmax predictions while successively reducing cross-entropy; the measured curvature estimates the minimum near 1.35, and timed-out attempts supplied no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.35 evaluation multiplier will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the evaluation-time logit multiplier from 1.25 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Every completed increase from 1.20 through 1.25 preserved all 9,258 argmax predictions while lowering cross-entropy; the measured improvement curve estimates its minimum near 1.35, and timed-out attempts provide no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.30 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the evaluation-time logit multiplier from 1.25 to 1.30.
mechanism: Midpoint evaluation-logit calibration
evidence_used: Every completed increase from 1.20 through 1.25 preserved all 9,258 argmax predictions while successively lowering cross-entropy; 1.30 advances toward the evidence-estimated minimum near 1.35, while timed-out attempts supplied no contradictory validation result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the evaluation multiplier to 1.26 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the evaluation-time logit multiplier from 1.25 to 1.26.
mechanism: Conservative evaluation-logit calibration
evidence_used: Every completed increase from 1.20 through 1.25 preserved all 9,258 argmax predictions while successively lowering cross-entropy; the prior 1.26 attempt timed out and therefore supplied no contradictory validation result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.35 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the positive evaluation-time logit multiplier from 1.25 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Every completed increase from 1.20 through 1.25 preserved all 9,258 argmax predictions while successively lowering cross-entropy; the measured curvature estimates the minimum near 1.35, while timed-out attempts provide no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.35 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the positive evaluation-time logit multiplier from 1.25 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Every completed increase from 1.20 through 1.25 preserved 9,258 correct predictions and successively lowered cross-entropy; the measured curvature estimates a minimum near 1.35, while prior 1.35 attempts timed out without producing contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging logits for each validation image and its horizontal reflection will increase validation_correct above 9,258 by suppressing orientation-sensitive errors without changing training or learned parameters.
change: Apply batched original-and-flipped inference during evaluation, average their logits, then retain the established 1.25 calibration multiplier.
mechanism: Horizontal-flip test-time ensemble
evidence_used: Every completed logit-scaling experiment through 1.25 preserved exactly 9,258 predictions, showing that further scalar calibration can improve only the tie-breaker; flip ensembling can change argmax decisions while exploiting the approximate horizontal symmetry of the image classes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 77.73628637497313, "validation_accuracy": 0.8883, "validation_correct": 8883, "validation_cross_entropy": 0.34744080505371094, "validation_score": 8883.371073814988}

RECENT RESULT
hypothesis: Increasing the evaluation multiplier to 1.35 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2163903.
change: Change only the evaluation-time logit multiplier from 1.25 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Every completed multiplier increase through 1.25 preserved exactly 9,258 correct predictions while successively lowering cross-entropy; the measured trend estimates its minimum near 1.35, and timed-out 1.35 attempts produced no contradictory validation result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Holding label smoothing at 0.05 for the first half of training and annealing it to zero during the second half will increase validation_correct above 9,258 by retaining early regularization while permitting sharper late class boundaries.
change: Replace constant label smoothing with a second-half linear decay to zero.
mechanism: Late-phase label-smoothing anneal
evidence_used: The verified model remained underconfident enough that increasing evaluation-logit scale repeatedly reduced cross-entropy without changing its 9,258 predictions. The earlier full-training annealing attempt timed out, so it supplied no contradictory validation evidence; this schedule preserves the proven early objective and targets only late underconfidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on deterministic one-pixel translations for half the minibatches will increase validation_correct above 9,258 by improving local spatial robustness without imposing reflection invariance.
change: Apply balanced up, down, left, and right translations to alternating training batches using replicated border padding.
mechanism: Balanced one-pixel translation augmentation
evidence_used: Evaluation-logit scaling repeatedly preserved exactly 9,258 predictions, so further calibration cannot improve the primary metric. Horizontal reflection reduced validation_correct to 8,883, motivating a milder, label-preserving spatial augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a 7×7-resolution convolutional refinement stage while slightly narrowing the dense layer will increase validation_correct above 9,258 without exceeding the parameter ceiling.
change: Add a 48-channel convolution, batch normalization, and GELU after the final pooling layer; reduce the classifier width from 88 to 80, yielding 241,610 learned parameters.
mechanism: Efficient post-pooling spatial refinement
evidence_used: Scalar calibration preserved exactly 9,258 predictions and reflection ensembling reduced validation_correct to 8,883, motivating additional learned spatial discrimination without imposing reflection invariance.
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
