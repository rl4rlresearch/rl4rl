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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.69989304197952, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21256791381835938, "validation_score": 9260.412348037831}
prior_hypothesis: Raising the EMA coefficient to 0.03 will increase validation_correct above 9,259 by tracking the better-converged final training states more closely while retaining noise suppression.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Averaging the original evaluation logits with lower-weight logits from four one-pixel translations will increase validation_correct above 9,258 by reducing pooling-phase sensitivity while preserving orientation-dependent class information.
change: During evaluation only, combine the original prediction at double weight with predictions from replicated-border shifts in all four axial directions, then apply the established 1.25 calibration.
mechanism: Conservative one-pixel translation test-time ensemble
evidence_used: Horizontal-reflection ensembling reduced validation_correct to 8,883, showing orientation must be preserved; the one-pixel training-translation attempt timed out and supplied no contradictory accuracy evidence, while translation averaging can change argmax predictions without increasing training time or parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.35 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the positive evaluation-time logit multiplier from 1.25 to 1.35.
mechanism: Evaluation-logit temperature calibration
evidence_used: Every completed multiplier increase through 1.25 preserved all 9,258 argmax predictions while lowering cross-entropy; the measured trend estimates the minimum near 1.35, and timed-out 1.35 attempts produced no contradictory validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 76.67864641710185, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.2135167869567871, "validation_score": 9258.412025614622}

RECENT RESULT
hypothesis: Increasing the EMA update coefficient from 0.01 to 0.02 will raise validation_correct above 9,258 by emphasizing better-converged late checkpoints while retaining enough averaging to suppress optimizer noise.
change: Halve the effective EMA lag without changing architecture, training exposure, evaluation cost, or parameter count.
mechanism: Faster late-training exponential weight averaging
evidence_used: Evaluation-logit calibration improved only cross-entropy while preserving all 9,258 predictions, so the next change must alter learned decision boundaries; the current halfway-started EMA still averages substantially older states in a short 3,126-step run.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the EMA coefficient from 0.01 to 0.02 will raise validation_correct above 9,258 by emphasizing better-converged late checkpoints while retaining substantial noise suppression.
change: Double the EMA update coefficient without changing architecture, training exposure, evaluation cost, or parameter count.
mechanism: Faster late-training exponential weight averaging
evidence_used: Logit calibration improved cross-entropy but preserved exactly 9,258 predictions, so further gains require changed decision boundaries; the current EMA begins halfway through the short 3,126-step run and updates only every four steps, making 0.01 strongly favor older states. The prior 0.02 verification timed out and supplied no contradictory result.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 85.50181333301589, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.2126968231201172, "validation_score": 9259.412304205362}

RECENT RESULT
hypothesis: Increasing the EMA coefficient from 0.01 to 0.02 will reproduce at least 9,259 correct predictions and lower validation cross-entropy versus the current 0.01 design.
change: Double the EMA interpolation coefficient while leaving architecture, training exposure, and evaluation calibration unchanged.
mechanism: Faster late-training exponential weight averaging
evidence_used: The reported 0.02 EMA run achieved 9,259 correct predictions and 0.2126968 cross-entropy, improving on the current design’s 9,258 correct and 0.2135168 cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the EMA interpolation coefficient to 0.02 will reproduce at least 9,259 correct predictions and reduce cross-entropy below the current 0.2135168.
change: Double the EMA coefficient from 0.01 to 0.02 while preserving the architecture, schedule, exposure, and evaluation calibration.
mechanism: Faster late-training exponential weight averaging
evidence_used: The reported 0.02 EMA run achieved 9,259 correct predictions with 0.2126968 cross-entropy, improving on the current 0.01 design’s 9,258 correct and 0.2135168 cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the EMA coefficient to 0.02 will reproduce at least 9,259 correct predictions and reduce validation cross-entropy below 0.2135168.
change: Double the EMA interpolation coefficient while preserving the architecture, schedule, exposure, and evaluation calibration.
mechanism: Faster late-training exponential weight averaging
evidence_used: A completed 0.02 EMA run achieved 9,259 correct predictions and 0.2126968 cross-entropy, improving on the current 0.01 design’s 9,258 correct and 0.2135168 cross-entropy.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 84.24292566697113, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.2126968231201172, "validation_score": 9259.412304205362}

RECENT RESULT
hypothesis: Raising the EMA coefficient to 0.03 will increase validation_correct above 9,259 by tracking the better-converged final training states more closely while retaining noise suppression.
change: Increase only the EMA interpolation coefficient from 0.02 to 0.03.
mechanism: Faster late-training exponential weight averaging
evidence_used: Increasing EMA from 0.01 to 0.02 improved validation_correct from 9,258 to 9,259 and cross-entropy from 0.2135168 to 0.2126968, motivating the next controlled step in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.69989304197952, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21256791381835938, "validation_score": 9260.412348037831}

RECENT RESULT
hypothesis: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by tracking later, better-converged training states while retaining substantial noise suppression.
change: Increase only the EMA interpolation coefficient from 0.03 to 0.04.
mechanism: Faster late-training exponential weight averaging
evidence_used: Successive EMA increases from 0.01 to 0.02 and then 0.03 each added one correct prediction, reaching 9,260, while also reducing cross-entropy from 0.2135168 to 0.2125679; this motivates one more controlled step in the same direction.
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
