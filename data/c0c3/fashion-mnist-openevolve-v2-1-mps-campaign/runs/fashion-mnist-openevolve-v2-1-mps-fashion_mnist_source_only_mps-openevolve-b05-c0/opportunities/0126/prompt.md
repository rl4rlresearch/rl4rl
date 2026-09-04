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

RECENT RESULT
hypothesis: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by emphasizing better-converged late states while retaining noise suppression.
change: Increase only the EMA interpolation coefficient from 0.03 to 0.04.
mechanism: Faster late-training exponential weight averaging
evidence_used: Increasing EMA from 0.01 to 0.02 and then 0.03 added one correct prediction at each step and reduced cross-entropy; the prior 0.04 verification timed out without contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by tracking later training states more closely while retaining noise suppression.
change: Increase only the EMA interpolation coefficient from 0.03 to 0.04.
mechanism: Faster late-training exponential weight averaging
evidence_used: EMA increases from 0.01 to 0.02 and 0.03 each added one correct prediction and reduced cross-entropy; two 0.04 attempts timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by tracking later, better-converged states while retaining noise suppression.
change: Increase only the EMA interpolation coefficient from 0.03 to 0.04.
mechanism: Faster late-training exponential weight averaging
evidence_used: Increasing EMA from 0.01 to 0.02 and then 0.03 added one correct prediction at each step and reduced cross-entropy; prior 0.04 attempts timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the EMA coefficient from 0.03 to 0.035 will produce more than 9,260 correct predictions by tracking later training states more closely without discarding the noise suppression that improved prior EMA variants.
change: Increase only the EMA interpolation coefficient from 0.03 to the intermediate value 0.035.
mechanism: Fine-grained late-training exponential weight averaging
evidence_used: Moving the coefficient from 0.01 to 0.02 and then 0.03 added one correct prediction at each step and reduced cross-entropy; because repeated 0.04 attempts timed out without accuracy evidence, 0.035 is the most informative finer-grained continuation of the verified trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 76.98259279201739, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21254101028442382, "validation_score": 9257.412357186899}

RECENT RESULT
hypothesis: An EMA coefficient of 0.0325 will exceed 9,260 correct predictions by tracking slightly later states than the best 0.03 design without the accuracy loss observed at 0.035.
change: Increase only the EMA interpolation coefficient from 0.03 to 0.0325.
mechanism: Fine-grained EMA boundary search
evidence_used: EMA coefficients 0.01, 0.02, and 0.03 progressively improved validation_correct to 9,260, while 0.035 fell to 9,257; testing the midpoint isolates the apparent optimum boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the halfway-state BatchNorm buffer rewind from 25% to 10% will increase validation_correct above 9,260 by better aligning normalization statistics with the late-state EMA weights.
change: Keep the best verified 0.03 parameter EMA and reduce only the final interpolation toward halfway-training floating-point buffers.
mechanism: Late-state BatchNorm alignment
evidence_used: EMA coefficients 0.01, 0.02, and 0.03 successively improved validation_correct by favoring later parameter states, while 0.035 reduced accuracy; this motivates preserving 0.03 and independently shifting the fixed BatchNorm-buffer mixture toward later statistics.
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
