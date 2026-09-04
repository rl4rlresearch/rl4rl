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

RECENT RESULT
hypothesis: Reducing the halfway-state BatchNorm buffer rewind from 25% to 10% will exceed 9,260 correct predictions by aligning normalization statistics more closely with the late-state EMA weights.
change: Preserve the best verified 0.03 parameter EMA and reduce only the final BatchNorm-buffer interpolation toward halfway-training statistics.
mechanism: Late-state BatchNorm alignment
evidence_used: The 0.03 EMA achieved the best verified result of 9,260 correct; increasing it to 0.035 fell to 9,257, so independently refining buffer alignment is more informative. The prior 0.10 buffer attempt timed out and supplied no contradictory validation evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 74.80157445906661, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21200638999938964, "validation_score": 9253.412539079105}

RECENT RESULT
hypothesis: Reducing the halfway-state BatchNorm buffer rewind from 0.25 to 0.20 will retain at least 9,260 correct predictions while lowering validation cross-entropy below 0.2125679.
change: Keep the best verified 0.03 parameter EMA and change only the final floating-point buffer interpolation from 0.25 to 0.20.
mechanism: Conservative BatchNorm buffer alignment
evidence_used: The 0.25 buffer mixture achieved 9,260 correct, while 0.10 reduced cross-entropy to 0.2120064 but lost seven correct predictions; 0.20 conservatively tests whether part of that cross-entropy gain can be captured without the accuracy loss.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the halfway-state BatchNorm buffer rewind from 0.25 to 0.225 will retain at least 9,260 correct predictions while lowering validation cross-entropy below 0.2125679.
change: Preserve the best verified 0.03 parameter EMA and change only the final floating-point buffer interpolation coefficient from 0.25 to 0.225.
mechanism: Conservative BatchNorm buffer interpolation
evidence_used: The 0.25 mixture achieved 9,260 correct, while 0.10 lowered cross-entropy to 0.2120064 but lost seven correct predictions; 0.225 is a conservative step toward the cross-entropy-improving direction, and the uncompleted 0.20 attempt provides no contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the halfway-state buffer interpolation from 0.25 to 0.30 will exceed 9,260 correct predictions by better aligning BatchNorm statistics with the temporally averaged parameters.
change: Preserve the best verified 0.03 parameter EMA and increase only the final floating-point buffer interpolation coefficient from 0.25 to 0.30.
mechanism: Slightly earlier BatchNorm-statistics alignment
evidence_used: The verified 0.25 mixture achieved 9,260 correct, whereas moving toward later buffers with 0.10 lost seven correct despite lower cross-entropy; because correct predictions dominate the objective, testing a conservative move in the opposite direction is the most informative next change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 76.50680629210547, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21286784744262696, "validation_score": 9255.412246067084}



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
