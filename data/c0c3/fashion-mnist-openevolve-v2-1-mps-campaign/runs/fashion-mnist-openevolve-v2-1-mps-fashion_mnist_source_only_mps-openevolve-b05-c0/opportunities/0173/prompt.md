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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 66.35910083306953, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200785331726074, "validation_score": 9260.412538581026}
prior_hypothesis: A 6.984375% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120079094.

## Recent verification evidence

RECENT RESULT
hypothesis: An evaluation-logit multiplier of 1.416408 will preserve all 9,260 predictions while lowering validation cross-entropy below 0.212016981506.
change: Refine only the evaluation-logit multiplier from 1.4164 to 1.416408, leaving training and class decisions unchanged.
mechanism: Quadratic temperature interpolation
evidence_used: Multipliers 1.4160, 1.4164, and 1.4166 produced cross-entropies 0.2120170723, 0.212016981506, and 0.212017001724 with 9,260 correct; quadratic interpolation places the local minimum near 1.416408.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.86813062499277, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201698303222657, "validation_score": 9260.412535473512}

RECENT RESULT
hypothesis: Incorporating the terminal optimizer iterate into the sparse EMA with a conservative 3% interpolation will exceed 9,260 correct predictions by capturing late decision-boundary refinement without materially reducing EMA smoothing.
change: Force one final EMA update when training completes, while preserving the architecture, optimizer, schedule, BatchNorm mixture, and evaluation calibration.
mechanism: Final-iterate EMA inclusion
evidence_used: The verified sparse EMA reaches 9,260 correct but its four-step update phase omits the final three iterates; a larger 10% terminal interpolation timed out without validation evidence, motivating this smaller runtime-neutral test.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 63.41994745796546, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201305084228517, "validation_score": 9260.41253681192}

RECENT RESULT
hypothesis: Increasing only the forced terminal EMA interpolation from 3% to 6% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120130508.
change: Use a 6% EMA interpolation for the final optimizer iterate while retaining 3% for regular quarter-rate EMA updates.
mechanism: Final-iterate EMA dose refinement
evidence_used: The newly verified 3% final-iterate inclusion improved the objective at unchanged accuracy, while the earlier 10% experiment timed out without validation evidence; 6% is a runtime-neutral intermediate test of the observed benefit.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 79.89010045793839, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200913772583008, "validation_score": 9260.412538143844}

RECENT RESULT
hypothesis: Increasing the forced terminal EMA interpolation from 6% to 9% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120091377.
change: Use a 9% EMA interpolation for the final optimizer iterate while retaining 3% for regular quarter-rate EMA updates.
mechanism: Final-iterate EMA dose continuation
evidence_used: Raising the terminal interpolation from 3% to 6% preserved 9,260 correct predictions and reduced cross-entropy from 0.2120130508 to 0.2120091377, motivating another runtime-neutral step in the same direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 7.5% terminal EMA interpolation will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120091377.
change: Increase only the forced final-iterate EMA interpolation from 6% to 7.5%; retain 3% quarter-rate EMA updates and all other training behavior.
mechanism: Intermediate final-iterate EMA interpolation
evidence_used: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct and reduced cross-entropy from 0.2120130508 to 0.2120091377. The 9% attempt timed out without contrary validation evidence, so 7.5% is a conservative intermediate continuation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising terminal EMA interpolation from 6% to 7.5% will retain 9,260 correct predictions while reducing validation cross-entropy below 0.2120091377.
change: Increase only the forced final optimizer-iterate EMA interpolation to 7.5%, preserving regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Intermediate final-iterate EMA interpolation
evidence_used: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct and improved cross-entropy from 0.2120130508 to 0.2120091377. The prior 7.5% verification timed out, providing no validation evidence against this runtime-neutral continuation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 6.75% terminal EMA interpolation will retain 9,260 correct predictions while reducing validation cross-entropy below 0.2120091377.
change: Increase only the forced final optimizer-iterate EMA interpolation from 6% to 6.75%, preserving regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Conservative final-iterate EMA continuation
evidence_used: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct predictions and reduced cross-entropy from 0.2120130508 to 0.2120091377; 6.75% is a smaller continuation than the repeatedly timed-out 7.5% attempt.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 81.00116574997082, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200816192626953, "validation_score": 9260.412538475983}

RECENT RESULT
hypothesis: A 7.125% terminal EMA interpolation will retain 9,260 correct predictions while reducing validation cross-entropy below 0.2120081619.
change: Increase only the forced final optimizer-iterate EMA interpolation from 6.75% to 7.125%, preserving regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Conservative terminal EMA interpolation refinement
evidence_used: Terminal interpolation increases from 3% to 6% to 6.75% successively lowered cross-entropy from 0.2120130508 to 0.2120091377 to 0.2120081619 without changing the 9,260 correct predictions; 7.125% is a conservative midpoint toward the repeatedly timed-out 7.5% test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 6.9375% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120081619.
change: Increase only the forced final optimizer-iterate EMA interpolation from 6.75% to 6.9375%, retaining regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Midpoint terminal-iterate EMA refinement
evidence_used: Increasing terminal interpolation from 6% to 6.75% preserved 9,260 correct and lowered cross-entropy from 0.2120091377 to 0.2120081619; 6.9375% is the midpoint toward the 7.125% attempt, whose timeout provided no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 74.93969758297317, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200790939331055, "validation_score": 9260.412538561939}

RECENT RESULT
hypothesis: A 7.03125% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120079094.
change: Increase only the forced final optimizer-iterate EMA interpolation from 6.9375% to 7.03125%, retaining regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Fine-grained terminal-iterate EMA interpolation
evidence_used: Raising terminal interpolation from 6.75% to 6.9375% preserved 9,260 correct and lowered cross-entropy from 0.2120081619 to 0.2120079094; 7.03125% is a conservative midpoint toward the unmeasured 7.125% setting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 6.984375% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120079094.
change: Increase only the forced final optimizer-iterate EMA interpolation from 6.9375% to 6.984375%, retaining regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Fine-grained terminal-iterate EMA continuation
evidence_used: Raising terminal interpolation from 6.75% to 6.9375% preserved 9,260 correct and lowered cross-entropy from 0.2120081619 to 0.2120079094; 6.984375% is the conservative midpoint toward the timed-out 7.03125% setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 66.35910083306953, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200785331726074, "validation_score": 9260.412538581026}

RECENT RESULT
hypothesis: A 7.0078125% terminal EMA interpolation will preserve 9,260 correct predictions while reducing validation cross-entropy below 0.2120078533.
change: Increase only the forced final optimizer-iterate EMA interpolation from 6.984375% to 7.0078125%, retaining regular 3% quarter-rate EMA updates and all other behavior.
mechanism: Fine-grained terminal-iterate EMA continuation
evidence_used: Raising terminal interpolation from 6.9375% to 6.984375% preserved 9,260 correct and lowered cross-entropy from 0.2120079094 to 0.2120078533; 7.0078125% is the conservative midpoint toward the timed-out 7.03125% setting.
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
