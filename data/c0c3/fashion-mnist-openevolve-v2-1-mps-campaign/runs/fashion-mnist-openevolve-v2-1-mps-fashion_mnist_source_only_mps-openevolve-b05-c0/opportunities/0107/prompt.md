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
hypothesis: Scaling evaluation logits by 1.28 will preserve exactly 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.28.
mechanism: Conservative evaluation-logit calibration
evidence_used: Completed increases from 1.20 to 1.22 and 1.24 preserved all 9,258 predictions while cross-entropy decreased from 0.2193095 to 0.2180025 and 0.2168836; 1.28 is a smaller untested step toward the quadratic-estimated minimum near 1.35, while later attempts produced no contradictory validation results.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decaying label smoothing from 0.05 to zero will increase validation_correct above 9,258 by retaining early regularization while allowing sharper class boundaries late in the fixed two-epoch training budget.
change: Replace constant label smoothing with a linear schedule that reaches zero at the final optimizer step.
mechanism: Annealed label smoothing
evidence_used: Increasing evaluation-logit scale repeatedly lowered cross-entropy without changing the 9,258 predictions, indicating underconfident outputs; annealing the training-time smoothing directly addresses that underconfidence with negligible runtime overhead while targeting the primary accuracy objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.25 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the evaluation-time logit multiplier from 1.24 to 1.25.
mechanism: Conservative evaluation-logit calibration
evidence_used: Completed increases from 1.20 to 1.22 and 1.24 preserved all 9,258 predictions while successively lowering cross-entropy; later attempts timed out without contradictory validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 70.92263820813969, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.2163902946472168, "validation_score": 9258.411052276724}

RECENT RESULT
hypothesis: Increasing the evaluation multiplier from 1.25 to 1.26 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.2163903.
change: Increase only the positive evaluation-time logit multiplier from 1.25 to 1.26.
mechanism: Conservative evaluation-logit calibration
evidence_used: Completed increases through 1.20, 1.22, 1.24, and 1.25 preserved all 9,258 predictions while successively reducing cross-entropy; timed-out 1.26 attempts provided no contradictory validation evidence.
result: training did not finish within the verification time limit

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
