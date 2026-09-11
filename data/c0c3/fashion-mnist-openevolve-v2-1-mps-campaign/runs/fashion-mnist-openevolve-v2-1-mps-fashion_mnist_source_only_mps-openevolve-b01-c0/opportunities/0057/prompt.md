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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 67.89242379087955, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20454037246704102, "validation_score": 9290.415096090948}
prior_hypothesis: Reducing the ensemble temperature from 0.85 to 0.80 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2076125.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the arithmetic ensemble logits by 1/0.9 will retain the baseline’s 9,290 correct predictions while reducing validation cross-entropy below 0.2241766, thereby improving validation_score.
change: Apply temperature 0.9 to the final live/EMA multi-view ensemble without changing training, parameters, forward-pass count, or predicted classes.
mechanism: Decision-preserving post-ensemble temperature sharpening
evidence_used: Geometric aggregation reduced cross-entropy to 0.217986 but lost four correct predictions; positive temperature scaling can pursue that calibration improvement while preserving the arithmetic ensemble’s argmax decisions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing from 0.04 to 0 while retaining approximately the baseline’s 0.02 average will exceed 9,290 correct predictions by regularizing early learning while allowing sharper class separation near training’s end.
change: Replace constant 0.02 label smoothing with a linear 0.04-to-0 schedule across the fixed training exposure.
mechanism: Linearly annealed label smoothing
evidence_used: Reducing head dropout from 0.10 to 0.05 lowered validation correct from 9,290 to 9,275, indicating that regularization remains useful; scheduling smoothing preserves its overall strength while avoiding persistent late-stage target bias.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying temperature 0.95 to the arithmetic ensemble will preserve its 9,290 correct predictions while lowering cross-entropy below 0.2241766, producing a strict validation_score improvement.
change: Scale the final ensemble log-probabilities by 1/0.95 without changing training, parameters, forward-pass count, or predicted classes.
mechanism: Conservative decision-preserving temperature sharpening
evidence_used: Geometric aggregation lowered cross-entropy from 0.2241766 to 0.217986, suggesting the baseline ensemble benefits from sharper consensus; the prior temperature-0.9 attempt timed out and therefore did not test this mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 75.05371608398855, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2174928207397461, "validation_score": 9290.410680039737}

RECENT RESULT
hypothesis: Reducing the arithmetic ensemble temperature from 0.95 to 0.90 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2174928.
change: Scale the final ensemble log-probabilities by 1/0.90 instead of 1/0.95, with no training, parameter, or inference-pass changes.
mechanism: Stronger decision-preserving temperature sharpening
evidence_used: Temperature 0.95 preserved 9,290 correct predictions and reduced cross-entropy from 0.2241766 to 0.2174928; because positive scaling cannot change argmax decisions, 0.90 directly tests whether further sharpening improves the tie-break objective. Its earlier verification timed out, so the mechanism remains unevaluated.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 71.47688325005583, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.21195276412963868, "validation_score": 9290.412557332924}

RECENT RESULT
hypothesis: Reducing the arithmetic ensemble temperature from 0.90 to 0.85 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2119528.
change: Scale the final ensemble log-probabilities by 1/0.85 instead of 1/0.90, without changing training, parameters, inference passes, or predicted classes.
mechanism: Further decision-preserving temperature sharpening
evidence_used: Successive reductions from temperature 1.00 to 0.95 and then 0.90 preserved exactly 9,290 correct predictions while lowering cross-entropy from 0.2241766 to 0.2174928 and then 0.2119528, motivating one further equal-sized step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 74.4472357921768, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2076125373840332, "validation_score": 9290.414040086966}

RECENT RESULT
hypothesis: Reducing the ensemble temperature from 0.85 to 0.80 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2076125.
change: Scale the final arithmetic ensemble log-probabilities by 1/0.80 instead of 1/0.85, without changing training, parameters, inference passes, or predicted classes.
mechanism: Continued decision-preserving temperature sharpening
evidence_used: Temperatures 0.95, 0.90, and 0.85 each preserved exactly 9,290 correct predictions while successively reducing cross-entropy from 0.2174928 to 0.2119528 to 0.2076125, motivating another equal-sized step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 67.89242379087955, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20454037246704102, "validation_score": 9290.415096090948}

RECENT RESULT
hypothesis: Reducing ensemble temperature from 0.80 to 0.75 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2045404.
change: Scale the final arithmetic ensemble log-probabilities by 1/0.75 instead of 1/0.80.
mechanism: Near-optimal decision-preserving temperature sharpening
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically reducing cross-entropy; the diminishing improvements suggest 0.75 is a useful next point near the calibration optimum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.75 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.75.
mechanism: Decision-preserving calibration sharpening
evidence_used: Temperatures from 0.95 through 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; the prior 0.75 verification timed out and provided no performance evidence against the change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.75 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.75.
mechanism: Decision-preserving ensemble temperature sharpening
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; the two 0.75 attempts timed out without providing performance evidence against the change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.775 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.775.
mechanism: Intermediate decision-preserving temperature sharpening
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; 0.775 tests a smaller step toward the repeatedly timed-out 0.75 setting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.79 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.79.
mechanism: Conservative decision-preserving temperature sharpening
evidence_used: Temperatures from 0.95 through 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; repeated larger reductions timed out, so a minimal step tests whether the established calibration trend continues.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An ensemble temperature of 0.70 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.
change: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.70.
mechanism: Evidence-fitted decision-preserving temperature calibration
evidence_used: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while lowering cross-entropy to 0.2174928, 0.2119528, 0.2076125, and 0.2045404; the diminishing improvements extrapolate to a calibration minimum near 0.70, while positive logit scaling preserves argmax predictions.
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
