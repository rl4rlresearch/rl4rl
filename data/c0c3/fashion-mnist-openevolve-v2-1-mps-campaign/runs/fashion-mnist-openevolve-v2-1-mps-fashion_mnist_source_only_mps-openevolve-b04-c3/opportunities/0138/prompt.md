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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 72.46243020799011, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.2065652229309082, "validation_score": 9251.414399479198}
prior_hypothesis: Restoring the verified 75/25 EMA–live ensemble and increasing its inference scale to 1.04 will preserve exactly 9,251 correct predictions while lowering cross-entropy below 0.20657955322265625.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.0855537080206, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655190887451172, "validation_score": 9251.414404052011}
prior_hypothesis: Increasing the positive inference scale from 1.044 to 1.048 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.20655614318847657.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.7220510840416, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.99391554109752, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20657955322265625, "validation_score": 9251.414394557461}
prior_hypothesis: Blending the stronger 0.99 EMA prediction with 25% of the final live-model prediction will exceed 9,247 correct predictions, or tie while lowering validation cross-entropy.

## Recent verification evidence

RECENT RESULT
hypothesis: Evaluating the qualified 0.99 EMA will reproduce Reference Design 3’s improvement from 9,243 to at least 9,247 correct predictions.
change: Add non-persistent EMA state, swap it in for validation, and update it after every optimizer step while preserving the calibrated flip ensemble.
mechanism: 0.99-decay exponential moving-average evaluation
evidence_used: Reference Design 3 achieved the best verified result—9,247 correct and 0.206655 cross-entropy—using this exact EMA implementation; later unverifiable attempts provide no contradictory performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Evaluating the qualified 0.99 EMA will reproduce Reference Design 3’s 9,247 correct predictions and improve over the current 9,243.
change: Add non-persistent EMA state, swap it in during validation, update it after every optimizer step, and restore the verified 1.03592 logit scale.
mechanism: 0.99-decay EMA evaluation with calibrated flip ensembling
evidence_used: Reference Design 3 achieved the best verified score—9,247 correct with 0.206655 cross-entropy—using this exact EMA configuration; unverifiable attempts provide no contradictory performance result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact qualified Reference Design 3 implementation will improve validation_correct from 9,243 to at least 9,247 and reduce validation cross-entropy.
change: Track non-persistent EMA copies of parameters and BatchNorm state after every optimizer step, swap them in during validation, and use the verified power-mean and logit calibration.
mechanism: 0.99-decay exponential moving-average evaluation with calibrated flip ensembling
evidence_used: Reference Design 3 achieved the strongest verified result—9,247 correct and 0.206655 cross-entropy—while the current non-EMA design achieved 9,243 correct; unverifiable repetitions supplied no contradictory performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Blending the stronger 0.99 EMA prediction with 25% of the final live-model prediction will exceed 9,247 correct predictions, or tie while lowering validation cross-entropy.
change: During validation, compute calibrated flip ensembles for both EMA and live parameters, then combine their class probabilities with 0.75/0.25 weights.
mechanism: EMA–endpoint snapshot probability ensemble
evidence_used: The 0.99 EMA improved validation_correct from 9,243 to 9,247, while calibration changes alone could not alter rankings; snapshot blending extends the only successful ranking-changing mechanism without changing training or learned-parameter count.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.99391554109752, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20657955322265625, "validation_score": 9251.414394557461}

RECENT RESULT
hypothesis: Replacing precise 7×7 flattening with joint 3×3 mean/max descriptors and widening the head from 38 to 105 units, while retaining the qualified EMA–endpoint ensemble, will exceed 9,251 correct predictions or tie it with lower cross-entropy.
change: Replace the assumption that most capacity should encode exact final-grid coordinates with translation-tolerant coarse spatial evidence. The unchanged stem feeds adaptive mean/max pooling, a 105-wide classifier, and the verified 0.99 EMA plus 25% live-model probability blend. The model has 248,599 learned parameters.
mechanism: Coarse-grid mean/max evidence pooling with a wider nonlinear head
evidence_used: Calibration-only changes plateaued at 9,243 correct, whereas EMA–endpoint blending reached 9,251. The earlier horizontal-pooling design timed out after also adding residual computation; this patch keeps the convolutional workload unchanged and uses approximately the same dense multiply count as the current head, isolating the alternative representation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding per-image ±1-pixel translations to the qualified EMA–endpoint design will exceed 9,251 correct predictions, or tie while lowering cross-entropy below 0.20657955322265625.
change: Restore the verified 0.99 EMA and 75/25 EMA–live flip ensemble, then add inexpensive replicated-border translations during training.
mechanism: Mild translation augmentation with EMA–endpoint probability ensembling
evidence_used: EMA–endpoint blending produced the best verified result of 9,251 correct; translation-tolerant representations remain unverified, so mild training-time translation tests the same invariance without their architectural or inference-time cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the live-model contribution from 25% to 30% will exceed 9,251 correct predictions, or tie while reducing cross-entropy below 0.20657955322265625.
change: Restore the verified 0.99 EMA evaluation machinery and calibrated flip ensemble, then blend EMA and live probabilities at 70/30 instead of 75/25.
mechanism: Reweighted EMA–endpoint probability ensemble
evidence_used: Adding 25% live predictions to the stronger EMA increased validation_correct from 9,247 to 9,251; a cautious increase to 30% tests whether the demonstrated endpoint diversity remains underweighted.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the live-model contribution from 25% to 20% will exceed 9,251 correct predictions, or tie while lowering cross-entropy below 0.20657955322265625.
change: Change the validation ensemble from 75/25 to 80/20 EMA/live probabilities while preserving training and inference cost.
mechanism: EMA–endpoint probability blend reweighting
evidence_used: The stronger EMA alone reached 9,247 correct, while adding 25% of the weaker 9,243-correct live model raised the result to 9,251; a slightly more EMA-weighted blend tests whether 25% introduces excess endpoint error while retaining its useful diversity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing EMA decay from 0.99 to 0.995 while retaining the verified 75/25 EMA–live blend will exceed 9,251 correct predictions, or tie while lowering cross-entropy below 0.20657955322265625.
change: Add validation-time EMA state swapping and EMA–live flip-ensemble blending from Reference Design 3, but extend the EMA horizon with decay 0.995.
mechanism: Longer-horizon EMA–endpoint probability ensemble
evidence_used: The 0.99 EMA improved the live model from 9,243 to 9,247 correct, and adding 25% live predictions reached the best verified result of 9,251; a longer averaging horizon tests whether further suppression of late-training noise improves that demonstrated mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified 75/25 EMA–live ensemble and increasing its inference scale to 1.04 will preserve exactly 9,251 correct predictions while lowering cross-entropy below 0.20657955322265625.
change: Add the qualified 0.99 EMA state tracking and EMA–live flip ensemble, then recalibrate the ranking-preserving output scale from 1.03592 to 1.04.
mechanism: EMA–endpoint probability ensemble with recalibrated confidence
evidence_used: The 75/25 EMA–live ensemble achieved the best verified result of 9,251 correct, but retained the 1.03592 scale calibrated for the weaker non-ensemble predictor; a positive scale cannot alter predicted classes, so this isolates secondary cross-entropy improvement without risking validation_correct.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 72.46243020799011, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.2065652229309082, "validation_score": 9251.414399479198}

RECENT RESULT
hypothesis: Restoring the verified 75/25 EMA–live ensemble and increasing its positive inference scale from 1.04 to 1.044 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.2065652229309082.
change: Add the qualified 0.99 EMA tracking and validation-time EMA/live flip ensemble, then test a slightly higher ranking-preserving logit scale.
mechanism: EMA–endpoint probability ensemble with one-step confidence extrapolation
evidence_used: Increasing the verified ensemble’s scale from 1.03592 to 1.04 preserved 9,251 correct predictions and reduced cross-entropy from 0.20657955322265625 to 0.2065652229309082; another small increase directly tests whether confidence remains under-calibrated without changing predicted classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 81.21722341608256, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655614318847657, "validation_score": 9251.414402597693}

RECENT RESULT
hypothesis: Increasing the positive inference scale from 1.044 to 1.048 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.20655614318847657.
change: Raise only the final EMA–live ensemble logit scale, leaving training, class rankings, and parameter count unchanged.
mechanism: One-step confidence calibration extrapolation
evidence_used: Scale increases from 1.03592 to 1.04 and then 1.044 both preserved 9,251 correct predictions while successively reducing cross-entropy, indicating the ensemble remains slightly under-confident.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.0855537080206, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655190887451172, "validation_score": 9251.414404052011}



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
