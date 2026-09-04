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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 79.54549562511966, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19609205169677735, "validation_score": 9286.418028026597}
prior_hypothesis: Temperature 0.912 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960927105.

## Recent verification evidence

RECENT RESULT
hypothesis: Fusing compressed 7×7 contour features with 3×3 semantic features will exceed 9,286 correct predictions while remaining under 250,000 parameters.
change: Replace the single endpoint classifier with learned fine-detail and deep-semantic branches, concatenating their embeddings for class prediction; retain the proven training and TTA procedures.
mechanism: Multi-resolution contour–semantic late fusion
evidence_used: Temperature calibration plateaued at 9,286 correct, while deeper residual refinements timed out. This challenges the load-bearing assumption that all evidence must pass through three pooling operations, adding a distinct representation path with only modest low-cost computation (243,898 parameters).
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Vectorizing the existing translated and flipped views will create enough runtime margin to test the previously untested 2.25 center weighting, which will exceed 9,286 correct predictions by reducing boundary-artifact influence.
change: Evaluate the ten distinct TTA views in two batched passes, weight both centered views at 2.25, normalize by 12.5, and retain the proven 0.912 temperature.
mechanism: Batched center-biased test-time augmentation fusion
evidence_used: The 0.912 baseline achieved 9,286 correct in 79.5 seconds, while three center-weighting attempts timed out without contradicting their accuracy hypothesis; batching equivalent view computations directly addresses that runtime failure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 80.26498845894821, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.19606860160827635, "validation_score": 9281.418036222443}

RECENT RESULT
hypothesis: Reducing both centered-view weights from 2.0 to 1.75 will reverse the five-prediction loss observed at weight 2.25 and exceed 9,286 validation-correct predictions.
change: Downweight the original and horizontally flipped centered views and normalize the ensemble by its new total weight of 11.5.
mechanism: Mild translated-view emphasis in test-time augmentation
evidence_used: Increasing centered-view weights to 2.25 reduced validation-correct from 9,286 to 9,281; testing the symmetric adjustment in the opposite direction is the most direct low-cost follow-up.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting both centered views at 1.75 will reverse the five-prediction loss seen at 2.25 and exceed 9,286 correct predictions; batching the ten distinct views will allow this previously timed-out hypothesis to complete.
change: Vectorize evaluation into two five-view model passes, downweight both centered views from 2.0 to 1.75, and normalize the ensemble by 11.5 while retaining temperature 0.912.
mechanism: Batched translated-view-emphasized test-time augmentation
evidence_used: Center weight 2.25 reduced validation-correct from 9,286 to 9,281, motivating the opposite adjustment; the 1.75 attempt timed out, while batched TTA completed successfully for the 2.25 experiment.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.36499479110353, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19612666091918945, "validation_score": 9286.418015931202}

RECENT RESULT
hypothesis: Temperature 0.912028 will preserve all 9,286 predictions while reducing cross-entropy below 0.19609205169677735, and vectorized view evaluation will prevent timeout.
change: Evaluate the ten existing weighted views in two batched model passes while preserving their fusion order, then apply temperature 0.912028.
mechanism: Batched test-time augmentation with interpolated temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.91213 locate the cross-entropy minimum near 0.912028; batched TTA previously completed in 73.36 seconds, whereas unbatched calibration attempts timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 71.7999381669797, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19609205627441406, "validation_score": 9286.418028024998}

RECENT RESULT
hypothesis: Averaging view logits instead of probabilities will exceed 9,286 correct predictions by suppressing isolated, overconfident view errors while preserving the training-matched 2:1 center-to-translation weighting.
change: Vectorize the existing ten views into two additional model passes and replace probability-mixture fusion with weighted logit averaging at the proven 0.912 temperature.
mechanism: Weighted logit-space TTA consensus
evidence_used: Center weights 1.75 and 2.25 failed to improve the 9,286-correct baseline, indicating weight tuning has plateaued; vectorized evaluation completed in 71.8 seconds, making the fusion rule the next low-cost, argmax-changing variable.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighted logit averaging will exceed 9,286 correct predictions by suppressing isolated overconfident view errors, while evaluating all ten views in exactly two batched passes will avoid the prior timeout.
change: Replace ten sequential evaluation passes and probability-space fusion with two five-view batched passes and 2:1 center-weighted logit averaging at temperature 0.912.
mechanism: Two-pass weighted logit-space TTA consensus
evidence_used: Center-weight tuning plateaued at 9,286 correct, and batched probability TTA completed in 71.8 seconds; the subsequent logit-space hypothesis timed out without testing its accuracy claim.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighted logit averaging will exceed 9,286 correct predictions by suppressing isolated overconfident view errors, while processing all nine remaining views in one batched forward pass will avoid the timeouts that prevented testing this fusion rule.
change: Reuse the centered-view logits, batch the other nine translated/flipped views into one model call, and combine logits with the training-matched 2:1 center weighting at temperature 0.912.
mechanism: Single-batch weighted logit-space TTA consensus
evidence_used: Probability-space weight changes plateaued at 9,286 correct, while both logit-space attempts timed out; batched probability TTA completed in 71.8 seconds, motivating more aggressive vectorization of the still-untested fusion rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the tail-average interpolation from 0.5 to 0.75 will exceed 9,286 correct predictions by reducing final-update variance without increasing parameters or computation.
change: Give the arithmetic average of late-training checkpoints 75% rather than 50% weight in the final model.
mechanism: Stronger late-training weight averaging
evidence_used: Symmetric TTA reweighting plateaued at 9,286 correct or regressed to 9,281, while representation changes repeatedly timed out; adjusting the existing tail average changes the learned decision boundaries at negligible runtime cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 62.173414249904454, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.1959786865234375, "validation_score": 9276.418067650899}

RECENT RESULT
hypothesis: Reducing tail-average interpolation to 0.25 will retain variance reduction while reversing the boundary drift seen at 0.75, yielding at least 9,287 correct predictions.
change: Give the late-training checkpoint average 25% rather than 50% weight in the final model.
mechanism: Reduced late-checkpoint averaging
evidence_used: Increasing interpolation from 0.5 to 0.75 lowered validation-correct from 9,286 to 9,276 despite improving cross-entropy, indicating that movement toward the tail average harms the primary accuracy objective and motivating a step in the opposite direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing lossy max-pooling and the positional dense head with lossless pixel rearrangement, residual feature learning, and coarse spatial-pyramid pooling will achieve at least 9,287 validation-correct predictions within the time limit.
change: Move capacity from the flattening MLP into two residual stages, preserve pixels during downsampling with PixelUnshuffle, classify pooled global and quadrant features, and batch the existing TTA views.
mechanism: Lossless space-to-depth residual pyramid classifier
evidence_used: The 9,286-correct baseline assumes repeated max-pooling is sufficient, while the multi-resolution alternative timed out after adding a second branch. This replacement tests preserved spatial evidence without adding a parallel computational path and uses 233,194 parameters with fewer convolutional operations than the baseline.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing tail-average interpolation from 0.5 to 0.25 will reverse the accuracy loss seen at 0.75 and achieve at least 9,287 correct predictions; two-pass batched TTA will let the previously timed-out test complete.
change: Evaluate the ten unchanged weighted views in two batched passes and reduce final interpolation toward the late-checkpoint average to 25%.
mechanism: Reduced tail averaging with batched probability-space TTA
evidence_used: Increasing tail interpolation to 0.75 reduced correct predictions from 9,286 to 9,276, motivating movement in the opposite direction; batched probability TTA previously completed in 71.8 seconds with all 9,286 predictions preserved.
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
