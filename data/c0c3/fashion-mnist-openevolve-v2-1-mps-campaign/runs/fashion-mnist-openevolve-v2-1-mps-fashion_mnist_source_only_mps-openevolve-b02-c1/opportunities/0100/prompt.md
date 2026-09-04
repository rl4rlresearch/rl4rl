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

RECENT RESULT
hypothesis: Covering all 7×7 terminal features with overlapping 3×3 pooling will achieve at least 9,287 correct predictions by eliminating the bottom/right feature loss of the current 2×2 pooling.
change: Replace only the final 7×7-to-3×3 max-pool with symmetric overlapping windows; parameter count and the proven training/TTA procedure remain unchanged.
mechanism: Symmetric overlapping terminal pooling
evidence_used: TTA reweighting plateaued at 9,286 correct and larger representation changes timed out, motivating a negligible-cost architectural correction: the current final pool discards one full feature row and column.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Covering all 7×7 terminal features with overlapping 3×3 pooling will achieve at least 9,287 validation-correct predictions, while batched probability-space TTA will complete within the verification limit.
change: Replace the final lossy 2×2 pool with a 3×3 stride-2 pool and evaluate the nine additional TTA views in two batched forward passes without changing fusion weights or temperature.
mechanism: Symmetric overlapping terminal pooling with batched TTA
evidence_used: The prior terminal-pooling test timed out without testing its accuracy claim; batched probability TTA previously completed in 71.8 seconds while preserving all 9,286 baseline predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 67.50699508283287, "validation_accuracy": 0.9234, "validation_correct": 9234, "validation_cross_entropy": 0.21175304641723633, "validation_score": 9234.412625329458}

RECENT RESULT
hypothesis: Reducing tail-average interpolation from 0.5 to 0.25 will reverse the boundary drift observed at 0.75 and achieve at least 9,287 correct predictions; batching the nine auxiliary views into two memory-bounded calls will complete within the time limit.
change: Give the late-checkpoint average 25% final weight and replace nine sequential auxiliary-view evaluations with batches of four unflipped and five flipped views while preserving fusion order, weights, and temperature.
mechanism: Reduced tail averaging with bounded batched TTA
evidence_used: Increasing tail interpolation to 0.75 reduced validation-correct from 9,286 to 9,276, directly motivating movement below 0.5. Earlier 0.25 attempts timed out without accuracy evidence, while bounded batched probability TTA has completed successfully.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing late-checkpoint averaging will reverse the accuracy drift observed when its weight increased and achieve at least 9,287 correct predictions, while bounded batched TTA preserves the proven ensemble within the time limit.
change: Keep the final optimizer iterate instead of blending checkpoint averages, and evaluate the nine auxiliary views in two memory-bounded batched calls while preserving fusion order, weights, and temperature.
mechanism: Final-iterate training with bounded batched probability TTA
evidence_used: Increasing tail-average weight from 0.5 to 0.75 reduced validation-correct from 9,286 to 9,276, motivating movement fully toward the final iterate; batched probability TTA previously completed while retaining all 9,286 predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Padding translated views with each image’s observed background level instead of normalized zero will exceed 9,286 correct predictions by removing artificial border strips from both training augmentation and TTA.
change: Subtract each image’s minimum intensity before padding and restore it afterward, so translations use the image’s background value without assuming a particular normalization.
mechanism: Normalization-aware background translation
evidence_used: Temperature and TTA-weight tuning plateaued at 9,286 correct, while altered terminal pooling regressed to 9,234; this motivates correcting the existing augmentation itself without changing model capacity, view weights, or runtime materially.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing normalized-zero translation borders with edge-replicated borders will achieve at least 9,287 validation-correct predictions without the overhead of per-image background estimation.
change: Use replication padding for both training translations and evaluation TTA while preserving the architecture, optimizer, view weights, and temperature.
mechanism: Edge-replicated translation padding
evidence_used: The background-aware padding hypothesis remained untested because verification timed out; replication padding addresses the same artificial-border issue with a native, lower-overhead operation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 60.663319458952174, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.19815347366333008, "validation_score": 9282.417308809756}

RECENT RESULT
hypothesis: Adding a low-cost residual spatial/channel refinement block before the unchanged terminal pool will exceed 9,286 validation-correct predictions while remaining below 250,000 parameters and adding little runtime.
change: Add a 10,464-parameter depthwise-separable residual block at the 7×7 feature stage, preserving the proven optimizer, augmentation, pooling, averaging, and TTA procedure.
mechanism: Pre-pooling residual depthwise refinement
evidence_used: Changing terminal pooling regressed from 9,286 to 9,234, while larger representation redesigns could not be verified; this tests additional representation capacity without disturbing the successful pooling path or adding a parallel branch.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding skip connections around the existing same-width convolutions will improve limited-budget optimization and achieve at least 9,287 validation-correct predictions without increasing parameters or materially affecting runtime.
change: Wrap the existing 32→32 and 64→64 convolutional refinements in residual connections while preserving all convolutional capacity, training settings, pooling, averaging, and TTA.
mechanism: Parameter-neutral residual refinement
evidence_used: The 9,286-correct baseline remains strongest, terminal-pooling changes regressed, and an added residual refinement timed out; reusing existing convolutions as residual branches isolates the optimization benefit without its additional parameters or convolutional work.
result: training did not finish within the verification time limit

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
