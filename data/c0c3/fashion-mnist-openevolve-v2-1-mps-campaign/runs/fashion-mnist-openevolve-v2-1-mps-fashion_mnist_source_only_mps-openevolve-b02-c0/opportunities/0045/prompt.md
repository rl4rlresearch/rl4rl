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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.0311525000725, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.228797705078125, "validation_score": 9252.406901801602}
prior_hypothesis: Scaling the ensemble log-probabilities by 1.05 will preserve all 9,252 class predictions exactly while lowering validation cross-entropy by correcting mild underconfidence from label smoothing and probability-space view averaging.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics alongside the successful late parameter snapshots will exceed 9,252 correct predictions by evaluating the averaged weights with activation statistics from the same trajectory window.
change: Retain the model on the optimizer and jointly average/copy BatchNorm running means and variances at every existing late checkpoint.
mechanism: BatchNorm-aligned late-checkpoint averaging
evidence_used: Sparse late-checkpoint parameter averaging improved validation_correct from 9,249 to 9,252, but the current implementation leaves BatchNorm statistics at their final, unaveraged values; aligning those statistics directly refines the only recently successful mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.08310595783405, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.2346428108215332, "validation_score": 9248.4049754274}

RECENT RESULT
hypothesis: Averaging snapshots from only the final 5% of training will exceed 9,252 correct predictions by retaining the variance reduction of successful weight averaging while reducing bias from earlier parameters and mismatch with final BatchNorm statistics.
change: Move the start of sparse parameter averaging from 90% to 95% of the trajectory, preserving its cadence and all other training and inference behavior.
mechanism: Narrow late-checkpoint weight averaging
evidence_used: Final-10% parameter averaging improved validation_correct from 9,249 to 9,252, whereas also averaging BatchNorm statistics reduced it to 9,248; narrowing the parameter window is a targeted way to align averaged weights more closely with the beneficial final BatchNorm state.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.28444195794873, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2356298973083496, "validation_score": 9252.40465191162}

RECENT RESULT
hypothesis: Averaging snapshots from only the final 2.5% of training will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by further reducing mismatch between averaged parameters and final BatchNorm statistics.
change: Move the start of sparse parameter averaging from 95% to 97.5% of the trajectory while preserving its cadence and all other behavior.
mechanism: Tighter terminal-checkpoint weight averaging
evidence_used: Narrowing the averaging window from the final 10% to 5% retained 9,252 correct predictions and lowered cross-entropy from 0.2360583 to 0.2356299, while averaging BatchNorm statistics hurt accuracy; this motivates another controlled narrowing of parameter averaging alone.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging equally many snapshots over only the final 2.5% of training will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by reducing BatchNorm mismatch without sacrificing the variance reduction of the successful final-5% average.
change: Start parameter averaging at 97.5% progress and double its sampling frequency from every four steps to every two steps, preserving approximately the current number of averaged snapshots.
mechanism: Dense terminal-checkpoint weight averaging
evidence_used: Narrowing averaging from the final 10% to 5% retained 9,252 correct and reduced cross-entropy from 0.2360583 to 0.2356299; the prior 2.5% attempt retained the four-step cadence and timed out, so preserving snapshot count is the most targeted next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.79407591698691, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.23564663162231445, "validation_score": 9247.404646431434}

RECENT RESULT
hypothesis: Giving later snapshots progressively more weight across the successful final-5% window will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by reducing mismatch with final BatchNorm statistics without discarding the earlier snapshots whose removal hurt accuracy.
change: Replace uniform arithmetic snapshot averaging with a linear recency-weighted average while preserving the averaging window, cadence, architecture, and training procedure.
mechanism: Linearly recency-weighted late-checkpoint averaging
evidence_used: Uniform final-5% averaging achieved 9,252 correct with lower cross-entropy than final-10% averaging, whereas restricting averaging to the final 2.5% reduced accuracy to 9,247; gradual recency weighting provides an intermediate refinement without shortening the successful window.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging convolutional and linear parameters over the successful final-5% window while retaining final BatchNorm affine parameters will exceed 9,252 correct predictions, or tie with lower cross-entropy, by reducing checkpoint noise without averaging another component coupled to final running statistics.
change: Record non-BatchNorm trainable parameters when constructing the optimizer and restrict the existing late snapshot average to those parameters.
mechanism: BatchNorm-affine-preserving late weight averaging
evidence_used: Final-5% parameter averaging achieved 9,252 correct, while averaging BatchNorm running statistics reduced accuracy to 9,248 and shortening the averaging window to 2.5% reduced it to 9,247; preserving final BatchNorm affine parameters is a targeted intermediate treatment of the observed normalization mismatch.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 75% of the successful final-5% parameter average with 25% of the final checkpoint will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by retaining snapshot variance reduction while reducing mismatch with final BatchNorm statistics.
change: Replace full averaged-parameter substitution with a 75/25 interpolation between the late average and final trained parameters.
mechanism: Final-anchored late weight averaging
evidence_used: Uniform final-5% averaging reached 9,252 correct, while shortening the window to 2.5% fell to 9,247 and averaging BatchNorm statistics fell to 9,248; a final-checkpoint anchor tests an intermediate correction without discarding useful earlier snapshots.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the ensemble log-probabilities by 1.05 will preserve all 9,252 class predictions exactly while lowering validation cross-entropy by correcting mild underconfidence from label smoothing and probability-space view averaging.
change: Apply a positive temperature scale to the final ten-view ensemble output without changing training, parameters, runtime, or predicted classes.
mechanism: Mild post-ensemble logit sharpening
evidence_used: Final-5% averaging tied the best correct count while improving cross-entropy, demonstrating that the tie-breaker is actionable; the current label smoothing and arithmetic probability ensemble both soften confidence, motivating a conservative sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.0311525000725, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.228797705078125, "validation_score": 9252.406901801602}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.05 to 1.10 will preserve exactly 9,252 predictions while further lowering validation cross-entropy.
change: Raise the inference-only temperature scale from 1.05 to 1.10 without changing training, runtime, parameters, or class decisions.
mechanism: Incremental post-ensemble logit sharpening
evidence_used: Scaling from 1.00 to 1.05 preserved 9,252 correct predictions and reduced cross-entropy from 0.2356299 to 0.2287977, indicating remaining underconfidence and motivating another conservative step in the same direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble log-probabilities by 1.075 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2287977.
change: Increase the inference-only ensemble scale from 1.05 to 1.075, midway toward the inconclusive 1.10 attempt.
mechanism: Conservative post-ensemble logit sharpening
evidence_used: Increasing the scale from 1.00 to 1.05 preserved 9,252 correct and lowered cross-entropy from 0.2356299 to 0.2287977; the 1.10 verification timed out without performance evidence, motivating a conservative intermediate scale.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the ensemble scale from 1.05 to 1.06 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2287977.
change: Increase the inference-only positive scale applied to the ten-view ensemble log-probabilities from 1.05 to 1.06.
mechanism: Fine-grained post-ensemble logit sharpening
evidence_used: Scaling from 1.00 to 1.05 preserved all 9,252 predictions and reduced cross-entropy from 0.2356299 to 0.2287977; the larger 1.075 and 1.10 trials timed out without contrary performance evidence, motivating a smaller continuation of the successful direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble log-probabilities by 1.10 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2287977.
change: Increase the inference-only ensemble scale from 1.05 to 1.10.
mechanism: Post-ensemble logit sharpening
evidence_used: Scaling from 1.00 to 1.05 preserved 9,252 correct predictions and reduced cross-entropy from 0.2356299 to 0.2287977; the prior 1.10 run timed out and therefore supplied no contrary performance evidence.
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
