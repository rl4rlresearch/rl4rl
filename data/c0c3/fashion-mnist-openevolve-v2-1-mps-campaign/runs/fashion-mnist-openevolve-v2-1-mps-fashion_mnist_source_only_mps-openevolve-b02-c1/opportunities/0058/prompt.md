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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 76.45461608306505, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19714545440673828, "validation_score": 9286.417660191717}
prior_hypothesis: Interpolating the final parameters halfway toward a sparse mean of the last 20% of training will exceed 9,280 correct predictions while retaining some of the observed cross-entropy benefit of tail averaging.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging only the final 10% of training will exceed 9,286 correct predictions by reducing bias from earlier, higher-learning-rate parameters while retaining late-trajectory smoothing.
change: Start sparse tail averaging at 90% rather than 80%, preserving the sampling interval and 50:50 final blend.
mechanism: Narrow late-trajectory weight averaging
evidence_used: Half-strength averaging over the final 20% improved validation_correct from 9,280 to 9,286, while full averaging lost a prediction, suggesting smoothing helps but excessive trajectory bias hurts.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.15 to 0.18 will exceed 9,286 correct predictions by modestly strengthening regularization without the optimization risk of a larger increase.
change: Raise the existing classifier dropout probability to 0.18 while leaving the architecture, optimizer, augmentation, TTA, and tail averaging unchanged.
mechanism: Conservative classifier-dropout increase
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, directly showing that weaker classifier regularization hurts; the previously proposed 0.20 setting timed out, motivating a conservative untested increase.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 75% toward the sparse tail mean will retain at least 9,286 correct predictions while reducing validation cross-entropy below 0.197145.
change: Increase the final parameter interpolation coefficient from 0.5 to 0.75.
mechanism: Stronger sparse tail interpolation
evidence_used: Half-strength averaging achieved 9,286 correct, while full averaging lost only one prediction but improved cross-entropy to 0.19631; testing the midpoint between them targets better calibration with less boundary drift than full averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 68.2550341670867, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.1970211971282959, "validation_score": 9276.417703547104}

RECENT RESULT
hypothesis: Softening the evaluation ensemble with temperature 1.05 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Divide the final evaluation logits by 1.05 without changing training, parameters, augmentation, averaging, or class predictions.
mechanism: Post-ensemble temperature calibration
evidence_used: The current implementation has the best observed correct count, while stronger tail interpolation slightly reduced cross-entropy but damaged accuracy; temperature scaling targets only the tie-break metric and leaves every argmax unchanged.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the fixed flattened MLP with a lightweight relational token head will exceed 9,286 correct predictions because class-specific queries can select contextualized image parts instead of relying on one static mixing of absolute spatial cells.
change: Preserve the proven convolutional backbone, augmentation, optimizer, TTA, and tail averaging, but contextualize the final nine spatial features with self-attention and compute each class logit from a learned query attending to those relational features.
mechanism: Relational spatial tokens with class-query attention
evidence_used: Parameter-averaging adjustments plateaued at 9,286 correct and stronger averaging fell to 9,276, while prior backbone and fixed-pooling changes hurt. This challenges the remaining load-bearing assumption—the flattened first-order prediction head—without disturbing the backbone, and uses only nine spatial tokens to avoid the covariance branch’s expensive computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising classifier dropout from 0.15 to 0.17 will exceed 9,286 correct predictions by modestly improving regularization without materially slowing or destabilizing fixed-exposure optimization.
change: Increase the existing classifier dropout probability to 0.17 while preserving the architecture, optimizer, augmentation, TTA, and tail averaging.
mechanism: Conservative classifier-dropout strengthening
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, directly showing that weaker classifier regularization hurts; larger increases only timed out, so a conservative nearby increase remains untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Slightly soften evaluation logits without changing training, parameters, runtime, or predicted classes.
mechanism: Mild post-ensemble temperature calibration
evidence_used: The current design has the best correct count, while stronger tail averaging lowered cross-entropy but damaged accuracy; positive temperature scaling targets only the tie-break metric. The prior 1.05 attempt timed out, leaving this conservative setting untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Divide the final evaluation ensemble logits by 1.03; training and predicted classes remain unchanged.
mechanism: Mild post-ensemble temperature calibration
evidence_used: Stronger tail averaging reduced cross-entropy to 0.197021 but lost predictions, so calibration should target the tie-break metric without moving decision boundaries; the prior 1.03 attempt timed out and remains untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Vectorizing the unchanged 12-weight TTA ensemble and applying temperature 1.03 will preserve all 9,286 argmax predictions, reduce validation cross-entropy below 0.197145, and avoid the recurrent verification timeouts.
change: Evaluate the ten augmented views in two batched backbone calls instead of ten sequential calls, then divide the ensemble logits by 1.03.
mechanism: Vectorized temperature-calibrated test-time augmentation
evidence_used: The current design has the best observed correct count, while prior temperature-1.03 attempts timed out without testing accuracy; positive temperature scaling preserves argmax exactly, and batching the existing views reduces evaluation overhead without changing their weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Pairwise TTA batching will avoid the timeout of five-view batches while preserving the 9,286 predictions, and temperature 1.03 will lower cross-entropy without changing argmax classes.
change: Evaluate the unchanged weighted 12-view ensemble in batches of at most two views, then divide its logits by 1.03.
mechanism: Memory-bounded pairwise TTA with temperature calibration
evidence_used: The sequential ensemble achieved the best 9,286 correct, while the prior five-view vectorization timed out; smaller batches retain its computation with lower peak memory, and positive temperature scaling preserves argmax exactly.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending BatchNorm running statistics with their sparse tail averages will exceed 9,286 correct predictions by aligning evaluation normalization with the already beneficial 50:50 tail-averaged parameters.
change: Track BatchNorm running means and variances at every existing tail sample and blend them 50:50 with their tail averages at the final step.
mechanism: Tail-consistent BatchNorm state averaging
evidence_used: Half-strength parameter averaging improved validation_correct from 9,280 to 9,286, but the current implementation leaves BatchNorm running statistics at their final-iterate values, creating a normalization mismatch with the blended weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Apply temperature 1.03 to the existing sequential 12-weight TTA ensemble without changing training or evaluation workload.
mechanism: Mild post-ensemble temperature calibration
evidence_used: The current design has the best observed correct count; stronger tail averaging reduced cross-entropy but changed decision boundaries, while prior temperature attempts timed out and therefore provide no accuracy or calibration evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 79.8308060830459, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19794324645996095, "validation_score": 9286.41738204333}



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
