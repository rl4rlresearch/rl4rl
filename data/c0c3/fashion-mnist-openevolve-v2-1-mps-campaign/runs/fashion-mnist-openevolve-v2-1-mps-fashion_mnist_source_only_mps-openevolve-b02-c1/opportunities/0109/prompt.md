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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 61.48627037508413, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.19316529846191408, "validation_score": 9310.419053420883}
prior_hypothesis: Widening the inexpensive dense head to 147 units and normalizing its activations will exceed 9,286 validation-correct predictions while staying below the 250,000-parameter ceiling and verification time limit.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing the lossy positional dense head with a covariance-based second-order branch plus a 2×2 spatial branch will exceed 9,286 validation-correct predictions while remaining within the time and parameter limits.
change: Preserve the terminal 7×7 feature map, classify learned channel co-occurrences through normalized covariance pooling, and combine them with a compact coarse-layout MLP; training, augmentation, averaging, and TTA remain unchanged.
mechanism: Centered bilinear covariance pooling with coarse spatial classification
evidence_used: Overlapping terminal max-pooling regressed to 9,234 and TTA tuning plateaued at 9,286, challenging the assumption that stronger decisions come from the same lossy 3×3 positional representation. This 238,132-parameter alternative tests quadratic feature relationships without adding convolutional branches or their observed timeout risk.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Widening the inexpensive dense head to 147 units and normalizing its activations will exceed 9,286 validation-correct predictions while staying below the 250,000-parameter ceiling and verification time limit.
change: Preserve the proven convolutional, pooling, training, averaging, and TTA paths; use the remaining parameter budget to widen the classifier and add BatchNorm, producing 249,601 learned parameters.
mechanism: Budget-saturating normalized classifier head
evidence_used: Changing terminal pooling regressed to 9,234, while added convolutional refinement timed out; this motivates preserving the successful feature extractor and adding low-cost capacity only in the dense head.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 61.48627037508413, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.19316529846191408, "validation_score": 9310.419053420883}

RECENT RESULT
hypothesis: Adding a direct classifier over all 7×7 terminal features while preserving the proven 3×3 positional path will achieve at least 9,311 validation-correct predictions within the parameter and time limits.
change: Move the unchanged terminal max-pool outside the feature extractor, reduce the dense head by one unit, and add its logits to a bias-free global-average classifier covering the complete pre-pool feature map.
mechanism: Full-field global-logit shortcut
evidence_used: The normalized 147-unit head improved the best result to 9,310 correct, whereas replacing its terminal pooling regressed to 9,234; an additive global shortcut preserves that successful path while cheaply recovering evidence from the row and column omitted by its 7×7-to-3×3 pool.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing tail-average weight from 0.5 to 0.25 will preserve the final iterate more strongly and achieve at least 9,311 validation-correct predictions.
change: Change only the final parameter interpolation weight, retaining the verified architecture, sequential TTA, optimizer, and runtime profile.
mechanism: Reduced tail-checkpoint interpolation
evidence_used: Raising tail-average weight to 0.75 reduced validation-correct from 9,286 to 9,276. Prior 0.25 tests were confounded by slower batched TTA and timed out, while the current sequential design completed with 9,310 correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 72.61721666692756, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.19320929565429687, "validation_score": 9307.419037969132}

RECENT RESULT
hypothesis: Applying 0.05 label smoothing will improve generalization enough to achieve at least 9,311 correct validation predictions while preserving the verified architecture and runtime profile.
change: Replace hard-label cross-entropy with mildly label-smoothed cross-entropy; leave the 249,601-parameter model, optimizer, schedule, averaging, augmentation, and TTA unchanged.
mechanism: Mild label-smoothed supervision
evidence_used: The normalized 147-unit head reached the current best 9,310 correct, while architectural changes regressed or timed out and reducing tail averaging fell to 9,307; this motivates an orthogonal, negligible-cost training-loss change that preserves every verified inference component.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering classifier dropout from 0.15 to 0.10 will achieve at least 9,311 correct predictions by allowing the newly widened head to learn more fully during the fixed two-exposure budget without materially changing runtime.
change: Reduce only the dropout probability in the verified 249,601-parameter normalized classifier head.
mechanism: Reduced dense-head regularization
evidence_used: Widening and normalizing the dense head improved validation-correct from 9,286 to 9,310, indicating useful remaining head capacity; with strong translation, flip, BatchNorm, weight decay, and checkpoint-averaging regularization already present, slightly less dropout is a targeted test of underfitting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Synchronizing BatchNorm running statistics with the existing tail-averaged parameters will achieve at least 9,311 correct validation predictions without materially increasing runtime.
change: Track floating-point BatchNorm buffers during tail sampling and blend their averages into the final model with the same 0.5 interpolation used for learned parameters.
mechanism: BatchNorm-aligned tail averaging
evidence_used: The 249,601-parameter normalized head reached 9,310 correct, and reducing tail interpolation to 0.25 fell to 9,307, showing that averaging helps; however, the current procedure averages BatchNorm parameters without their running statistics, leaving an avoidable state mismatch.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics alongside the tail-averaged parameters will exceed 9,310 correct validation predictions by eliminating the state mismatch between averaged weights and final-iterate normalization buffers.
change: Track floating-point BatchNorm buffers at each existing tail sample and blend their averages into the final model using the same 0.5 interpolation as learned parameters.
mechanism: BatchNorm-aligned tail checkpoint averaging
evidence_used: The normalized 147-unit head achieved the best result of 9,310 correct, and reducing parameter averaging to 0.25 regressed to 9,307; the prior BatchNorm-alignment verification timed out, so its accuracy effect remains untested while requiring negligible extra computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A three-parameter input-dependent channel gate after the terminal pool will achieve at least 9,311 validation-correct predictions while retaining the verified runtime profile and 249,601-parameter architecture.
change: Insert an identity-initialized global-context channel gate after the existing terminal max-pool, adding only three learned parameters and leaving training, TTA, and the normalized 147-unit head unchanged.
mechanism: Identity-initialized efficient channel attention
evidence_used: The normalized dense head produced the best result at 9,310 correct, while a 10,464-parameter spatial/channel residual block timed out; this tests adaptive channel refinement with negligible computation and without disturbing the successful pooling or classifier paths.
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
