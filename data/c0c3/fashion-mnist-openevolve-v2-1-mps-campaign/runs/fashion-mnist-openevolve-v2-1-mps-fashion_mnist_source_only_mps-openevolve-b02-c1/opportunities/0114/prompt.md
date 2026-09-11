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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 78.30724849994294, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19236609153747558, "validation_score": 9311.419334299717}
prior_hypothesis: Replacing probability averaging with weighted logit averaging will suppress transformation-specific confidence outliers and achieve at least 9,311 correct validation predictions without increasing parameters or runtime.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing probability averaging with weighted logit averaging will suppress transformation-specific confidence outliers and achieve at least 9,311 correct validation predictions without increasing parameters or runtime.
change: Aggregate the unchanged twelve weighted evaluation views in logit space while preserving view weights and temperature.
mechanism: Consensus-weighted logit-domain TTA
evidence_used: The normalized 147-unit head reached 9,310 correct, while architectural additions repeatedly timed out and prior TTA weight tuning plateaued; changing only the aggregation rule tests an orthogonal, computation-neutral improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 78.30724849994294, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19236609153747558, "validation_score": 9311.419334299717}

RECENT RESULT
hypothesis: Replacing fixed 3×3 flattening with two layers of content-dependent global token interaction over the complete 7×7 feature map will exceed 9,311 correct validation predictions while remaining below 250,000 parameters.
change: Remove the terminal max-pool and dense classifier, reduce terminal width to 88 channels, and classify a learned global token after two pre-normalized self-attention blocks; retain the proven training procedure and weighted logit TTA.
mechanism: Full-resolution convolutional token transformer
evidence_used: Expanding the fixed dense head improved accuracy only to 9,310 correct and logit TTA reached 9,311, while another fixed terminal-pooling scheme regressed to 9,234. This challenges the load-bearing assumption that static pooling and flattening adequately model spatial relationships by introducing input-dependent global interactions at 241,778 parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing BatchNorm momentum to 0.01 will better align evaluation statistics with the tail-averaged parameters and achieve at least 9,312 correct predictions without increasing parameters or computation.
change: Change every convolutional and classifier BatchNorm layer from the default 0.1 momentum to 0.01; leave architecture, optimization, augmentation, averaging, and TTA unchanged.
mechanism: Tail-aligned BatchNorm statistics
evidence_used: Tail parameter averaging at weight 0.5 produced 9,311 correct, while reducing its weight regressed to 9,307; explicit buffer-averaging attempts timed out, motivating a computation-neutral way to make running statistics represent a longer, tail-aligned training window.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using unit temperature for weighted logit TTA will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.192366.
change: Increase the evaluation temperature from 0.912 to 1.0 without changing training or class decisions.
mechanism: Logit-ensemble temperature recalibration
evidence_used: Weighted logit averaging produced the current best 9,311 correct while retaining the 0.912 temperature from probability aggregation; logit averaging is typically sharper, and positive temperature scaling preserves every argmax.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using current validation-batch statistics in the classifier BatchNorm will eliminate its mismatch with tail-averaged weights and exceed 9,311 correct predictions without additional parameters or training computation.
change: Disable running-stat tracking only for the 147-unit classifier BatchNorm, preserving the verified feature extractor, optimizer, augmentation, averaging, and weighted-logit TTA.
mechanism: Evaluation-time dense-feature normalization
evidence_used: The tail-averaged normalized head achieved the best 9,311 correct, while reducing averaging regressed to 9,307; prior attempts to align BatchNorm buffers timed out, leaving the identified parameter/statistics mismatch unresolved.
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
