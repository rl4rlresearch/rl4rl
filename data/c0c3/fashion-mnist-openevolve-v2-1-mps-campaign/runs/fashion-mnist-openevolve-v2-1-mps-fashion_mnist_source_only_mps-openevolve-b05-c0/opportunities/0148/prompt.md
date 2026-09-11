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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.17715312493965, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201698150634765, "validation_score": 9260.412535474032}
prior_hypothesis: An evaluation-logit multiplier of 1.4164 will preserve all 9,260 correct predictions while lowering validation cross-entropy below 0.2120170723.

## Recent verification evidence

RECENT RESULT
hypothesis: An evaluation-logit multiplier of 1.41635 will preserve all 9,260 correct predictions while lowering validation cross-entropy below 0.2120169815.
change: Change only the positive inference-time logit multiplier from 1.4164 to 1.41635.
mechanism: Analytic validation-temperature refinement
evidence_used: The verified multipliers 1.40, 1.417, and 1.45 define a convex cross-entropy curve whose quadratic minimum is approximately 1.41635; the successful 1.4164 result closely confirms that estimate, and positive scaling preserves every argmax prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 78.31807733303867, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201698265075683, "validation_score": 9260.412535473642}

RECENT RESULT
hypothesis: Lowering label smoothing from 0.05 to 0.03 will increase validation_correct above 9,260 by strengthening correct-class gradients while retaining moderate regularization.
change: Reduce only the training loss label-smoothing coefficient; preserve the verified EMA, BatchNorm-buffer mixture, and 1.4164 evaluation calibration.
mechanism: Reduced label smoothing for stronger class separation
evidence_used: The 1.4164 multiplier minimizes cross-entropy only by sharpening logits and cannot change predictions, while BatchNorm and reflection changes lost accuracy; the optimal multiplier above 1 also indicates underconfident logits, motivating less target smoothing as an orthogonal accuracy-focused change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering label smoothing from 0.05 to 0.03 will increase validation_correct above 9,260 by strengthening correct-class gradients while retaining moderate regularization.
change: Reduce only the training loss label-smoothing coefficient; preserve the verified EMA, BatchNorm-buffer mixture, and 1.4164 evaluation calibration.
mechanism: Reduced label smoothing for stronger class separation
evidence_used: The optimal evaluation multiplier above 1 indicates underconfident logits, while calibration cannot alter predictions. The previous 0.03 attempt timed out and therefore provides no contradictory validation evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 73.38936833408661, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.21358166885375976, "validation_score": 9242.412003586436}

RECENT RESULT
hypothesis: Training with independent random translations of up to two pixels will increase validation_correct above 9,260 by improving local translation invariance without imposing the harmful reflection invariance observed previously.
change: Replace the unchanged training inputs with vectorized per-image random crops from two-pixel reflection padding; preserve the verified optimizer, EMA, BatchNorm interpolation, label smoothing, and evaluation calibration.
mechanism: Reflection-padded random translation augmentation
evidence_used: Horizontal-reflection augmentation reduced validation_correct from 9,260 to 9,223, showing that orientation-changing invariance is harmful; small translations are a less semantic, orthogonal augmentation aimed at accuracy rather than calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Independent random translations of up to two pixels will increase validation_correct above 9,260 by improving local translation invariance without imposing the harmful reflection invariance.
change: Replace unchanged training inputs with vectorized per-image random crops from two-pixel reflection padding.
mechanism: Reflection-padded random translation augmentation
evidence_used: Horizontal reflections reduced validation_correct to 9,223, while the prior translation attempt timed out and supplied no contradictory accuracy evidence; translations preserve image semantics better than reflections.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 76.21448441618122, "validation_accuracy": 0.9104, "validation_correct": 9104, "validation_cross_entropy": 0.24583653106689454, "validation_score": 9104.401336762514}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.05 to 0.07 will exceed 9,260 correct predictions by improving generalization during the fixed two-pass exposure budget.
change: Increase only the training loss label-smoothing coefficient while preserving the verified optimizer, EMA, BatchNorm interpolation, and evaluation calibration.
mechanism: Stronger label-smoothing regularization
evidence_used: Reducing label smoothing symmetrically from 0.05 to 0.03 lowered validation_correct from 9,260 to 9,242, indicating that weaker target regularization is harmful and motivating a controlled test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 71.71565187489614, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.21435780296325685, "validation_score": 9244.411740262038}

RECENT RESULT
hypothesis: Adding a learned residual path across the two 14×14 convolutional layers will exceed 9,260 correct predictions by improving feature and gradient propagation within the fixed two-pass training budget.
change: Replace the two later sequential convolutions with an equivalent residual block using a 1×1 BatchNorm projection, while preserving all training, EMA, calibration, and batch-size settings.
mechanism: Projected residual convolutional block
evidence_used: Both directions of label-smoothing adjustment and both tested input augmentations reduced validation correctness; this motivates an orthogonal architecture-level optimization that preserves spatial semantics and remains below the parameter ceiling.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 240930, "training_seconds": 86.1797984589357, "validation_accuracy": 0.9246, "validation_correct": 9246, "validation_cross_entropy": 0.22015029296875, "validation_score": 9246.409785583695}

RECENT RESULT
hypothesis: Applying 10% dropout to the parameter-dominant classifier hidden layer will increase validation_correct above 9,260 by reducing co-adaptation without imposing harmful image-space invariances.
change: Add training-only dropout after the classifier’s hidden GELU while preserving the architecture size, optimizer, EMA, label smoothing, and evaluation calibration.
mechanism: Mild dense-head dropout regularization
evidence_used: Both label-smoothing directions reduced correctness, translation and reflection augmentation were harmful, and the projected residual block reached only 9,246; mild parameter-free regularization of the large dense head is an orthogonal, localized test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 77.17444075015374, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21971884651184082, "validation_score": 9237.409930535574}

RECENT RESULT
hypothesis: Reallocating parameters from the flattening dense head into a 96-channel learned spatial downsampling stage will exceed 9,260 correct predictions by extracting stronger local features within the fixed exposure budget.
change: Add a stride-2 convolutional stage after the existing feature extractor and resize the dense head, increasing learned parameters from 239,634 to 249,808 while preserving training and calibration settings.
mechanism: Learned convolutional downsampling with capacity rebalancing
evidence_used: Dense-head dropout reduced correctness to 9,237 and a projected residual rewiring reached only 9,246; since augmentation and label-smoothing changes also failed, redistributing the parameter-dominant dense capacity into additional spatial feature learning is the most distinct architecture test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 96-channel spatial downsampling stage and resized dense head will exceed 9,260 correct predictions by allocating more capacity to local feature extraction while remaining below 250,000 parameters.
change: Add a stride-2 48→96 convolution after the existing feature extractor and change the classifier to a 1,536→114→10 head, yielding 249,808 learned parameters.
mechanism: Learned convolutional downsampling with capacity rebalancing
evidence_used: The previous verification of this architecture timed out without validation evidence; because label-smoothing, augmentation, dropout, and residual-block changes all reduced correctness, retrying this distinct capacity redistribution remains the most informative next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the classifier hidden layer from 88 to 92 units will exceed 9,260 correct predictions by increasing capacity while remaining below 250,000 parameters and avoiding expensive spatial computation.
change: Expand both dense-head dimensions consistently, increasing learned parameters from 239,634 to 249,086 without changing training dynamics.
mechanism: Parameter-budget dense-head widening
evidence_used: The 249,808-parameter convolutional redistribution repeatedly timed out, while dense-head dropout reduced correctness to 9,237; using the remaining parameter budget to add deterministic head capacity is a lower-cost alternative that preserves every activation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249086, "training_seconds": 71.2331101668533, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2205847106933594, "validation_score": 9237.409639737103}

RECENT RESULT
hypothesis: Holding the peak learning rate for the first 20% of training before cosine decay will exceed 9,260 correct predictions by preventing premature optimization slowdown within the fixed two-pass exposure budget.
change: Replace immediate cosine decay with a 20%-hold, 80%-cosine schedule while preserving the optimizer, EMA, architecture, and evaluation calibration.
mechanism: Short-horizon delayed cosine decay
evidence_used: Label-smoothing changes, augmentations, dropout, residual rewiring, and head widening all reduced validation correctness; the learning-rate schedule remains an untested, compute-neutral optimization lever.
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
