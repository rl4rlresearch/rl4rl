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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.8552653749939, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.1980760238647461, "validation_score": 9280.417335786746}
prior_hypothesis: A batch size of 48 will finish within the verification limit and exceed 9,238 correct predictions by providing 33% more optimizer updates than batch size 64 without the roughly doubled runtime that caused batch size 32 to time out.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,280 correct predictions by improving regularization without changing runtime or parameter count.
change: Raise only the classifier-head dropout probability, preserving the proven architecture, optimizer, schedule, augmentation, batch size, and validation ensemble.
mechanism: Moderately stronger classifier-head dropout
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating that head regularization is beneficial; a conservative increase tests the supported direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 0.05 label smoothing will exceed 9,280 correct predictions by regularizing the fixed two-pass training horizon without adding meaningful runtime.
change: Add mild label smoothing to the existing cross-entropy loss while preserving the proven architecture, batch size, optimizer, schedule, augmentation, and validation ensemble.
mechanism: Mild target-distribution smoothing
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating that weaker regularization hurts; label smoothing tests complementary regularization without the runtime risk of larger models, smaller batches, or parameter averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing each unshifted view’s ensemble weight from 2 to 3 will exceed 9,280 correct predictions by favoring the validation images’ native alignment while retaining shifted-view robustness.
change: Reweight the existing probability ensemble so centered original and flipped predictions each receive weight 3, with shifted predictions unchanged.
mechanism: Stronger center-prior test-time ensemble
evidence_used: The proven 9,280-correct design already privileges centered views over shifted views; recent training-side changes either timed out or reduced accuracy, motivating a conservative, computation-neutral refinement of that established ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 62.270978291053325, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.19783181915283204, "validation_score": 9276.417420869946}

RECENT RESULT
hypothesis: Reducing the terminal learning rate from 1.5e-4 to 3e-5 will exceed 9,280 correct predictions by stabilizing the final iterate while retaining the proven batch-48 optimization trajectory.
change: Lower the cosine schedule floor from 5% to 1% of the peak learning rate, with all other settings unchanged.
mechanism: Lower-noise cosine schedule tail
evidence_used: Sparse tail EMA reduced validation cross-entropy from 0.19808 to 0.19631 while losing only one correct prediction, indicating useful late-training variance reduction; a lower terminal learning rate targets that variance without averaging overhead or replacing the final weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding BatchNorm1d after the hidden classifier layer will exceed 9,280 correct predictions by accelerating and regularizing head optimization during the fixed two-pass exposure without materially increasing runtime.
change: Insert a 128-feature batch-normalization layer between the classifier’s first linear layer and GELU, adding only 256 learned parameters.
mechanism: Batch-normalized classifier representation
evidence_used: Reducing classifier dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, showing that the head benefits from regularization; batch normalization provides complementary regularization and better-conditioned optimization at negligible computational cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the cosine learning-rate floor from 5% to 3% will exceed 9,280 correct predictions by stabilizing the final iterate without materially reducing earlier optimization.
change: Reduce only the terminal learning rate from 1.5e-4 to 9e-5, preserving all other training and evaluation behavior.
mechanism: Conservative late-training noise reduction
evidence_used: Sparse tail EMA lowered cross-entropy from 0.19808 to 0.19631 while losing only one correct prediction, indicating beneficial late-iterate smoothing; the unscored 1% floor was more aggressive, so 3% is a conservative intermediate test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 67.0232464580331, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.1997829502105713, "validation_score": 9266.41674204481}

RECENT RESULT
hypothesis: Averaging augmented-view logits instead of probabilities will exceed 9,280 correct predictions by rewarding class agreement across views while preserving the proven center-weighted ensemble and training procedure.
change: Replace the probability-space test-time ensemble with the same 2:1 center-weighted arithmetic mean in logit space.
mechanism: Weighted logit-space test-time fusion
evidence_used: Increasing centered-view weights from 2 to 3 reduced validation_correct to 9,276, supporting the established 2:1 weighting; changing only the fusion domain is a computation-neutral test of how those views should be combined.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding identity shortcuts to the two same-width convolutional refinements will exceed 9,280 correct predictions by improving gradient flow and optimization within the fixed two-pass exposure, without materially increasing runtime or parameter count.
change: Replace the 32-channel and 64-channel Conv-BatchNorm-GELU refinements with residual blocks containing the same learned layers.
mechanism: Parameter-neutral residual feature refinement
evidence_used: Batch size 48 outperformed batch size 64 through additional optimizer updates, suggesting optimization is limiting; this change improves optimization while avoiding the extra convolutions that caused the larger residual pyramid to time out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 76.04279074980877, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.20241839981079102, "validation_score": 9266.415828633426}

RECENT RESULT
hypothesis: Replacing fixed max pooling with overlapping strided convolutions will exceed 9,280 correct predictions by learning which local patterns survive each resolution change while reducing computation enough to avoid the recent timing failures.
change: Move downsampling into the existing refinement convolutions at all three stages, preserving parameter count, output shape, classifier, training procedure, and validation ensemble.
mechanism: All-convolutional learned downsampling
evidence_used: The residual refinement kept all fixed max-pooling bottlenecks and fell to 9,266 correct, while added prediction branches repeatedly timed out. This challenges the load-bearing assumption that hard-coded max pooling is the right image representation, using a parameter-neutral mechanism that is also computationally cheaper.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying meaningful decoupled weight decay only to convolutional and linear weight tensors will exceed 9,280 correct predictions by strengthening the regularization that prior dropout evidence indicates is beneficial, without penalizing normalization scales or biases.
change: Replace uniform 2e-4 AdamW decay with 1e-2 decay for parameters having at least two dimensions and zero decay for biases and normalization parameters.
mechanism: Selective matrix-weight regularization
evidence_used: Reducing classifier dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating insufficient regularization hurts; selective AdamW decay tests complementary regularization with negligible runtime impact.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Softening evaluation logits with temperature 1.05 will preserve all 9,280 argmax predictions while reducing validation cross-entropy below 0.198076.
change: Divide only the final test-time ensemble logits by 1.05; training and predicted classes remain unchanged.
mechanism: Conservative post-ensemble temperature calibration
evidence_used: Sparse tail EMA reduced cross-entropy to 0.19631 while changing only one prediction, showing probability quality can improve near the established decision boundary; temperature scaling targets that tie-breaker without altering argmax decisions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining a 4×4 final feature map while narrowing the classifier to 80 units will exceed 9,280 correct predictions by preserving spatial boundary information within the parameter ceiling.
change: Enable ceiling mode on the final max pool and resize the classifier for the resulting 4×4 features, yielding 244,442 learned parameters.
mechanism: Boundary-preserving final pooling
evidence_used: Learned downsampling timed out, while residual refinements retaining the 3×3 pooling bottleneck fell to 9,266 correct; this tests the spatial bottleneck with negligible convolutional overhead.
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
