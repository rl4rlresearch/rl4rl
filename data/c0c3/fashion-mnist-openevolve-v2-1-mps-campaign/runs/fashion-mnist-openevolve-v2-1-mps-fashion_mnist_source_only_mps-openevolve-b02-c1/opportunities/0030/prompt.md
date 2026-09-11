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
hypothesis: Batch size 44 will exceed 9,280 correct predictions by providing approximately 9% more optimizer steps than batch size 48 while remaining within the runtime boundary that batch size 40 exceeded.
change: Reduce only the training batch size from 48 to 44, preserving the architecture, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Runtime-bracketed intermediate-batch optimization
evidence_used: Reducing batch size from 64 to 48 increased validation_correct from 9,238 to 9,280, while batch size 40 timed out; 44 is the nearest conservative midpoint between the best completed configuration and the observed runtime failure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing GELU with in-place ReLU will make batch-size-44 training finish within the runtime limit and exceed 9,280 correct predictions through approximately 9% more optimizer steps.
change: Use batch size 44 and replace all feature and classifier GELU activations with lower-cost in-place ReLU activations; preserve augmentation, optimizer, schedule, capacity, and validation ensemble.
mechanism: Faster activations fund higher-update optimization
evidence_used: Batch 48 reached 9,280 correct after outperforming batch 64 through additional updates, while batch 44 timed out with the architecture unchanged; reducing activation overhead directly targets that runtime failure while retaining the proposed higher-update regime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a direct global mean–max decision path to the proven spatial classifier will exceed 9,280 correct predictions by capturing translation-tolerant feature presence without the runtime cost of the timed-out residual pyramid.
change: Preserve the successful feature extractor and position-specific head, but augment every prediction with a learned, zero-initialized linear residual computed from global mean and maximum channel statistics.
mechanism: Zero-initialized invariant residual classification branch
evidence_used: Repeated optimizer and capacity changes failed or timed out, while the global mean–max design was never accuracy-tested because its seven-convolution extractor exceeded the time limit; this patch isolates that alternative prediction mechanism with only 1,920 parameters and negligible training computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 25% of the sparse tail EMA into the final weights will retain at least 9,280 correct predictions while lowering validation cross-entropy, improving validation_score.
change: Sample a parameter EMA every 32 steps during the second half of training, then interpolate it 25% into the final parameters instead of fully replacing them.
mechanism: Conservative tail-EMA weight interpolation
evidence_used: Full sparse tail EMA finished within the time limit and reduced cross-entropy from 0.19808 to 0.19631 but lost one correct prediction; a conservative interpolation should capture some smoothing benefit while staying closer to the higher-accuracy final weights.
result: training did not finish within the verification time limit

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
