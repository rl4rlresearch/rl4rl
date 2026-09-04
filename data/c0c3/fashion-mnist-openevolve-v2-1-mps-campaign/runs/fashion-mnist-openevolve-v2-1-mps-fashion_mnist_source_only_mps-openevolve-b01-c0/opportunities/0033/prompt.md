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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 53.25754679203965, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.22417662353515624, "validation_score": 9290.40843779434}
prior_hypothesis: Expanding the dense bottleneck from 48 to 58 units will exceed 9,265 correct predictions by using the remaining parameter budget to improve class separation without altering the successful augmentation and optimization procedure.

## Recent verification evidence

RECENT RESULT
hypothesis: Weighting the unshifted validation view twice as heavily as each cardinal translation will exceed 9,290 correct predictions by retaining translation robustness while emphasizing the original image geometry.
change: Duplicate the centered view in the evaluation ensemble, changing its weight from 20% to 33.3% while leaving training and all other settings unchanged.
mechanism: Moderately center-weighted test-time augmentation
evidence_used: Cardinal translation-flip averaging previously improved validation correct from 9,138 to 9,167, while adding diagonal views reduced it; this shows evaluation-view composition affects accuracy and motivates a conservative weighting refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging raw logits across live/EMA cardinal and flipped views will exceed 9,290 correct predictions by preventing a single overconfident transformed view from dominating the ensemble.
change: Replace probability-space test-time averaging with arithmetic averaging of raw logits; training remains unchanged.
mechanism: Logit-space test-time ensemble consensus
evidence_used: Cardinal translation-flip averaging previously improved validation correct from 9,138 to 9,167, while adding diagonal views reduced it, showing that ensemble behavior materially affects accuracy and motivating a direct test of its aggregation rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using batch size 50 will exceed 9,290 correct predictions by increasing optimizer updates from 1,564 to exactly 2,000 while eliminating undersized end-of-epoch batches.
change: Change the batch size from 64 to 50, preserving the successful model, augmentation, optimizer, schedule, and ensemble.
mechanism: Exact-divisor smaller minibatches
evidence_used: The 247,836-parameter design achieved 9,290 correct in 1,564 steps, while multiple architecture and regularization changes regressed; increasing update resolution without altering those validated components is the most informative remaining optimization change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing EMA decay from 0.99 to 0.995 will exceed 9,290 correct predictions by averaging roughly twice as many late-training updates, reducing parameter and BatchNorm-state noise without changing exposure or runtime materially.
change: Lengthen the effective EMA window from approximately 100 to 200 optimizer steps while preserving the verified model, training augmentation, schedule, and evaluation ensemble.
mechanism: Longer-horizon exponential moving average
evidence_used: Test-time ensembling previously improved validation correct from 9,138 to 9,167, while recent architecture and regularization changes all regressed from 9,290; refining the existing ensemble’s temporal component is therefore an informative low-risk change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.04852300020866, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.22472033462524413, "validation_score": 9282.408256469549}

RECENT RESULT
hypothesis: Weighting the centered translation twice will exceed 9,290 correct predictions by emphasizing validation geometry while retaining cardinal-shift robustness, and reusing its computed probabilities will avoid the prior timeout.
change: Duplicate the already-computed centered and centered-flipped log-probability tensors in each live/EMA ensemble, increasing their combined translation weight from 20% to 33.3% without additional forward passes.
mechanism: Compute-free center-weighted probability ensemble
evidence_used: Cardinal translation-flip averaging previously improved correct predictions from 9,138 to 9,167, while diagonal views reduced accuracy; the direct center-weighting experiment timed out, leaving its accuracy hypothesis unresolved and motivating an equivalent compute-efficient implementation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Excluding the centered training view and sampling the four one-pixel cardinal translations uniformly will exceed 9,290 correct predictions by further strengthening the translation robustness favored by the existing evidence.
change: Remove the centered offset from training augmentation while preserving evaluation views and all other settings.
mechanism: Shift-emphasized cardinal augmentation
evidence_used: Increasing the centered-view share from 20% to 50% reduced validation correct from 9,290 to 9,278, directly motivating a controlled change in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 51.80013829190284, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.2223417610168457, "validation_score": 9270.409050902084}

RECENT RESULT
hypothesis: Decreasing EMA decay from 0.99 to 0.985 will exceed 9,290 correct predictions by tracking late-training improvements more closely while retaining useful temporal smoothing.
change: Shorten the EMA’s effective averaging window from roughly 100 to 67 optimizer steps, preserving architecture, augmentation, optimizer schedule, and evaluation views.
mechanism: Shorter-horizon exponential moving average
evidence_used: Increasing EMA decay from 0.99 to 0.995 reduced validation correct from 9,290 to 9,282, directly motivating a controlled move toward a shorter rather than longer averaging horizon.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing AdamW beta2 from 0.999 to 0.99 will exceed 9,290 correct predictions by making per-parameter learning rates respond faster during the limited 1,564-update training run.
change: Set AdamW betas to (0.9, 0.99) while preserving the model, exposure, schedule, augmentation, regularization, and EMA ensemble.
mechanism: Faster-adapting AdamW second-moment estimate
evidence_used: The batch-size-50 experiment attempted to improve optimization through more updates but timed out, while recent dropout, smoothing, augmentation, and EMA changes failed to beat 9,290; faster moment adaptation tests the same limited-update bottleneck without additional computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.83970891707577, "validation_accuracy": 0.9269, "validation_correct": 9269, "validation_cross_entropy": 0.22179617462158202, "validation_score": 9269.409233561526}

RECENT RESULT
hypothesis: Reducing the second convolutional stage from 64 to 63 channels and expanding the dense bottleneck from 58 to 60 units will exceed 9,290 correct predictions by extending the demonstrated benefit of greater dense class-separation capacity while remaining below the parameter ceiling.
change: Reallocate a small amount of convolutional capacity to two additional dense bottleneck units, producing a 249,759-parameter model without changing training or evaluation.
mechanism: Convolution-to-head parameter reallocation
evidence_used: Expanding the dense bottleneck from 48 to 58 units produced the current 9,290-correct design, while subsequent dropout, smoothing, augmentation, EMA, and optimizer changes all failed to improve it; this motivates another budget-constrained head expansion.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249759, "training_seconds": 55.08811104204506, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.22385634269714355, "validation_score": 9264.40854468172}

RECENT RESULT
hypothesis: Applying a 4×4 mean-valued erasure to 25% of training images will exceed 9,290 correct predictions by improving local-occlusion robustness without disturbing the validated translation, flip, smoothing, or EMA settings.
change: Add inexpensive per-image random erasing after the existing cardinal-shift and horizontal-flip augmentation.
mechanism: Mild random-erasing regularization
evidence_used: Removing centered training views reduced validation correct from 9,290 to 9,270, while dropout and label-smoothing changes also regressed; this motivates preserving those validated components and testing an orthogonal, low-cost augmentation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 75.31700700009242, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.22545674438476562, "validation_score": 9252.40801113731}

RECENT RESULT
hypothesis: Adding parameter-efficient channel attention while preserving all 64 convolutional channels will exceed 9,290 correct predictions by learning image-dependent feature emphasis within the remaining parameter budget.
change: Add a 2,128-parameter squeeze-and-excitation module after the final pooling layer, bringing the model to 249,964 parameters without changing training or evaluation.
mechanism: Identity-centered squeeze-and-excitation
evidence_used: Expanding the dense bottleneck to 58 units produced the 9,290-correct best result, while reallocating convolutional width to a larger head fell to 9,264; this motivates using the remaining budget for adaptive capacity without sacrificing the validated convolutional width.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding residual connections between equal-width convolutional layers will exceed 9,290 correct predictions by improving feature preservation and gradient flow during the limited 1,564-step training run without changing capacity or validated widths.
change: Add residual additions within both 32-channel and 64-channel convolutional stages while preserving all parameters, optimization, augmentation, and evaluation behavior.
mechanism: Parameter-free intra-stage residual feature reuse
evidence_used: Reducing convolutional width to enlarge the head regressed from 9,290 to 9,264 correct, while changing AdamW adaptation regressed to 9,269; this motivates retaining the best architecture and optimizer while improving its information flow without additional parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.15142254112288, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2243206771850586, "validation_score": 9267.408389737524}



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
