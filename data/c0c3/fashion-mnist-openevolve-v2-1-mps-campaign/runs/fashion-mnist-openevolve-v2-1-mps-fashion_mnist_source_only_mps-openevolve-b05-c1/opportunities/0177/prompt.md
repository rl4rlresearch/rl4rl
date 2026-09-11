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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.13296675006859, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18573341827392578, "validation_score": 9360.421679942805}
prior_hypothesis: Increasing the agreement coefficient from 0.12 to 0.16 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18575621643066406.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding cross-channel context to the existing channel gate will exceed 9,360 correct predictions while preserving the verified model’s initial behavior and full spatial classifier.
change: Add a 64→16→64 excitation branch, zero-initialize its output layer, and combine it with the existing per-channel gate; parameters increase from 247,546 to 249,674.
mechanism: Identity-initialized low-rank cross-channel recalibration
evidence_used: Channel-projection and pooled-head redesigns underperformed, indicating that channel identities and the verified classifier should be preserved; this adds narrowly targeted capacity without changing either.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 249674, "training_seconds": 58.2336482910905, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1884157241821289, "validation_score": 9328.42072819286}

RECENT RESULT
hypothesis: Redistributing one eighth of the off-center TTA weight to diagonal one-pixel translations will exceed 9,360 correct predictions by reducing two-axis pooling-phase sensitivity while preserving the verified logit-space aggregation and calibration.
change: Extend evaluation TTA with four diagonal translations at 0.125 weight each and reduce cardinal-view weights to 0.875, keeping the total off-center weight and center weight unchanged.
mechanism: Low-weight diagonal translation marginalization
evidence_used: Learned anti-aliased downsampling targeted phase sensitivity but timed out, while probability-space TTA lost one correct prediction; this tests translation stability without changing the verified architecture, training procedure, logit aggregation, or total ensemble weight.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating one eighth of off-center training crops to diagonal one-pixel translations will exceed 9,360 correct predictions by improving two-axis pooling-phase robustness without changing the verified model, evaluation ensemble, or augmentation strength.
change: Preserve the one-third centered-crop probability and two-thirds shifted-crop probability, while replacing one eighth of cardinal shifts with uniformly distributed diagonal shifts.
mechanism: Low-rate diagonal translation augmentation
evidence_used: The diagonal evaluation ensemble and learned anti-aliased pooling both targeted phase sensitivity but timed out, while broader architecture, loss, and EMA changes reduced accuracy; training on sparse diagonal views tests the same mechanism without additional inference cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.5462752499152, "validation_accuracy": 0.934, "validation_correct": 9340, "validation_cross_entropy": 0.18374926300048827, "validation_score": 9340.42238674661}

RECENT RESULT
hypothesis: Halving classifier dropout from 0.10 to 0.05 will exceed 9,360 correct predictions by improving optimization of the narrow 30-unit head while retaining mild regularization.
change: Reduce only the classifier-head dropout probability; preserve architecture, parameter count, optimizer, EMA, augmentation, TTA, and calibration.
mechanism: Reduced bottleneck dropout under fixed exposure
evidence_used: Added regularization through label smoothing fell to 9,332 correct and focal reweighting fell to 9,331, while the uniform-loss baseline reached 9,360; this motivates cautiously reducing existing training noise.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 76.93204570794478, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.18807834701538087, "validation_score": 9311.420847666534}

RECENT RESULT
hypothesis: Reducing the EMA decay cap from 0.98 to 0.979 will retain all 9,360 correct predictions while lowering validation cross-entropy, strictly improving validation_score.
change: Shorten only the late-training parameter-averaging horizon; preserve the architecture, loss, augmentation, schedule, TTA, BatchNorm handling, and calibration.
mechanism: Fine-grained EMA horizon interpolation
evidence_used: The 0.975 EMA achieved lower cross-entropy but lost three correct predictions, while 0.99 lost eleven and worsened cross-entropy; testing 0.979 isolates a narrow point between the accuracy-leading 0.98 setting and the better-calibrated 0.975 setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 65.7714123330079, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.18581063003540038, "validation_score": 9358.421652485933}

RECENT RESULT
hypothesis: Replacing the assumption that fixed crop augmentation and ten-view TTA sufficiently handle translation with bounded, learned two-axis feature alignment will exceed 9,360 correct predictions by presenting the position-sensitive classifier with a more consistent spatial representation.
change: Add an identity-initialized localization network that predicts a bounded continuous translation for each 7×7 feature map and resamples it before channel gating, statistics extraction, and classification; parameters increase from 247,546 to 249,400.
mechanism: Learned per-image subpixel feature registration
evidence_used: The verified model still relies on extensive translated-view averaging, while sparse diagonal crop augmentation reduced validation_correct to 9,340 and anti-aliased pooling did not finish. This motivates adaptive per-image registration rather than additional fixed views or another classifier-head modification.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Redistributing one eighth of vertical-view weight to horizontal translations will exceed 9,360 correct predictions by preserving class-bearing vertical alignment while retaining horizontal translation robustness.
change: Keep the existing ten views, center weight, total ensemble weight, logit-space aggregation, and calibration, but weight vertical shifts at 0.875 and horizontal shifts at 1.125.
mechanism: Anisotropic cardinal-view logit averaging
evidence_used: Diagonal translation augmentation reduced validation_correct from 9,360 to 9,340, indicating that translation invariance is not uniformly beneficial; probability-space TTA also underperformed, motivating a targeted weight redistribution within the verified logit-space ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.152112000156194, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.18580753479003906, "validation_score": 9357.421653586549}

RECENT RESULT
hypothesis: Centered per-image temperature adjustments based on ten-view prediction agreement will retain all 9,360 predictions while lowering validation cross-entropy below 0.18585695190429688.
change: Preserve the verified weighted logit ensemble, then slightly sharpen high-consensus examples and soften low-consensus examples with a strictly positive scale that cannot change argmax predictions.
mechanism: TTA-consensus-conditioned logit calibration
evidence_used: Global scaling preserved 9,360 correct and improved cross-entropy, while anisotropic TTA weighting lost three correct; this motivates richer calibration without altering ensemble decisions.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 76.05017645796761, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18581797180175783, "validation_score": 9360.421649875352}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.04 to 0.08 will retain all 9,360 predictions while further lowering validation cross-entropy below 0.18581797180175783.
change: Double only the strength of the strictly positive per-image confidence scale, preserving training, TTA logits, global calibration, and argmax predictions.
mechanism: Stronger TTA-consensus-conditioned temperature scaling
evidence_used: Adding the 0.04 agreement-conditioned scale preserved 9,360 correct predictions and improved cross-entropy from the global-scaling result of 0.18585695190429688 to 0.18581797180175783, indicating that greater consensus is positively associated with correctness and motivating a cautious step farther in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 51.011631625005975, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18578439636230468, "validation_score": 9360.421661814351}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.08 to 0.12 will preserve all 9,360 predictions because it applies a strictly positive per-image scalar, while lowering validation cross-entropy below 0.18578439636230468.
change: Increase only the consensus-conditioned confidence-scaling coefficient from 0.08 to 0.12.
mechanism: Stronger TTA-agreement-conditioned temperature scaling
evidence_used: Raising the coefficient from 0.04 to 0.08 preserved 9,360 correct predictions and reduced cross-entropy from 0.18581797180175783 to 0.18578439636230468, supporting another cautious step in the same direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.08 to 0.12 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18578439636230468.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.08 to 0.12.
mechanism: Stronger TTA-agreement-conditioned temperature scaling
evidence_used: Raising the coefficient from 0.04 to 0.08 preserved all 9,360 correct predictions and reduced cross-entropy from 0.18581797180175783 to 0.18578439636230468; the prior 0.12 attempt timed out without producing contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.18385050003417, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18575621643066406, "validation_score": 9360.4216718353}

RECENT RESULT
hypothesis: Increasing the agreement coefficient from 0.12 to 0.16 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18575621643066406.
change: Increase only the strictly positive consensus-conditioned confidence-scaling coefficient from 0.12 to 0.16.
mechanism: Incremental TTA-agreement-conditioned temperature scaling
evidence_used: Successive increases from 0.04 to 0.08 and then 0.12 preserved all 9,360 correct predictions while reducing cross-entropy from 0.18581797180175783 to 0.18578439636230468 and then 0.18575621643066406, supporting another equal-sized step.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.13296675006859, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18573341827392578, "validation_score": 9360.421679942805}



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
