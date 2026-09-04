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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.25880720792338, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19353691482543944, "validation_score": 9330.418922945566}
prior_hypothesis: Using a 0.04 EMA update rate for the classifier while retaining 0.02 for feature parameters will exceed 9,328 correct predictions by tracking the rapidly changing dropout- and smoothing-free terminal classifier without exposing the backbone to the weaker global 0.04 averaging regime.

## Recent verification evidence

RECENT RESULT
hypothesis: Lowering the cosine schedule floor from 10% to 5% will exceed 9,330 correct predictions by reducing late parameter drift while preserving enough updates for convergence.
change: Keep the optimizer, EMA rates, and schedule shape unchanged, but reduce the final learning rate from 2.1e-4 to 1.05e-4.
mechanism: Lower terminal learning-rate floor
evidence_used: Classifier EMA peaked at 0.04 and regressed at both slower and faster nearby rates, suggesting further EMA-rate tuning is unlikely to help; stabilizing the underlying late trajectory directly targets the remaining averaging lag.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 84.86875083297491, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.19555552558898925, "validation_score": 9302.418215623866}

RECENT RESULT
hypothesis: Raising only convolutional-backbone EMA rates from 0.02 to 0.025 will exceed 9,330 correct predictions by better aligning feature kernels with final BatchNorm statistics without disturbing the proven classifier or normalization averaging.
change: Use EMA rate 0.025 for the four spatial convolution weights, retain 0.04 for classifier parameters, and retain 0.02 for view fusion and BatchNorm affine parameters.
mechanism: Faster convolution-only EMA alignment
evidence_used: EMA-averaging BatchNorm buffers fell to 9,327 and using final BatchNorm affine parameters fell to 9,328, while classifier-specific faster EMA reached 9,330; this motivates a smaller, isolated acceleration of upstream convolution weights.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 64.12241741688922, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19375289802551268, "validation_score": 9326.418847150719}

RECENT RESULT
hypothesis: Using a 0.05 EMA rate only for the final linear layer while retaining 0.04 for the hidden classifier projection will exceed 9,330 correct predictions by reducing output-layer lag without weakening the averaging of learned spatial features.
change: Split classifier EMA rates: 0.05 for `classifier.4` parameters, 0.04 for the rest of the classifier, and 0.02 elsewhere.
mechanism: Layer-specific classifier EMA tracking
evidence_used: Applying 0.05 to the entire classifier reached 9,329 versus 9,330 at 0.04, while applying 0.04 only to the output layer reached 9,327; this suggests the hidden projection benefits from the proven 0.04 rate while leaving the rapidly adapting output layer’s preferred rate unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.8132424580399, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19358426361083986, "validation_score": 9330.418906327139}

RECENT RESULT
hypothesis: Increasing the spatial classifier width from 48 to 56 units will exceed 9,330 correct predictions by reducing the post-fusion representation bottleneck while remaining below the 250,000-parameter ceiling.
change: Widen both classifier linear layers consistently from 48 to 56 hidden units, raising the parameter count from 224,442 to 249,618 without changing the proven training procedure.
mechanism: Wider position-sensitive classifier projection
evidence_used: Global pooling reduced correctness to 9,290 and decision-level flip symmetrization reached only 9,285, showing that preserving and effectively projecting the full 7×7 spatial representation matters; this change adds capacity directly to that proven position-sensitive path.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249618, "training_seconds": 61.529344874899834, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.19519722785949706, "validation_score": 9301.418340997072}

RECENT RESULT
hypothesis: Retaining local mean evidence alongside maxima at the final downsampling will exceed 9,330 correct predictions by distinguishing activation density from peak presence while preserving the proven 7×7 spatial representation.
change: Replace the old assumption that one max statistic sufficiently represents each 2×2 region with concatenated max-and-mean feature maps, and expand cross-view fusion to learn from both; its initialization exactly reproduces the existing max-only path.
mechanism: Dual-statistic spatial downsampling
evidence_used: Widening the spatial classifier fell to 9,301 correct and global pooling fell to 9,290, suggesting added capacity should enrich spatial evidence before classification rather than enlarge the dense bottleneck or discard layout.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 232634, "training_seconds": 61.13984312489629, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19762962112426757, "validation_score": 9311.417491343886}

RECENT RESULT
hypothesis: Fixing the invariant feature path as an identity shortcut while learning only a zero-initialized disagreement correction will exceed 9,330 correct predictions by reducing redundant projection drift without sacrificing flip-specific information.
change: Change `view_fusion` to project only the 64-channel disagreement tensor, then add its output residually to the invariant features.
mechanism: Anchored residual disagreement fusion
evidence_used: Increasing the EMA rate of `view_fusion` reduced correctness from 9,330 to 9,326, suggesting that changes to the invariant projection are harmful; anchoring that path preserves the proven spatial representation while retaining a learnable correction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a residual 7×7 convolutional bottleneck before the proven position-sensitive classifier will exceed 9,330 correct predictions by learning local spatial interactions without enlarging the underperforming dense projection.
change: Add a zero-initialized 64→32→32→64 residual convolutional bottleneck after view fusion, increasing parameters from 224,442 to 237,754 while initially preserving the current function exactly.
mechanism: Zero-initialized local spatial refinement
evidence_used: Widening the spatial classifier fell to 9,301 correct and dual-statistic pooling reached 9,311, while global pooling fell to 9,290; this favors enriching the existing spatial representation with local processing instead of adding dense capacity or changing its pooling statistics.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237754, "training_seconds": 68.79223525011912, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.1966290481567383, "validation_score": 9309.417840433316}

RECENT RESULT
hypothesis: Lowering only the four spatial convolution EMA rates from 0.02 to 0.015 will exceed 9,330 correct predictions by smoothing late feature-kernel drift while retaining the proven 0.04 classifier rate and 0.02 rates for normalization and view fusion.
change: Use EMA rate 0.015 for the four spatial convolution weights, 0.04 for classifier parameters, and 0.02 for all remaining learned parameters.
mechanism: Stronger convolutional-backbone temporal averaging
evidence_used: Raising the spatial convolution EMA rate to 0.025 reduced correctness from 9,330 to 9,326, while classifier-specific faster averaging remains best; this directly motivates testing stronger averaging in the opposite direction only for backbone kernels.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 72.71698191715404, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.19324242057800292, "validation_score": 9324.419026336456}

RECENT RESULT
hypothesis: Raising the cosine schedule floor from 10% to 15% will exceed 9,330 correct predictions by preserving useful late optimization while retaining EMA stabilization.
change: Increase the final learning rate from 2.1e-4 to 3.15e-4 without changing the schedule’s warm plateau, shape, optimizer, or EMA rates.
mechanism: Higher terminal learning-rate floor
evidence_used: Lowering the floor from 10% to 5% reduced correctness from 9,330 to 9,302, indicating that stronger late learning—not earlier freezing—is the informative next direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 74.29568379209377, "validation_accuracy": 0.9313, "validation_correct": 9313, "validation_cross_entropy": 0.195617431640625, "validation_score": 9313.418193969717}

RECENT RESULT
hypothesis: Training on independently translated crops will exceed 9,330 correct predictions by regularizing the proven position-sensitive classifier against small alignment changes without discarding spatial layout.
change: Add replicate-padded random translations of up to two pixels per axis to every training image while preserving batch shape and the existing flip handling.
mechanism: Per-example integer translation augmentation
evidence_used: Global pooling fell to 9,290 correct and widening the spatial classifier fell to 9,301, indicating that spatial layout is valuable but additional classifier capacity is ineffective; translation augmentation preserves that layout while reducing brittle position dependence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 79.70787879102863, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22268453178405762, "validation_score": 9208.408936227623}

RECENT RESULT
hypothesis: Averaging supervised loss across two classifier dropout masks and softly aligning their predictions will exceed 9,330 correct predictions by reducing dropout-induced classifier variance without perturbing the position-sensitive image representation.
change: Run the inexpensive dense classifier twice during training and add a modest symmetric-KL consistency penalty while retaining the proven dropout, smoothing, optimizer, EMA, and inference behavior.
mechanism: Two-mask dropout consistency regularization
evidence_used: Spatial changes underperformed—translation augmentation reached 9,208, classifier widening 9,301, and local refinement 9,309—while the evidence identifies dropout’s linear taper as beneficial; this tests stronger use of that regularizer without changing spatial evidence or parameter count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 72.25649212487042, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1967281795501709, "validation_score": 9307.417805821358}

RECENT RESULT
hypothesis: Lowering only the view-fusion EMA rate from 0.02 to 0.015 will exceed 9,330 correct predictions by suppressing harmful late drift in the invariant/disagreement projection while preserving the proven classifier and backbone averaging rates.
change: Use EMA rate 0.015 for `view_fusion` parameters, 0.04 for classifier parameters, and 0.02 for all remaining learned parameters.
mechanism: Stronger temporal averaging for cross-view fusion
evidence_used: Increasing the view-fusion EMA rate reduced correctness from 9,330 to 9,326, while anchoring its invariant path was motivated by the same projection-drift concern; testing stronger averaging is the direct unexplored opposite direction without changing the established architecture.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 63.459984792163596, "validation_accuracy": 0.9329, "validation_correct": 9329, "validation_cross_entropy": 0.19342952499389648, "validation_score": 9329.418960642022}



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
