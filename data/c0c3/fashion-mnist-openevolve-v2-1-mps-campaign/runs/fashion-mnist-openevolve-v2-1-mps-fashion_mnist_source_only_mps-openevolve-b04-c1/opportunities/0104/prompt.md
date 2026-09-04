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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.22494162502699, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19341388931274414, "validation_score": 9328.418966131094}
prior_hypothesis: Holding label smoothing at 0.02 for the first half of training and annealing it to zero during EMA collection will exceed 9,323 correct predictions by removing late soft-target bias while retaining early regularization.

## Recent verification evidence

RECENT RESULT
hypothesis: Quadratically annealing label smoothing after mid-training will exceed 9,328 correct predictions by emphasizing hard-label separation earlier during EMA collection while retaining the successful 0.02 early regularization.
change: Preserve the existing dropout schedule but change label smoothing from a linear to a quadratic decay over the second half of training.
mechanism: Accelerated late label-smoothing removal
evidence_used: Annealing late label smoothing to zero produced the strongest 9,328-correct design, whereas stronger persistent weight decay reduced accuracy; this motivates removing soft-target bias faster without adding computation or persistent regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning-rate multiplier from 10% to 11.25% will exceed 9,328 correct predictions by preserving slightly stronger late hard-label updates without the larger jump of the unresolved 12.5% attempt.
change: Increase the cosine schedule’s terminal learning rate from 2.10e-4 to 2.3625e-4 while keeping its initial rate and all other behavior unchanged.
mechanism: Moderately elevated cosine learning-rate floor
evidence_used: Lowering the floor from 10% to 2% reduced validation correct from 9,328 to 9,298, while the 12.5% attempt timed out rather than producing negative accuracy evidence; a smaller upward step is the most direct test of the observed learning-rate direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Ending label smoothing at 80% progress will exceed 9,328 correct predictions by allowing more hard-label optimization during EMA collection while retaining early regularization.
change: Decay label smoothing linearly from 0.02 at mid-training to zero at 80% progress, while preserving the existing dropout schedule.
mechanism: Earlier hard-label transition during EMA
evidence_used: Late label-smoothing annealing produced the strongest 9,328-correct design, and the accelerated quadratic-decay attempt timed out without negative accuracy evidence; this runtime-neutral edit directly retests earlier soft-target removal.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 79.08531620795839, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.19512491340637206, "validation_score": 9316.41836630999}

RECENT RESULT
hypothesis: Constraining mirror fusion to add a learned disagreement correction onto the invariant features will exceed 9,328 correct predictions by removing a redundant invariant-channel transformation while preserving the winning early-fusion representation.
change: Replace the 128→64 fusion projection with a zero-initialized 64→64 disagreement projection and add its output residually to the invariant feature map.
mechanism: Residual disagreement-only mirror fusion
evidence_used: The 224,442-parameter early-fusion model achieved 9,328 correct, while widening it to 249,618 parameters fell to 9,300 and adding global-max features reached only 9,325; this motivates reducing redundant capacity while retaining invariant/disagreement fusion.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 220346, "training_seconds": 73.53945741686039, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20131970710754393, "validation_score": 9290.41620893842}

RECENT RESULT
hypothesis: Adding a lightweight nonlinear refinement block at 7×7 resolution will exceed 9,328 correct predictions by improving spatial feature interactions without the overfitting and computational cost of generic classifier widening.
change: Add a zero-initialized depthwise-separable residual block after the second pooling stage, increasing learned parameters from 224,442 to 229,370 while preserving the baseline mapping at initialization.
mechanism: Post-pooling depthwise-separable residual refinement
evidence_used: Widening the dense classifier to 249,618 parameters reduced correct predictions to 9,300, indicating that generic head capacity is unhelpful; a parameter-efficient convolutional block instead adds targeted nonlinear feature extraction at negligible relative compute.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning-rate multiplier from 10% to 11.25% will exceed 9,328 correct predictions by preserving slightly stronger late hard-label updates.
change: Increase the cosine schedule’s terminal learning rate from 2.10e-4 to 2.3625e-4 while preserving its initial learning rate and all other behavior.
mechanism: Moderately elevated cosine learning-rate floor
evidence_used: Lowering the floor to 2% reduced validation correct from 9,328 to 9,298; the prior 11.25% verification timed out and therefore supplied no contrary accuracy evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 70.74130841600709, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19360797004699706, "validation_score": 9320.418898007174}

RECENT RESULT
hypothesis: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late-update variance while retaining enough responsiveness to the cosine-decayed trajectory.
change: Double the effective averaging horizon of learned parameters without changing training computation, normalization buffers, or the optimizer schedule.
mechanism: Longer-horizon parameter EMA
evidence_used: Changing the learning-rate floor in either tested direction reduced accuracy, while BatchNorm momentum changes also failed to improve the 9,328-correct baseline; this motivates stabilizing the successful trajectory through parameter averaging rather than altering its updates or normalization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A depthwise-separable refinement block at 7×7 resolution will exceed 9,328 correct predictions by adding targeted spatial interactions, while deriving mirrored features through exact horizontal equivariance will keep training within the time limit.
change: Compute convolutional features once per image, obtain the mirrored view by flipping the feature map, and add a zero-initialized 64-channel depthwise-separable residual block after final pooling.
mechanism: Equivariance-cached spatial residual refinement
evidence_used: The same 4,928-parameter spatial refinement previously timed out without accuracy evidence, while equivariance-cached feature extraction completed in 62.99 seconds; generic dense widening reduced accuracy to 9,300, favoring efficient convolutional refinement over more head capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A parameter-neutral 1×1/2×2/4×4 average-and-max pyramid will exceed 9,328 correct predictions by preserving coarse part layout while making class predictions less sensitive to exact 7×7 feature positions.
change: Replace the flattened single-resolution representation with multiscale regional average/max statistics and adjust the hidden width to 56, leaving the successful mirror fusion and training procedure intact.
mechanism: Dual-statistic spatial-pyramid classification
evidence_used: Dense-head widening fell to 9,300 correct, while adding only global-max evidence reached 9,325; this tests a structurally different head that retains spatial configuration missing from global pooling without relying on generic parameter growth.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying random translations of up to two pixels during the first 75% of training will exceed 9,328 correct predictions by improving positional robustness while the final clean-data phase preserves precise classification boundaries.
change: Add efficient replicate-padded per-image random crops before the existing horizontal flip, disabling translation augmentation for the final quarter of training.
mechanism: Early per-image translation augmentation with clean late fine-tuning
evidence_used: The unchanged early-fusion architecture remains best at 9,328 correct, while structural head and refinement changes failed or timed out; augmentation tests a distinct, parameter-neutral source of generalization and replaces some value lost because the existing flip is largely redundant with symmetric paired-view fusion.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by averaging over more late-training updates and suppressing trajectory variance without changing optimization or runtime materially.
change: Double the effective parameter-averaging horizon during the second half of training.
mechanism: Longer-horizon parameter EMA
evidence_used: Learning-rate-floor changes in both directions failed to improve the 9,328-correct baseline, while the prior runtime-neutral 0.01 EMA attempt timed out and produced no contrary accuracy evidence, leaving this stabilization hypothesis unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 68.9803614160046, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1924131622314453, "validation_score": 9322.41931774643}

RECENT RESULT
hypothesis: Increasing the EMA update rate from 0.02 to 0.03 will exceed 9,328 correct predictions by tracking the improving late-training trajectory more responsively while still averaging roughly 33 optimizer updates.
change: Increase only the parameter-EMA interpolation rate from 0.02 to 0.03.
mechanism: Shorter-horizon parameter EMA
evidence_used: Extending the EMA horizon with a 0.01 update rate reduced validation correct from 9,328 to 9,322, directly motivating a runtime-neutral test in the opposite direction.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 82.7579179580789, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19399065170288085, "validation_score": 9325.418763747679}



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
