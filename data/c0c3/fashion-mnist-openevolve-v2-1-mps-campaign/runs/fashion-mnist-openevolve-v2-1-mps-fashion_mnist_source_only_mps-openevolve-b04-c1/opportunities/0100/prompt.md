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
hypothesis: Shared-head classification of each mirrored orientation before logit averaging will exceed 9,328 correct predictions by retaining asymmetric spatial evidence, while deriving the second feature map through exact horizontal equivariance removes the runtime bottleneck that prevented the prior late-pooling design from finishing.
change: Remove early invariant/disagreement fusion, classify the original and horizontally reflected feature maps independently with one shared head, average their logits, and compute the convolutional features only once.
mechanism: Equivariance-cached late logit pooling
evidence_used: The prior late-logit-pooling attempt directly tested this more expressive invariant representation but timed out; the convolutional feature extractor consists entirely of horizontally equivariant operations, so reflecting its output avoids the redundant second convolutional pass that caused repeated mirrored-view designs to approach the time limit.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 62.99464391707443, "validation_accuracy": 0.9218, "validation_correct": 9218, "validation_cross_entropy": 0.21755882720947264, "validation_score": 9218.410657775892}

RECENT RESULT
hypothesis: Averaging BatchNorm running means and variances on the same 0.02 EMA horizon as learned parameters will exceed 9,328 correct predictions by evaluating the averaged model with normalization statistics from the same late-training trajectory.
change: EMA-average floating-point BatchNorm running statistics instead of copying only their final values; continue copying integer counters directly.
mechanism: EMA-aligned BatchNorm buffer averaging
evidence_used: Live paired-view BatchNorm tied the best 9,328 correct predictions, while setting BatchNorm momentum to 0.02 reached 9,326, showing normalization-state alignment is influential but unresolved; directly averaging the buffers with the parameters tests that alignment without changing training computation or learned parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Image-conditioned multiplicative channel gating will exceed 9,328 correct predictions by letting global invariant and disagreement evidence determine which fused channels matter for each image.
change: Add a zero-initialized squeeze-excitation branch over the paired mirror representation and use it to recalibrate the existing fused feature map while preserving baseline behavior at initialization.
mechanism: Reflection-context channel recalibration
evidence_used: The static early-fusion model remains best at 9,328, while widening its classifier fell to 9,300 and cached late pooling fell to 9,218; this motivates retaining early fusion while replacing generic capacity with targeted nonlinear interactions between global reflection context and spatial features.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining early invariant/disagreement fusion while adding zero-initialized image-conditioned channel gates will exceed 9,328 correct predictions; deriving the mirrored features through horizontal equivariance will keep the design within the verification time limit without changing its initial classifier behavior.
change: Compute convolutional features once, derive the mirrored map by flipping those features, and add a compact 128→16→64 gate that adaptively recalibrates the fused channels.
mechanism: Equivariance-cached reflection-context channel recalibration
evidence_used: Static early fusion achieved the best result of 9,328 correct, whereas replacing it with cached late pooling fell to 9,218. The prior channel-recalibration attempt timed out, while equivariance caching completed in 62.99 seconds, motivating caching solely for efficiency while preserving the winning fusion representation.
result: training did not finish within the verification time limit

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
