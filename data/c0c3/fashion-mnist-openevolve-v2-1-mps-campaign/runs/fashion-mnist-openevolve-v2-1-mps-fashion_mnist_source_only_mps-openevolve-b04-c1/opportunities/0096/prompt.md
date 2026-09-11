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
hypothesis: Increasing AdamW weight decay to 1e-2 will exceed 9,328 correct predictions by controlling late co-adaptation after dropout and label smoothing begin annealing away.
change: Raise AdamW’s weight decay from 2e-4 to 1e-2 while preserving the winning architecture, learning-rate schedule, EMA, and stochastic-regularization schedules.
mechanism: Persistent deterministic weight shrinkage
evidence_used: Expanding the classifier to 249,618 parameters reduced validation correct from 9,328 to 9,300, suggesting additional capacity is not beneficial; stronger persistent regularization tests that signal without adding runtime-heavy operations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 75.56032000016421, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.19485718688964843, "validation_score": 9290.418460051533}

RECENT RESULT
hypothesis: Eliminating AdamW weight decay will exceed 9,328 correct predictions by allowing better late hard-label fitting after dropout and label smoothing anneal away.
change: Set AdamW weight decay from 2e-4 to zero while preserving the architecture, learning-rate schedule, EMA, and stochastic regularization.
mechanism: Remove deterministic parameter shrinkage
evidence_used: Increasing weight decay to 1e-2 reduced validation correct from 9,328 to 9,290, providing directional evidence that persistent shrinkage conflicts with this short fixed-exposure training regime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.9665104590822, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.1946596031188965, "validation_score": 9300.418529260296}

RECENT RESULT
hypothesis: Matching BatchNorm momentum to the 0.02 parameter-EMA update rate will exceed 9,328 correct predictions by aligning validation-time running statistics with the averaged weights.
change: Set all four BatchNorm layers’ momentum from the default 0.1 to 0.02 without changing parameters or computation.
mechanism: EMA-matched BatchNorm statistic horizon
evidence_used: Live paired-view BatchNorm tied the best accuracy at 9,328 but slightly worsened cross-entropy, showing normalization-state handling affects the result; the current final buffers use a much shorter averaging horizon than the 0.02 EMA parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.54905045800842, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19380205039978027, "validation_score": 9326.418829905539}

RECENT RESULT
hypothesis: Appending each fused channel’s global maximum to the full spatial representation will exceed 9,328 correct predictions by adding translation-tolerant evidence without distorting inputs or sacrificing spatial detail.
change: Add 64 parameter-free global-max features to the classifier input, increasing learned parameters from 224,442 to 227,514 with negligible additional computation.
mechanism: Channelwise global-salience skip features
evidence_used: Two-pixel translation augmentation reduced correct predictions from 9,328 to 9,206, while widening the dense head to 249,618 parameters reached only 9,300; this motivates a compact architectural source of positional robustness rather than more augmentation or generic capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 227514, "training_seconds": 79.30748958396725, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1944935516357422, "validation_score": 9325.418587441778}

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
