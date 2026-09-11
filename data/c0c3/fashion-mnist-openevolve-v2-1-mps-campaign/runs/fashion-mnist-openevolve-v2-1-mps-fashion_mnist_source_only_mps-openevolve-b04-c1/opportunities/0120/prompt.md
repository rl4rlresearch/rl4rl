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
hypothesis: Efficiently applying translations of up to two pixels during the first 75% of training will exceed 9,328 correct predictions by improving positional robustness while retaining a clean late fine-tuning phase.
change: Replace redundant mirrored convolution passes with exact feature-map flipping, and replace redundant horizontal-flip augmentation with deterministic batchwise replicate-padded translations that add minimal runtime overhead.
mechanism: Equivariance-cached cyclic translation augmentation
evidence_used: Per-image translation augmentation timed out without accuracy evidence, while equivariance-cached feature extraction completed in 62.99 seconds; this tests the same promising invariance mechanism using a cheaper translation implementation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging BatchNorm running means and variances alongside the learned parameters will exceed 9,328 correct predictions by evaluating a normalization state aligned with the successful parameter-EMA trajectory.
change: Extend the existing 0.02 EMA to every floating-point model-state tensor while continuing to copy integer BatchNorm counters directly.
mechanism: EMA-consistent BatchNorm statistics
evidence_used: EMA-rate changes to 0.01 and 0.03 reduced validation correct to 9,322 and 9,325, so the 0.02 parameter horizon should be preserved; the current implementation pairs those averaged parameters with final-step normalization buffers, making state alignment a distinct runtime-neutral target.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 62.15120491711423, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19391162071228027, "validation_score": 9327.418791467748}

RECENT RESULT
hypothesis: Adding content-dependent interactions between all 7×7 feature locations will exceed 9,328 correct predictions by modeling relationships between garment parts that the existing local convolutions and static flattened classifier cannot express efficiently.
change: Preserve the successful mirrored-view fusion, but insert a four-head positional self-attention residual before classification. Its output projection starts at zero so training begins as the verified baseline while learning a genuinely nonlocal prediction mechanism.
mechanism: Zero-initialized positional self-attention over fused spatial features
evidence_used: Dense-head widening reduced performance to 9,300 and disagreement-only fusion reached 9,290, so neither generic classifier capacity nor removing the invariant pathway helped. This keeps that pathway and instead challenges the shared assumption that purely local features followed by fixed dense aggregation are sufficient.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Content-dependent channel gating of the fused 7×7 representation will exceed 9,328 correct predictions by emphasizing class-relevant feature channels without the runtime and optimization costs of spatial refinement or attention.
change: Add a 2,128-parameter squeeze-and-excitation gate after mirrored-view fusion, initialized to preserve the current representation exactly at the start of training.
mechanism: Identity-initialized global channel recalibration
evidence_used: Dense-head widening fell to 9,300 correct and global-max evidence reached only 9,325, while spatial refinement and attention timed out; this motivates a lightweight, structurally targeted use of global context instead of more static head capacity or expensive spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining 0.05 classifier dropout at training end will exceed 9,328 correct predictions by regularizing late hard-label optimization without the soft-target bias that made slower label-smoothing decay underperform.
change: Keep the existing early dropout and label-smoothing schedules, but anneal classifier dropout from 0.15 to 0.05 instead of zero during the second half.
mechanism: Late hard-label dropout floor
evidence_used: Ending label smoothing earlier reduced correct predictions to 9,316, indicating late regularization matters, while retaining more smoothing reached only 9,325; dense-head widening also fell to 9,300, motivating unbiased late classifier regularization rather than more capacity or softer targets.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 73.2502585418988, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.19472210578918456, "validation_score": 9321.418507364664}

RECENT RESULT
hypothesis: Mixing each image with a neighboring training example at 10% strength during the first half of training will exceed 9,328 correct predictions by encouraging smoother input-space boundaries, while stopping MixUp before EMA collection avoids late soft-target bias.
change: Apply inexpensive deterministic within-batch MixUp only before 50% progress and train against the corresponding mixed labels; preserve the existing label-smoothing, dropout, learning-rate, and EMA schedules.
mechanism: Early low-strength MixUp with clean EMA fine-tuning
evidence_used: Translation augmentation repeatedly exceeded the runtime limit, while changing late label smoothing or dropout failed to improve the 9,328-correct baseline; early MixUp tests input-space regularization without geometric augmentation overhead or altering the successful clean late-training phase.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding an elementwise interaction between mirrored feature views will exceed 9,328 correct predictions by exposing bilateral coactivations that linear invariant/disagreement fusion cannot represent directly.
change: Add a zero-initialized 64-channel mirrored-feature product stream to the existing fusion layer, preserving the baseline function at initialization.
mechanism: Multiplicative bilateral coactivation fusion
evidence_used: Disagreement-only fusion fell to 9,290 while the full invariant pathway reached 9,328, and generic dense-head widening fell to 9,300; this motivates targeted nonlinear view interaction without removing successful features or materially increasing runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the parameter-EMA interpolation rate from 0.020 to 0.022 will exceed 9,328 correct predictions by modestly favoring later training states without the shorter-horizon degradation observed at 0.030.
change: Change only the parameter-EMA interpolation rate from 0.020 to 0.022.
mechanism: Fine-grained EMA horizon tuning
evidence_used: EMA rates of 0.010 and 0.030 produced 9,322 and 9,325 correct versus 9,328 at 0.020; these opposite-side probes place the best observed horizon near 0.020 and motivate a small upward refinement rather than another structural or runtime-heavy change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 56.562048624968156, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19355815544128419, "validation_score": 9328.418915490394}

RECENT RESULT
hypothesis: Decreasing the parameter-EMA interpolation rate from 0.020 to 0.018 will exceed 9,328 correct predictions by modestly suppressing noisy late updates without approaching the overly long horizon that underperformed at 0.010.
change: Change only the parameter-EMA interpolation rate from 0.020 to 0.018.
mechanism: Symmetric EMA horizon refinement
evidence_used: Increasing the rate to 0.022 tied the 0.020 baseline at 9,328 correct but worsened cross-entropy, while 0.010 and 0.030 reached only 9,322 and 9,325; a symmetric downward refinement is the most targeted remaining probe near the observed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 62.53067254205234, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19324942359924316, "validation_score": 9326.419023877248}

RECENT RESULT
hypothesis: Matching BatchNorm’s running-statistics momentum to the successful 0.02 parameter EMA will exceed 9,328 correct predictions by reducing normalization noise without the lag introduced by averaging already-smoothed BatchNorm buffers.
change: Change all BatchNorm2d layers from the default 0.10 momentum to 0.02; preserve every other architectural and training detail.
mechanism: EMA-horizon BatchNorm tracking
evidence_used: Averaging BatchNorm buffers alongside parameters nearly tied the baseline at 9,327 correct, indicating normalization-state alignment matters; direct 0.02 tracking tests alignment with the parameter EMA while avoiding nested exponential smoothing.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Cosine-annealing label smoothing during the second half will exceed 9,328 correct predictions by preserving useful smoothing early in the annealing phase while removing soft-target bias more quickly near the EMA-dominated endpoint.
change: Keep dropout’s linear decay unchanged, but replace label smoothing’s linear decay with a cosine schedule from 0.02 to zero.
mechanism: Cosine label-smoothing handoff
evidence_used: Ending smoothing earlier reduced validation correct to 9,316, while retaining more smoothing reached only 9,325; a cosine handoff targets the useful middle ground without adding runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.05 will exceed 9,328 correct predictions by suppressing noisy late hard-label updates while preserving the successful 0.02 EMA horizon.
change: Preserve the existing schedule shape and peak learning rate, but halve its terminal learning-rate floor from 2.1e-4 to 1.05e-4.
mechanism: Lower-noise cosine learning-rate tail
evidence_used: Fine-grained EMA changes around 0.02 and altered late dropout or label smoothing did not improve the baseline, motivating a runtime-neutral reduction in optimizer noise during the same successful EMA collection phase.
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
