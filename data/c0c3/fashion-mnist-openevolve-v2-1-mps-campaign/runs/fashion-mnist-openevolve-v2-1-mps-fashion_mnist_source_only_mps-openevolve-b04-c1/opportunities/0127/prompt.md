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

RECENT RESULT
hypothesis: Adding translation-tolerant second-order channel interactions alongside the existing spatial classifier will exceed 9,328 correct predictions by detecting class-specific feature co-occurrences that its flattened first-order representation cannot efficiently express.
change: Preserve the verified classifier and mirrored-view pathway, but add a zero-initialized residual logit branch that projects fused features into two 16-channel spaces, globally pools their pairwise products, and classifies the normalized bilinear descriptor.
mechanism: Low-rank bilinear channel co-occurrence head
evidence_used: Widening the conventional dense head fell to 9,300 correct, while spatial attention and full-resolution multiplicative fusion exceeded the runtime limit. This challenges the shared assumption that more static head capacity is sufficient by introducing explicit quadratic evidence with only 4,608 additional parameters and negligible computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Expanding the second residual block’s receptive field with dilation will exceed 9,328 correct predictions by capturing larger garment-part relationships without adding parameters or meaningful computation.
change: Change the 64-channel residual convolution from standard 3×3 sampling to dilation 2 with matching padding, preserving feature dimensions and all training behavior.
mechanism: Dilated late-stage spatial context
evidence_used: Spatial attention and refinement exceeded the runtime limit, while dense-head widening fell to 9,300 correct; dilation tests richer spatial features at the same parameter count and nearly identical computational cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 63.768704750109464, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19794159393310548, "validation_score": 9328.417382619096}

RECENT RESULT
hypothesis: Learning a per-channel blend of max- and average-pooled evidence will exceed 9,328 correct predictions by retaining salient garment details while reducing pooling aliasing in channels where distributed shape evidence is more useful.
change: Replace only the final fixed max pooling operation with a 64-parameter adaptive max/average blend initialized to reproduce the verified model exactly.
mechanism: Identity-initialized channelwise mixed final pooling
evidence_used: Global-max evidence reached only 9,325 and dense-head widening fell to 9,300, while spatial refinement and attention timed out; this motivates a lightweight improvement to spatial aggregation rather than more head capacity or expensive feature processing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing label smoothing from 0.02 to 0.03 only before EMA collection will exceed 9,328 correct predictions by strengthening early representation regularization while preserving the successful late smoothing decay.
change: Use 0.03 label smoothing during the first half of training, then retain the existing 0.02-to-zero schedule.
mechanism: Front-loaded label smoothing
evidence_used: Ending smoothing earlier fell to 9,316 correct, showing early regularization is valuable, while retaining more smoothing late reached only 9,325; this isolates additional smoothing to the pre-EMA phase.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 63.31309066712856, "validation_accuracy": 0.9293, "validation_correct": 9293, "validation_cross_entropy": 0.1963447608947754, "validation_score": 9293.417939724688}

RECENT RESULT
hypothesis: Averaging the original prediction with half-weight contributions from one-pixel upward and downward views will exceed 9,328 correct predictions by reducing residual vertical-alignment sensitivity without altering training.
change: Preserve training exactly, but during evaluation ensemble logits from the original image and symmetric one-pixel vertical translations, weighting the original view 0.50 and each translation 0.25.
mechanism: Center-weighted vertical translation ensemble
evidence_used: Training-time translation augmentation repeatedly exceeded the runtime limit, while numerous runtime-neutral changes to EMA, label smoothing, dropout, capacity, and dilation failed to improve 9,328 correct; evaluation-only translation tests the still-unresolved geometric-tolerance hypothesis on only 10,000 examples with no backward pass.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging floating-point BatchNorm buffers at 0.025 will exceed 9,328 correct predictions by matching their combined tracking lag to the successful 0.02 parameter-EMA horizon.
change: Apply a 0.025 EMA to floating-point model buffers during parameter averaging while continuing to copy integer buffers directly.
mechanism: Horizon-aligned BatchNorm-statistics EMA
evidence_used: Averaging BatchNorm buffers at the parameter rate reached 9,327 correct, suggesting normalization alignment is useful but slightly over-lagged; accounting for BatchNorm’s existing momentum makes 0.025 a closer horizon match.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Preserving the mirrored-feature average as an immutable identity path while learning only a residual correction will exceed 9,328 correct predictions by preventing optimization from degrading the strongest view representation.
change: Zero-initialize the existing fusion convolution and add its output residually to the invariant features, preserving the baseline function and parameter count at initialization.
mechanism: Fixed invariant skip with learned residual view fusion
evidence_used: Disagreement-only fusion fell to 9,290 correct while the full invariant pathway reached 9,328, showing that invariant features are essential; a fixed skip protects them while retaining learnable disagreement interactions without extra runtime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 82.41610429203138, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19525115814208985, "validation_score": 9311.418322121333}



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
