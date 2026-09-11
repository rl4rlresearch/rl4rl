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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 77.54128845804371, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600157852172852, "validation_score": 9360.42158459909}
prior_hypothesis: A center weight of 1.8173828125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860016449.

## Recent verification evidence

RECENT RESULT
hypothesis: A center weight of 1.8203125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860023609.
change: Reduce the centered original and flipped logit weights from 1.828125 to 1.8203125 while preserving every evaluation view and the existing normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Successive reductions from 1.875 to 1.84375 and then 1.828125 retained 9,360 correct while monotonically lowering cross-entropy; this half-step toward the unresolved 1.8125 candidate continues the verified direction with minimal classification-boundary risk.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 66.76356629119255, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1860017868041992, "validation_score": 9360.421584525051}

RECENT RESULT
hypothesis: A learned content-conditioned spatial gate will exceed 9,360 correct predictions by suppressing background activations and emphasizing discriminative regions before both prediction heads.
change: Add a 99-parameter, 7×7 spatial-attention gate derived from channel mean and maximum maps, initialized to preserve the current model exactly.
mechanism: Identity-initialized spatial recalibration
evidence_used: Center-weight refinements plateaued at 9,360 correct with only tiny cross-entropy gains, suggesting inference aggregation is saturated; this challenges the shared assumption that global channel gating is sufficient and every spatial location should enter the learned representation unconditionally.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.81640625 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860017868.
change: Halve the remaining step from the verified 1.8203125 center weight toward the unresolved 1.8125 candidate, preserving all views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Successive reductions from 1.875 through 1.84375, 1.828125, and 1.8203125 retained 9,360 correct while monotonically lowering cross-entropy; the 1.8125 attempts timed out and provide no contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A center weight of 1.818359375 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860017868.
change: Halve the step from the verified 1.8203125 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Successive reductions from 1.875 through 1.84375, 1.828125, and 1.8203125 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 run timed out and provided no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 76.01240162504837, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600164489746093, "validation_score": 9360.421584575495}

RECENT RESULT
hypothesis: A center weight of 1.8173828125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860016449.
change: Halve the step from the verified 1.818359375 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Reductions from 1.875 through 1.84375, 1.828125, 1.8203125, and 1.818359375 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 verification timed out and supplied no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 77.54128845804371, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600157852172852, "validation_score": 9360.42158459909}

RECENT RESULT
hypothesis: A center weight of 1.81689453125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.
change: Halve the remaining interval from the verified 1.8173828125 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Every verified reduction from 1.875 through 1.8173828125 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 run timed out and therefore provides no contrary accuracy evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 63.04985083290376, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18600154457092286, "validation_score": 9359.421584611157}

RECENT RESULT
hypothesis: A center weight of 1.817138671875 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.
change: Bisect the interval between the best verified 1.8173828125 weight and the 1.81689453125 weight that lost one prediction.
mechanism: Classification-boundary bisection
evidence_used: Reducing the weight to 1.81689453125 improved cross-entropy but reduced correct predictions to 9,359, establishing a nearby accuracy boundary; midpoint testing is the most informative refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A zero-initialized covariance head will exceed 9,360 correct predictions by exposing second-order channel co-activation evidence unavailable to the existing per-channel statistics and 30-unit spatial bottleneck.
change: Project gated feature maps to 15 channels, compute their 120 unique spatial covariance terms, normalize them, and add learned residual logits. This raises the model to 249,956 parameters.
mechanism: Learned projected channel-covariance residual head
evidence_used: Repeated center-weight refinements plateaued at 9,360 correct with only microscopic cross-entropy gains, indicating inference aggregation is saturated. The old assumption is that marginal channel statistics are sufficient; this patch instead models joint channel interactions with little additional computation.
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
