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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.5085795421619, "validation_accuracy": 0.9178, "validation_correct": 9178, "validation_cross_entropy": 0.2245953540802002, "validation_score": 9178.408298135653}
prior_hypothesis: Restoring the 9,320-correct shared dual-statistic attention model and adding mild per-image translations will exceed 9,320 correct predictions by improving spatial robustness without changing parameter count or the successful paired-view objective.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.78843287518248, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1926749450683594, "validation_score": 9307.419225709458}
prior_hypothesis: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the 9,320-correct shared dual-statistic attention model and adding mild per-image translations will exceed 9,320 correct predictions by improving spatial robustness without changing parameter count or the successful paired-view objective.
change: Restore Reference Design 1, then apply independent replicate-padded translations of up to two pixels before its random flip and paired-orientation training.
mechanism: Per-image integer-translation augmentation with paired flip supervision
evidence_used: Shared average-plus-maximum attention with paired supervision achieved the best result at 9,320 correct. Translations previously appeared only in a confounded 9,024-correct experiment that also changed batch size and classifier capacity, so isolating translation augmentation on the best regimen is informative.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.5085795421619, "validation_accuracy": 0.9178, "validation_correct": 9178, "validation_cross_entropy": 0.2245953540802002, "validation_score": 9178.408298135653}

RECENT RESULT
hypothesis: Allowing the 9,320-correct shared attention model to learn a bounded relative weight for global-maximum evidence will exceed 9,320 correct predictions while preserving its beneficial shared kernel and initial behavior.
change: Add one scalar parameter, initialized to reproduce the current average-plus-maximum gate exactly, that learns the maximum descriptor’s weight in the range zero to two.
mechanism: Bounded learned dual-statistic balancing
evidence_used: Shared average-plus-maximum attention achieved 9,320 correct, whereas fully independent descriptor kernels fell to 9,300; a single learned balance retains weight sharing while adding only one controlled degree of freedom.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249856, "training_seconds": 63.763402458047494, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.19342187652587892, "validation_score": 9310.41896332708}

RECENT RESULT
hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.
change: Restore the best shared channel-attention kernel and replace its global-maximum descriptor with a parameter-free top-four activation mean; preserve paired-view training and flip-ensemble evaluation.
mechanism: Shared average-plus-top-k channel attention
evidence_used: Shared average-plus-maximum attention reached 9,320 correct, outperforming average-only attention at 9,286; separate kernels fell to 9,300 and learned descriptor balancing reached 9,310, motivating preserved sharing with a more robust salient-activation statistic.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}

RECENT RESULT
hypothesis: Extending the verified top-four saliency statistic to the spatial-attention summary will exceed 9,322 correct predictions by reducing sensitivity to single-channel activation outliers while preserving salient evidence.
change: Restore average-plus-top-four channel attention and replace the spatial gate’s hard channel maximum with the mean of its four strongest channel activations.
mechanism: Top-four robust channel-and-spatial attention
evidence_used: Average-plus-top-four channel attention achieved 9,322 correct versus 9,320 for hard maxima and 9,286 for average-only attention, motivating the same parameter-free robust saliency statistic in the spatial branch.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending the spatial maximum with the top-four mean will exceed 9,322 correct predictions by retaining strong peak evidence while reducing single-activation sensitivity.
change: Remove the harmful translation augmentation and replace hard-maximum channel evidence with an equal blend of the maximum and top-four mean, using the existing shared attention kernel.
mechanism: Peak-preserving top-four channel attention
evidence_used: Top-four channel attention achieved 9,322 correct versus 9,320 for hard maxima, while hard maxima had slightly lower cross-entropy; the blend isolates a middle ground between their saliency profiles. Translation augmentation fell sharply to 9,178 correct.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the two strongest spatial activations per channel will exceed 9,322 correct predictions by retaining more peak evidence than top-four averaging while preserving some robustness over the 9,320-correct hard maximum.
change: Change only the salient channel descriptor from a top-four mean to a top-two mean, preserving the best architecture, paired-view training, and flip-ensemble inference.
mechanism: Shared average-plus-top-two channel attention
evidence_used: Top-four channel attention achieved 9,322 correct versus 9,320 for hard maxima, while hard maxima had slightly lower cross-entropy; top-two averaging tests the closest computational midpoint without the extra spatial top-k work that timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing each channel’s isolated maximum with its strongest 2×2 local mean will exceed 9,322 correct predictions by preserving the robustness of four-activation evidence while emphasizing spatially coherent features.
change: Compute the salient channel descriptor as the maximum local 2×2 average, retaining shared attention, paired-view training, and flip-ensemble inference without adding parameters.
mechanism: Spatially coherent four-activation channel saliency
evidence_used: Global top-four averaging improved 9,320 to 9,322 correct, indicating that aggregating several activations is preferable to a hard maximum; later top-k extensions timed out, motivating a pooling-based four-activation statistic.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Evaluating a 0.90-decay EMA sampled every ten optimizer steps will exceed 9,322 correct predictions by suppressing late-training parameter noise while preserving the best verified architecture and training objective.
change: Restore shared average-plus-top-four channel attention, maintain a parameter-neutral EMA of floating model state during training, and use that averaged state for flip-ensemble validation.
mechanism: Periodic exponential weight averaging on the top-four attention model
evidence_used: Top-four channel attention produced the best verified result at 9,322 correct; descriptor reparameterization, translation, and added spatial refinement regressed, motivating a conservative optimization-level refinement that leaves the successful computation and paired cross-entropy intact.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the 9,322-correct top-four attention model and softly aligning its paired-orientation predictions will exceed 9,322 correct by strengthening the flip invariance already rewarded by ensemble validation.
change: Remove harmful translations, restore top-four channel saliency, and add a lightweight Jensen–Shannon consistency penalty to the existing paired-view cross-entropy.
mechanism: Flip-pair Jensen–Shannon consistency regularization
evidence_used: Top-four channel attention achieved the best verified result at 9,322 correct, while translations reduced correctness to 9,178; paired-orientation supervision previously improved 9,271 to 9,286, motivating stronger alignment of those already-computed views.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging paired-orientation logits will exceed 9,322 correct predictions by favoring cross-orientation consensus over a single view’s disproportionately confident probability.
change: Replace probability-space flip ensembling with arithmetic logit averaging; training, parameters, and forward-pass count remain unchanged.
mechanism: Geometric flip-ensemble inference
evidence_used: Paired-orientation supervision and flip-ensemble validation were beneficial, while adding Jensen–Shannon alignment timed out; logit averaging provides a computationally free consistency bias at inference.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing hard channel maxima with an unsorted top-four mean will exceed the current 9,320 correct predictions while avoiding unnecessary sorting overhead from the 9,322-correct reference implementation.
change: Use the mean of each channel’s four strongest spatial activations as the shared salient descriptor, with `sorted=False` to reduce top-k runtime; preserve all other training and inference behavior.
mechanism: Unsorted top-four channel saliency
evidence_used: Top-four channel attention achieved the best verified result at 9,322 correct versus 9,320 for hard maxima, but later top-k variants timed out, motivating the same validated statistic with sorting disabled because only its mean is consumed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.
change: Restore shared global-average-plus-maximum channel attention and replace plain paired-view cross-entropy with an equal blend of individual-view cross-entropy and validation-matched flip-ensemble negative log-likelihood.
mechanism: Probability-ensemble-aware flip-pair training
evidence_used: Shared average-plus-maximum attention reached 9,320 correct and paired-view probability ensembling was beneficial; top-four attention improved by only two additional correct predictions but repeatedly encountered runtime failures, motivating a computationally light objective-level refinement on the reliable 9,320-correct design.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.78843287518248, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1926749450683594, "validation_score": 9307.419225709458}



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
