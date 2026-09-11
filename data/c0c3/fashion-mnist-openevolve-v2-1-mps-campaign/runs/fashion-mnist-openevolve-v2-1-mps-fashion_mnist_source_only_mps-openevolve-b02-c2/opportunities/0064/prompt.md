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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.78843287518248, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1926749450683594, "validation_score": 9307.419225709458}
prior_hypothesis: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.09728554193862, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19226545028686523, "validation_score": 9320.419369696472}
prior_hypothesis: Restoring hard-maximum attention and weighting BatchNorm updates by actual batch size will exceed 9,322 correct predictions by retaining the 9,320-correct fast architecture while preventing the two 80-example tail batches from disproportionately shifting evaluation statistics.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring plain paired-view cross-entropy and applying 0.02 label smoothing will exceed 9,322 correct predictions by removing the harmful ensemble-aware objective while mildly regularizing the fast 9,320-correct hard-maximum model.
change: Replace the balanced individual-and-ensemble loss with a single label-smoothed cross-entropy over both orientations.
mechanism: Light label smoothing with paired-view supervision
evidence_used: Plain paired-view cross-entropy with hard-maximum attention reached 9,320 correct and lower cross-entropy, whereas ensemble-aware training fell to 9,307; light label smoothing is a low-overhead, previously untested loss refinement that preserves the faster architecture.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using 125-example batches with hard-maximum channel saliency will exceed 9,322 correct predictions by providing 800 uniform optimizer steps without partial batches while retaining the faster, lower-cross-entropy attention design.
change: Replace top-four saliency with global maximum attention and set the batch size to 125, which exactly divides each 50,000-example training pass.
mechanism: Batch-aligned hard-maximum attention
evidence_used: Hard-maximum attention achieved 9,320 correct with lower cross-entropy and 3.5 seconds less training time than top-four; batch size 125 removes the two 80-example tail batches and adds 18 full optimizer updates within that timing margin.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the 9,322-correct top-four attention while removing the functionally redundant preparation flip will achieve at least 9,323 correct predictions and finish within the time limit.
change: Replace global-maximum channel evidence with top-four averaging, and pass training batches through unchanged because the loss already evaluates both horizontal orientations.
mechanism: Redundancy-free top-four channel saliency
evidence_used: Top-four saliency produced the best verified result at 9,322 correct; the recent translation trial established that preparation-time flipping is redundant with paired-view training, so removing it provides runtime headroom for the stronger attention reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring fast global-maximum attention and mildly softening each orientation’s probabilities before averaging will exceed 9,322 correct predictions by reducing domination from an overconfident orientation while retaining probability averaging’s robustness.
change: Replace local-patch saliency with verified global-maximum saliency and use a temperature-1.25 flip ensemble, recalibrated back to the original logit scale.
mechanism: Temperature-softened probability flip ensemble
evidence_used: Global-maximum attention achieved 9,320 correct with lower cross-entropy and faster training than the 9,322-correct top-four model; the unverified geometric ensemble tested only the hard endpoint, motivating a low-cost intermediate aggregation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring top-four saliency and plain paired-view cross-entropy while using tanh-approximated GELUs will exceed 9,322 correct predictions and finish within the verification limit by preserving the strongest verified model while reducing activation overhead.
change: Replace maximum saliency with top-four averaging, remove the harmful ensemble-aware loss, and use tanh-approximated GELU throughout the network.
mechanism: Fast approximate-GELU top-four channel saliency
evidence_used: Top-four saliency with plain paired-view cross-entropy achieved the best verified result of 9,322 correct, but later top-four trials timed out; hard-maximum attention finished faster but peaked at 9,320, motivating a compute-focused restoration of the strongest design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using 200-example batches with proportionally scaled learning rate and moment decay will exceed 9,322 correct predictions while finishing reliably by reducing training from 782 to 500 uniform optimizer steps.
change: Preserve the best top-four attention model and paired-view loss, increase batch size to 200, and rescale AdamW’s learning rate and betas to approximately preserve optimizer dynamics per example.
mechanism: Example-time-matched large-batch AdamW
evidence_used: The current top-four model achieved the best result at 9,322 correct but required 78.8 seconds, while numerous subsequent 782–800-step trials timed out; batch size 200 exactly divides 50,000 and removes 282 optimizer steps without discarding examples.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding independent two-pixel translations to the fast 9,320-correct hard-maximum model will exceed 9,322 correct predictions by learning small positional invariance without materially increasing training time.
change: Replace the redundant preparation-time horizontal flip with per-image random crops from a two-pixel replicated border; paired-view loss still supplies both horizontal orientations.
mechanism: Per-image replicated-border translation augmentation
evidence_used: Hard-maximum attention reached 9,320 correct in 75.3 seconds, while top-four gained only two predictions at 78.8 seconds and later attention variants timed out; the paired loss already evaluates both horizontal orientations, making preparation-time flipping available for a low-overhead orthogonal augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring hard-maximum attention and weighting BatchNorm updates by actual batch size will exceed 9,322 correct predictions by retaining the 9,320-correct fast architecture while preventing the two 80-example tail batches from disproportionately shifting evaluation statistics.
change: Replace strongest-local-patch attention with verified global-maximum attention and scale BatchNorm momentum on each step to preserve approximately equal running-stat influence per example.
mechanism: Sample-weighted BatchNorm with fast peak attention
evidence_used: Hard-maximum attention reached 9,320 correct in 75.3 seconds, outperforming the current local-patch model’s 9,312 while leaving runtime headroom; the recent BatchNorm hypothesis identified undersized tail batches as a parameter-neutral source of evaluation noise.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.09728554193862, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19226545028686523, "validation_score": 9320.419369696472}

RECENT RESULT
hypothesis: Blending global maxima with per-channel RMS activations and restoring plain paired-view cross-entropy will exceed 9,322 correct predictions while avoiding top-k’s runtime cost.
change: Replace hard-maximum channel evidence with a fast equal blend of maximum and RMS descriptors, and remove the ensemble-aware loss that reduced accuracy.
mechanism: RMS-tempered channel saliency
evidence_used: Top-four saliency improved hard maximum from 9,320 to 9,322 correct, suggesting reduced outlier sensitivity, while repeated top-k variants timed out; plain paired-view cross-entropy reached 9,320 whereas the current ensemble-aware objective fell to 9,307.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A ramped 0.99 parameter EMA on the faster 9,320-correct hard-maximum architecture will exceed 9,322 correct predictions by suppressing late-update noise while retaining enough runtime headroom to finish verification.
change: Restore global-maximum channel saliency, maintain a low-overhead foreach EMA after optimizer steps, and swap the averaged parameters in for validation.
mechanism: Evaluation-time exponential weight averaging
evidence_used: Hard-maximum attention achieved 9,320 correct in 75.3 seconds versus top-four attention’s 9,322 in 78.8 seconds; recent top-four and augmentation trials timed out, motivating an orthogonal stabilization method on the faster qualified architecture.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training with fast global-maximum attention but replacing it with top-four averaging during evaluation will exceed 9,322 correct predictions while retaining the completed hard-maximum model’s training runtime.
change: Keep global-maximum channel evidence during training and compute the more robust top-four descriptor only during validation.
mechanism: Evaluation-only top-four channel saliency
evidence_used: Top-four saliency achieved the best verified result at 9,322 correct versus 9,320 for hard maximum, but repeated top-k training variants exceeded the time limit; restricting top-k to evaluation preserves its inference benefit without adding training overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Rebuilding BatchNorm running statistics over the low-learning-rate final quarter will exceed 9,322 correct predictions while retaining the completed hard-maximum architecture’s runtime.
change: Reset BatchNorm statistics at 75% training progress, then accumulate sample-weighted running statistics over the remaining examples.
mechanism: Final-quarter cumulative BatchNorm recalibration
evidence_used: Sample-weighting only the two tail batches left the 9,320-correct result unchanged versus ordinary BatchNorm, indicating tail influence was not the main issue; the fast hard-maximum model leaves room to test a broader, parameter-free reduction in evaluation-statistic noise.
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
