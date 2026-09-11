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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.90509041585028, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19264950561523436, "validation_score": 9320.419234651627}
prior_hypothesis: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.56534891715273, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922137680053711, "validation_score": 9320.419387876082}
prior_hypothesis: Mildly favoring the more decisive horizontal view will finish within the reliable hard-maximum runtime and exceed 9,322 correct predictions by resolving flip disagreements more accurately.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

## Recent verification evidence

RECENT RESULT
hypothesis: Mean top-two channel evidence will finish within the reliable runtime envelope and exceed 9,322 correct predictions by retaining top-four saliency’s reduced outlier sensitivity with less ranking overhead; probability ensembling with temperature 1.05 will improve tie-breaking cross-entropy without changing argmax predictions.
change: Replace hard channel maxima with each channel’s mean top-two activations, then use a mildly softened arithmetic-probability flip ensemble at evaluation.
mechanism: Runtime-balanced top-two channel saliency
evidence_used: Top-four saliency achieved the best verified count of 9,322 but repeatedly risked timeout, whereas hard maxima reliably reached 9,320 faster. Arithmetic-probability fusion also lowered hard-maximum cross-entropy without changing its correct count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Preserving the qualified model’s stochastic training path while fusing its linear attention calls will retain at least 9,322 correct predictions and finish reliably; temperature 1.05 will leave argmax predictions unchanged while reducing cross-entropy below 0.192551.
change: Apply the shared bias-free attention kernel once to the sum of mean and top-four descriptors, and soften only the final evaluation logits.
mechanism: Fused top-four attention with temperature calibration
evidence_used: The current top-four design has the best verified count of 9,322. Earlier optimized retries also removed the pre-loss random flip, changing the RNG sequence consumed by dropout and regressing to 9,301; this patch keeps that successful stochastic path intact while reducing one kernel call.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a modest agreement penalty between horizontally paired predictions will exceed 9,322 correct predictions by learning flip invariance directly while retaining the reliable hard-maximum architecture and runtime.
change: Preserve paired cross-entropy and add a 0.10-weighted Jensen–Shannon divergence between each image’s two orientation predictions.
mechanism: Paired-view Jensen–Shannon consistency
evidence_used: The reliable hard-maximum model reaches 9,320 correct in 66.6 seconds, while attention-based attempts to gain the remaining predictions repeatedly time out; this targets the already-present paired views with negligible additional computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.54820091696456, "validation_accuracy": 0.9303, "validation_correct": 9303, "validation_cross_entropy": 0.1912450294494629, "validation_score": 9303.419728928675}

RECENT RESULT
hypothesis: A 0.99-decay EMA of the reliable hard-maximum model will exceed 9,322 correct predictions by reducing late-step parameter noise without changing training examples, learned-parameter count, or augmentation.
change: Maintain a fused EMA of all learned parameters after every optimizer step and install the averaged weights when evaluation begins.
mechanism: Late-trajectory exponential weight averaging
evidence_used: The hard-maximum design reliably reaches 9,320 correct in 66.6–75.3 seconds, while attention variants frequently time out and consistency or schedule changes fell to 9,303–9,315; preserving its exact training path while smoothing only the final weights is the clearest untested improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 74.32841341686435, "validation_accuracy": 0.9304, "validation_correct": 9304, "validation_cross_entropy": 0.19334426651000977, "validation_score": 9304.418990574666}

RECENT RESULT
hypothesis: Cycling through all nine one-pixel translations will exceed 9,322 correct predictions by improving spatial robustness while retaining the reliable hard-maximum architecture and runtime.
change: Apply a deterministic, balanced 3×3 translation schedule after the existing random flip, using replicated border padding and no additional RNG draws.
mechanism: Balanced one-pixel translation augmentation
evidence_used: The reliable hard-maximum model repeatedly reaches 9,320 correct, while attention, consistency, schedule, and EMA changes either regress or risk timeout; mild translation augmentation is an orthogonal, low-cost source of invariance not tested by those results.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 57.42849454190582, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.20898301315307616, "validation_score": 9229.413570740498}

RECENT RESULT
hypothesis: Avoiding unnecessary top-four sorting will reduce runtime while preserving at least 9,322 correct predictions, and temperature 1.05 will lower cross-entropy without changing ensemble argmaxes.
change: Compute the existing top-four descriptor with `sorted=False` and divide only the final evaluation logits by 1.05.
mechanism: Unsorted top-four saliency with argmax-invariant temperature calibration
evidence_used: The current top-four design achieved the best count of 9,322 but finished near the time limit; the earlier unsorted attempt also removed the random pre-flip and changed dropout RNG, so it did not isolate this runtime optimization, while evaluation temperature remains prediction-invariant.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting hard-max evidence by 0.85 will exceed 9,322 correct predictions by reducing single-activation dominance like successful top-four saliency, while the fused bias-free attention call preserves the reliable runtime envelope.
change: Fuse the shared mean/max channel-attention calls and modestly downweight the maximum descriptor.
mechanism: Attenuated peak channel attention
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but often exceeded the time limit, while hard maxima reliably finished with 9,320; the eighth-power substitute regressed, motivating a cheaper direct adjustment of peak influence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.56459229183383, "validation_accuracy": 0.9318, "validation_correct": 9318, "validation_cross_entropy": 0.1926055290222168, "validation_score": 9318.419250110646}

RECENT RESULT
hypothesis: Preserving top-four saliency in the forward pass while routing its gradient through the inexpensive hard maximum will finish within the time limit and achieve at least 9,322 correct predictions; temperature 1.05 will then reduce cross-entropy below 0.192551 without changing predictions.
change: Compute unsorted top-four evidence from detached features, use hard-max evidence as its straight-through gradient, fuse the bias-free attention calls, and soften only the final ensemble logits.
mechanism: Straight-through top-four channel saliency with temperature calibration
evidence_used: Exact top-four saliency produced the best verified count of 9,322 but repeatedly timed out, whereas hard-max attention reliably finished with 9,320; this retains the successful top-four forward statistic while eliminating its ranking backward pass.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Selecting the four strongest row maxima per channel will finish within the reliable runtime envelope and exceed 9,322 correct predictions by retaining robust upper-tail evidence while reducing each top-k operation from 49 candidates to 7.
change: Replace global hard-max attention with top-four saliency over row maxima, and use the verified equal probability-space flip ensemble.
mechanism: Row-compressed top-four channel saliency
evidence_used: Exact spatial top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard maxima reliably finished at 9,320; compressing each channel to seven row maxima preserves salient, horizontally flip-invariant candidates with substantially less ranking work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gradually replacing hard-max channel evidence with top-four evidence during the final half of training will exceed 9,322 correct predictions while avoiding the repeated full-training top-four timeout.
change: Train with inexpensive hard-max saliency for the first half, smoothly transition to top-four saliency during the second half, and use pure top-four saliency for validation.
mechanism: Late-phase saliency curriculum
evidence_used: Full top-four training achieved the best verified count of 9,322 but repeatedly timed out, whereas hard-max training reliably finished in 66.6–75.3 seconds with 9,320 correct; limiting top-four computation to late optimization preserves adaptation time while targeting a safer runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the margin-weight coefficient from 0.10 to 0.25 will exceed 9,322 correct predictions by resolving more flip-view disagreements in favor of the more decisive prediction while preserving the exact training path and reliable runtime.
change: Strengthen only the evaluation-time probability-mixture weighting; leave architecture, optimization, augmentation, and example accounting unchanged.
mechanism: Stronger confidence-adaptive flip fusion
evidence_used: Moving from equal probability fusion to coefficient 0.10 retained 9,320 correct while improving cross-entropy from 0.192262 to 0.192214, indicating that confidence weighting was directionally useful but too mild to change enough argmax decisions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.92806970886886, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19227309799194336, "validation_score": 9319.41936700647}

RECENT RESULT
hypothesis: Averaging four regional maxima will finish within the reliable runtime envelope and exceed 9,322 correct predictions by approximating top-four saliency without a costly ranking operation.
change: Replace each channel’s single global maximum with the mean of a 2×2 adaptive-max grid, leaving training and evaluation otherwise unchanged.
mechanism: Spatially balanced four-region channel saliency
evidence_used: Global top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard maxima reliably reached 9,320; regional max pooling supplies four robust salient activations using the already-reliable pooling primitive instead of `topk`.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.54781024996191, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.1925997917175293, "validation_score": 9301.419252127556}



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
