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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.56534891715273, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922137680053711, "validation_score": 9320.419387876082}
prior_hypothesis: Mildly favoring the more decisive horizontal view will finish within the reliable hard-maximum runtime and exceed 9,322 correct predictions by resolving flip disagreements more accurately.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.90509041585028, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19264950561523436, "validation_score": 9320.419234651627}
prior_hypothesis: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding uniformly cycled two-pixel translations to the reliable hard-maximum model will exceed 9,322 correct predictions by improving spatial robustness without the runtime risk of ranking-based attention.
change: Extend the existing paired-flip training augmentation with a deterministic 5×5 translation cycle using inexpensive replicated padding and cropping.
mechanism: Cyclic integer-translation augmentation
evidence_used: Hard-maximum attention reliably reached 9,320 correct, while most attention variants regressed or timed out and top-four improved by only two predictions; the paired loss already supplies both horizontal orientations, making spatial translation an informative orthogonal augmentation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.49137470801361, "validation_accuracy": 0.9156, "validation_correct": 9156, "validation_cross_entropy": 0.23165286254882814, "validation_score": 9156.40595854173}

RECENT RESULT
hypothesis: Restoring the qualified top-four descriptor will recover 9,322 correct predictions, while temperature 1.05 will preserve those predictions and reduce cross-entropy below 0.192551.
change: Use mean top-four channel evidence, restore ordinary BatchNorm momentum, and mildly soften the arithmetic-probability ensemble.
mechanism: Temperature-calibrated top-four channel saliency
evidence_used: Reference Design 2 is the only verified implementation reaching 9,322 correct; the earlier calibrated top-four experiment timed out, so isolating calibration on the exact qualified implementation is the most informative retry.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using each channel’s strongest contiguous 2×2 activation average will finish within the reliable hard-maximum runtime envelope and exceed 9,322 correct predictions by approximating top-four saliency without ranking overhead.
change: Replace global-maximum channel evidence with maximum local 2×2-average evidence, fuse the linear shared-attention calls, and use the proven arithmetic-probability flip ensemble.
mechanism: Peak 2×2 regional saliency attention
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly risked timeout; hard maximum reliably finished at 9,320, while global mean–maximum interpolation regressed. A peak local average preserves concentrated upper-tail evidence with dense gradients and pooling-only computation. Arithmetic probability averaging lowered hard-maximum cross-entropy without changing its correct count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.145087167155, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.19244651679992675, "validation_score": 9302.419306017466}

RECENT RESULT
hypothesis: Dividing the qualified top-four model’s probability-mixture logits by 1.05 will preserve its 9,322 correct predictions while lowering validation cross-entropy below 0.192551.
change: Apply temperature 1.05 only to evaluation logits, leaving initialization, training, parameters, and predictions by argmax unchanged.
mechanism: Argmax-invariant ensemble temperature calibration
evidence_used: The current top-four design has the best verified count, and arithmetic probability averaging previously improved ensemble cross-entropy; calibration remains untested because prior attempts timed out rather than producing contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fusing the exact top-four attention computation and removing redundant paired-view preprocessing will finish within the time limit, retain at least 9,322 correct predictions, and temperature 1.05 will lower cross-entropy below 0.192551.
change: Restore exact top-four channel evidence using unsorted selection, fuse the bias-free shared attention calls, eliminate the redundant random pre-flip, and soften only evaluation logits.
mechanism: Runtime-efficient exact top-four saliency with argmax-invariant calibration
evidence_used: Reference Design 2 uniquely reached 9,322 correct with exact top-four saliency, while subsequent faithful retries timed out rather than disproving its accuracy; the paired loss already presents both orientations, so its preceding random flip is computationally redundant.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 83.55110208294354, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.19272081451416015, "validation_score": 9301.419209586951}

RECENT RESULT
hypothesis: Averaging maxima from four interleaved spatial lattices will exceed 9,322 correct predictions by approximating top-four saliency without ranking overhead, while ordinary BatchNorm and temperature 1.05 preserve the faster qualified path and improve tie-breaking cross-entropy.
change: Replace each channel’s single maximum with four checkerboard-stratified maxima, fuse the linear attention calls, restore ordinary BatchNorm updates, and mildly soften evaluation logits.
mechanism: Checkerboard-stratified four-peak channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly approached or exceeded the time limit; hard-maximum attention reliably finished faster, while sample-weighted BatchNorm slightly worsened cross-entropy. Stratified maxima retain four upper-tail signals using only inexpensive reductions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Holding the peak learning rate through the first 50,000 examples before cosine decay will exceed 9,322 correct predictions by providing more optimization under the fixed two-exposure budget, while probability-space flip ensembling will improve cross-entropy if correct counts tie.
change: Delay cosine annealing until halfway through training and replace mean-logit validation ensembling with the verified arithmetic probability mixture.
mechanism: First-epoch learning-rate hold with second-epoch cosine consolidation
evidence_used: Hard-maximum attention repeatedly completed near 75–78 seconds with 9,320 correct, whereas the two-prediction top-four improvement repeatedly risked timeout. Reference Designs 1 and 2 also showed that probability ensembling lowers hard-maximum cross-entropy without changing its correct count, motivating an optimization-only change that preserves the reliable architecture and runtime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 79.86112145916559, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.1904667350769043, "validation_score": 9315.420003335892}

RECENT RESULT
hypothesis: Mildly weighting each validation orientation by its top-two log-probability margin will exceed 9,322 correct predictions by favoring the more decisive view on flip disagreements while preserving the qualified top-four model and training runtime.
change: Replace equal arithmetic-probability flip fusion with a normalized, modestly confidence-weighted probability mixture during evaluation only.
mechanism: Margin-weighted flip probability ensemble
evidence_used: Top-four saliency produced the best verified count of 9,322, and arithmetic probability fusion improved cross-entropy over mean-logit fusion without changing the hard-maximum model’s count; this isolates reliability-aware probability fusion without disturbing successful training.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing hard channel maxima with an eighth-power mean will finish within the reliable pooling runtime envelope and exceed 9,322 correct predictions by approximating top-four evidence with dense gradients and reduced single-outlier sensitivity.
change: Replace global-maximum channel evidence with a parameter-free eighth-power mean while preserving the qualified architecture, training objective, schedule, and probability-space flip ensemble.
mechanism: Eighth-power upper-tail channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly risked timeout, while hard maxima reliably reached 9,320. Unlike the regressing peak-2×2 approximation, a power mean retains global upper-tail evidence without ranking or imposing spatial locality.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.40948354196735, "validation_accuracy": 0.9317, "validation_correct": 9317, "validation_cross_entropy": 0.19269057350158691, "validation_score": 9317.41922021613}

RECENT RESULT
hypothesis: Mildly favoring the more decisive horizontal view will finish within the reliable hard-maximum runtime and exceed 9,322 correct predictions by resolving flip disagreements more accurately.
change: Restore ordinary BatchNorm momentum and replace equal validation-view averaging with a normalized, modestly margin-weighted probability mixture.
mechanism: Confidence-adaptive flip probability fusion
evidence_used: Reference Design 1 reached 9,320 correct faster and with slightly lower cross-entropy than sample-weighted BatchNorm; confidence-weighted fusion remains untested because its previous attempt was coupled to repeatedly timing-out top-four attention.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.56534891715273, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922137680053711, "validation_score": 9320.419387876082}

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
