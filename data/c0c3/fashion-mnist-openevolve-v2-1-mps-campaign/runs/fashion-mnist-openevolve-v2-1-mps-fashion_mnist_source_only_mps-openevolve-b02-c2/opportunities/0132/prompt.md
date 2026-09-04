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
hypothesis: A 75/25 maximum-to-global-mean saliency blend will exceed 9,322 correct predictions by approximating top-four activation magnitude and distributing gradients more broadly, while arithmetic probability ensembling will improve cross-entropy if counts tie.
change: Replace hard-maximum channel evidence with a fused, parameter-free shrinkage estimate and switch validation from mean logits to the proven arithmetic probability mixture.
mechanism: Conservative upper-tail saliency shrinkage
evidence_used: Exact top-four saliency achieved the best result at 9,322 correct, but repeated ranking-based variants timed out; more aggressive approximations regressed to 9,297–9,305, motivating a conservative interpolation that preserves maximum evidence and hard-maximum runtime. Arithmetic probability averaging previously lowered hard-maximum cross-entropy without changing its correct count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.58690866688266, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19324755897521972, "validation_score": 9312.419024532033}

RECENT RESULT
hypothesis: Preserving exact top-four saliency while learning a three-parameter contrast kernel between average and salient evidence will exceed 9,322 correct predictions by relaxing the shared-kernel constraint without changing initialization or material runtime.
change: Add a zero-initialized residual kernel that specializes channel attention according to the difference between top-four and average descriptors, while fusing their existing shared response.
mechanism: Descriptor-contrast residual channel attention
evidence_used: Exact top-four saliency produced the best verified count of 9,322, whereas alternative saliency approximations regressed; this patch preserves that descriptor and makes only its fusion more expressive.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging only three late cosine-schedule checkpoints will exceed 9,322 correct predictions by reducing final optimizer noise without the timeout caused by updating an EMA after every step.
change: Replace AdamW with a compatible subclass that samples weights at 80%, 90%, and 100% of training, forms a final-biased 25/25/50 average, and installs it after the exact final optimizer step.
mechanism: Sparse tail-weight exponential averaging
evidence_used: The unchanged hard-maximum model reliably reached 9,320 correct, while the per-step ramped EMA experiment timed out; sparse checkpoint averaging tests the same promising stabilization mechanism with only three whole-model updates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training with inexpensive hard-maximum attention for the first 40% of steps, then adapting under exact top-four saliency for the remaining 60%, will exceed 9,322 correct predictions while avoiding the timeout risk of full-run top-four pooling.
change: Restore ordinary BatchNorm momentum and switch channel attention from global maxima to exact top-four evidence after 40% of training; evaluation always uses top-four saliency.
mechanism: Late top-four saliency curriculum
evidence_used: Full-run top-four saliency achieved the best verified count of 9,322 but took 78.8 seconds, while hard-maximum training finished faster at 9,320; evaluation-only switching regressed to 9,315, showing that the model needs training exposure to the top-four descriptor.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying exact top-four saliency to one view of every training pair and hard-maximum saliency to the other will finish within the time limit and exceed 9,322 correct predictions by preserving per-example top-four exposure while regularizing the shared attention kernel; arithmetic probability ensembling will reduce cross-entropy when counts tie.
change: Use top-four channel evidence for the first paired view and inexpensive hard-maximum evidence for its flipped counterpart during training, use top-four evidence for both validation views, fuse the shared bias-free attention calls, and return an arithmetic probability ensemble.
mechanism: Asymmetric paired saliency co-training
evidence_used: Full top-four training achieved the best verified count of 9,322 but cost 78.8 seconds, while validation-only top-four regressed to 9,315, showing that training exposure is necessary. Hard-maximum training reliably finished near 9,320, and arithmetic probability averaging previously improved its cross-entropy without changing predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a modest symmetric KL penalty between paired horizontal views while retaining ordinary per-view cross-entropy will exceed 9,322 correct predictions by reducing orientation and dropout disagreement without materially increasing runtime.
change: Remove the redundant pre-loss random flip because the paired loss already includes both orientations, then add a 0.10-weight symmetric consistency penalty between their logits.
mechanism: Flip-pair consistency regularization
evidence_used: Exact top-four attention reached the best count of 9,322, while the stronger flip-mixture objective regressed to 9,299; this motivates preserving the successful cross-entropy objective and adding only a conservative disagreement penalty. Removing redundant view swapping offsets its small loss-computation cost.
result: training did not finish within the verification time limit

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
