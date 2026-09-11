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
hypothesis: Adding a 25% validation-aligned arithmetic-probability mixture loss will exceed 9,322 correct predictions while retaining reliable hard-maximum attention runtime; probability averaging will also reduce cross-entropy if correct counts tie.
change: Optimize a blend of per-view cross-entropy and paired flip-mixture NLL, and use the same arithmetic probability mixture during validation.
mechanism: Flip-mixture-aware paired supervision
evidence_used: Arithmetic probability ensembling reduced hard-maximum cross-entropy from 0.1926495 to 0.1922617 without changing its 9,320 correct predictions, while numerous additional saliency and augmentation computations timed out or regressed; directly supervising that proven aggregation is an inexpensive orthogonal test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.1965839159675, "validation_accuracy": 0.9299, "validation_correct": 9299, "validation_cross_entropy": 0.1935381923675537, "validation_score": 9299.418922497158}

RECENT RESULT
hypothesis: Preserving the 9,322-correct top-four model while applying temperature 1.05 will lower validation cross-entropy below 0.192551; unsorted selection, fused attention, and batched flip inference will reduce timeout risk.
change: Keep exact top-four saliency, disable unnecessary top-k sorting, fuse the bias-free channel-attention calls, evaluate both flip views in one forward pass, and mildly soften the probability-mixture logits.
mechanism: Batched calibrated top-four ensemble
evidence_used: Exact top-four saliency achieved the best verified count of 9,322, while arithmetic probability averaging gave the lowest hard-maximum cross-entropy without changing its predictions; repeated top-four calibration attempts timed out, motivating algebraically equivalent runtime reductions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Evaluating the reliably trained hard-maximum model with top-four channel saliency will exceed 9,322 correct predictions by capturing the inference-side benefit of the best reference without its training-time ranking overhead.
change: Retain global-maximum attention throughout training, but replace it with exact top-four averaging during evaluation.
mechanism: Validation-only top-four channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322, while hard-maximum training reliably finished with 9,320; repeated training-time top-four variants timed out, motivating an evaluation-only isolation of the descriptor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 55.78863395797089, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19561468048095704, "validation_score": 9315.418194932}

RECENT RESULT
hypothesis: A short-horizon EMA of the unchanged hard-maximum model will exceed 9,322 correct predictions by reducing final-step parameter noise while retaining the reliably completed training path.
change: Restore ordinary BatchNorm behavior and use an AdamW subclass that tracks a ramped parameter EMA without affecting training, then installs the averaged weights after the exact final optimizer step.
mechanism: Ramped final-weight exponential averaging
evidence_used: Ordinary-BatchNorm hard-maximum attention reliably finished in 75.3 seconds with 9,320 correct and slightly better cross-entropy than sample-weighted BatchNorm; attention-descriptor changes mostly regressed or timed out, motivating a low-overhead orthogonal change that leaves every training forward pass unchanged.
result: training did not finish within the verification time limit

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
