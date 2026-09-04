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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.09728554193862, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19226545028686523, "validation_score": 9320.419369696472}
prior_hypothesis: Restoring hard-maximum attention and weighting BatchNorm updates by actual batch size will exceed 9,322 correct predictions by retaining the 9,320-correct fast architecture while preventing the two 80-example tail batches from disproportionately shifting evaluation statistics.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.90509041585028, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19264950561523436, "validation_score": 9320.419234651627}
prior_hypothesis: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.

## Recent verification evidence

RECENT RESULT
hypothesis: Exact top-four saliency with its sparse backbone gradient detached will exceed 9,322 correct predictions while avoiding the runtime failures of later top-k variants.
change: Restore unsorted top-four channel evidence, stop its gradient into the feature backbone, and algebraically fuse the two bias-free attention calls.
mechanism: Stop-gradient top-four channel saliency
evidence_used: Reference Design 2 achieved the best verified count of 9,322 with exact top-four saliency; repeated top-k variants timed out, motivating preservation of its forward evidence while removing selection backward work and one redundant attention invocation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging each channel’s two strongest activations will exceed 9,322 correct predictions by retaining the rank-based saliency advantage of top-four attention while staying closer to the reliable hard maximum and avoiding the repeatedly timeout-prone top-k operator.
change: Replace hard channel maxima with an exact max-plus-second-max descriptor computed using reductions and masking, fuse the linear attention calls, and restore ordinary BatchNorm momentum.
mechanism: Exact top-two channel saliency without top-k sorting
evidence_used: Exact top-four saliency achieved the best verified result of 9,322 correct versus 9,320 for hard maxima, while smooth max–RMS and L4 approximations fell to 9,317 and 9,299; ordinary BatchNorm matched sample-weighted BatchNorm at 9,320 with slightly better cross-entropy and lower overhead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.80949499993585, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.19339137344360352, "validation_score": 9294.418974035783}

RECENT RESULT
hypothesis: Exact top-four saliency computed without `topk` will finish reliably and recover at least 9,322 correct predictions, exceeding the current 9,320-correct design.
change: Restore top-four attention using four max-and-scatter reductions, fuse the bias-free attention calls, restore arithmetic probability flip ensembling, and remove the redundant preparatory flip.
mechanism: Iterative-reduction exact top-four channel saliency
evidence_used: Reference Design 3 reached 9,322 correct with top-four saliency, while later `topk` variants timed out; the reduction-based top-two implementation finished in 71.8 seconds, showing this computational route avoids the problematic sorting overhead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 70.76510787499137, "validation_accuracy": 0.9305, "validation_correct": 9305, "validation_cross_entropy": 0.19256290893554687, "validation_score": 9305.419265093902}

RECENT RESULT
hypothesis: Dividing the arithmetic flip-mixture logits by 1.05 will preserve all 9,322 argmax predictions while lowering validation cross-entropy, strictly improving validation_score at the same correct count.
change: Apply temperature 1.05 only to the final validation logits; training remains exactly unchanged.
mechanism: Mild global ensemble temperature calibration
evidence_used: Arithmetic probability flip ensembling previously lowered cross-entropy relative to mean-logit ensembling at the same 9,320 correct predictions, while accuracy-focused attention alternatives consistently failed to surpass the current top-four model’s 9,322 correct.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging four sparsely sampled low-learning-rate states with the terminal state will strictly exceed the current 9320.419371 score by stabilizing late optimization without adding forward passes or learned parameters.
change: Accumulate floating model parameters and BatchNorm state every 16 steps during the final 64 steps, then install their average with the terminal state when evaluation begins.
mechanism: Final-valley checkpoint averaging
evidence_used: The hard-maximum model reliably completed in 75.3 seconds with 9,320 correct, whereas top-k-based improvements repeatedly timed out; this isolates low-overhead late-state averaging on the reliable architecture.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 65.73454754194245, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.19211699752807618, "validation_score": 9309.419421920027}

RECENT RESULT
hypothesis: Restoring the qualified top-four attention design with ordinary BatchNorm momentum will improve the current 9,320 correct predictions to the previously verified 9,322.
change: Replace hard global-max channel evidence with the mean of each channel’s four strongest activations and remove sample-weighted BatchNorm momentum.
mechanism: Exact top-four channel saliency
evidence_used: Reference Design 2 achieved the best verified result—9,322 correct—while the current hard-maximum, sample-weighted-BatchNorm design achieved 9,320; smooth and reduction-based substitutes failed to reproduce the top-four gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 0.02 label smoothing to the reliable hard-maximum model will exceed 9,322 correct predictions, while arithmetic probability averaging will reduce cross-entropy relative to mean-logit ensembling.
change: Add low-overhead label smoothing to paired-view cross-entropy and restore arithmetic probability averaging for validation flips.
mechanism: Mild label-smoothed paired supervision with arithmetic flip ensembling
evidence_used: Hard-maximum training reliably reached 9,320 correct while repeated attention alternatives regressed or timed out; arithmetic probability ensembling retained 9,320 correct and reduced cross-entropy from 0.1926495 to 0.1922617, motivating an orthogonal calibration-oriented training regularizer.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the redundant preparatory flip with random one-pixel translations will exceed 9,322 correct predictions by adding positional diversity while preserving exact paired horizontal-view supervision and the best verified top-four attention architecture.
change: Generate an independently translated crop for each training image using replicate padding; the existing loss still trains on both that crop and its horizontal flip.
mechanism: Per-example one-pixel translation augmentation
evidence_used: Exact top-four saliency achieved the best verified count of 9,322, while subsequent attention and checkpoint variants regressed or timed out; this preserves that architecture and tests a low-overhead, orthogonal augmentation improvement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Cheap deterministic translations on the reliable hard-maximum architecture will exceed 9,322 correct predictions by improving positional robustness without the timeout overhead of per-example augmentation or top-k attention.
change: Replace the redundant preparatory random flip with a replicate-padded crop cycling through all nine one-pixel offsets; paired-view supervision still supplies both horizontal orientations.
mechanism: Cyclic batch-shared one-pixel translation augmentation
evidence_used: The hard-maximum model reliably completed with 9,320 correct, while the per-example translation experiment on top-four attention timed out; batch-shared slicing isolates the promising orthogonal augmentation at substantially lower overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a mild symmetric KL penalty between the already-computed paired-view predictions will exceed 9,322 correct predictions by encouraging flip-invariant decision boundaries without the runtime cost of additional model passes.
change: Restore ordinary BatchNorm momentum and augment paired-view cross-entropy with a 0.10-weighted, stop-gradient symmetric KL consistency loss.
mechanism: Symmetric flip-consistency regularization
evidence_used: Ordinary BatchNorm reached 9,320 correct with slightly lower cross-entropy and runtime than sample-weighted BatchNorm, while arithmetic flip ensembling improved cross-entropy at the same count; repeated top-k and augmentation variants timed out, motivating a low-overhead loss that reuses existing paired logits.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing each channel’s single global maximum with the mean of four regional maxima will exceed 9,322 correct predictions by retaining multiple salient responses without timeout-prone global ranking; arithmetic flip ensembling will improve cross-entropy when counts tie.
change: Pool one maximum from each of four spatial regions, fuse the bias-free channel-attention calls, and evaluate with an arithmetic probability mixture.
mechanism: Spatially distributed four-region channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 versus 9,320 for global maxima, but sorting-based versions repeatedly timed out and iterative global reductions fell to 9,305; arithmetic probability ensembling previously preserved 9,320 correct while lowering cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using single-view training during the final 10% of the near-zero-learning-rate cosine tail will prevent timeout while retaining at least 9,322 correct predictions, and temperature 1.05 will lower cross-entropy enough to exceed 9322.419269.
change: Preserve exact top-four attention, halve forward work in the final cosine tail with batch-size-adjusted BatchNorm momentum, and mildly soften only the validation flip-mixture logits.
mechanism: Calibrated single-view cosine-tail refinement
evidence_used: Top-four attention produced the best verified count of 9,322 but required 78.8 seconds and repeatedly timed out afterward; the reliable hard-maximum design finished near 75 seconds, while arithmetic probability ensembling reduced cross-entropy at unchanged accuracy. The final 10% contributes less than 0.4% of the cosine schedule’s integrated learning rate.
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
