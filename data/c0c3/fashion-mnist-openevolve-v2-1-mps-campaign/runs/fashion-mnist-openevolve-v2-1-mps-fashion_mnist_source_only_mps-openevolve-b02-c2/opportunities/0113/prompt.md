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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.09728554193862, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19226545028686523, "validation_score": 9320.419369696472}
prior_hypothesis: Restoring hard-maximum attention and weighting BatchNorm updates by actual batch size will exceed 9,322 correct predictions by retaining the 9,320-correct fast architecture while preventing the two 80-example tail batches from disproportionately shifting evaluation statistics.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.90509041585028, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19264950561523436, "validation_score": 9320.419234651627}
prior_hypothesis: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.

## Recent verification evidence

RECENT RESULT
hypothesis: Using single-view training during the final 10% of the near-zero-learning-rate cosine tail will prevent timeout while retaining at least 9,322 correct predictions, and temperature 1.05 will lower cross-entropy enough to exceed 9322.419269.
change: Preserve exact top-four attention, halve forward work in the final cosine tail with batch-size-adjusted BatchNorm momentum, and mildly soften only the validation flip-mixture logits.
mechanism: Calibrated single-view cosine-tail refinement
evidence_used: Top-four attention produced the best verified count of 9,322 but required 78.8 seconds and repeatedly timed out afterward; the reliable hard-maximum design finished near 75 seconds, while arithmetic probability ensembling reduced cross-entropy at unchanged accuracy. The final 10% contributes less than 0.4% of the cosine schedule’s integrated learning rate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending ordinary per-view cross-entropy with arithmetic flip-mixture negative log-likelihood will exceed 9,322 correct predictions by directly optimizing the ensemble used during validation without additional model passes.
change: Reuse the paired logits to combine equal-weight individual-view and probability-mixture losses.
mechanism: Validation-aligned paired-view mixture supervision
evidence_used: Arithmetic probability flip ensembling reduced cross-entropy from 0.1926495 to 0.1922617 at the same 9,320 correct predictions; aligning training with that beneficial validation rule is an orthogonal, low-overhead change to the reliable hard-maximum design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Selecting the exact four strongest activations via one kth-value threshold will exceed 9,322 correct predictions by preserving the best reference design’s saliency while avoiding its timeout-prone `topk` operation.
change: Replace hard maxima with threshold-selected top-four means and restore ordinary BatchNorm momentum.
mechanism: Order-statistic top-four channel saliency
evidence_used: Exact `topk`-based top-four attention achieved the best verified result of 9,322 correct; hard maxima reached 9,320, while iterative maximum extraction regressed to 9,305. Ordinary BatchNorm matched sample-weighted BatchNorm with slightly better cross-entropy and less overhead.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Unsorted top-four selection with a fused bias-free attention call will retain at least the reference design’s 9,322 correct predictions while reducing its timeout-prone overhead; arithmetic probability flip ensembling will improve cross-entropy when prediction counts tie.
change: Replace global-max channel evidence with exact unsorted top-four saliency, fuse the two linear channel-attention calls, and use arithmetic probability averaging at validation.
mechanism: Unsorted exact top-four channel saliency
evidence_used: Reference Design 3 achieved the best verified count, 9,322, using exact top-four saliency, whereas hard maxima reached 9,320 and iterative max extraction fell to 9,305. Repeated sorted-top-k runs timed out, motivating `sorted=False` and attention-call fusion while preserving the successful selection rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing the redundant preparatory flip will retain at least 9,322 correct predictions while reducing overhead enough for the verified top-four attention design to finish reliably.
change: Pass canonical images directly to the loss; paired supervision still trains on every image and its horizontal flip.
mechanism: Canonical paired-view training
evidence_used: Exact top-four saliency achieved the best verified result of 9,322 correct, but subsequent runs repeatedly timed out; the preparatory flip only swaps the order of the two views later constructed by training_loss.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 80% global-maximum evidence with 20% global-average evidence will exceed 9,322 correct predictions by reducing maximum outlier sensitivity while retaining salient activations, and fusing the bias-free attention calls will preserve the reliable runtime.
change: Approximate top-four channel saliency with a zero-overhead maximum/average blend and compute the combined channel gate with one convolution.
mechanism: Fused shrinkage channel saliency
evidence_used: Reference Design 2 reached the best verified count of 9,322 by replacing a single maximum with top-four averaging, but repeated top-k implementations timed out; the hard-maximum design reliably finishes with 9,320, motivating a cheap interpolation toward smoother saliency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A batch size of 125 will exceed 9,322 correct predictions by eliminating both partial epoch-ending batches and providing 800 uniform optimizer updates, while redundant-flip removal and mathematically equivalent attention fusion keep runtime below the verification limit.
change: Use batches that exactly divide the 50,000-image split, restore ordinary BatchNorm momentum, remove the preparatory flip that merely swaps paired-view order, and fuse the two bias-free channel-attention calls.
mechanism: Exact-epoch paired training with fused channel gating
evidence_used: The ordinary-BatchNorm hard-maximum reference reliably finished in 75.3 seconds with 9,320 correct, whereas sample-weighted tail handling did not improve that count; the current batch size creates two 80-example tail batches, so exact divisibility tests stronger tail removal without timeout-prone top-four selection.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Switching from hard-maximum to exact top-four channel saliency halfway through training will exceed 9,322 correct predictions while avoiding the runtime failures of full-training top-four selection.
change: Train the first half with fast maximum saliency, fine-tune the second half and evaluate with unsorted exact top-four saliency, eliminate the redundant preparatory flip, fuse the linear attention calls, and use arithmetic probability averaging for validation.
mechanism: Late-phase exact top-four saliency curriculum
evidence_used: Reference Design 3 achieved the best verified count of 9,322 using top-four saliency, while hard-maximum designs reliably completed around 75–78 seconds with 9,320; repeated full-training top-four variants timed out, motivating a half-duration curriculum that retains late adaptation while reducing selection overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying temperature 1.05 to the flip-mixture logits will preserve the current design’s 9,322 correct predictions while reducing validation cross-entropy below 0.192551.
change: Mildly soften the existing arithmetic probability-mixture output without changing training, parameters, runtime materially, or validation argmaxes.
mechanism: Validation-only temperature calibration
evidence_used: Exact top-four saliency produced the best verified count of 9,322; positive temperature scaling preserves those predictions and cleanly isolates the calibration idea from the timed-out experiment that combined it with altered tail training.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing single-pixel maxima with the strongest 2×2 mean will exceed 9,322 correct predictions while avoiding the runtime cost of exact top-four ranking.
change: Use overlapping 2×2 average pooling followed by global maximum for channel saliency, and fuse the bias-free attention calls.
mechanism: Strongest contiguous four-cell channel saliency
evidence_used: Reference Design 2 reached 9,322 correct with top-four averaging versus 9,320 for hard maxima, but subsequent ranking-based implementations timed out; a strongest-local-mean descriptor preserves four-response smoothing using optimized pooling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.7785815000534, "validation_accuracy": 0.9297, "validation_correct": 9297, "validation_cross_entropy": 0.19252699699401857, "validation_score": 9297.419277719717}

RECENT RESULT
hypothesis: Blending the global maximum with mean row and column maxima will exceed 9,322 correct predictions by approximating distributed top-four evidence without ranking overhead or the locality bias of 2×2 pooling.
change: Replace hard-maximum channel evidence with a 50/25/25 global/row/column maximum blend and restore ordinary BatchNorm momentum.
mechanism: Axis-stratified upper-tail channel saliency
evidence_used: Exact top-four saliency achieved 9,322 correct versus 9,320 for hard maxima, while contiguous 2×2 pooling fell to 9,297; ordinary BatchNorm also ran faster with slightly lower cross-entropy than sample-weighted updates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the redundant preparatory flip with cycling one-pixel translations will exceed 9,322 correct predictions by adding low-cost spatial regularization, while arithmetic probability ensembling will lower cross-entropy when prediction counts tie.
change: Cycle through all nine offsets of a padded 3×3 translation grid during training and replace validation mean-logit ensembling with the verified arithmetic probability mixture.
mechanism: Deterministic one-pixel translation augmentation
evidence_used: Paired supervision makes the existing preparatory flip only swap view order; meanwhile, hard-maximum designs reliably reached 9,320 correct and arithmetic probability ensembling reduced cross-entropy from 0.1926495 to 0.1922617 without changing that count. Attention-based alternatives mostly timed out or regressed, motivating an orthogonal, inexpensive augmentation.
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
