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
hypothesis: Top-four channel saliency with algebraically fused attention and redundant preparation flipping removed will exceed 9,322 correct predictions while completing within the verification limit.
change: Replace global-maximum channel evidence with the mean of each channel’s four strongest activations, combine both shared linear-attention inputs into one call, and make batch preparation a no-op.
mechanism: Runtime-optimized top-four channel saliency
evidence_used: Reference Design 2 achieved the best verified result—9,322 correct—using top-four saliency versus 9,320 for hard maxima; subsequent top-k variants timed out, motivating removal of a redundant flip and one attention-kernel invocation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A maximum over overlapping 2×2 feature averages will exceed 9,322 correct predictions by preserving the successful four-activation saliency effect while favoring spatially coherent evidence and avoiding top-k’s runtime cost.
change: Replace hard channel maxima with peak 2×2 average pooling, fuse the bias-free attention calls, and restore ordinary BatchNorm momentum.
mechanism: Local four-pixel peak pooling
evidence_used: Top-four saliency achieved the best result at 9,322 correct, but later top-k implementations timed out; ordinary BatchNorm matched sample-weighted BatchNorm at 9,320 correct while finishing faster.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.14246404101141, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.19244651679992675, "validation_score": 9302.419306017466}

RECENT RESULT
hypothesis: Replacing the redundant preparatory flip with one-pixel translations during the first 75% of training will exceed 9,322 correct predictions by adding genuine spatial diversity, while clean final updates preserve alignment with validation images.
change: Cycle deterministically through centered and eight neighboring translations using normalization-safe replicated padding, then train on unshifted batches for the final quarter; paired horizontal-view supervision remains unchanged.
mechanism: Early one-pixel translation augmentation with clean-data refinement
evidence_used: The current hard-maximum design reaches 9,320 correct, only two behind the best top-four model, while paired-view training already includes both horizontal orientations, making the existing random preparatory flip merely swap their order. This motivates a low-cost augmentation orthogonal to the timeout-prone attention changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using batch size 125 will exceed 9,322 correct predictions by giving every training example equal optimizer-step weight and providing 800 uniform updates without materially increasing runtime.
change: Change the batch size from 128 to 125, which exactly divides the 50,000-image split and eliminates both 80-example tail batches.
mechanism: Tail-free divisor batching
evidence_used: The top-four design achieved 9,322 correct with batch size 128, while weighting BatchNorm updates for the smaller tail batches left the hard-maximum design at 9,320; this suggests testing the remaining tail effect—disproportionate optimizer updates—by eliminating partial batches entirely.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Unsorted top-four attention will retain the best design’s 9,322 correct predictions, while temperature 1.05 lowers its validation cross-entropy and reduced top-k ordering work improves completion reliability.
change: Restore top-four channel evidence using unsorted selection and mildly soften only the arithmetic flip-mixture logits at validation.
mechanism: Calibrated unsorted top-four channel saliency
evidence_used: Reference Design 2 achieved the best verified count with top-four saliency; arithmetic flip ensembling produced lower cross-entropy than geometric ensembling, motivating class-preserving temperature calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending each channel’s maximum with its global RMS will exceed 9,322 correct predictions by approximating top-four saliency with less outlier sensitivity and denser gradients, without top-k’s timeout-prone sorting cost.
change: Replace hard-maximum attention with a fused max–RMS saliency descriptor and restore ordinary BatchNorm momentum for lower runtime.
mechanism: Global max–RMS channel saliency
evidence_used: Top-four saliency achieved the best verified count of 9,322 versus 9,320 for hard maxima, while spatially constrained peak pooling fell to 9,302 and top-k variants repeatedly timed out; sample-weighted BatchNorm also provided no accuracy gain.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.09702225006185, "validation_accuracy": 0.9317, "validation_correct": 9317, "validation_cross_entropy": 0.1924756732940674, "validation_score": 9317.41929576527}

RECENT RESULT
hypothesis: Adding one-pixel up/down views to the hard-maximum model’s validation ensemble will exceed 9,322 correct predictions by reducing pooling-phase sensitivity, while arithmetic probability averaging preserves the previously observed cross-entropy advantage over mean-logit ensembling.
change: During evaluation, jointly infer the original, horizontal-flip, one-pixel-up, and one-pixel-down views in both orientations, then average their class probabilities.
mechanism: Batched vertical-shift probability ensembling
evidence_used: The hard-maximum design reliably finishes near 78 seconds with 9,320 correct, and arithmetic flip averaging achieved lower cross-entropy than geometric averaging at the same count; the attempted training-time translation experiment timed out, motivating translation diversity only during validation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 63.78145879181102, "validation_accuracy": 0.9305, "validation_correct": 9305, "validation_cross_entropy": 0.20198182525634767, "validation_score": 9305.41597966749}

RECENT RESULT
hypothesis: Softening logits only when the original and flipped views predict different classes will preserve all 9,322 ensemble predictions while lowering validation cross-entropy on these uncertainty-signaling cases.
change: Keep training and arithmetic flip ensembling unchanged, but apply temperature 1.10 only to validation samples whose two views disagree.
mechanism: Disagreement-conditioned flip calibration
evidence_used: Arithmetic probability ensembling achieved lower cross-entropy than geometric ensembling at the same 9,320 correct predictions, indicating that handling flip-view uncertainty affects the tie-breaker; positive per-sample temperature scaling cannot change the ensemble argmax.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing hard maxima with a differentiable L4 upper-tail descriptor will exceed 9,322 correct predictions by capturing multiple strong activations like top-four pooling while avoiding its timeout-prone selection cost.
change: Use positive L4 pooling normalized to four effective salient sites and fuse the two bias-free channel-attention calls into one linear operation.
mechanism: Four-site-normalized L4 channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 versus 9,320 for hard maxima, while max–RMS pooling reached 9,317 and repeated top-k variants timed out; a higher-order pooled descriptor targets the successful upper tail more selectively than RMS without sorting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.78321512509137, "validation_accuracy": 0.9299, "validation_correct": 9299, "validation_cross_entropy": 0.19290080947875976, "validation_score": 9299.419146333063}

RECENT RESULT
hypothesis: Restoring top-four channel saliency and evaluating a sparsely sampled EMA of the final training half will exceed 9,322 correct predictions by combining the best verified attention descriptor with lower low-learning-rate checkpoint variance.
change: Restore exact top-four attention, ordinary BatchNorm momentum, and maintain a low-overhead EMA of parameters and floating BatchNorm state every 16 steps during the final half of training.
mechanism: Top-four saliency with sampled late-state EMA
evidence_used: Top-four saliency produced the best verified count of 9,322; sample-weighted BatchNorm remained at 9,320 while increasing runtime, so ordinary BatchNorm provides headroom for sparse late-state averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Smoothly introducing top-four saliency during the final half of training will exceed 9,322 correct predictions while avoiding the repeated runtime failures of using top-k throughout training.
change: Train initially with fast hard-maximum attention, linearly transition to exact top-four attention over the final half, use top-four attention at evaluation, and restore arithmetic probability flip ensembling.
mechanism: Progressive hard-max-to-top-four channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322, while hard maxima reliably completed at 9,320 and later full-training top-k variants repeatedly timed out; arithmetic probability ensembling also had lower cross-entropy than mean-logit ensembling at the same hard-max correct count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.47079716692679, "validation_accuracy": 0.9317, "validation_correct": 9317, "validation_cross_entropy": 0.19215523567199708, "validation_score": 9317.41940846715}

RECENT RESULT
hypothesis: Adding a 25% evaluation-aligned arithmetic flip-mixture loss while retaining 75% per-view supervision will exceed 9,322 correct predictions without material runtime increase.
change: Reuse the existing paired logits to optimize both individual-view NLL and the NLL of their arithmetic probability mixture.
mechanism: Flip-mixture-aligned paired loss
evidence_used: Arithmetic probability ensembling achieved lower cross-entropy than mean-logit ensembling at the same 9,320 correct predictions, while top-four attention produced the best 9,322 count; directly supervising that mixture is a low-overhead objective change to the best architecture.
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
