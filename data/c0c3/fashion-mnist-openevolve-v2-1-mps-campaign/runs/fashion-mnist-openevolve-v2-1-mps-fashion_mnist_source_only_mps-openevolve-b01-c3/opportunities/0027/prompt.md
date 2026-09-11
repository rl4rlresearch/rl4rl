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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.06300445785746, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.23108364181518554, "validation_score": 9209.406146246296}
prior_hypothesis: Adding an inexpensive 7×7 spatial refinement block to the best multi-scale model will exceed 9,202 correct predictions by improving local feature interactions that global channel gating could not capture.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.886122666997835, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.2185217487335205, "validation_score": 9275.410333258737}
prior_hypothesis: Averaging only the final 5% of iterates will exceed 9,282 correct predictions by retaining the proven variance reduction of tail averaging while reducing mismatch with terminal BatchNorm statistics and halving averaging overhead.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249466, "training_seconds": 60.428222542162985, "validation_accuracy": 0.9202, "validation_correct": 9202, "validation_cross_entropy": 0.23369549179077148, "validation_score": 9202.4052863963}
prior_hypothesis: Adding input-conditioned channel gating to the qualified multi-scale model will exceed 9,202 correct predictions by emphasizing class-relevant texture and silhouette channels while preserving the proven model exactly at initialization.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 65.26041350001469, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.21929060821533203, "validation_score": 9282.410074511057}
prior_hypothesis: Averaging model parameters over the final 10% of cosine-decay updates will exceed 9,262 correct predictions by reducing terminal minibatch noise without changing the proven architecture, augmentation, or inference ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing single-location global maxima with the mean of each channel’s four strongest spatial responses will exceed 9,209 correct predictions by preserving salient features while reducing sensitivity to noisy or misaligned activation spikes.
change: Restore the verified 249,961-parameter multi-scale spatial-refinement model and its hierarchical test-time ensemble, but use top-four spatial averaging for the peak-statistics half of the classifier input.
mechanism: Robust top-k spatial peak pooling
evidence_used: Reference Design 3 achieved the best result—9,209 correct—using late spatial refinement and concatenated mean/max statistics; the large earlier gain from global-statistical pooling motivates refining its parameter-free peak statistic without adding the costly second refinement stage that timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing hard maxima with softmax-weighted peak statistics will exceed 9,209 correct predictions by retaining salient responses while distributing learning across several strong spatial locations.
change: Restore the verified 249,961-parameter spatial-refinement model and training recipe, but replace costly top-k pooling with efficient differentiable soft peak pooling.
mechanism: Smooth top-response spatial pooling
evidence_used: Spatial refinement achieved the best result at 9,209 correct, while robust top-four pooling was promising but timed out; smooth peak pooling tests the same robustness hypothesis without discrete top-k selection.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing depthwise spatial filtering followed by pointwise mixing with one grouped 3×3 convolution will exceed 9,209 correct predictions by learning spatial and cross-channel interactions jointly while preserving similar capacity and reducing sequential refinement operations.
change: Replace the late depthwise/pointwise refinement with a single 8-group 3×3 residual convolution and reduce classifier width from 61 to 60, yielding 249,854 learned parameters.
mechanism: Joint grouped spatial-channel residual refinement
evidence_used: Late spatial refinement produced the best result at 9,209 correct, whereas channel-only gating tied the earlier 9,202 result with worse cross-entropy; the proposed block strengthens the successful spatial mechanism without the extra stages that caused the two-stage design to time out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Expanding the verified late refinement from 3×3 to 5×5 will exceed 9,209 correct predictions by modeling broader spatial interactions without the sequential operations that caused the two-stage design to time out.
change: Restore the best spatial-refinement architecture, use a direct 5×5 depthwise kernel, and reduce classifier width to 53, yielding 249,873 learned parameters.
mechanism: Single-stage 5×5 depthwise spatial refinement
evidence_used: The 3×3 spatial-refinement model achieved the best result at 9,209 correct; a proposed two-stage effective-5×5 refinement timed out, motivating a computationally simpler single-stage test of the same broader-context hypothesis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training the verified spatial-refinement model only on centered and one-pixel cardinal translations will exceed 9,209 correct predictions by removing the larger and diagonal shifts that conflict with its best inference ensemble.
change: Restore the 249,961-parameter multi-scale spatial-refinement architecture and hierarchical ensemble, then replace independent ±2-pixel augmentation with uniform sampling from the same five centered/cardinal positions used at validation.
mechanism: Evaluation-matched cardinal translation training
evidence_used: Reference Design 3 achieved 9,209 correct, while adding diagonal inference translations reduced an earlier model from 9,111 to 9,109; the current 25-position training augmentation includes both diagonal and two-pixel shifts despite the strongest evaluation evidence favoring five cardinal positions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 68.40002791699953, "validation_accuracy": 0.9262, "validation_correct": 9262, "validation_cross_entropy": 0.22061247711181642, "validation_score": 9262.409630418642}

RECENT RESULT
hypothesis: Replacing local late refinement with one global, position-aware attention step will exceed 9,262 correct predictions by capturing content-dependent relationships between every pair of 7×7 feature locations.
change: Build a 249,878-parameter multi-scale CNN whose late features undergo three-head bottleneck self-attention with learned 2D positions, while retaining the proven cardinal-translation training and hierarchical validation ensemble.
mechanism: Position-aware bottleneck self-attention refinement
evidence_used: Local spatial refinement improved correctness from 9,202 to 9,209 and evaluation-matched augmentation raised it to 9,262, showing that spatial structure matters; broader convolutional refinements timed out, motivating a cheaper nonlocal mechanism with a global receptive field in one stage.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249878, "training_seconds": 68.60009224992245, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21704905166625976, "validation_score": 9257.41082978481}

RECENT RESULT
hypothesis: Averaging model parameters over the final 10% of cosine-decay updates will exceed 9,262 correct predictions by reducing terminal minibatch noise without changing the proven architecture, augmentation, or inference ensemble.
change: Retain the verified 249,961-parameter design and optimizer schedule, but maintain an online average of the final training iterates and install it after the last optimizer step.
mechanism: Low-learning-rate tail weight averaging
evidence_used: Evaluation-matched cardinal training produced the best result at 9,262 correct, while several more computationally expensive architectural and pooling changes timed out; tail averaging directly stabilizes that successful training trajectory with no extra forward passes or learned parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 65.26041350001469, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.21929060821533203, "validation_score": 9282.410074511057}

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics alongside the final 10% of model parameters will exceed 9,282 correct predictions by eliminating the normalization-state mismatch in the best tail-averaged model.
change: Restore the verified spatial-refinement architecture with cardinal augmentation and tail averaging, then average and install all floating-point model buffers together with the learned parameters.
mechanism: BatchNorm-consistent tail weight averaging
evidence_used: Reference Design 2 achieved the best result at 9,282 correct using final-10% parameter averaging, but retained terminal BatchNorm statistics; synchronizing those statistics directly strengthens the only optimization change that has surpassed 9,262.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Re-estimating BatchNorm statistics cumulatively over the same final 10% window used for parameter averaging will exceed 9,282 correct predictions by better matching normalization state to the averaged model without the overhead that caused explicit buffer averaging to time out.
change: Restore cardinal-translation augmentation and final-10% parameter averaging, while resetting BatchNorm statistics immediately before that window and accumulating fresh tail statistics automatically during normal training forwards.
mechanism: Tail-window BatchNorm recalibration for averaged weights
evidence_used: Cardinal augmentation raised correctness to 9,262 and tail parameter averaging further raised it to 9,282; explicit averaging of BatchNorm buffers targeted the remaining state mismatch but timed out, motivating this cheaper recalibration method.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 75% of the proven tail-averaged parameters with 25% of the terminal parameters will exceed 9,282 correct predictions by retaining variance reduction while reducing mismatch with terminal BatchNorm statistics.
change: Restore the 249,961-parameter spatial-refinement model, cardinal translation augmentation, and final-10% parameter averaging, but install a terminal-anchored blend instead of the pure parameter average.
mechanism: Endpoint-anchored tail weight averaging
evidence_used: Pure tail averaging improved the cardinally trained spatial model from 9,262 to 9,282 correct, while both attempts to align BatchNorm state with averaged weights timed out; endpoint anchoring targets the same mismatch without extra forwards or buffer processing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling the native centered view 50% of the time while retaining all four one-pixel cardinal shifts will exceed 9,282 correct predictions by improving clean-image learning during the fixed two-pass exposure without discarding the translation robustness responsible for the 9,262-result gain.
change: Retain the verified architecture, inference ensemble, schedule, and tail averaging, but change training-position probabilities from 20% each to 50% centered and 12.5% for each cardinal shift.
mechanism: Center-biased cardinal translation augmentation
evidence_used: Restricting augmentation to centered/cardinal positions raised correctness from 9,209 to 9,262, and tail averaging then raised it to 9,282; because validation images retain their native alignment and exposure is limited, biasing that proven augmentation set toward the centered view is the most direct low-cost refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging only the final 5% of iterates will exceed 9,282 correct predictions by retaining the proven variance reduction of tail averaging while reducing mismatch with terminal BatchNorm statistics and halving averaging overhead.
change: Restore the verified 249,961-parameter spatial-refinement model with cardinal augmentation, then average and install parameters over the final 5% rather than 10% of optimizer steps.
mechanism: Short-window tail weight averaging
evidence_used: Final-10% parameter averaging improved the cardinally trained spatial-refinement model from 9,262 to 9,282 correct, the strongest verified result; normalization-alignment and endpoint-blending variants timed out, motivating a cheaper, terminal-proximal averaging window.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.886122666997835, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.2185217487335205, "validation_score": 9275.410333258737}



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
