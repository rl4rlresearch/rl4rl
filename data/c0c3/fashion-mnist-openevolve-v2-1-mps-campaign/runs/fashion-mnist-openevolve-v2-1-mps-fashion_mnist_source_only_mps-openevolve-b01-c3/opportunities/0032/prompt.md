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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 78.42631683289073, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21967102966308594, "validation_score": 9281.409946606782}
prior_hypothesis: Uniformly averaging two of every three iterates across the final 15% of training will exceed 9,282 correct predictions by capturing broader low-learning-rate trajectory diversity while performing the same 80 averaging updates as the proven final-10% average.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.95856470800936, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21857940216064453, "validation_score": 9285.41031384505}
prior_hypothesis: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Uniformly averaging the final 15% of iterates will exceed 9,282 correct predictions because the 10% window outperformed the more terminal-proximal 5% window, indicating that broader low-learning-rate temporal coverage improves decision accuracy.
change: Restore evaluation-matched cardinal translation training and add online parameter averaging over the final 15% of optimizer steps while preserving the proven architecture, schedule, and inference ensemble.
mechanism: Extended-window tail weight averaging
evidence_used: Final-10% averaging achieved 9,282 correct versus 9,275 for final-5% averaging on the same 249,961-parameter design; extending the window is the direct next test of the observed accuracy benefit from broader averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Uniformly averaging two of every three iterates across the final 15% of training will exceed 9,282 correct predictions by capturing broader low-learning-rate trajectory diversity while performing the same 80 averaging updates as the proven final-10% average.
change: Restore the verified 249,961-parameter spatial-refinement model, cardinal translation augmentation, and hierarchical ensemble, then average a uniformly subsampled final-15% tail whose final iterate is always included.
mechanism: Subsampled extended-tail weight averaging
evidence_used: Final-10% averaging achieved 9,282 correct versus 9,275 for final-5% averaging, suggesting broader temporal coverage helps; full final-15% averaging timed out, so two-of-three subsampling preserves its span with the proven averaging workload.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 78.42631683289073, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21967102966308594, "validation_score": 9281.409946606782}

RECENT RESULT
hypothesis: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.
change: Keep the verified architecture, augmentation, schedule, and averaging window, but replace uniform tail averaging with a linear recency-weighted mean at identical computational cost.
mechanism: Recency-weighted tail parameter averaging
evidence_used: Final-10% uniform averaging achieved the best result at 9,282 correct, outperforming final-5% averaging at 9,275; meanwhile BatchNorm-alignment and endpoint-blending attempts timed out, motivating a zero-overhead recency bias within the successful window.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.95856470800936, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21857940216064453, "validation_score": 9285.41031384505}

RECENT RESULT
hypothesis: Preserving 2×2 feature layout alongside global mean and peak statistics will exceed 9,285 correct predictions because it exposes class-relevant spatial arrangement that the shared globally invariant readout discards.
change: Replace the global-only classifier input with a 2×2 spatial pyramid plus global summaries, resize its bottleneck to remain under 250,000 parameters, and evaluate it using the proven final-10% recency-weighted averaging procedure.
mechanism: Coarse spatial-pyramid classification head
evidence_used: Local spatial refinement improved correctness from 9,202 to 9,209, and position-matched augmentation reached 9,262, indicating that spatial structure matters; however, all available designs still erase feature layout with mean/max pooling. The 9,285-result establishes recency-weighted averaging as the strongest training baseline for testing this alternative representation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing global mean pooling with per-channel learned 7×7 spatial attention will exceed 9,285 correct predictions by preserving class-relevant feature layout, while linear final-10% averaging retains the strongest verified optimization baseline.
change: Add zero-initialized separable row/column attention for each late feature channel, reduce the classifier width to remain below 250,000 parameters, and restore the verified final-10% linearly recency-weighted parameter average.
mechanism: Channel-specific separable spatial attention pooling
evidence_used: Final-10% linear recency weighting achieved the best result at 9,285 correct; spatial refinement and evaluation-matched translations showed that spatial structure matters, yet the current global mean statistic discards feature location entirely.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249884, "training_seconds": 77.96925141708925, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.22021239547729493, "validation_score": 9257.409764727725}



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
