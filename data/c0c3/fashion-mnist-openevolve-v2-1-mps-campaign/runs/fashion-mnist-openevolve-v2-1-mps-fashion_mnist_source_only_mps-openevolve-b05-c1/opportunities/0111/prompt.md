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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 75.35178270796314, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18601707305908202, "validation_score": 9360.421579091362}
prior_hypothesis: Weighted logit averaging will exceed 9,358 correct predictions by reinforcing class-margin agreement across translated and flipped views without altering training or parameter count.

## Recent verification evidence

RECENT RESULT
hypothesis: Retaining the final live BatchNorm statistics when installing EMA parameters will exceed 9,345 correct predictions by avoiding compounded temporal lag in the activation statistics.
change: Continue averaging learned parameters, but stop averaging and replacing BatchNorm buffers at the end of training.
mechanism: Live BatchNorm calibration with EMA weights
evidence_used: The 9,345-correct baseline averages both parameters and already-smoothed BatchNorm running statistics, while dropout and label-smoothing variants failed to improve it; correcting this orthogonal train-to-evaluation state mismatch adds no parameters or material runtime.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.08622666704468, "validation_accuracy": 0.9351, "validation_correct": 9351, "validation_cross_entropy": 0.18788676643371582, "validation_score": 9351.420915540208}

RECENT RESULT
hypothesis: Replacing exact 7×7 flattening with 1×1, 2×2, and 4×4 regional averages will exceed 9,351 correct predictions by providing translation-tolerant spatial structure and allowing a wider 64-dimensional learned representation within the parameter ceiling.
change: Replace the 30-unit flattened classifier with a 64-unit classifier over multiscale pooled features, while retaining the successful statistics branch, channel gate, EMA handling, and training procedure.
mechanism: Multiscale spatial-pyramid representation
evidence_used: The best design uses translated training and evaluation views but still classifies an exact-position 7×7 flattening through only 30 units; live BatchNorm improved it to 9,351 correct, while more expensive class-part attention timed out. Spatial-pyramid pooling tests coarse spatial invariance without attention or additional convolutional work.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Lowering the EMA decay cap to 0.985 will exceed 9,351 correct predictions by reducing parameter lag while retaining late-training noise suppression.
change: Reduce the parameter EMA decay cap from 0.99 to 0.985; architecture, BatchNorm buffers, optimizer, schedule, and runtime remain unchanged.
mechanism: Short-horizon parameter EMA
evidence_used: Keeping live BatchNorm statistics improved the best result from 9,345 to 9,351 correct, showing that reducing evaluation-state lag is beneficial; shortening the parameter EMA horizon tests the same mechanism directly.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the EMA decay cap to 0.985 will exceed 9,351 correct predictions by reducing evaluation-state lag while retaining late-training noise suppression.
change: Reduce only the parameter EMA decay cap from 0.99 to 0.985.
mechanism: Short-horizon parameter EMA
evidence_used: Keeping live BatchNorm statistics improved validation correct from 9,345 to 9,351, supporting reduced evaluation-state lag; the prior 0.985 EMA attempt timed out despite adding no computation, so its effect remains unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 71.43735983385704, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.18711766891479492, "validation_score": 9357.421188238615}

RECENT RESULT
hypothesis: Lowering the EMA decay cap from 0.985 to 0.98 will exceed 9,357 correct predictions by further reducing evaluation-weight lag while preserving smoothing over roughly 50 late-training updates.
change: Reduce only the parameter EMA decay cap from 0.985 to 0.98.
mechanism: Further-shortened parameter EMA horizon
evidence_used: Removing EMA from BatchNorm buffers improved validation correct from 9,345 to 9,351, and lowering the parameter EMA cap from 0.99 to 0.985 further improved it to 9,357; this consistent direction motivates testing another measured reduction in parameter lag.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 73.11182583402842, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.1867821262359619, "validation_score": 9358.421307322504}

RECENT RESULT
hypothesis: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by further reducing evaluation-weight lag while retaining smoothing over roughly 40 late-training updates.
change: Reduce only the parameter EMA decay cap from 0.98 to 0.975.
mechanism: Incrementally shorter parameter EMA horizon
evidence_used: Keeping BatchNorm buffers live improved validation correct from 9,345 to 9,351, then lowering parameter EMA from 0.99 to 0.985 improved it to 9,357 and lowering it again to 0.98 improved it to 9,358 with better cross-entropy; a small additional reduction directly tests whether that consistent trend continues.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by further reducing evaluation-weight lag while preserving smoothing over roughly 40 late-training updates.
change: Reduce only the parameter EMA decay cap from 0.98 to 0.975.
mechanism: Incrementally shorter parameter EMA horizon
evidence_used: Live BatchNorm buffers improved validation correct from 9,345 to 9,351, then EMA caps of 0.985 and 0.98 improved it to 9,357 and 9,358 respectively; the prior 0.975 attempt timed out without producing contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the EMA decay cap from 0.98 to 0.9775 will exceed 9,358 correct predictions by reducing evaluation-weight lag while retaining more smoothing than the repeatedly unverified 0.975 setting.
change: Reduce only the parameter EMA decay cap from 0.98 to 0.9775.
mechanism: Interpolated short-horizon parameter EMA
evidence_used: Lowering the cap from 0.99 to 0.985 improved validation correct from 9,351 to 9,357, and 0.98 improved it again to 9,358 with lower cross-entropy; 0.9775 conservatively continues that direction while interpolating between the best verified setting and the twice-timed-out 0.975 candidate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by further reducing evaluation-weight lag while retaining smoothing over roughly 40 late-training updates.
change: Reduce only the learned-parameter EMA decay cap from 0.98 to 0.975.
mechanism: Incrementally shorter parameter EMA horizon
evidence_used: Live BatchNorm buffers improved validation correct from 9,345 to 9,351, and reducing the EMA cap from 0.99 to 0.985 and then 0.98 improved it to 9,357 and 9,358. The 0.975 attempts timed out despite having identical computational cost, so this evidence-guided setting remains unresolved.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by continuing the verified benefit of reducing evaluation-weight lag while retaining smoothing over roughly 40 updates.
change: Reduce only the learned-parameter EMA decay cap from 0.98 to 0.975.
mechanism: Incrementally shorter parameter EMA horizon
evidence_used: Keeping BatchNorm buffers live improved validation correct from 9,345 to 9,351, and lowering the parameter EMA cap from 0.99 to 0.985 and then 0.98 improved it to 9,357 and 9,358; prior 0.975 attempts timed out despite identical computational cost and therefore provide no contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighted logit averaging will exceed 9,358 correct predictions by reinforcing class-margin agreement across translated and flipped views without altering training or parameter count.
change: Replace probability-space TTA averaging with weighted raw-logit averaging while preserving the views, center weighting, normalization, and output temperature.
mechanism: Confidence-preserving logit-space TTA consensus
evidence_used: The 9,358-correct design still averages per-view probabilities; the prior weighted-logit attempt timed out without accuracy evidence, so this orthogonal, computation-neutral aggregation change remains unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 75.35178270796314, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18601707305908202, "validation_score": 9360.421579091362}

RECENT RESULT
hypothesis: Replacing hard max-only downsampling with learned per-channel max/average mixtures will exceed 9,360 correct predictions by reducing shift aliasing while retaining salient local responses.
change: Add a lightweight mixed-pooling layer and use it for both spatial reductions, adding only 96 learned parameters.
mechanism: Channel-adaptive mixed pooling
evidence_used: The best design relies heavily on translated training views and multi-shift evaluation, suggesting that hard downsampling remains a load-bearing source of shift sensitivity. This tests invariance inside the learned hierarchy rather than revisiting unverified head aggregation changes.
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
