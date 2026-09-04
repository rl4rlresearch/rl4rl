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

RECENT RESULT
hypothesis: Increasing each centered-view weight from 2 to 3 will exceed 9,360 correct predictions by reducing the influence of less reliable translated views while preserving flip consensus.
change: Increase the centered original and flipped logits to weight 3 and renormalize the ten-view logit average.
mechanism: Center-prior logit-space TTA
evidence_used: Weighted raw-logit aggregation improved validation correct from 9,358 to 9,360; the resulting best design already privileges centered views, motivating a focused test of a stronger center prior without changing training, parameters, or inference passes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 62.080008334014565, "validation_accuracy": 0.9355, "validation_correct": 9355, "validation_cross_entropy": 0.18615564842224122, "validation_score": 9355.421529839414}

RECENT RESULT
hypothesis: Reducing centered-view weight from 2.0 to 1.5 will exceed 9,360 correct predictions by increasing translated-view influence without discarding the useful center prior.
change: Weight the centered original and flipped logits by 1.5 and renormalize the ten-view logit average.
mechanism: Reduced center-prior logit-space TTA
evidence_used: Increasing centered-view weight from 2 to 3 reduced validation correct from 9,360 to 9,355, indicating that stronger center emphasis is harmful and motivating a measured move in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 70.26704333396628, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.1859872371673584, "validation_score": 9357.42158969703}

RECENT RESULT
hypothesis: Increasing centered-view weight from 2.0 to 2.125 will exceed 9,360 correct predictions by modestly strengthening the useful center prior without the overemphasis observed at weight 3.0.
change: Weight the centered original and flipped logits by 2.125 and renormalize the ten-view logit average.
mechanism: Fine-grained center-prior logit-space TTA
evidence_used: Center weight 2.0 achieved 9,360 correct, while weights 1.5 and 3.0 achieved only 9,357 and 9,355; this brackets the best setting and motivates a conservative local refinement near 2.0.
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
