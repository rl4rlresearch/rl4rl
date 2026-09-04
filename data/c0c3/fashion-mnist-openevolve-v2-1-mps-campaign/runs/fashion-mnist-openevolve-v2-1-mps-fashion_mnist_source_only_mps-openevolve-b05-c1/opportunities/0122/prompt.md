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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 66.76356629119255, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1860017868041992, "validation_score": 9360.421584525051}
prior_hypothesis: A center weight of 1.8203125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860023609.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A center weight of 1.875 will retain at least 9,360 correct predictions while reducing validation cross-entropy below 0.186017, strictly improving validation_score.
change: Reduce centered original and flipped logit weights from 2.0 to 1.875 and renormalize the ten-view logit average.
mechanism: Conservative center-prior interpolation
evidence_used: Weight 2.0 achieved 9,360 correct, while weight 1.5 lowered cross-entropy to 0.185987 but lost three correct predictions; 1.875 conservatively moves toward the better-calibrated setting while minimizing classification-boundary changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 63.512122832937166, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600597343444825, "validation_score": 9360.421583036848}

RECENT RESULT
hypothesis: A center weight of 1.8125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860059734.
change: Reduce the centered original and flipped logit weights from 1.875 to 1.8125, preserving all views and normalization.
mechanism: Conservative center-prior refinement
evidence_used: Weight 1.875 retained the 9,360 correct predictions of weight 2.0 while lowering cross-entropy, whereas weight 1.5 lowered cross-entropy further but lost three correct; a half-step toward 1.5 tests additional calibration benefit conservatively.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.8125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860059734.
change: Reduce the centered original and flipped logit weights from 1.875 to 1.8125 while preserving all views and normalization.
mechanism: Conservative center-prior refinement
evidence_used: Weight 1.875 retained the 9,360 correct predictions of weight 2.0 while lowering cross-entropy, whereas weight 1.5 lowered cross-entropy further but lost three predictions; the previous 1.8125 verification timed out and supplied no contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.84375 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860059734.
change: Reduce the centered original and flipped logit weights from 1.875 to 1.84375 while preserving all views and normalization.
mechanism: Fine-grained center-prior interpolation
evidence_used: Reducing center weight from 2.0 to 1.875 retained 9,360 correct and lowered cross-entropy, while 1.5 lowered cross-entropy further but lost three predictions; this smaller step toward 1.5 targets calibration improvement with less boundary risk than the twice-timed-out 1.8125 candidate.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 70.53252716711722, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600352783203125, "validation_score": 9360.421583906174}

RECENT RESULT
hypothesis: A center weight of 1.828125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860035278.
change: Reduce the centered original and flipped logit weights from 1.84375 to 1.828125 while preserving all views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Lowering center weight from 1.875 to 1.84375 retained 9,360 correct and reduced cross-entropy from 0.1860059734 to 0.1860035278; 1.828125 continues that verified direction by only half the previously successful step, while the timed-out 1.8125 trials provide no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 68.28671604208648, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1860023609161377, "validation_score": 9360.421584320973}

RECENT RESULT
hypothesis: A center weight of 1.8203125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860023609.
change: Reduce the centered original and flipped logit weights from 1.828125 to 1.8203125 while preserving every evaluation view and the existing normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Successive reductions from 1.875 to 1.84375 and then 1.828125 retained 9,360 correct while monotonically lowering cross-entropy; this half-step toward the unresolved 1.8125 candidate continues the verified direction with minimal classification-boundary risk.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 66.76356629119255, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1860017868041992, "validation_score": 9360.421584525051}

RECENT RESULT
hypothesis: A learned content-conditioned spatial gate will exceed 9,360 correct predictions by suppressing background activations and emphasizing discriminative regions before both prediction heads.
change: Add a 99-parameter, 7×7 spatial-attention gate derived from channel mean and maximum maps, initialized to preserve the current model exactly.
mechanism: Identity-initialized spatial recalibration
evidence_used: Center-weight refinements plateaued at 9,360 correct with only tiny cross-entropy gains, suggesting inference aggregation is saturated; this challenges the shared assumption that global channel gating is sufficient and every spatial location should enter the learned representation unconditionally.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.81640625 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860017868.
change: Halve the remaining step from the verified 1.8203125 center weight toward the unresolved 1.8125 candidate, preserving all views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Successive reductions from 1.875 through 1.84375, 1.828125, and 1.8203125 retained 9,360 correct while monotonically lowering cross-entropy; the 1.8125 attempts timed out and provide no contrary accuracy evidence.
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
