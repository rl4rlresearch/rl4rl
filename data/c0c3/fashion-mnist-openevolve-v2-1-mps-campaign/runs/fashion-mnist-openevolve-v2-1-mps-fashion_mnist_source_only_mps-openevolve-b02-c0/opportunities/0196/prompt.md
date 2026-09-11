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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 67.687772374833, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332077026367, "validation_score": 9254.412448242021}
prior_hypothesis: Transferring one float32 weight step from both orientations of the first vertical crop to the opposing vertical crop will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the probability-ensemble contribution by one effective float32 increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer 5.960464477539063e-8 of evaluation weight from the geometric ensemble to the probability ensemble while preserving their total weight.
mechanism: One-step probability–geometric ensemble interpolation
evidence_used: Both one-ULP logit-scale directions and the tested neighboring spatial-weight redistributions regressed from 0.21227332077026367, motivating refinement of the remaining orthogonal ensemble-mixture coordinate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the probability-ensemble contribution by one effective float32 increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer 5.960464477539063e-8 of evaluation weight from the geometric ensemble to the probability ensemble while preserving their total weight.
mechanism: One-step probability–geometric ensemble interpolation
evidence_used: The neighboring spatial-weight and logit-scale refinements all regressed; this orthogonal ensemble-mixture adjustment remains unresolved because its prior verification timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 64.66724270791747, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2122733238220215, "validation_score": 9254.412448240982}

RECENT RESULT
hypothesis: Decreasing the probability-ensemble contribution by one effective float32 increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer 5.960464477539063e-8 of evaluation weight from the probability ensemble to the geometric ensemble while preserving their total weight.
mechanism: One-step geometric–probability ensemble interpolation
evidence_used: Increasing the probability contribution by the same increment preserved correctness but worsened cross-entropy to 0.2122733238220215, making the exact opposite direction the most informative neighboring setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 70.09816987486556, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332305908203, "validation_score": 9254.412448241243}

RECENT RESULT
hypothesis: Equalizing the two center-view weights will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer one float32 weight step from the higher-weight unflipped center view to its flipped counterpart, preserving their sum and the ensemble denominator.
mechanism: Center-view flip-orientation weight equalization
evidence_used: Both directions of the tested logit-scale and probability–geometric refinements regressed while preserving correctness; the center-view orientation imbalance is a remaining independent local ensemble coordinate.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 63.65408700006083, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2122733253479004, "validation_score": 9254.412448240462}

RECENT RESULT
hypothesis: Increasing the existing center-view orientation imbalance by one float32 step per orientation will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer one float32 weight step from the flipped center view to the higher-weight unflipped center view while preserving their sum and the ensemble denominator.
mechanism: Center-view flip-orientation weight separation
evidence_used: Equalizing the center-view weights preserved correctness but worsened cross-entropy to 0.2122733253479004, making the exact opposite redistribution the most informative neighboring setting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the existing center-view orientation imbalance by one float32 step per orientation will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer one float32 weight step from the flipped center view to the higher-weight unflipped center view while preserving their combined weight and the ensemble denominator.
mechanism: Center-view flip-orientation weight separation
evidence_used: Equalizing these center-view weights worsened cross-entropy to 0.2122733253479004 without changing correctness; the opposite neighboring redistribution remains unresolved because its verification timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 70.0981508330442, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332077026367, "validation_score": 9254.412448242021}

RECENT RESULT
hypothesis: Extending the paired vertical redistribution by one effective float32 increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer another 2.384185791015625e-7 of weight from both orientations of the first vertical crop to both orientations of its opposing crop, preserving total ensemble weight and denominator.
mechanism: Symmetric vertical-shift weight extrapolation
evidence_used: The current paired vertical redistribution produced the best measured cross-entropy, while concentrating it in only flipped or unflipped views regressed; extending the successful orientation-symmetric direction is the most informative remaining local test.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the successful paired vertical redistribution by 2.384185791015625e-7 per orientation will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.
change: Transfer equal weight from both orientations of the first vertical crop to both orientations of the opposing vertical crop, preserving total ensemble weight.
mechanism: Symmetric vertical-shift weight extrapolation
evidence_used: The current symmetric redistribution achieved the best measured cross-entropy, while concentrating it in either flip orientation regressed; the previous extrapolation could not be verified, so this direction remains unresolved.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the successful paired vertical redistribution by one float32 step per orientation will preserve 9,254 correct predictions and reduce cross-entropy below 0.21227332077026367.
change: Transfer 2.384185791015625e-7 of weight from both orientations of the first vertical crop to both orientations of the opposing vertical crop, preserving total ensemble weight and denominator.
mechanism: Symmetric vertical-shift weight extrapolation
evidence_used: The current symmetric redistribution achieved the best measured cross-entropy, while concentrating it in either flip orientation regressed; its direct extrapolation remains unresolved because the prior implementations could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the successful paired vertical redistribution by 2.384185791015625e-7 per orientation will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.
change: Transfer equal weight from both orientations of the first vertical crop to both orientations of the opposing vertical crop while preserving total ensemble weight and denominator.
mechanism: Symmetric vertical-shift weight extrapolation
evidence_used: The current symmetric vertical weighting produced the best measured cross-entropy, while orientation-specific redistributions regressed; its direct extrapolation remains unresolved because prior attempts could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the only spatial redistribution associated with the best measured cross-entropy by 2.384185791015625e-7 per orientation will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.
change: Transfer equal weight from both orientations of the first vertical crop to both orientations of the opposing vertical crop, preserving total ensemble weight and denominator.
mechanism: Symmetric vertical-shift weight extrapolation
evidence_used: The current symmetric vertical redistribution achieved the best verified cross-entropy; orientation-specific redistributions regressed, while direct extrapolation remains empirically unresolved because prior implementations could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Transferring 2.384185791015625e-7 per orientation from the first vertical crop to its opposing crop will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.
change: Decrease both first-vertical-crop weights and equally increase both opposing-vertical-crop weights, preserving total ensemble weight and denominator.
mechanism: Symmetric vertical-shift weight extrapolation
evidence_used: The current symmetric vertical redistribution produced the best verified cross-entropy, while orientation-specific redistributions regressed; direct extrapolation remains unresolved only because prior implementations could not be verified.
result: the implementation could not be verified



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
