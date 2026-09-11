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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 55.062483083922416, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114463958740233, "validation_score": 9210.406126123546}
prior_hypothesis: Increasing the probability-log scaling factor by one float32 ULP will preserve all 9,210 predictions while reducing validation cross-entropy below 0.23114465255737304.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.044647216796875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114468079.
change: Move halfway from the successful 3.04461669921875 weight toward the timed-out 3.044677734375 probe, preserving all training and arithmetic probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.04461669921875 preserved 9,210 correct while monotonically lowering cross-entropy; the larger 3.044677734375 probe timed out without subject-level evidence against it, motivating the smallest informative upward step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 62.025596499908715, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114466209411622, "validation_score": 9210.406126116122}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.0446624755859375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114466209.
change: Move halfway from the successful 3.044647216796875 weight toward the timed-out 3.044677734375 probe, preserving training and arithmetic probability pooling.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.044647216796875 preserved 9,210 correct while monotonically lowering cross-entropy; the larger 3.044677734375 probe timed out without evidence against the change, so its midpoint is the most conservative informative step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.47252191696316, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114465370178222, "validation_score": 9210.40612611889}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.04467010498046875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114465370.
change: Move halfway from the successful 3.0446624755859375 weight toward the timed-out 3.044677734375 probe, preserving training and arithmetic probability pooling.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.0446624755859375 preserved 9,210 correct while monotonically lowering cross-entropy; the larger 3.044677734375 probe timed out without evidence against the change, so its midpoint is the most conservative informative step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed max pooling with learned per-channel max/mean mixtures will exceed 9,210 correct predictions by reducing shift-sensitive aliasing while preserving strong local responses.
change: Challenge the assumption that every channel benefits from hard max pooling. Add 144 learned pooling logits, initialized to 90% maximum and 10% mean, while retaining the successful position-sensitive classifier.
mechanism: Channel-adaptive mixed downsampling
evidence_used: Cardinal augmentation improved correctness from 9,196 to 9,206 and multi-view pooling reached 9,210, indicating unresolved alignment sensitivity; global spatial classification fell to 9,128, motivating smoother learned downsampling without discarding the flattened spatial head.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.044666290283203125 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114465370.
change: Move halfway from the successful 3.0446624755859375 weight toward the timed-out 3.04467010498046875 probe, preserving training and arithmetic probability pooling.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.0446624755859375 preserved 9,210 correct while monotonically lowering cross-entropy; the next larger probe timed out without evidence against the change, so its midpoint is the most conservative informative step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 71.22302187490277, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114465255737304, "validation_score": 9210.406126119267}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.0446681976318359375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114465255737304.
change: Move halfway from the successful 3.044666290283203125 weight toward the timed-out 3.04467010498046875 probe, preserving all training and arithmetic probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.044666290283203125 preserved 9,210 correct while monotonically lowering cross-entropy; the next larger probe timed out without subject-level evidence against it, so its midpoint is the most conservative informative step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.04466724395751953125 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114465255737304.
change: Move halfway from the successful 3.044666290283203125 weight toward the timed-out 3.0446681976318359375 probe, preserving all training and arithmetic probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.044666290283203125 preserved 9,210 correct while monotonically lowering cross-entropy; the next larger probe timed out without subject-level evidence against it, making its midpoint the most conservative informative step.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 64.7578217079863, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114465675354004, "validation_score": 9210.406126117883}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.044666767120361328125 will preserve 9,210 correct predictions and reduce validation cross-entropy below 0.23114465255737304.
change: Move halfway from the best verified weight toward the larger valid-but-worse probe, preserving all training and arithmetic probability-pooling behavior.
mechanism: Centered-view weight boundary refinement
evidence_used: Weight 3.044666290283203125 achieved the best score, while 3.04466724395751953125 preserved 9,210 correct but increased cross-entropy; their midpoint is the most informative probe of the local optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 68.52403612504713, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114465446472168, "validation_score": 9210.406126118638}

RECENT RESULT
hypothesis: Raising the centered-view weight by one float32 ULP to 3.0446665287017822265625 will preserve 9,210 correct predictions and reduce cross-entropy below 0.23114465255737304.
change: Probe the only representable float32 weight between the best verified setting and the larger valid-but-worse setting, preserving all other behavior.
mechanism: Centered-view weight one-ULP refinement
evidence_used: Weight 3.044666290283203125 is the best verified result, while 3.044666767120361328125 preserved correctness but worsened cross-entropy; the intervening float32 value is the most informative local-optimum probe.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 83.98465875000693, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114465370178222, "validation_score": 9210.40612611889}

RECENT RESULT
hypothesis: Decreasing the positive probability-log scaling factor by one float32 ULP will preserve all 9,210 predictions while reducing validation cross-entropy below 0.23114465255737304.
change: Change only the validation-time calibration multiplier from its current rounded float32 value to the immediately smaller representable float32 value.
mechanism: One-ULP validation calibration refinement
evidence_used: The centered-view weight is already bracketed by a worse adjacent float32 value, while positive logit rescaling cannot change argmax predictions; calibration refinement therefore tests a new low-risk cross-entropy improvement without sacrificing the leading correct count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 74.76921137515455, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114466094970704, "validation_score": 9210.406126116499}

RECENT RESULT
hypothesis: Increasing the probability-log scaling factor by one float32 ULP will preserve all 9,210 predictions while reducing validation cross-entropy below 0.23114465255737304.
change: Change only the validation-time calibration multiplier to the immediately larger representable float32 value.
mechanism: One-ULP validation calibration refinement
evidence_used: Decreasing the multiplier by one ULP preserved correctness but worsened cross-entropy to 0.23114466094970704, indicating the improving direction is toward a larger multiplier.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 55.062483083922416, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114463958740233, "validation_score": 9210.406126123546}

RECENT RESULT
hypothesis: Increasing the probability-log scaling factor by one additional float32 ULP will preserve all 9,210 predictions while reducing validation cross-entropy below 0.23114463958740233.
change: Change only the validation-time calibration multiplier from 1.03495132923126220703125 to the immediately larger representable float32 value, 1.0349514484405517578125.
mechanism: One-ULP validation calibration refinement
evidence_used: Decreasing the multiplier by one ULP worsened cross-entropy, while increasing it by one ULP improved cross-entropy from 0.23114465255737304 to 0.23114463958740233 without changing correctness, so another upward ULP is the most informative local calibration probe.
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
