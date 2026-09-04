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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.47252191696316, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114465370178222, "validation_score": 9210.40612611889}
prior_hypothesis: Raising the centered-view weight to 3.0446624755859375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114466209.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.04296875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114737778.
change: Move the centered validation-view weight halfway from the successful 3.0390625 setting toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Increasing the weight from 3.03125 to 3.0390625 preserved 9,210 correct and lowered cross-entropy from 0.23115117645 to 0.23114737778; 3.125 lowered cross-entropy further but lost one prediction, motivating a smaller upward boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.71539566596039, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114547729492188, "validation_score": 9210.406125847207}

RECENT RESULT
hypothesis: Adding a learned spatial gate while preserving the position-sensitive flattened head will exceed 9,210 correct predictions by emphasizing informative foreground locations before classification.
change: Add a zero-initialized 3×3 spatial gate driven by channel-wise mean and maximum maps; apply its bounded residual scale after the existing channel gate. This adds 19 learned parameters.
mechanism: Content-adaptive spatial recalibration
evidence_used: Cardinal augmentation improved correctness from 9,196 to 9,206 and multi-view pooling reached 9,210, indicating unresolved spatial-alignment sensitivity. Global spatial classification fell to 9,128, so this challenges the channel-only recalibration assumption while retaining the successful spatially explicit classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.044921875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114547729.
change: Move the centered validation-view weight halfway from the successful 3.04296875 setting toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Increasing the weight from 3.0390625 to 3.04296875 preserved 9,210 correct and lowered cross-entropy from 0.23114737778 to 0.23114547729; the lower-cross-entropy result at 3.125 supports continuing upward cautiously before its one-prediction loss.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.0439453125 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114547729.
change: Move the centered validation-view weight halfway from the successful 3.04296875 setting toward the inconclusive 3.044921875 setting, preserving all training and arithmetic probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases from 3.03125 through 3.04296875 preserved 9,210 correct while monotonically lowering cross-entropy; 3.125 lost one prediction, so a minimal upward probe is justified.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 64.64874179195613, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114500770568847, "validation_score": 9210.406126002112}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.04443359375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114500771.
change: Move the centered validation-view weight halfway from the successful 3.0439453125 setting toward the inconclusive 3.044921875 setting, preserving all training and probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.0439453125 preserved 9,210 correct while monotonically lowering cross-entropy; the 3.044921875 verification timed out, making its midpoint the smallest informative upward probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 65.14004704193212, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2311447654724121, "validation_score": 9210.40612608202}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.044677734375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114476547.
change: Move the centered validation-view weight halfway from the successful 3.04443359375 setting toward the inconclusive 3.044921875 setting, preserving all training and arithmetic probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.04443359375 preserved 9,210 correct while monotonically lowering cross-entropy; the 3.044921875 verification timed out, so its midpoint is the smallest informative upward probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.0445556640625 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114476547.
change: Move the centered validation-view weight halfway from the successful 3.04443359375 setting toward the timed-out 3.044677734375 probe, preserving all training and arithmetic probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.04443359375 preserved 9,210 correct while monotonically lowering cross-entropy; the 3.044677734375 run timed out without subject-level evidence against the change, motivating the smallest informative upward probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 69.85358583298512, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114470787048338, "validation_score": 9210.406126101021}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.04461669921875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114470787.
change: Move halfway from the successful 3.0445556640625 weight toward the timed-out 3.044677734375 probe, preserving all training and arithmetic probability pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Successive increases through 3.0445556640625 preserved 9,210 correct while monotonically lowering cross-entropy; the next larger probe timed out without evidence against the change, so its midpoint is the most conservative informative step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 68.60703395796008, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114468078613282, "validation_score": 9210.406126109956}

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
