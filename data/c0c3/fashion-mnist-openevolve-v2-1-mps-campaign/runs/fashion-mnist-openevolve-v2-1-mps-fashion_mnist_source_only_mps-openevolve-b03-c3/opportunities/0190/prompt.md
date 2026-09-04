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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.21409408305772, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072585422515869, "validation_score": 9284.414161492756}
prior_hypothesis: An evaluation scale of 1.167286 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258541.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 49.4281909998972, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725957565307618, "validation_score": 9284.414161138237}
prior_hypothesis: An evaluation scale of 1.1645 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207286140.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.361905792029575, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725854110717773, "validation_score": 9284.414161493147}
prior_hypothesis: An evaluation scale of 1.16727 will preserve all 9,284 predictions while reducing cross-entropy below 0.207258652.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.8443427500315, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072586524963379, "validation_score": 9284.414161454933}
prior_hypothesis: An evaluation scale of 1.1682 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207259576.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the evaluation logit scale from 1.249 to 1.248 will preserve exactly 9,284 correct predictions while lowering validation cross-entropy below 0.208056427.
change: Decrease only the positive evaluation logit multiplier by 0.001, leaving training, TTA weights, and class rankings unchanged.
mechanism: Accuracy-preserving continuation of global logit recalibration
evidence_used: Four consecutive scale reductions through 1.249 preserved all 9,284 predictions and monotonically lowered cross-entropy; another positive global rescaling cannot alter argmax predictions and directly extends that validated trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 46.70599520811811, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2080379653930664, "validation_score": 9284.413894276773}

RECENT RESULT
hypothesis: An evaluation scale of 1.153 will preserve all 9,284 argmax predictions while lowering cross-entropy below 0.208037965.
change: Leave training and TTA unchanged; replace the positive evaluation-logit multiplier with the estimated cross-entropy optimum.
mechanism: Curvature-estimated global logit calibration
evidence_used: Reductions from 1.252 through 1.248 preserved 9,284 correct and consistently improved cross-entropy; the shrinking per-step gains imply curvature whose extrapolated zero-slope point is approximately 1.153.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 39.76445912499912, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072861400604248, "validation_score": 9284.414152025282}

RECENT RESULT
hypothesis: Content-conditioned recalibration of the 80 learned feature channels will exceed 9,284 correct predictions by emphasizing class-relevant shape and texture responses before spatial aggregation, while identity initialization preserves the validated computation at the start of training.
change: Add a 4,659-parameter mean–max channel gate upstream of both aggregation branches, retaining the layout-sensitive classifier and adopting Reference Design 1’s strongest validated TTA weights and calibration.
mechanism: Identity-initialized mean–max channel attention
evidence_used: The additive class-evidence head regressed to 9,268 correct, suggesting that replacing the established spatial decision path is harmful. This alternative preserves that path and instead challenges the shared assumption that every image should use fixed channel importance; its bounded multiplicative gate begins as an exact identity and keeps the model under 250,000 parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249811, "training_seconds": 54.78699766704813, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.20782216186523436, "validation_score": 9274.41396822793}

RECENT RESULT
hypothesis: An evaluation scale of 1.1645 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207286140.
change: Increase only the positive evaluation-logit multiplier from 1.153 to the curvature-refined optimum of approximately 1.1645.
mechanism: Quadratic-refined global logit calibration
evidence_used: Scale reductions near 1.248 showed measurable convex curvature, while the large jump to 1.153 improved cross-entropy to 0.207286140; fitting those observations places the estimated minimum near 1.1645, and positive global scaling preserves class rankings.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 49.4281909998972, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725957565307618, "validation_score": 9284.414161138237}

RECENT RESULT
hypothesis: An evaluation scale of 1.1682 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207259576.
change: Replace the evaluation-logit multiplier with the fitted cross-entropy minimum, leaving training, TTA weights, and class rankings unchanged.
mechanism: Three-point quadratic logit calibration
evidence_used: Scales 1.153, 1.1645, and 1.248 produced cross-entropies 0.207286140, 0.207259576, and 0.208037965 respectively; a quadratic fit places the minimum near 1.1682, and positive global scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.8443427500315, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072586524963379, "validation_score": 9284.414161454933}

RECENT RESULT
hypothesis: An evaluation scale of 1.16727 will preserve all 9,284 predictions while reducing cross-entropy below 0.207258652.
change: Replace the evaluation-logit multiplier with the vertex estimated from the three nearest validated calibration points.
mechanism: Local quadratic logit recalibration
evidence_used: Scales 1.153, 1.1645, and 1.1682 yielded cross-entropies 0.207286140, 0.207259576, and 0.207258652; their local quadratic fit has its minimum near 1.16727, and positive scaling preserves argmax rankings.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.361905792029575, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725854110717773, "validation_score": 9284.414161493147}

RECENT RESULT
hypothesis: An evaluation scale of 1.167286 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258541.
change: Restore the best validated center-compensated TTA weights and replace the evaluation multiplier with the quadratic vertex estimated from the three nearest calibration results.
mechanism: Nearest-point quadratic logit calibration
evidence_used: Scales 1.1645, 1.16727, and 1.1682 yielded cross-entropies 0.207259576, 0.207258541, and 0.207258652; their fitted minimum is approximately 1.167286, while positive global scaling preserves class rankings.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.21409408305772, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072585422515869, "validation_score": 9284.414161492756}

RECENT RESULT
hypothesis: Matching stochastic training translations to the validated anisotropic TTA weights will exceed 9,284 correct predictions by training the model for the same directional robustness emphasized during evaluation.
change: Replace the symmetric training translation probabilities with the current evaluation ensemble’s normalized center, vertical, horizontal, diagonal, and radius-two weights.
mechanism: Train–evaluation translation-distribution alignment
evidence_used: Calibration is saturated at 9,284 correct, while the best evaluation ensemble assigns substantially different weights to vertical and horizontal one-pixel shifts; training still samples those directions equally, leaving a concrete train–evaluation mismatch.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 33.25039462512359, "validation_accuracy": 0.9262, "validation_correct": 9262, "validation_cross_entropy": 0.20897913360595702, "validation_score": 9262.413572067624}

RECENT RESULT
hypothesis: Averaging calibrated class probabilities instead of logits across the existing weighted views will exceed 9,284 correct predictions by limiting the effect of confidently wrong translated views.
change: Preserve training, model parameters, TTA transforms, weights, and final temperature while replacing the geometric logit ensemble with an arithmetic probability mixture.
mechanism: Arithmetic probability TTA ensemble
evidence_used: Logit-scale optimization saturated at 9,284 correct, while aligning training translations to TTA reduced accuracy to 9,262; this motivates a rank-changing evaluation-only ensemble that retains the validated augmentation weights.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 32.542964624939486, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20945048828125, "validation_score": 9280.413410887708}

RECENT RESULT
hypothesis: Averaging the final quarter of the cosine training trajectory will exceed 9,284 correct predictions by reducing final-iterate noise while retaining the validated architecture, augmentation, and logit-space TTA.
change: Restore the best validated 1.16727 calibration and maintain a decay-0.95 EMA of parameters and floating BatchNorm buffers during the final training quarter, installing the averaged state after the last optimizer step.
mechanism: Late-stage parameter-and-BatchNorm exponential moving average
evidence_used: Calibration refinements saturated at 9,284 correct, while channel attention, translation-distribution changes, and probability-space TTA reduced accuracy; this motivates preserving the established computation and stabilizing only its late learned trajectory.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 45.893878041999415, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.2076131259918213, "validation_score": 9275.414039885158}

RECENT RESULT
hypothesis: Setting the evaluation scale to 1.16727 will preserve 9,284 correct predictions while reducing validation cross-entropy below the current 0.207259576.
change: Replace only the evaluation logit multiplier with the best verified scale.
mechanism: Best empirically validated global logit calibration
evidence_used: Reference Design 1 achieved the highest available validation_score, 9284.414161493147, at scale 1.16727; the nearby 1.167286 scale produced marginally higher cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the batch size to 128 will exceed 9,284 correct predictions by increasing optimizer updates from 522 to approximately 782 within the fixed 100,000-example budget, while the paired-image loss still gives BatchNorm an effective 256-image batch.
change: Change only the training batch size from 192 to 128, preserving the validated architecture, loss, augmentation, TTA, and calibration.
mechanism: Increased finite-exposure optimization granularity
evidence_used: Calibration is saturated at 9,284 correct, while channel attention, aligned translation sampling, probability-space TTA, and late EMA all reduced accuracy; batch granularity remains an untested way to improve convergence without altering the established classifier computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245152, "training_seconds": 44.35637570801191, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.2046711456298828, "validation_score": 9264.415051030162}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
