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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.361905792029575, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725854110717773, "validation_score": 9284.414161493147}
prior_hypothesis: An evaluation scale of 1.16727 will preserve all 9,284 predictions while reducing cross-entropy below 0.207258652.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 49.4281909998972, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725957565307618, "validation_score": 9284.414161138237}
prior_hypothesis: An evaluation scale of 1.1645 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207286140.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.21409408305772, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072585422515869, "validation_score": 9284.414161492756}
prior_hypothesis: An evaluation scale of 1.167286 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258541.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.8443427500315, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072586524963379, "validation_score": 9284.414161454933}
prior_hypothesis: An evaluation scale of 1.1682 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207259576.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the evaluation logit scale from 1.253 to 1.254 will retain exactly 9,284 correct predictions while reducing validation cross-entropy below 0.208131822.
change: Apply a small positive temperature adjustment to the final evaluation logits without changing training or class rankings.
mechanism: Accuracy-preserving global logit recalibration
evidence_used: TTA-weight optimization has reached a sharp 9,284/9,283 boundary; a positive global scale preserves every non-tied argmax while directly probing the remaining cross-entropy tie-breaker.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 33.35787345794961, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20815124206542968, "validation_score": 9284.413855469904}

RECENT RESULT
hypothesis: Reducing the evaluation logit scale from 1.253 to 1.252 will preserve all 9,284 argmax predictions while lowering cross-entropy below 0.208131822.
change: Decrease only the final evaluation logit multiplier, leaving training and TTA weights unchanged.
mechanism: Accuracy-preserving reverse logit recalibration
evidence_used: Increasing the scale to 1.254 preserved 9,284 correct but worsened cross-entropy to 0.208151242, strongly motivating an equal-sized probe in the opposite direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.91189074981958, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20811298179626464, "validation_score": 9284.413868576477}

RECENT RESULT
hypothesis: Extending Reference Design 1’s successful scale reduction from 1.252 to 1.251 will retain exactly 9,284 correct predictions while lowering validation cross-entropy below 0.208112982.
change: Restore Reference Design 1’s validated TTA weights and decrease only its positive evaluation logit multiplier by another 0.001.
mechanism: Accuracy-preserving continuation of global logit recalibration
evidence_used: Reducing the multiplier from 1.253 to 1.252 preserved all 9,284 predictions and reduced cross-entropy from 0.208132217 to 0.208112982; another small positive-scale reduction cannot change argmax rankings and directly tests whether calibration continues improving.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 37.00071358308196, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20809393920898436, "validation_score": 9284.413875100083}

RECENT RESULT
hypothesis: Extending Reference Design 3’s scale reduction from 1.251 to 1.250 will preserve exactly 9,284 correct predictions while lowering validation cross-entropy below 0.208093939.
change: Restore Reference Design 3’s validated TTA weights and reduce only its evaluation logit multiplier by 0.001.
mechanism: Accuracy-preserving continuation of global logit recalibration
evidence_used: Consecutive reductions from 1.253 to 1.252 and then 1.251 preserved all 9,284 predictions while monotonically lowering cross-entropy from 0.208132217 to 0.208112982 and 0.208093939; positive global scaling leaves argmax rankings unchanged.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 39.32671537506394, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2080750877380371, "validation_score": 9284.41388155842}

RECENT RESULT
hypothesis: Reducing the evaluation logit scale from 1.250 to 1.249 will preserve exactly 9,284 correct predictions while lowering validation cross-entropy below 0.208075088.
change: Decrease only the positive evaluation logit multiplier by 0.001, leaving training, TTA weights, and class rankings unchanged.
mechanism: Accuracy-preserving continuation of global logit recalibration
evidence_used: Three consecutive reductions from 1.253 through 1.250 preserved all 9,284 predictions while monotonically lowering cross-entropy; another positive global rescaling cannot alter non-tied argmax predictions and directly continues the validated calibration trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 43.96069316705689, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20805642738342287, "validation_score": 9284.413887951478}

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
