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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 67.73309200000949, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.1876555145263672, "validation_score": 9349.420997497915}
prior_hypothesis: Setting the crop fusion power to 1.3477 will preserve 9,349 correct predictions while lowering validation cross-entropy below 0.1876555618286133.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.12604154110886, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.1876555103302002, "validation_score": 9349.420997499403}
prior_hypothesis: Setting the crop fusion power to 1.3515 will preserve 9,349 correct predictions while reducing validation cross-entropy below 0.1876555145263672.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 53.07685683295131, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765550994873048, "validation_score": 9349.420997499537}
prior_hypothesis: Using the verified 1.3515 crop-fusion power with the current reciprocal-multiplication calibration will preserve 9,349 correct predictions and reduce cross-entropy below 0.1876555103302002.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 54.766303499927744, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765551071166992, "validation_score": 9349.420997499266}
prior_hypothesis: Using fusion power 1.3515 with the current interpolated calibration will preserve 9,349 correct predictions and reduce cross-entropy below 0.18765550994873048.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the verified 1.34-order crop fusion will increase validation_correct from 9,348 to 9,349 and achieve cross-entropy near 0.18765556.
change: Average native/reflected probabilities within each crop, fuse the five weighted crop consensuses with a normalized 1.34-order power mean, and apply the verified temperature calibration.
mechanism: Quadratic-extrapolated crop power-mean fusion
evidence_used: Reference Design 2 achieved the best verified result—9,349 correct and validation_score 9349.420997481147—while powers 1.0, 1.087, and 1.174 all remained at 9,348 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the crop fusion power to 1.3477 will preserve 9,349 correct predictions while lowering validation cross-entropy below 0.1876555618286133.
change: Move the fusion power from 1.34 to the local cross-entropy minimum estimated from the three most relevant measured powers.
mechanism: Quadratic-refined crop power fusion
evidence_used: Cross-entropy decreased from 0.1876840835571289 at 1.087 to 0.18766820755004882 at 1.174 and 0.1876555618286133 at 1.34; a quadratic fit to these nonuniformly spaced observations places the minimum near 1.3477, while the small adjustment should retain the additional correct prediction gained at 1.34.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 67.73309200000949, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.1876555145263672, "validation_score": 9349.420997497915}

RECENT RESULT
hypothesis: Learning a bounded translation of the 7×7 feature grid before local refinement will exceed 9,349 correct predictions by aligning displaced garment structure while preserving the spatial layout whose removal reduced accuracy to 9,085.
change: Replace the fixed-coordinate refinement block with an identity-initialized, content-conditioned feature aligner followed by the established gated refinement, and use the best qualified 1.3477 crop fusion and calibration.
mechanism: Content-conditioned feature canonicalization
evidence_used: Global pooling fell to 9,085, nonlocal attention reached only 9,325, and parallel local refinement reached only 9,321, indicating that spatial layout matters but additional aggregation or filtering is insufficient; meanwhile translation fusion raised the unchanged classifier to 9,349, motivating learned internal alignment rather than continued fusion tuning.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact qualified Reference Design 3 inference path will increase validation_correct from 9,348 to 9,349 and reproduce validation cross-entropy near 0.1876555145263672.
change: Set crop fusion power to 1.3477 and replace interpolated calibration with Reference Design 3’s exact temperature division.
mechanism: Verified 1.3477-order crop power fusion
evidence_used: Reference Design 3 used identical training and 249,881-parameter architecture, achieving the best available result of 9,349 correct; lower fusion powers up to 1.174 remained at 9,348.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the crop fusion power to 1.3515 will preserve 9,349 correct predictions while reducing validation cross-entropy below 0.1876555145263672.
change: Replace arithmetic crop averaging with orientation-first weighted power-mean fusion at the updated local quadratic minimum, followed by the calibration used by the best qualified design.
mechanism: Locally refined crop power-mean fusion
evidence_used: Powers 1.34 and 1.3477 both achieved 9,349 correct, with 1.3477 lowering cross-entropy to 0.1876555145263672; fitting the measured 1.174, 1.34, and 1.3477 results places the local minimum near 1.3515.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.12604154110886, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.1876555103302002, "validation_score": 9349.420997499403}

RECENT RESULT
hypothesis: Setting the crop fusion power to 1.3509 will preserve 9,349 correct predictions while reducing validation cross-entropy below 0.1876555103302002.
change: Move the crop-consensus power from 1.3515 to the quadratic minimum fitted from the three nearest verified powers, preserving training, crop weights, orientation fusion, and calibration.
mechanism: Three-point quadratic crop-power refinement
evidence_used: Powers 1.34, 1.3477, and 1.3515 produced cross-entropies 0.1876555618286133, 0.1876555145263672, and 0.1876555103302002 respectively, with all relevant endpoints retaining 9,349 correct; fitting these measurements places the local minimum near 1.3509.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.8287799579557, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765551223754884, "validation_score": 9349.420997498726}

RECENT RESULT
hypothesis: Setting the crop fusion power to 1.3515 will preserve 9,349 correct predictions and reduce validation cross-entropy from 0.1876555145263672 to approximately 0.1876555103302002.
change: Replace the current 1.3477 crop-consensus power with the highest-scoring verified value, preserving all other training, fusion, and calibration behavior.
mechanism: Verified crop-power restoration
evidence_used: Reference Design 1 used fusion power 1.3515 with otherwise identical code and achieved the best available validation_score, while the nearby 1.3509 result was slightly worse.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Using the verified 1.3515 crop-fusion power with the current reciprocal-multiplication calibration will preserve 9,349 correct predictions and reduce cross-entropy below 0.1876555103302002.
change: Change only the crop-consensus power from 1.174 to 1.3515, retaining the current reciprocal temperature multiplier.
mechanism: Combined optimal crop fusion and reciprocal calibration
evidence_used: Reference Design 1 verified that power 1.3515 yields 9,349 correct and the best available cross-entropy; the current design separately verified that reciprocal multiplication slightly improves calibration at otherwise identical predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 53.07685683295131, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765550994873048, "validation_score": 9349.420997499537}

RECENT RESULT
hypothesis: Using fusion power 1.3515 with the current interpolated calibration will preserve 9,349 correct predictions and reduce cross-entropy below 0.18765550994873048.
change: Change only the crop-consensus fusion power from 1.174 to the best verified value, retaining the current finer temperature interpolation.
mechanism: Combined optimal crop fusion and interpolated temperature calibration
evidence_used: Fusion power 1.3515 achieved 9,349 correct with the best available cross-entropy, while the current interpolation reduced cross-entropy at power 1.174 without changing predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 54.766303499927744, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765551071166992, "validation_score": 9349.420997499266}

RECENT RESULT
hypothesis: Raising the crop fusion power to 1.50 will exceed 9,349 correct predictions by crossing additional beneficial decision boundaries beyond the one observed between powers 1.174 and 1.34.
change: Increase only the crop-consensus power, preserving the verified architecture, training procedure, crop weights, orientation averaging, and calibration.
mechanism: Accuracy-directed stronger crop-consensus fusion
evidence_used: Increasing fusion power from 1.174 to 1.34 raised validation_correct from 9,348 to 9,349, while all tested values through 1.3515 retained that gain; because correctness dominates cross-entropy, the unexplored higher-power direction is more informative than further local calibration refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 56.47474341606721, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766387405395507, "validation_score": 9348.420994534668}

RECENT RESULT
hypothesis: Setting crop fusion power to 1.42575 will exceed 9,349 correct predictions if another beneficial decision boundary occurs before the regression observed at power 1.50.
change: Move crop-consensus power halfway between the best 1.3515 setting and the accuracy-regressing 1.50 endpoint, preserving all other behavior.
mechanism: Upper-interval crop-power boundary probe
evidence_used: Raising power from 1.174 to 1.34 gained one correct prediction, while 1.50 lost it; probing the untested midpoint is the most informative accuracy-directed test within that transition interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 46.45982337510213, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765763931274415, "validation_score": 9349.420996744726}

RECENT RESULT
hypothesis: Increasing the best verified reciprocal logit multiplier by one float32 ULP will preserve 9,349 correct predictions and reduce validation cross-entropy below 0.18765550994873048.
change: Replace interpolated temperature division with direct multiplication by the next representable float32 value above the best verified multiplier.
mechanism: One-ULP temperature calibration probe
evidence_used: Direct multiplication by 1.32772159576416015625 produced the lowest verified cross-entropy, while lower effective logit scales were worse; the adjacent upward value tests whether the calibration optimum lies just beyond it without changing argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.12678216700442, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18765551147460938, "validation_score": 9349.420997498997}

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
