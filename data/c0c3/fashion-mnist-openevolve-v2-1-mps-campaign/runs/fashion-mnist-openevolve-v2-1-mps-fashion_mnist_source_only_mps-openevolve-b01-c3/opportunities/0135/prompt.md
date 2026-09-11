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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.64375383383594, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059268951416, "validation_score": 9287.414351159512}
prior_hypothesis: A 1.5578756246377452× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.16700104204938, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935928344725, "validation_score": 9287.414356848474}
prior_hypothesis: A 7.727% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 61.30429754196666, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668936004638672, "validation_score": 9287.414356848212}
prior_hypothesis: A 7.708% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.97645550011657, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059280395508, "validation_score": 9287.41435115912}
prior_hypothesis: A 1.557939852073332× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059269.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1.557939852073332× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059269.
change: Advance only the centered-view pooling weight one eighth from the best verified value toward the nearest accuracy-losing value, preserving the 1.22775 logit calibration.
mechanism: Conservative center-view boundary refinement
evidence_used: Reference Design 3 retained 9,287 correct at 1.5578756246377452× with the best verified cross-entropy, while 1.5583894441224401× lost one prediction; this conservative interpolation probes the remaining improvement interval with limited accuracy risk.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.97645550011657, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059280395508, "validation_score": 9287.41435115912}

RECENT RESULT
hypothesis: A 1.5578643× center-view weight will retain all 9,287 correct predictions and reduce validation cross-entropy below 0.2067059268951416.
change: Set the centered-view pooling weight to the estimated cross-entropy minimum while preserving the verified 1.22775 logit calibration.
mechanism: Quadratic interpolation of center-view calibration
evidence_used: The three safe weights 1.5578022218542173, 1.5578756246377452, and 1.557939852073332 produced cross-entropies 0.20670592765808105, 0.2067059268951416, and 0.2067059280395508 respectively; quadratic interpolation places the minimum near 1.5578643, inside this accuracy-preserving interval.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.35771483299322, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670592880249022, "validation_score": 9287.414351158857}

RECENT RESULT
hypothesis: Averaging per-view class probabilities instead of logits will exceed 9,287 correct predictions by reducing the influence of overconfident errors from individual shifted or flipped views.
change: Preserve the verified model, training procedure, center-view weight, and final calibration while replacing geometric logit aggregation with an arithmetic mixture of per-view probabilities.
mechanism: Arithmetic probability TTA pooling
evidence_used: Center-weight refinements plateaued at 9,287 correct and a larger weight lost one prediction, while the spatial-pooling redesign fell to 9,252; this motivates retaining the learned representation and testing the remaining load-bearing assumption—the form of multi-view aggregation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 56.184109332971275, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2067205276489258, "validation_score": 9286.414346146057}

RECENT RESULT
hypothesis: Downscaling logits only for examples with disagreeing TTA views will retain all 9,287 predictions while reducing validation cross-entropy below 0.2067059268951416.
change: Preserve the verified logit pooling and apply a positive per-image temperature based on agreement between all ten view predictions.
mechanism: View-agreement-conditioned logit calibration
evidence_used: Probability pooling lost one correct prediction, while center-weight refinement reached an accuracy boundary; positive scalar calibration uses view disagreement without changing the pooled argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 51.37766083306633, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670405883789061, "validation_score": 9287.414351800955}

RECENT RESULT
hypothesis: Doubling the disagreement adjustment to 0.30 will retain exactly 9,287 correct predictions because it applies only a positive per-image logit scale, while reducing validation cross-entropy below 0.20670405883789061.
change: Restore the best verified center-view weight and strengthen the verified agreement-conditioned calibration from 15% to 30%.
mechanism: Stronger view-agreement-conditioned logit calibration
evidence_used: Reference Design 1 retained 9,287 correct and improved cross-entropy from 0.20670592765808105 to 0.20670405883789061 with a 15% agreement adjustment; a doubled adjustment is an informative accuracy-invariant line-search step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30% disagreement adjustment will preserve exactly 9,287 correct predictions via positive scalar scaling while reducing cross-entropy below 0.20670405883789061.
change: Restore the best verified center-view weight and double the agreement-conditioned calibration strength from 15% to 30%.
mechanism: Stronger view-agreement-conditioned logit calibration
evidence_used: The verified 15% adjustment preserved 9,287 correct and improved cross-entropy; the prior 30% attempt only timed out, leaving this accuracy-invariant line-search step unresolved.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.
change: Double the agreement-conditioned calibration strength from 15% to 30%, leaving training and pooled predictions unchanged.
mechanism: Stronger view-agreement-conditioned logit calibration
evidence_used: The verified 15% adjustment preserved 9,287 correct and improved cross-entropy; both 30% attempts only timed out, so this accuracy-invariant line-search point remains unresolved.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A zero-initialized, translation-compatible saliency pool will exceed 9,287 correct predictions by learning image-dependent regions while retaining fixed mean pooling as its initial state.
change: Replace global mean pooling with an 18-parameter spatial-attention-weighted mean, retain global maxima, and use the best verified agreement-conditioned calibration.
mechanism: Content-conditioned spatial attention pooling
evidence_used: Static vertical pooling fell to 9,252 correct, suggesting absolute-position summaries are harmful; content-conditioned attention instead preserves shift compatibility and can remain uniform when spatial selection is unhelpful.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249979, "training_seconds": 65.37634091707878, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2101951587677002, "validation_score": 9266.413156503211}

RECENT RESULT
hypothesis: A 22.5% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.
change: Restore the best verified center-view weight and interpolate calibration strength halfway between the verified 15% adjustment and the unresolved 30% probe.
mechanism: Midpoint agreement-conditioned logit calibration
evidence_used: The verified 15% adjustment preserved 9,287 correct and improved cross-entropy; three 30% attempts timed out without subject-level evidence, so 22.5% is an informative intermediate line-search point.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 69.78742716717534, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20675360946655275, "validation_score": 9287.414334787216}

RECENT RESULT
hypothesis: A 7.708% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.
change: Restore the center-view weight used by the best reference and apply a weaker, quadratically estimated agreement-conditioned logit scale.
mechanism: Quadratically interpolated agreement-conditioned calibration
evidence_used: The 0%, 15%, and 22.5% adjustments produced cross-entropies of 0.20670592765808105, 0.20670405883789061, and 0.20675360946655275; quadratic interpolation estimates the minimum near 7.708%, while positive scaling preserves the pooled argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 61.30429754196666, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668936004638672, "validation_score": 9287.414356848212}

RECENT RESULT
hypothesis: A 7.727% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.
change: Move the agreement-conditioning strength from 7.708% to the quadratic minimum estimated from the verified 0%, 7.708%, and 15% results.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: Those calibration strengths produced cross-entropies of 0.20670592765808105, 0.20668936004638672, and 0.20670405883789061; local quadratic interpolation places the minimum near 7.727%, and positive scaling leaves every pooled argmax unchanged.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 7.727% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.
change: Replace the current 15% agreement-conditioning strength with the quadratically estimated 7.727% optimum, leaving training and pooled logits unchanged.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: The verified 0%, 7.708%, and 15% adjustments yielded cross-entropies of 0.20670592765808105, 0.20668936004638672, and 0.20670405883789061; their quadratic minimum is near 7.727%, and its only prior verification timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.16700104204938, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935928344725, "validation_score": 9287.414356848474}



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
