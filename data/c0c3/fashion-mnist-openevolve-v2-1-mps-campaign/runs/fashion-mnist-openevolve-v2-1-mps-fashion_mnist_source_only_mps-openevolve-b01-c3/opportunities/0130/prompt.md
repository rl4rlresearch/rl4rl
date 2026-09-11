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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 51.37766083306633, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670405883789061, "validation_score": 9287.414351800955}
prior_hypothesis: Downscaling logits only for examples with disagreeing TTA views will retain all 9,287 predictions while reducing validation cross-entropy below 0.2067059268951416.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.35771483299322, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670592880249022, "validation_score": 9287.414351158857}
prior_hypothesis: A 1.5578643× center-view weight will retain all 9,287 correct predictions and reduce validation cross-entropy below 0.2067059268951416.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.97645550011657, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059280395508, "validation_score": 9287.41435115912}
prior_hypothesis: A 1.557939852073332× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059269.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1.5578022218542173× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059322.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 3 retained 9,287 correct and achieved the best verified cross-entropy at 1.557131110690534×, while 1.5625× lost one prediction; advancing one eighth of the remaining interval continues the refinement sequence with limited accuracy risk.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 70.55100329197012, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670592765808105, "validation_score": 9287.41435115925}

RECENT RESULT
hypothesis: A 1.5583894441224401× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 1 achieved the best verified cross-entropy at 1.5578022218542173× while retaining 9,287 correct; advancing one eighth of the remaining interval toward the accuracy-losing 1.5625× boundary continues the consistently successful refinement sequence with limited accuracy risk.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.13762945798226, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20670592346191408, "validation_score": 9286.414351160693}

RECENT RESULT
hypothesis: Two learned top/bottom feature summaries plus global peak features will exceed 9,287 correct predictions by preserving class-relevant vertical structure discarded by global mean pooling.
change: Replace global mean pooling with a mirrored, end-to-end learned seven-row pooling kernel; feed both vertical summaries and global maxima to a resized bias-free hidden layer, totaling 249,995 learned parameters.
mechanism: Learned vertical spatial-pyramid pooling
evidence_used: Center-view refinements plateaued at 9,287 correct and the 1.5583894441224401× probe lost a prediction, indicating evaluation aggregation is saturated. The load-bearing untested assumption is that spatially invariant mean/max statistics contain enough shape information; this patch instead learns coarse vertical organization while retaining the proven peak descriptor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249995, "training_seconds": 64.23200345900841, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.20585158843994142, "validation_score": 9252.414644724768}

RECENT RESULT
hypothesis: A 1.5578756246377452× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.
change: Advance the center-view pooling weight one eighth toward the nearest accuracy-losing weight, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view boundary refinement
evidence_used: The current 1.5578022218542173× weight achieved 9,287 correct, while 1.5583894441224401× lost one prediction; a one-eighth advance limits accuracy risk while probing the newly established boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.5578756246377452× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.
change: Advance the center-view pooling weight one eighth toward the nearest accuracy-losing weight while preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view boundary refinement
evidence_used: The 1.5578022218542173× reference retained 9,287 correct with the best verified cross-entropy, while 1.5583894441224401× lost one prediction; the intermediate 1.5578756246377452× attempt only timed out and remains the closest unresolved conservative probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.64375383383594, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059268951416, "validation_score": 9287.414351159512}

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
