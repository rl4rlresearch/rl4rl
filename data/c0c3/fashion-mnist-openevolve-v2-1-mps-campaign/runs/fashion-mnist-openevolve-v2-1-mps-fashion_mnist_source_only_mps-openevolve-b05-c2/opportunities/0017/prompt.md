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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.64090537489392, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2374292724609375, "validation_score": 9192.40406349771}
prior_hypothesis: A diagonal weight of 0.71640625 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23742967 achieved at 0.7171875.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.69897558307275, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23743045463562013, "validation_score": 9192.40406311169}
prior_hypothesis: A diagonal weight of 0.71875 will retain 9,192 correct predictions while lowering cross-entropy below the 0.2374336 achieved at 0.725.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.06414550007321, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23744633865356446, "validation_score": 9192.404057925085}
prior_hypothesis: Weighting diagonal views at 0.75 will retain the extra correct prediction of uniform 3×3 averaging while lowering cross-entropy, producing a validation score above 9191.404013889833.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.15519516705535, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742967071533203, "validation_score": 9192.404063367667}
prior_hypothesis: A diagonal weight of 0.7171875 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23743045 achieved at 0.71875.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging all 25 translations from the same ±2-pixel distribution used during training, together with their flips, will exceed 9,191 correct predictions by reducing residual shift variance.
change: Replace two-view evaluation with uniform logit averaging over the complete 5×5 translation grid and both horizontal orientations.
mechanism: Full augmentation-support translation-and-flip ensemble
evidence_used: Expanding flip evaluation to cardinal one-pixel shifts improved correct predictions from 9,177 to 9,190, and completing the 3×3 grid reached 9,191; the training distribution spans the larger ±2-pixel grid proposed here.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 49.98154616705142, "validation_accuracy": 0.9188, "validation_correct": 9188, "validation_cross_entropy": 0.24227803649902344, "validation_score": 9188.402486388159}

RECENT RESULT
hypothesis: Using the best residual 3×3 TTA design while restricting training translations to the same ±1-pixel support will exceed 9,191 correct predictions by avoiding harmful extreme shifts.
change: Replace the current CNN with the 249,260-parameter residual network and full 3×3 translation/flip ensemble, then narrow random training translations from ±2 to ±1 pixels.
mechanism: Matched one-pixel translation invariance
evidence_used: The residual design with uniform 3×3 TTA achieved 9,191 correct, while expanding evaluation to the full ±2 support fell to 9,188, suggesting two-pixel shifts can dilute useful predictions and motivating matched narrower augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Giving diagonal views half weight will exceed 9,191 correct predictions by retaining their useful corrections while reducing the cross-entropy degradation observed with uniform 3×3 averaging.
change: Keep centered and cardinal translation logits at full weight, but down-weight the four diagonal translations and their flips by 50%.
mechanism: Shrinkage-weighted translation-and-flip ensemble
evidence_used: Adding diagonal views raised correct predictions from 9,190 to 9,191 but worsened cross-entropy from 0.237434 to 0.237581, indicating that diagonal evidence is useful yet noisier than centered and cardinal views.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.60364616592415, "validation_accuracy": 0.919, "validation_correct": 9190, "validation_cross_entropy": 0.23733386154174804, "validation_score": 9190.404094655081}

RECENT RESULT
hypothesis: Weighting diagonal views at 0.75 will retain the extra correct prediction of uniform 3×3 averaging while lowering cross-entropy, producing a validation score above 9191.404013889833.
change: Replace the harmful 5×5 evaluation ensemble with a 3×3 translation-and-flip ensemble that gives centered and cardinal views full weight and diagonal views three-quarter weight.
mechanism: Three-quarter-weight diagonal translation ensemble
evidence_used: Uniform 3×3 averaging achieved 9,191 correct, while half-weight diagonals reduced cross-entropy but lost one correct prediction; an intermediate weight tests whether the accuracy gain can be retained with less noisy diagonal influence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.06414550007321, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23744633865356446, "validation_score": 9192.404057925085}

RECENT RESULT
hypothesis: A diagonal weight of 0.70 will retain the 9,192 correct predictions achieved at 0.75 while lowering cross-entropy by moving toward the better-calibrated but less accurate 0.50 ensemble.
change: Replace the current CNN with the 249,260-parameter residual design and use a full 3×3 translation-and-flip ensemble with diagonal views weighted at 0.70.
mechanism: Conservative diagonal-view shrinkage
evidence_used: Diagonal weights of 0.50, 0.75, and 1.00 produced 9,190, 9,192, and 9,191 correct respectively, while lower weights reduced cross-entropy; a small reduction from 0.75 targets the observed accuracy/calibration boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.042545708827674, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742110137939454, "validation_score": 9191.40406616587}

RECENT RESULT
hypothesis: Increasing the diagonal-view weight from 0.75 to 0.80 will retain the 9,192 correct predictions and may add corrections from diagonal shifts while remaining less noisy than uniform weighting.
change: Use 0.80-weighted diagonal translation views and update the ensemble normalization from 16.0 to 16.4.
mechanism: Local diagonal-view weight refinement
evidence_used: Diagonal weights 0.50, 0.70, 0.75, and 1.00 yielded 9,190, 9,191, 9,192, and 9,191 correct respectively, making a small upward refinement from the best observed weight the most informative next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.9035667499993, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2374723831176758, "validation_score": 9192.404049421079}

RECENT RESULT
hypothesis: A diagonal weight of 0.725 will retain the 9,192 correct predictions achieved at 0.75 while lowering cross-entropy by moving halfway toward the better-calibrated 0.70 ensemble.
change: Increase diagonal translation-view weight from 0.70 to 0.725 and adjust ensemble normalization from 15.6 to 15.8.
mechanism: Accuracy-boundary diagonal-view shrinkage
evidence_used: Weight 0.75 produced 9,192 correct at 0.237446 cross-entropy, whereas 0.70 produced 9,191 correct at a lower 0.237421; bisecting this narrow accuracy boundary is the most informative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.99292870890349, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23743359680175782, "validation_score": 9192.404062085669}

RECENT RESULT
hypothesis: A diagonal weight of 0.7125 will retain the 9,192 correct predictions achieved at 0.725 while lowering cross-entropy toward the better-calibrated 0.70 ensemble.
change: Weight the four diagonal translation views and their flips at 0.7125 while keeping centered and cardinal views at full weight, with normalization adjusted to 15.7.
mechanism: Accuracy-boundary diagonal-view shrinkage
evidence_used: Weight 0.725 achieved 9,192 correct at 0.237434 cross-entropy, while 0.70 achieved 9,191 correct at the lower 0.237421 cross-entropy; bisecting them directly probes the accuracy boundary for a tie-breaking calibration gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 59.04874770808965, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742732315063478, "validation_score": 9191.404064134229}

RECENT RESULT
hypothesis: A diagonal weight of 0.71875 will retain 9,192 correct predictions while lowering cross-entropy below the 0.2374336 achieved at 0.725.
change: Reduce diagonal translation-view weight from 0.725 to 0.71875 and adjust ensemble normalization from 15.8 to 15.75.
mechanism: Accuracy-boundary diagonal-view bisection
evidence_used: Weight 0.725 achieved 9,192 correct, while 0.7125 achieved 9,191 correct with lower cross-entropy; bisecting this interval is the most direct test for the lowest weight that preserves the leading correct count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.69897558307275, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23743045463562013, "validation_score": 9192.40406311169}

RECENT RESULT
hypothesis: A diagonal weight of 0.715625 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23743045 achieved at 0.71875.
change: Reduce diagonal-view weight from 0.75 to 0.715625 and adjust ensemble normalization from 16.0 to 15.725.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.71875 achieved 9,192 correct, while 0.7125 achieved 9,191 correct with lower cross-entropy; their midpoint directly probes the lowest weight preserving the leading correct count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.24790508393198, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742888679504395, "validation_score": 9191.404063623644}

RECENT RESULT
hypothesis: A diagonal weight of 0.7171875 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23743045 achieved at 0.71875.
change: Increase diagonal-view weight from 0.7125 to 0.7171875 and adjust ensemble normalization from 15.7 to 15.7375.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.71875 achieved 9,192 correct, while 0.715625 achieved 9,191 with lower cross-entropy; their midpoint is the most informative probe of the lowest weight preserving the leading correct count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.15519516705535, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742967071533203, "validation_score": 9192.404063367667}

RECENT RESULT
hypothesis: A diagonal weight of 0.71640625 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23742967 achieved at 0.7171875.
change: Replace the baseline with the proven 249,260-parameter residual design and bisect the diagonal-view weight between 0.715625 and 0.7171875, normalizing the ensemble by 15.73125.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7171875 achieved 9,192 correct, while 0.715625 achieved 9,191 with slightly lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.64090537489392, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2374292724609375, "validation_score": 9192.40406349771}



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
