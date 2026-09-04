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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.71539566596039, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114547729492188, "validation_score": 9210.406125847207}
prior_hypothesis: Raising the centered-view weight to 3.04296875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114737778.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the probability-pooling power from 1 to 1.25 will exceed 9,210 correct predictions by emphasizing confident agreement while retaining the arithmetic ensemble’s resistance to low-probability vetoes.
change: Replace arithmetic probability averaging with a class-wise power mean across the same ten weighted validation views.
mechanism: Mild power-mean test-time pooling
evidence_used: Arithmetic probability pooling improved correctness from 9,206 to 9,210 over logit/geometric pooling, motivating a modest extrapolation beyond the arithmetic endpoint rather than returning toward the inferior geometric rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting centered validation views at exactly 25% will exceed 9,210 correct predictions by matching the augmentation distribution whose higher and lower centered-crop variants both performed worse.
change: Change center-to-cardinal validation weights from 3:2:2:2:2 to 4:3:3:3:3 while retaining arithmetic probability pooling and all ten views.
mechanism: Training-matched test-time view weighting
evidence_used: The 25%-center training distribution achieved 9,206 correct before probability pooling, while increasing center exposure to 27.3% produced 9,198 and decreasing it to 22.7% produced 9,196; the current validation weights assign 27.3% to centered views.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed winner-take-all pooling with learned per-channel max/mean mixtures will exceed 9,210 correct predictions by reducing one-pixel aliasing while preserving the successful position-sensitive classifier.
change: Replace all three max-pooling stages with channel-specific learned mixed pooling initialized at 90% max and 10% average; this adds 144 parameters for a total of 249,933.
mechanism: Channel-adaptive mixed downsampling
evidence_used: Cardinal-only augmentation improved correctness from 9,196 to 9,206, and arithmetic multi-view pooling further improved it to 9,210, indicating persistent alignment sensitivity. Global spatial classification previously regressed to 9,128, so this retains the flattened spatial head while challenging the load-bearing assumption that every downsampling stage should discard all but its local maximum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A mild convex confidence tilt will exceed 9,210 correct predictions by emphasizing confident agreement across views without restoring the low-probability veto behavior of geometric pooling.
change: Replace arithmetic view-probability averaging with a normalized average of `p + 0.25p²`, preserving all validation views and weights.
mechanism: Quadratically confidence-tilted probability pooling
evidence_used: Arithmetic probability pooling improved correctness from 9,206 to 9,210 over logit/geometric pooling. The fractional power-mean test timed out, so this computationally cheap multiply-add tests a smaller move in the same confidence-emphasizing direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Assigning exactly 25% of ensemble weight to centered views will exceed 9,210 correct predictions by matching the augmentation balance whose higher and lower center rates both performed worse.
change: Change center-to-cardinal validation weights from 3:2:2:2:2 to 4:3:3:3:3 while retaining arithmetic probability pooling.
mechanism: Training-matched test-time view weighting
evidence_used: The 25%-center training distribution reached 9,206 correct, versus 9,198 at 27.3% and 9,196 at 22.7%; the prior verification of this exact low-cost weighting timed out without yielding subject-level evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 79.60080970800482, "validation_accuracy": 0.9205, "validation_correct": 9205, "validation_cross_entropy": 0.23133830718994142, "validation_score": 9205.406062247133}

RECENT RESULT
hypothesis: Increasing centered-view weight from 27.3% to 28.9% will exceed 9,210 correct predictions by continuing the improvement observed when center weight increased from 25% to 27.3%.
change: Change center-to-cardinal validation weights from 3:2:2:2:2 to 3.25:2:2:2:2 while preserving arithmetic probability pooling.
mechanism: Slightly center-biased probability ensemble
evidence_used: Arithmetic probability pooling with 27.3% centered weight achieved 9,210 correct, while reducing centered weight to 25% achieved 9,205; a small upward adjustment directly tests the supported direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing centered-view weight from 27.3% to 28.1% will exceed 9,210 correct predictions by continuing the improvement observed from 25% to 27.3% without making the larger unverified jump to 28.9%.
change: Change the centered validation-view weight from 3.0 to 3.125 while preserving arithmetic probability pooling and all training behavior.
mechanism: Fine-grained centered-view reweighting
evidence_used: Arithmetic probability pooling achieved 9,210 correct at 27.3% centered weight, whereas 25% achieved 9,205; the 28.9% test timed out, motivating a smaller low-cost step in the supported direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 57.851363708032295, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.23110608520507814, "validation_score": 9209.406138842143}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.03125 will preserve 9,210 correct predictions while lowering validation cross-entropy, producing a strict validation-score improvement.
change: Move one quarter of the way from the current 3.0 centered-view weight toward the tested 3.125 weight, retaining arithmetic probability pooling and all training behavior.
mechanism: Conservative centered-view interpolation
evidence_used: Weight 3.125 reduced cross-entropy from 0.23116648 to 0.23110609 but lost one correct prediction; a smaller interpolation tests whether its calibration benefit can be captured before crossing that decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 81.87815125007182, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23115117645263672, "validation_score": 9210.406123967197}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.046875 will preserve 9,210 correct predictions while further lowering validation cross-entropy.
change: Move the centered validation-view weight one small step toward the lower-cross-entropy 3.125 setting, preserving arithmetic probability pooling and all training behavior.
mechanism: Conservative centered-view boundary refinement
evidence_used: Weight 3.03125 retained 9,210 correct and improved cross-entropy over 3.0, while 3.125 lowered cross-entropy further but lost one prediction; a smaller interpolation tests the supported calibration direction with limited boundary risk.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.0390625 will preserve 9,210 correct predictions while lowering validation cross-entropy below 0.23115117645.
change: Move halfway from the successful 3.03125 weight toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.
mechanism: Conservative centered-view boundary refinement
evidence_used: Weight 3.03125 retained 9,210 correct while improving cross-entropy over 3.0; weight 3.125 lowered cross-entropy further but lost one prediction, supporting a small upward calibration step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 72.10050574992783, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114737777709962, "validation_score": 9210.406125220283}

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.046875 will preserve 9,210 correct predictions while lowering validation cross-entropy below 0.23114737778.
change: Increase only the centered validation-view weight, preserving training and arithmetic probability pooling.
mechanism: Conservative centered-view boundary refinement
evidence_used: Increasing the weight from 3.03125 to 3.0390625 preserved 9,210 correct and reduced cross-entropy from 0.23115117645 to 0.23114737778; 3.125 reduced cross-entropy further but lost one prediction, motivating another small step toward that boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the centered-view weight to 3.04296875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114737778.
change: Move the centered validation-view weight halfway from the successful 3.0390625 setting toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.
mechanism: Fine-grained centered-view boundary refinement
evidence_used: Increasing the weight from 3.03125 to 3.0390625 preserved 9,210 correct and lowered cross-entropy from 0.23115117645 to 0.23114737778; 3.125 lowered cross-entropy further but lost one prediction, motivating a smaller upward boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.71539566596039, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23114547729492188, "validation_score": 9210.406125847207}

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
