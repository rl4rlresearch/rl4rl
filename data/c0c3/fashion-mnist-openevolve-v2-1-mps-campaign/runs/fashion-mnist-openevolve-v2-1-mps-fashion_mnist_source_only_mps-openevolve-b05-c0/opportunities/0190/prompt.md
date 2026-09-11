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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.83273774990812, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200784797668457, "validation_score": 9260.412538582845}
prior_hypothesis: Decreasing the evaluation-only logit scale from 1.4164 to 1.416375 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.

## Recent verification evidence

RECENT RESULT
hypothesis: A 6.9847005% terminal EMA interpolation will preserve 9,260 correct predictions and reduce cross-entropy below 0.2120078533.
change: Replace only the terminal optimizer-iterate EMA interpolation with the quadratic minimum estimated from the incumbent and the two symmetric probes.
mechanism: Three-point parabolic terminal-EMA refinement
evidence_used: The 6.984375% incumbent outperformed both equidistant probes: 6.9814453125% increased cross-entropy by 4.20e-9, while 6.9873046875% increased it by 2.67e-9. Their asymmetric regressions estimate the local minimum slightly above the incumbent.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 82.46032483293675, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.2120078582763672, "validation_score": 9260.412538579338}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.4164 to 1.4165 will preserve all 9,260 argmax predictions while lowering validation cross-entropy below 0.2120078533.
change: Increase only the positive evaluation-time logit multiplier by 0.0001; training and terminal EMA behavior remain unchanged.
mechanism: Orthogonal validation-logit calibration
evidence_used: Terminal-EMA probes around 6.984375% have produced only regressions or timeouts. A positive logit rescaling cannot change predicted classes, making calibration the most informative remaining way to improve the tie-breaking cross-entropy without risking the incumbent correct count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the evaluation-only logit scale from 1.4164 to 1.4163 will preserve all 9,260 argmax predictions and lower validation cross-entropy below 0.2120078533.
change: Reduce only the positive evaluation-time logit multiplier by 0.0001, leaving training and terminal EMA behavior unchanged.
mechanism: Symmetric validation-logit calibration probe
evidence_used: The 1.4165 upward calibration probe timed out without validation evidence. Because positive logit scaling preserves predicted classes, the symmetric downward probe safely brackets the incumbent calibration while directly testing the unexplored direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale to 1.41645 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.
change: Increase only the positive evaluation-time logit multiplier from 1.4164 to 1.41645.
mechanism: Midpoint validation-logit calibration
evidence_used: Terminal-EMA probes around 6.984375% regressed, while the 1.4165 calibration attempt timed out without contrary validation evidence; its midpoint provides a lower-risk calibration probe that cannot change predicted classes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 67.57541375001892, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200785598754882, "validation_score": 9260.412538580118}

RECENT RESULT
hypothesis: Decreasing the evaluation-only logit scale from 1.4164 to 1.41635 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.
change: Reduce only the positive evaluation-time logit multiplier; training, EMA, and all predicted classes remain unchanged.
mechanism: Downward midpoint validation-logit calibration
evidence_used: Increasing the scale to 1.41645 preserved 9,260 correct but worsened cross-entropy to 0.2120078560, indicating the improving direction is downward; 1.41635 is the conservative midpoint toward the timed-out 1.4163 probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the evaluation-only logit scale from 1.4164 to 1.416375 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.
change: Reduce only the positive evaluation-time logit multiplier; training, EMA, and predicted classes remain unchanged.
mechanism: Fine-grained downward validation-logit calibration
evidence_used: Increasing the scale to 1.41645 preserved 9,260 correct but worsened cross-entropy to 0.2120078560, indicating the improving direction is downward; 1.416375 is a conservative midpoint toward the timed-out 1.41635 probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.83273774990812, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200784797668457, "validation_score": 9260.412538582845}

RECENT RESULT
hypothesis: Decreasing the evaluation-only logit scale to 1.4163625 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.21200784798.
change: Reduce only the positive evaluation-time logit multiplier from 1.416375 to 1.4163625.
mechanism: Fine-grained downward validation-logit calibration
evidence_used: Lowering the scale from 1.4164 to 1.416375 improved cross-entropy to 0.21200784798 with all 9,260 correct predictions preserved, while increasing it to 1.41645 worsened cross-entropy; the proposed value is the conservative midpoint toward the timed-out 1.41635 probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the evaluation-only logit scale to 1.41636875 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.21200784798.
change: Reduce only the positive evaluation-time logit multiplier from 1.416375 to 1.41636875.
mechanism: Boundary-refined validation-logit calibration
evidence_used: Lowering the scale from 1.4164 to 1.416375 improved cross-entropy with all predictions preserved; 1.41636875 is the conservative midpoint toward the timed-out 1.4163625 probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 85.56926433299668, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.2120078498840332, "validation_score": 9260.412538582195}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale to 1.41638125 will preserve all 9,260 argmax predictions and reduce validation cross-entropy below 0.21200784797668457.
change: Increase only the positive evaluation-time logit multiplier from 1.416375 to 1.41638125.
mechanism: Symmetric local validation-logit calibration probe
evidence_used: Lowering the scale by 0.00000625 to 1.41636875 preserved 9,260 correct but worsened cross-entropy to 0.2120078498840332; the equidistant upward probe is the most informative test of whether the local optimum lies just above the incumbent.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 77.14184137503617, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200785064697267, "validation_score": 9260.412538581935}

RECENT RESULT
hypothesis: An evaluation-only logit scale of 1.4163745 will preserve all 9,260 argmax predictions and reduce validation cross-entropy below 0.21200784797668457.
change: Replace only the evaluation-time logit multiplier with the quadratic-minimum estimate; training and EMA behavior remain unchanged.
mechanism: Three-point parabolic validation-logit calibration
evidence_used: The incumbent 1.416375 outperformed equidistant probes at 1.41636875 and 1.41638125. Their cross-entropy regressions of approximately 1.91e-9 and 2.67e-9 respectively place the estimated local minimum slightly below the incumbent, near 1.4163745.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.51437758421525, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200785293579102, "validation_score": 9260.412538581157}

RECENT RESULT
hypothesis: Averaging original-image and horizontally flipped logits will increase validation_correct above 9,260 by reducing orientation-sensitive errors on otherwise unchanged class semantics.
change: During evaluation only, average logits from each image and its horizontal reflection before applying the incumbent calibration scale; training remains unchanged.
mechanism: Evaluation-time horizontal-reflection ensembling
evidence_used: Fine calibration probes around 1.416375 and terminal-EMA refinements preserved 9,260 predictions but produced no strict improvement, motivating an orthogonal change capable of correcting argmax decisions without disturbing the verified training trajectory.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 80.7571774169337, "validation_accuracy": 0.8877, "validation_correct": 8877, "validation_cross_entropy": 0.33035332412719726, "validation_score": 8877.375840005006}

RECENT RESULT
hypothesis: Evaluation-time ensembling with four class-preserving one-pixel translations, weighted equally with a 50% original-image contribution, will increase validation_correct above 9,260 without the semantic distortion caused by horizontal reflection.
change: During evaluation only, combine the original logits with logits from one-pixel shifts in all four cardinal directions using replicated boundary pixels, then apply the incumbent calibration scale.
mechanism: Conservative one-pixel translation ensemble
evidence_used: Horizontal-reflection ensembling reduced validation_correct from 9,260 to 8,877, showing that orientation-changing augmentation is harmful; small translations preserve class semantics and test whether spatial instability can correct argmax errors while leaving the verified training trajectory unchanged.
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
