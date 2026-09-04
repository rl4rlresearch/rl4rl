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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 78.62715308298357, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332611083985, "validation_score": 9254.412448240204}
prior_hypothesis: A 2.52490234375:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227335586547852.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the center-to-shift ratio from 2.5:1 to 2.53125:1 will exceed 9,254 correct predictions while retaining more useful shifted-view information than the unresolved higher-ratio designs.
change: Increase both center-view weights from 5.0 to 5.0625 and renormalize the ten-view ensemble from 26.0 to 26.125.
mechanism: Fine-grained center-weight extrapolation in test-time augmentation
evidence_used: Moving from uniform weighting to 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; all larger extrapolations timed out without contradictory validation evidence, motivating a smaller untested step in the supported direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the center-to-shift view-weight ratio from 2.5:1 to 2.515625:1 will exceed 9,254 correct predictions while preserving nearly all shifted-view ensemble information.
change: Increase both center-view weights from 5.0 to 5.03125 and renormalize the ten-view ensemble from 26.0 to 26.0625.
mechanism: Minimum-step center-weight extrapolation in test-time augmentation
evidence_used: Moving from uniform weighting to 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; every larger extrapolation timed out without contradictory score evidence, motivating the smallest untested step in the supported direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.1373300831765, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227385559082032, "validation_score": 9254.41244806006}

RECENT RESULT
hypothesis: A 2.5234375:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227385559082032.
change: Increase both center-view weights from 5.03125 to 5.046875 and renormalize the ten-view ensemble from 26.0625 to 26.09375.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Moving from 2.5:1 to 2.515625:1 preserved 9,254 correct predictions while lowering cross-entropy from 0.21227477645874024 to 0.21227385559082032; this midpoint toward the unresolved 2.53125:1 design tests whether that improvement continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 80.00604279199615, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227340812683104, "validation_score": 9254.412448212299}

RECENT RESULT
hypothesis: A 2.52734375:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227340812683104.
change: Increase both center-view weights from 5.046875 to 5.0546875 and renormalize the ten-view ensemble from 26.09375 to 26.109375.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Increasing the ratio from 2.515625:1 to 2.5234375:1 preserved 9,254 correct predictions and lowered cross-entropy from 0.21227385559082032 to 0.21227340812683104; this midpoint toward the unresolved 2.53125:1 design tests whether the improvement continues.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.525390625:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227340812683104.
change: Increase both center-view weights from 5.046875 to 5.05078125 and renormalize the ten-view ensemble from 26.09375 to 26.1015625.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Increasing the ratio from 2.515625:1 to 2.5234375:1 preserved 9,254 correct predictions and lowered cross-entropy from 0.21227385559082032 to 0.21227340812683104; this smaller midpoint toward the unresolved 2.52734375:1 design tests whether the improvement continues.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.5244140625:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227340812683104.
change: Increase both center-view weights from 5.046875 to 5.048828125 and renormalize the ten-view ensemble from 26.09375 to 26.09765625.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Increasing the ratio from 2.515625:1 to 2.5234375:1 preserved 9,254 correct predictions and lowered cross-entropy from 0.21227385559082032 to 0.21227340812683104; this midpoint toward the unresolved 2.525390625:1 design tests whether the improvement continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 74.85179783310741, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227335586547852, "validation_score": 9254.41244823008}

RECENT RESULT
hypothesis: A 2.52490234375:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227335586547852.
change: Increase both center-view weights from 5.048828125 to 5.0498046875 and renormalize the ten-view ensemble from 26.09765625 to 26.099609375.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Increasing the ratio from 2.5234375:1 to 2.5244140625:1 preserved 9,254 correct predictions and lowered cross-entropy from 0.21227340812683104 to 0.21227335586547852; this midpoint toward the unresolved 2.525390625:1 design tests whether the improvement continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 78.62715308298357, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332611083985, "validation_score": 9254.412448240204}

RECENT RESULT
hypothesis: A 2.525146484375:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332611083985.
change: Increase both center-view weights from 5.0498046875 to 5.05029296875 and renormalize the ten-view ensemble from 26.099609375 to 26.1005859375.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Increasing the ratio from 2.5244140625:1 to 2.52490234375:1 preserved 9,254 correct predictions and lowered cross-entropy from 0.21227335586547852 to 0.21227332611083985; this midpoint toward the unresolved 2.525390625:1 design tests whether the improvement continues.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.5250244140625:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332611083985.
change: Increase both center-view weights from 5.0498046875 to 5.050048828125 and renormalize the ten-view ensemble from 26.099609375 to 26.10009765625.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Increasing the ratio from 2.5244140625:1 to 2.52490234375:1 preserved 9,254 correct predictions and lowered cross-entropy from 0.21227335586547852 to 0.21227332611083985; this midpoint toward the timed-out 2.525146484375:1 trial tests whether the improvement continues.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.52496337890625:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332611083985.
change: Increase both center-view weights from 5.0498046875 to 5.0499267578125 and renormalize the ten-view ensemble from 26.099609375 to 26.099853515625.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Cross-entropy improved monotonically through ratios 2.5234375:1, 2.5244140625:1, and 2.52490234375:1 while retaining 9,254 correct; this midpoint toward the timed-out 2.5250244140625:1 trial is the smallest informative continuation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.524932861328125:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332611083985.
change: Increase both center-view weights from 5.0498046875 to 5.04986572265625 and renormalize the ten-view ensemble from 26.099609375 to 26.0997314453125.
mechanism: Binary refinement of center-weighted test-time ensembling
evidence_used: Cross-entropy improved monotonically through ratios 2.5234375:1, 2.5244140625:1, and 2.52490234375:1 while retaining 9,254 correct; this midpoint toward the timed-out 2.52496337890625:1 trial is the smallest informative continuation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.5249176025390625:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332611083985.
change: Increase both center-view weights from 5.0498046875 to 5.049835205078125 and renormalize the ten-view ensemble from 26.099609375 to 26.09967041015625.
mechanism: Minimum-step center-weight refinement
evidence_used: Cross-entropy improved monotonically through ratios 2.5234375:1, 2.5244140625:1, and 2.52490234375:1 while retaining 9,254 correct; this midpoint toward the timed-out 2.524932861328125:1 trial is the smallest untested continuation in the supported direction.
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
