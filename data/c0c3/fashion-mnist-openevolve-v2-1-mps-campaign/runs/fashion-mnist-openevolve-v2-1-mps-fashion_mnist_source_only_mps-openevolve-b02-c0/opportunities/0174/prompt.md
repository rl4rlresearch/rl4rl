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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 67.687772374833, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332077026367, "validation_score": 9254.412448242021}
prior_hypothesis: Transferring one float32 weight step from both orientations of the first vertical crop to the opposing vertical crop will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.

## Recent verification evidence

RECENT RESULT
hypothesis: Favoring flipped shifted views by the same increment that previously favored unflipped shifted views will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.
change: Transfer equal weight from each unflipped shifted view to its flipped counterpart while preserving every shifted pair’s total weight and the ensemble denominator.
mechanism: Reverse orientation-wide shifted-view weighting
evidence_used: The opposite orientation-wide redistribution preserved 9,254 correct predictions but worsened cross-entropy to 0.21227332191467285, providing direct local evidence that moving shifted-view weight in the reverse direction is the most informative next test.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 80.14510612515733, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2122733238220215, "validation_score": 9254.412448240982}

RECENT RESULT
hypothesis: Lowering only the flipped center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.
change: Decrease the flipped center weight from 5.049818992614746 to 5.049818515777588 while leaving the unflipped weight unchanged and renormalizing the ensemble.
mechanism: One-coordinate center-view refinement
evidence_used: The first antisymmetric redistribution toward the unflipped center improved cross-entropy, its second step tied, and symmetric downward movement caused less regression than symmetric upward movement; isolating the downward flipped-weight component tests the most promising unresolved intermediate direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 71.08179850014858, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332229614257, "validation_score": 9254.4124482415}

RECENT RESULT
hypothesis: Increasing only the unflipped center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.
change: Raise the unflipped center weight from 5.0498199462890625 to 5.049820423126221, leave the flipped center weight unchanged, and renormalize the ensemble denominator.
mechanism: One-coordinate unflipped center-view refinement
evidence_used: The combined first-step redistribution toward the unflipped center improved cross-entropy, while subsequently lowering only the flipped weight regressed to 0.21227332229614257; isolating the other component is the most informative unresolved center-weight direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing only the unflipped center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.
change: Raise the unflipped center weight to its next float32 value, leave all other view weights unchanged, and renormalize the ensemble denominator.
mechanism: One-coordinate unflipped center-view refinement
evidence_used: The successful antisymmetric redistribution favored the unflipped center view, while lowering only the flipped weight regressed; this complementary one-coordinate change remains unresolved because its sole verification timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.73037720797583, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332458496093, "validation_score": 9254.412448240722}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Transferring weight from the final vertical-shift pair to the first vertical-shift pair will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.
change: Increase both orientations of the first vertical crop and decrease both orientations of the opposing vertical crop by the same numeric amount, preserving total ensemble weight and the denominator.
mechanism: Antisymmetric vertical-shift view weighting
evidence_used: Center-view and orientation-wide directions have reached local regressions or plateaus; allocation between opposing spatial shifts is an orthogonal untested direction, and paired changes preserve the currently successful orientation balance.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Transferring one effective float32 step of weight from the final vertical-shift pair to the first vertical-shift pair will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.
change: Increase both first vertical-shift weights to 2.000000238418579 and decrease both opposing vertical-shift weights to 1.999999761581421, preserving total ensemble weight and the denominator.
mechanism: Antisymmetric vertical-shift view weighting
evidence_used: The identical vertical-shift direction was the most recent evidence-grounded proposal but timed out, so its effect remains unresolved; center and orientation-weight refinements have already plateaued or regressed.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 72.40071212500334, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332458496093, "validation_score": 9254.412448240722}

RECENT RESULT
hypothesis: Reversing the previously regressive vertical redistribution will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.
change: Transfer one effective float32 weight step from both orientations of the first vertical crop to both orientations of the opposing vertical crop, preserving total ensemble weight and the denominator.
mechanism: Reverse antisymmetric vertical-shift view weighting
evidence_used: The forward vertical redistribution preserved accuracy but worsened cross-entropy to 0.21227332458496093; testing its exact reverse is the most informative untested local spatial direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Transferring one float32 weight step from both orientations of the first vertical crop to the opposing vertical crop will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.
change: Decrease both first vertical-shift weights to 1.999999761581421 and increase both opposing vertical-shift weights to 2.000000238418579, preserving total ensemble weight.
mechanism: Reverse antisymmetric vertical-shift view weighting
evidence_used: The forward vertical redistribution preserved 9,254 correct predictions but worsened cross-entropy to 0.21227332458496093; its exact reverse is the strongest local direction, and the prior reverse test timed out without resolving its effect.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 67.687772374833, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332077026367, "validation_score": 9254.412448242021}

RECENT RESULT
hypothesis: Continuing the successful reverse vertical redistribution by the same weight increment will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Transfer another 2.384185791015625e-7 of weight from both orientations of the first vertical crop to both orientations of the opposing vertical crop, preserving total ensemble weight.
mechanism: Second-step reverse vertical-shift weighting
evidence_used: The first reverse vertical redistribution improved cross-entropy from 0.21227332153320314 to 0.21227332077026367 while preserving 9,254 correct predictions; continuing that direction is the most informative local test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 64.9227865000721, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2122733238220215, "validation_score": 9254.412448240982}

RECENT RESULT
hypothesis: Applying the next vertical redistribution only to the unflipped views will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Decrease the unflipped first-vertical-crop weight to 1.9999995231628418 and increase the unflipped opposing-crop weight to 2.000000476837158, preserving total ensemble weight while leaving their flipped counterparts unchanged.
mechanism: Orientation-selective half-step vertical interpolation
evidence_used: The first full vertical redistribution improved cross-entropy, but the second full step regressed to 0.2122733238220215; changing one orientation implements the informative aggregate midpoint between those two tested settings.
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
