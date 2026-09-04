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

RECENT RESULT
hypothesis: Applying the unresolved second vertical redistribution only to unflipped views will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Decrease the unflipped first vertical-crop weight by one float32 step and increase the unflipped opposing-crop weight by one float32 step, leaving flipped weights and total ensemble weight unchanged.
mechanism: Orientation-selective half-step vertical interpolation
evidence_used: The first full reverse vertical redistribution improved cross-entropy, while the second full step regressed; this orientation-selective midpoint is the most informative untested setting because its previous verification timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 82.21863900008611, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332305908203, "validation_score": 9254.412448241243}

RECENT RESULT
hypothesis: Applying the unresolved second vertical redistribution only to flipped views will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Decrease the flipped first vertical-crop weight by one effective float32 increment and increase the flipped opposing-crop weight equally, leaving unflipped weights and total ensemble weight unchanged.
mechanism: Flipped-view second-step vertical interpolation
evidence_used: The first paired reverse vertical redistribution improved cross-entropy, while both the second paired step and its unflipped-only midpoint regressed; the complementary flipped-only midpoint is the most informative remaining local vertical-weight setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 71.50840520812199, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332229614257, "validation_score": 9254.4124482415}

RECENT RESULT
hypothesis: Transferring one float32 weight step from both orientations of the first horizontal crop to the opposing horizontal crop will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.
change: Decrease both first horizontal-crop weights to 1.999999761581421 and increase both opposing horizontal-crop weights to 2.000000238418579, preserving total ensemble weight and the denominator.
mechanism: Reverse antisymmetric horizontal-shift view weighting
evidence_used: The analogous first reverse vertical redistribution improved cross-entropy while preserving 9,254 correct predictions; additional vertical steps and orientation-selective refinements regressed, motivating the corresponding untested horizontal direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 57.09326762496494, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332229614257, "validation_score": 9254.4124482415}

RECENT RESULT
hypothesis: Reversing the regressive horizontal redistribution will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.
change: Increase both orientations of the first horizontal crop by one float32 step and decrease both orientations of the opposing horizontal crop equally, preserving total ensemble weight and denominator.
mechanism: Opposite antisymmetric horizontal-shift view weighting
evidence_used: Decreasing the first horizontal pair while increasing the opposing pair regressed cross-entropy to 0.21227332229614257; the analogous reversal in the vertical dimension previously improved the objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing both first horizontal-crop weights by one float32 step while decreasing both opposing horizontal-crop weights equally will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.
change: Transfer one effective float32 weight step from both orientations of the opposing horizontal crop to the first horizontal crop, preserving total ensemble weight and denominator.
mechanism: Opposite antisymmetric horizontal-shift view weighting
evidence_used: The reverse horizontal redistribution is the most informative unresolved direction: the opposite change regressed cross-entropy to 0.21227332229614257, while its prior verification timed out without measuring its effect.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 80.75953095802106, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332229614257, "validation_score": 9254.4124482415}

RECENT RESULT
hypothesis: Concentrating the successful vertical redistribution in flipped views will preserve 9,254 correct predictions and lower cross-entropy below 0.21227332077026367.
change: Restore the unflipped vertical-shift weights to 2.0 and move their redistribution to the flipped counterparts, preserving every aggregate spatial weight and the denominator.
mechanism: Orientation-redistributed vertical-shift weighting
evidence_used: Extending the redistribution only in flipped views produced lower cross-entropy (0.21227332229614257) than extending it only in unflipped views (0.21227332305908203), suggesting the flipped orientation better tolerates additional vertical redistribution.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 61.38297408306971, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332153320314, "validation_score": 9254.412448241761}

RECENT RESULT
hypothesis: Moving the successful vertical redistribution from flipped views into unflipped views will preserve 9,254 correct predictions and reduce cross-entropy below 0.21227332077026367.
change: Restore the flipped vertical-shift weights to 2.0 and double the existing redistribution in their unflipped counterparts, preserving aggregate spatial weights and the denominator.
mechanism: Unflipped-view concentrated vertical-shift weighting
evidence_used: Concentrating the redistribution in flipped views regressed cross-entropy to 0.21227332153320314; testing the exact opposite orientation redistribution is the most informative remaining local direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 67.40390537492931, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227332305908203, "validation_score": 9254.412448241243}

RECENT RESULT
hypothesis: Decreasing the positive logit scale by one float32 step will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.21227332077026367.
change: Replace the evaluation logit multiplier with its immediately lower float32 value.
mechanism: One-ULP logit-temperature refinement
evidence_used: The current vertical weighting is locally best; paired, orientation-selective, and horizontal neighboring redistributions all regressed. Logit scaling is an orthogonal tie-break coordinate whose positive adjustment cannot change mathematical argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 73.22649074997753, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2122733238220215, "validation_score": 9254.412448240982}

RECENT RESULT
hypothesis: Increasing the positive evaluation logit scale by one float32 step will preserve all 9,254 argmax predictions and lower validation cross-entropy below 0.21227332077026367.
change: Replace the evaluation logit multiplier with its immediately higher float32 value.
mechanism: One-ULP upward logit-temperature refinement
evidence_used: Decreasing the multiplier by one float32 step preserved accuracy but worsened cross-entropy to 0.2122733238220215, making the exact opposite direction the most informative untested refinement.
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
