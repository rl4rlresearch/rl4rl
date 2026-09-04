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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 73.68122837506235, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Scaling ensemble logits by 1.03592 will preserve all 9,243 predictions while reducing cross-entropy below 0.20883301086425782.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 67.20907104178332, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883301086425782, "validation_score": 9243.413622059876}
prior_hypothesis: Scaling inference logits by 1.0360 will preserve exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20883404388427734.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}
prior_hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 81.03165445895866, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883301391601564, "validation_score": 9243.413622058832}
prior_hypothesis: Scaling ensemble logits by 1.03577 will preserve all 9,243 argmax predictions while reducing validation cross-entropy below 0.20883300514221192.

## Recent verification evidence

RECENT RESULT
hypothesis: The immediately lower float32 power-mean order will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Lower only the inference-time power-mean order by one float32 ULP.
mechanism: One-ULP flip-ensemble power-mean refinement
evidence_used: The current order is best verified; a three-ULP increase worsened cross-entropy, while a 26-ULP decrease worsened it only slightly, making the untested immediate lower neighbor the most informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.43927162513137, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2090454807281494, "validation_score": 9243.4135493726}

RECENT RESULT
hypothesis: A learned 24-channel local projection plus full-channel global summary feeding a 141-unit nonlinear head will exceed 9,243 correct predictions within the parameter ceiling.
change: Replace the 4,704-to-38 bottleneck with a shared per-location channel compressor, retain all 7×7 locations, concatenate a 96-channel global summary, and classify through a substantially wider latent representation.
mechanism: Multi-resolution channel-factorized spatial head
evidence_used: The current design spends 178,752 weights collapsing all spatial-channel features into only 38 units, while the 9,166-correct attention result indicates that discarding spatial detail is harmful. This 249,331-parameter alternative preserves every spatial location and avoids the failed direct-logit shortcut by learning nonlinear interactions through a 141-unit head.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249331, "training_seconds": 79.30040733283386, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.21118341751098632, "validation_score": 9235.412819390334}

RECENT RESULT
hypothesis: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.
change: Lower only the inference-time probability power-mean order, leaving architecture and training unchanged.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: This exact order achieved the highest verified score, 9243.41354937286, in Reference Designs 2 and 3; both the immediately lower float32 neighbor and nearby higher orders produced worse cross-entropy with the same 9,243 correct predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions while reducing validation cross-entropy below the current 0.20904550170898437.
change: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: Reference Designs 2 and 3 achieved the highest verified validation score, 9243.41354937286, with this exact order; neighboring tested orders were no better while retaining the same correct count.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling evaluation logits by 0.99 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Apply a small softening factor only to inference-time flip-ensemble logits, leaving training, architecture, and class rankings unchanged.
mechanism: Argmax-preserving ensemble temperature calibration
evidence_used: Power-mean refinements around 0.5831695556640625 repeatedly retained 9,243 correct but no longer improved cross-entropy, motivating an orthogonal calibration adjustment whose positive scale mathematically preserves every non-tied argmax.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 64.54505312512629, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20918512001037598, "validation_score": 9243.413501615034}

RECENT RESULT
hypothesis: Scaling inference logits by 1.01 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Multiply only the inference-time flip-ensemble logits by 1.01, leaving training and class rankings unchanged.
mechanism: Argmax-preserving ensemble temperature hardening
evidence_used: Scaling logits by 0.99 preserved 9,243 correct but worsened cross-entropy to 0.20918512001037598, providing directional evidence that a symmetric hardening probe above 1.0 may improve calibration.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.31657862500288, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20894218406677245, "validation_score": 9243.413584707847}

RECENT RESULT
hypothesis: Using the best verified power-mean order and scaling its logits by 1.0334 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20894218406677245.
change: Set the power-mean order to 0.5831695556640625 and apply a 1.0334 inference-only logit scale.
mechanism: Quadratically estimated argmax-preserving temperature hardening
evidence_used: Scaling from 0.99 to 1.0 to 1.01 progressively reduced cross-entropy from 0.20918512001037598 to 0.20904547996520997 to 0.20894218406677245 without changing the 9,243 correct predictions; quadratic interpolation estimates the calibration optimum near 1.0334.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.0334 inference-only logit scale with the best verified power-mean order will preserve all 9,243 argmax predictions and reduce cross-entropy below 0.20894218406677245.
change: Use power 0.5831695556640625 and multiply the flip-ensemble logits by 1.0334 without changing training.
mechanism: Quadratically calibrated inference-logit hardening
evidence_used: Scales 0.99, 1.0, and 1.01 produced progressively lower cross-entropies while retaining 9,243 correct; quadratic interpolation estimates the optimum near 1.0334, and its prior timeout provides no contradictory performance result.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.80138483387418, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883404388427734, "validation_score": 9243.413621706412}

RECENT RESULT
hypothesis: Scaling inference logits by 1.0360 will preserve exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20883404388427734.
change: Increase only the positive inference-time logit scale from 1.0334 to 1.0360.
mechanism: Cubically refined inference-temperature calibration
evidence_used: Verified scales 0.99, 1.0, 1.01, and 1.0334 retained 9,243 correct while progressively lowering cross-entropy; cubic interpolation of those measurements estimates the local minimum near 1.036.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 67.20907104178332, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883301086425782, "validation_score": 9243.413622059876}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.03592 will preserve all 9,243 predictions while reducing cross-entropy below 0.20883301086425782.
change: Replace the 1.01 inference-only logit scale with the estimated local optimum, leaving training and ensembling unchanged.
mechanism: Convex local refinement of inference temperature
evidence_used: Scales 1.0334 and 1.0360 achieved cross-entropies 0.20883404388427734 and 0.20883301086425782 with identical predictions; local convex interpolation places the minimum near 1.03592.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 73.68122837506235, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.03577 will preserve all 9,243 argmax predictions while reducing validation cross-entropy below 0.20883300514221192.
change: Apply a 1.03577 positive scale to inference-only flip-ensemble logits, leaving training and ensembling unchanged.
mechanism: Locally refined inference-temperature calibration
evidence_used: Scale 1.03592 achieved the best cross-entropy, while 1.0360 was slightly worse and 1.0334 was materially worse; quadratic interpolation of these verified measurements places the local minimum near 1.03577.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 81.03165445895866, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883301391601564, "validation_score": 9243.413622058832}

RECENT RESULT
hypothesis: Preserving all 96 semantic channels while pooling the redundant 7×7 grid to a vertically detailed 4×2 layout and widening the learned head from 38 to 235 units will exceed 9,243 correct predictions within the parameter ceiling.
change: Replace the full-resolution narrow bottleneck with adaptive 4×2 spatial pooling and a 235-unit nonlinear classifier, using 249,329 learned parameters; also apply the best verified argmax-preserving logit scale.
mechanism: Anisotropically pooled full-channel nonlinear head
evidence_used: The current 9,243-correct design spends most parameters mapping 4,704 spatial-channel values into only 38 units. The 9,235-correct alternative instead compressed channels from 96 to 24 while retaining every location; this patch tests the complementary assumption that late semantic channels are more valuable than fine spatial resolution.
result: the implementation could not be verified



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
