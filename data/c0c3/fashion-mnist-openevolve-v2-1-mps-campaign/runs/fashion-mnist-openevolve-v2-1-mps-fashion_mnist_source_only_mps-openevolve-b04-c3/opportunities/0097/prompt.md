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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.80138483387418, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883404388427734, "validation_score": 9243.413621706412}
prior_hypothesis: A 1.0334 inference-only logit scale with the best verified power-mean order will preserve all 9,243 argmax predictions and reduce cross-entropy below 0.20894218406677245.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}
prior_hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.13068708381616, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904547996520997, "validation_score": 9243.41354937286}
prior_hypothesis: A 0.5831695556640625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.31657862500288, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20894218406677245, "validation_score": 9243.413584707847}
prior_hypothesis: Scaling inference logits by 1.01 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.5831697082519531-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Increase only the inference-time probability power-mean order to the interpolated local optimum.
mechanism: Quadratic local calibration of flip-ensemble power mean
evidence_used: The verified order 0.5831695556640625 outperformed the bracketing probes 0.58316802978515625 and 0.583172607421875 while all retained 9,243 correct; quadratic interpolation estimates the minimum near 0.5831697082519531, whose prior timeout supplied no contradictory performance evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.94443312496878, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548263549805, "validation_score": 9243.413549371948}

RECENT RESULT
hypothesis: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions and reduce validation cross-entropy from 0.20904548225402833 to 0.20904547996520997.
change: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: Reference Design 2 used this exact order with the otherwise identical implementation and achieved the highest available validation score, 9243.41354937286; the failed repeat verifications were timeouts or unverifiable rather than contradictory performance results.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions while reducing validation cross-entropy below the current 0.20904550170898437.
change: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: Reference Design 2 used this exact order with the otherwise identical implementation and achieved the highest verified validation score, 9243.41354937286, with 9,243 correct and cross-entropy 0.20904547996520997.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions and reduce validation cross-entropy from 0.209074520111084 toward the best verified 0.20904547996520997.
change: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: Reference Design 3 achieved the highest verified validation score, 9243.41354937286, with this exact order; nearby verified orders were no better, while repeat failures were timeouts or unverifiable rather than contradictory results.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 78.4054907921236, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904547996520997, "validation_score": 9243.41354937286}

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
