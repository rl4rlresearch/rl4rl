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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}
prior_hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.68883183412254, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904550170898437, "validation_score": 9243.413549365423}
prior_hypothesis: A 0.583203125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.209045771.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.13068708381616, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904547996520997, "validation_score": 9243.41354937286}
prior_hypothesis: A 0.5831695556640625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.93156333290972, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.209074520111084, "validation_score": 9243.41353944003}
prior_hypothesis: A 0.625-order power mean will retain 9,242 correct predictions while reducing cross-entropy below the verified 0.75-order ensemble’s 0.209166.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.5831787109375-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904550170898437.
change: Lower only the inference-time flip-ensemble power-mean order from 0.58359375 to 0.5831787109375.
mechanism: Conservative power-mean boundary refinement
evidence_used: Order 0.583203125 retained 9,243 correct, while 0.5830078125 lost one prediction; 0.5831787109375 is a conservative refinement toward the lower-cross-entropy boundary, and the intervening timeout supplied no contradictory accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 74.14429220906459, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2090454845428467, "validation_score": 9243.413549371295}

RECENT RESULT
hypothesis: A 0.58316650390625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.
change: Lower only the inference-time probability power-mean order from 0.584375 to 0.58316650390625.
mechanism: Conservative flip-ensemble power-mean boundary refinement
evidence_used: Order 0.5831787109375 retained 9,243 correct with the best verified cross-entropy, while 0.5830078125 lost one prediction; the proposed value cautiously approaches the lower-cross-entropy boundary, and the intervening 0.583154296875 run timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.
change: Lower only the inference-time probability power-mean order from 0.625 to 0.583172607421875.
mechanism: Conservative flip-ensemble power-mean boundary refinement
evidence_used: Order 0.5831787109375 retained 9,243 correct with the best verified cross-entropy, while 0.5830078125 lost one prediction; the proposed value cautiously approaches the lower-cross-entropy boundary, and the nearby 0.58316650390625 attempt timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.
change: Lower only the inference-time probability power-mean order from 0.5831787109375 to 0.583172607421875.
mechanism: Conservative flip-ensemble power-mean boundary refinement
evidence_used: Order 0.5831787109375 achieved the best verified result with 9,243 correct, while 0.5830078125 lost one prediction; the closer 0.583172607421875 probe previously timed out and therefore remains the most informative unresolved refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}

RECENT RESULT
hypothesis: A 0.58316650390625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.
change: Lower only the inference-time probability power-mean order from 0.583203125 to 0.58316650390625.
mechanism: Conservative flip-ensemble power-mean boundary refinement
evidence_used: Order 0.583172607421875 retained 9,243 correct with the best verified cross-entropy, while 0.5830078125 lost one prediction; this nearby lower order previously timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.5831695556640625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904548225402833.
change: Lower only the inference-time probability power-mean order from 0.584375 to 0.5831695556640625.
mechanism: Conservative flip-ensemble power-mean boundary refinement
evidence_used: Order 0.583172607421875 achieved the best verified result with 9,243 correct, while the nearby lower order 0.58316650390625 timed out without contradictory accuracy evidence; their midpoint is the most informative conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.13068708381616, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904547996520997, "validation_score": 9243.41354937286}

RECENT RESULT
hypothesis: Reallocating the shared 38-unit bottleneck into a 28-unit nonlinear branch plus a direct full-resolution class-specific branch will exceed 9,243 correct predictions while remaining within the existing runtime and parameter envelope.
change: Preserve the verified convolutional stem and training procedure, but compute logits by summing nonlinear latent predictions with learned class-specific templates over every channel and 7×7 location; use the best verified flip-ensemble order.
mechanism: Residual class-specific spatial logit shortcut
evidence_used: The current model reaches 9,243 correct while concentrating 179,256 weights in a shared spatial bottleneck, whereas the 9,166-correct attention design indicates that discarding spatial detail is harmful. Unlike the timed-out spatial-refinement head, this reallocation preserves all locations without adding convolutional work and totals 244,920 parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.58316802978515625-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Lower only the inference-time probability power-mean order from 0.5831695556640625 to 0.58316802978515625.
mechanism: Conservative flip-ensemble power-mean refinement
evidence_used: The current order retains 9,243 correct with the best verified cross-entropy, while the nearby lower order 0.58316650390625 timed out without contradictory accuracy evidence; their midpoint is the most conservative informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.92408937495202, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2090454807281494, "validation_score": 9243.4135493726}

RECENT RESULT
hypothesis: Using the verified 0.5831695556640625 order will retain 9,243 correct predictions and reduce validation cross-entropy from 0.20904548225402833 to approximately 0.20904547996520997.
change: Replace only the inference-time probability power-mean order with the best verified value.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: Reference Design 2 used this exact order with the otherwise identical implementation and achieved the highest available validation score, 9243.41354937286.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 0.5831697082519531-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Increase only the inference-time probability power-mean order from 0.583203125 to the interpolated local optimum 0.5831697082519531.
mechanism: Quadratic local calibration of flip-ensemble power mean
evidence_used: Order 0.5831695556640625 outperformed both nearby probes: 0.58316802978515625 had cross-entropy 0.2090454807281494 and 0.583172607421875 had 0.20904548225402833, all retaining 9,243 correct. Quadratic interpolation of these three verified points places the estimated minimum near 0.5831697082519531.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting the inference-time power-mean order to 0.5831695556640625 will retain 9,243 correct predictions and reduce validation cross-entropy below the current 0.2090745201.
change: Replace only the inference-time probability power-mean order, leaving architecture and training unchanged.
mechanism: Verified flip-ensemble power-mean calibration
evidence_used: Reference Design 3 used this exact order with the otherwise identical implementation and achieved the highest verified score, 9243.41354937286, with cross-entropy 0.20904547996520997.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.5831697082519531-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.
change: Increase only the inference-time probability power-mean order to the interpolated local optimum.
mechanism: Quadratic local calibration of flip-ensemble power mean
evidence_used: The verified order 0.5831695556640625 outperformed the bracketing probes 0.58316802978515625 and 0.583172607421875 while all retained 9,243 correct; quadratic interpolation estimates the minimum near 0.5831697082519531, whose prior timeout supplied no contradictory performance evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.94443312496878, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548263549805, "validation_score": 9243.413549371948}



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
