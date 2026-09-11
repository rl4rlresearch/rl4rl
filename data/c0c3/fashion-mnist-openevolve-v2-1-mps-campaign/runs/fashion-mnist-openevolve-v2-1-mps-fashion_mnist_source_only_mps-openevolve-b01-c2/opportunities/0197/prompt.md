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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 79.0933020839002, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962220230102539, "validation_score": 9331.417982607227}
prior_hypothesis: Raising the adjustment to 0.029970703125 will preserve 9,331 correct predictions while reducing validation cross-entropy below 0.19622203216552733.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 65.84409662499093, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962220642089844, "validation_score": 9331.417982592831}
prior_hypothesis: Raising the bounded adjustment from 0.02984375 to 0.029921875 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962221206665039.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 67.32374141714536, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19622203216552733, "validation_score": 9331.417982604027}
prior_hypothesis: Raising the bounded adjustment to 0.0299609375 will preserve 9,331 correct predictions while reducing validation cross-entropy below 0.1962220642089844.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.70848262519576, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962224868774414, "validation_score": 9331.417982445142}
prior_hypothesis: Raising the bounded adjustment to 0.029375 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.19622298736572266.

## Recent verification evidence

RECENT RESULT
hypothesis: The reported 0.0275 adjustment will retain 9,331 correct predictions, reduce cross-entropy below 0.1962260711669922, and complete verification after eliminating redundant evaluation-time argmax operations.
change: Increase the adaptive adjustment from 0.025 to 0.0275 and reuse top-k indices to detect view disagreement.
mechanism: Runtime-trimmed stronger disagreement-gated flip fusion
evidence_used: A prior 0.0275 run reported 9,331 correct and 0.1962239959716797 cross-entropy; its failure was attributed to runtime rather than validation quality.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 69.25329458294436, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962239959716797, "validation_score": 9331.417981917837}

RECENT RESULT
hypothesis: A 0.03 higher-margin-view adjustment will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962239959716797, and top-k index reuse will keep verification within the time limit.
change: Increase the bounded adaptive adjustment from 0.025 to 0.03 and eliminate redundant evaluation-time argmax operations.
mechanism: Runtime-trimmed stronger disagreement-gated flip fusion
evidence_used: Increasing the adjustment from 0.025 to 0.0275 preserved 9,331 correct while lowering cross-entropy to 0.1962239959716797; top-k index reuse reduced that design’s training time to 69.25 seconds, addressing the timeout that prevented evaluating 0.03.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The verified 0.0275 adjustment will retain 9,331 correct predictions while reducing validation cross-entropy from 0.19625150680541992 to approximately 0.1962239959716797.
change: Replace fixed evaluation-time flip weighting with the best verified adaptive fusion rule, reusing top-k indices for efficient disagreement detection.
mechanism: Disagreement-gated margin-adaptive flip fusion
evidence_used: Reference Design 1 achieved the best qualified validation_score, preserving 9,331 correct predictions while lowering cross-entropy to 0.1962239959716797 and completing in 69.25 seconds.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The verified 0.0275 adjustment will retain 9,331 correct predictions while reducing validation cross-entropy from 0.1962515151977539 to approximately 0.1962239959716797.
change: Replace fixed evaluation-time flip weighting with the best qualified adaptive fusion rule, shifting weight toward the higher-margin view only during decisive view disagreements.
mechanism: Disagreement-gated margin-adaptive flip fusion
evidence_used: Reference Design 2 completed in 69.25 seconds with 9,331 correct and 0.1962239959716797 cross-entropy, the best qualified validation_score reported.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.61128929094411, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962239959716797, "validation_score": 9331.417981917837}

RECENT RESULT
hypothesis: Raising the bounded adjustment from 0.0275 to 0.02875 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962239959716797.
change: Increase only the higher-margin-view adjustment for decisive original/flip disagreements.
mechanism: Interpolated stronger disagreement-gated flip fusion
evidence_used: Increasing the adjustment from 0.025 to 0.0275 preserved 9,331 correct predictions and reduced cross-entropy from 0.1962260711669922 to 0.1962239959716797; 0.02875 cautiously extends that observed trend toward the unmeasured 0.03 setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 71.38771945796907, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19622298736572266, "validation_score": 9331.417982270263}

RECENT RESULT
hypothesis: Raising the bounded adjustment to 0.029375 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.19622298736572266.
change: Increase only the higher-margin-view adjustment for decisive original/flip disagreements from 0.0275 to 0.029375.
mechanism: Fine-grained stronger disagreement-gated flip fusion
evidence_used: Successive increases from 0.025 to 0.0275 to 0.02875 preserved 9,331 correct while monotonically reducing cross-entropy to 0.1962260711669922, 0.1962239959716797, and 0.19622298736572266; 0.029375 cautiously continues that verified trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.70848262519576, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962224868774414, "validation_score": 9331.417982445142}

RECENT RESULT
hypothesis: Increasing the bounded adjustment from 0.029375 to 0.0296875 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962224868774414.
change: Use efficient top-k disagreement detection and raise the decisive higher-margin-view adjustment to the midpoint between the best verified 0.029375 setting and the unverified 0.03 setting.
mechanism: Fine-grained margin-adaptive flip fusion
evidence_used: Adjustments of 0.025, 0.0275, 0.02875, and 0.029375 successively preserved 9,331 correct while monotonically lowering cross-entropy; 0.0296875 conservatively continues this trend without jumping directly to 0.03.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 75.20226345886476, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962222442626953, "validation_score": 9331.417982529918}

RECENT RESULT
hypothesis: Raising the bounded adjustment from 0.0296875 to 0.02984375 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962222442626953.
change: Replace fixed evaluation-time flip fusion with disagreement-gated fusion using the midpoint between the best verified adjustment and the unverified 0.03 endpoint.
mechanism: Boundary-refined margin-adaptive flip fusion
evidence_used: Adjustments from 0.025 through 0.0296875 monotonically reduced cross-entropy while preserving 9,331 correct predictions; 0.02984375 conservatively tests whether that trend continues toward 0.03.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.28793762507848, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962221206665039, "validation_score": 9331.417982573104}

RECENT RESULT
hypothesis: Raising the bounded adjustment from 0.02984375 to 0.029921875 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962221206665039.
change: Increase only the decisive higher-margin-view adjustment to the midpoint between the best verified setting and the unverified 0.03 endpoint.
mechanism: Endpoint-refined margin-adaptive flip fusion
evidence_used: Every verified increase from 0.025 through 0.02984375 preserved 9,331 correct predictions while monotonically reducing cross-entropy; the 0.03 attempts timed out without contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 65.84409662499093, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962220642089844, "validation_score": 9331.417982592831}

RECENT RESULT
hypothesis: Raising the bounded adjustment to 0.0299609375 will preserve 9,331 correct predictions while reducing validation cross-entropy below 0.1962220642089844.
change: Increase only the decisive higher-margin-view adjustment to the midpoint between the best verified setting and the 0.03 endpoint.
mechanism: Endpoint-bisected margin-adaptive flip fusion
evidence_used: The verified 0.029921875 adjustment preserved 9,331 correct and achieved the best cross-entropy, 0.1962220642089844; every verified increase from 0.025 through 0.029921875 improved cross-entropy monotonically without changing correctness.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 67.32374141714536, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19622203216552733, "validation_score": 9331.417982604027}

RECENT RESULT
hypothesis: Raising the bounded adjustment to 0.02998046875 will preserve 9,331 correct predictions while reducing validation cross-entropy below 0.19622203216552733.
change: Increase only the decisive higher-margin-view adjustment to the midpoint between the best verified 0.0299609375 setting and the 0.03 endpoint.
mechanism: Endpoint-refined margin-adaptive flip fusion
evidence_used: The verified 0.0299609375 adjustment achieved 9,331 correct and the best cross-entropy, 0.19622203216552733; every verified increase from 0.025 through 0.0299609375 monotonically improved cross-entropy without changing correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the adjustment to 0.029970703125 will preserve 9,331 correct predictions while reducing validation cross-entropy below 0.19622203216552733.
change: Increase only the decisive higher-margin-view adjustment to the midpoint between the best verified 0.0299609375 setting and the timed-out 0.02998046875 setting.
mechanism: Endpoint-bisected margin-adaptive flip fusion
evidence_used: The verified 0.0299609375 adjustment achieved the best score with 9,331 correct and 0.19622203216552733 cross-entropy; all verified increases improved cross-entropy without changing correctness, while 0.02998046875 produced no contrary quality evidence because verification timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 79.0933020839002, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962220230102539, "validation_score": 9331.417982607227}



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
