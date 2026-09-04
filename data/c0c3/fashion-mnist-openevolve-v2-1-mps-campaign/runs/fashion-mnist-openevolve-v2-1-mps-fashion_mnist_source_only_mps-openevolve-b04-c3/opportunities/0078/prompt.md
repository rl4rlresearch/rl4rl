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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.68883183412254, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904550170898437, "validation_score": 9243.413549365423}
prior_hypothesis: A 0.583203125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.209045771.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}
prior_hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.71922179101966, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904630241394043, "validation_score": 9243.413549091545}
prior_hypothesis: A 0.584375-order power mean with the verified 12.5% learning-rate floor will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090484.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.93156333290972, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.209074520111084, "validation_score": 9243.41353944003}
prior_hypothesis: A 0.625-order power mean will retain 9,242 correct predictions while reducing cross-entropy below the verified 0.75-order ensemble’s 0.209166.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.58359375-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090463.
change: Lower only the inference-time probability power-mean order from 0.5875 to 0.58359375.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.584375 achieved 9,243 correct at 0.2090463 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.2090452; their midpoint is the most informative probe of the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.43309808406048, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904577102661132, "validation_score": 9243.413549273304}

RECENT RESULT
hypothesis: A 0.583203125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.209045771.
change: Lower only the inference-time probability power-mean order from 0.625 to 0.583203125.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.58359375 achieved 9,243 correct at 0.209045771 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.209045237; their midpoint is the most informative next probe of the accuracy boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.583203125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.209045771.
change: Raise only the inference-time probability power-mean order from 0.58125 to 0.583203125, leaving the verified architecture and training procedure unchanged.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.58359375 achieved 9,243 correct at 0.209045771 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.209045237; their midpoint remains the most informative boundary probe because its prior verification timed out without accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.68883183412254, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904550170898437, "validation_score": 9243.413549365423}

RECENT RESULT
hypothesis: A 0.5830078125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.
change: Lower only the inference-time probability power-mean order from 0.583203125 to 0.5830078125.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.583203125 achieved 9,243 correct at 0.2090455017 cross-entropy, while 0.5828125 achieved 9,242 correct at the lower 0.2090452370; their midpoint is the most informative next probe of the accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 72.22093158308417, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.20904536933898926, "validation_score": 9242.4135494107}

RECENT RESULT
hypothesis: Replacing the 38-unit monolithic flattened bottleneck with local channel projection, residual refinement, and a 110-unit spatial reasoning layer will exceed 9,243 correct predictions within 244,280 parameters.
change: Preserve the complete 7×7 feature grid, project its channels from 96 to 32, refine them locally, and use a nearly three-times-wider global latent representation before classification.
mechanism: Spatially preserved factorized convolutional head
evidence_used: The current 245,040-parameter model reaches 9,243 correct while spending 179,256 weights on a single flatten-to-38 layer. The 9,166-correct attention design showed that aggressively pooling spatial features is harmful; this alternative reallocates those parameters without pooling away any locations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.58310546875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.
change: Lower only the inference-time probability power-mean order from 0.584375 to 0.58310546875, leaving training unchanged.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.583203125 retained 9,243 correct at 0.2090455017 cross-entropy, while 0.5830078125 achieved lower cross-entropy but lost one correct prediction; their midpoint is the most informative boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.58310546875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.
change: Lower only the inference-time probability power-mean order from 0.625 to 0.58310546875, leaving training unchanged.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.583203125 retained 9,243 correct at 0.2090455017 cross-entropy, while 0.5830078125 lowered cross-entropy but lost one prediction; their midpoint is the most informative boundary probe, and its prior timeout supplied no contradictory result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.583154296875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090455017.
change: Lower only the inference-time flip-ensemble power-mean order from 0.583203125 to 0.583154296875.
mechanism: Conservative power-mean boundary refinement
evidence_used: Order 0.583203125 retained 9,243 correct, whereas 0.5830078125 lost one prediction; the proposed conservative quarter-point tests for a calibration gain while staying closer to the successful endpoint, since the intervening 0.58310546875 attempts yielded only timeouts.
result: training did not finish within the verification time limit

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
