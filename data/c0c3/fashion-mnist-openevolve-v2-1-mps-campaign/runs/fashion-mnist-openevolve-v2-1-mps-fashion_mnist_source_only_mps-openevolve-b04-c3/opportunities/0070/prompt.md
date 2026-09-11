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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.43309808406048, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904577102661132, "validation_score": 9243.413549273304}
prior_hypothesis: A 0.58359375-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090463.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.68883183412254, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904550170898437, "validation_score": 9243.413549365423}
prior_hypothesis: A 0.583203125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.209045771.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.71922179101966, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904630241394043, "validation_score": 9243.413549091545}
prior_hypothesis: A 0.584375-order power mean with the verified 12.5% learning-rate floor will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090484.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.93156333290972, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.209074520111084, "validation_score": 9243.41353944003}
prior_hypothesis: A 0.625-order power mean will retain 9,242 correct predictions while reducing cross-entropy below the verified 0.75-order ensemble’s 0.209166.

## Recent verification evidence

RECENT RESULT
hypothesis: Lowering the power-mean order from 0.75 to 0.60 will retain at least 9,243 correct predictions while reducing cross-entropy below 0.2090745.
change: Change only the inference-time probability power-mean order to 0.60, preserving the verified training procedure, architecture, schedule, and parameter count.
mechanism: 0.60-order probability power-mean flip ensemble
evidence_used: Order 0.625 achieved the best result with 9,243 correct and 0.2090745 cross-entropy, improving both metrics over order 0.75; order 0.60 is a conservative continuation toward the lower-cross-entropy geometric ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.86788470903412, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20905703506469728, "validation_score": 9243.41354542052}

RECENT RESULT
hypothesis: A 0.5875-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below 0.209057.
change: Replace arithmetic probability averaging with a 0.5875-order generalized mean of original and horizontally flipped class probabilities, leaving training unchanged.
mechanism: Finely tuned sub-arithmetic probability power-mean flip ensemble
evidence_used: Orders 0.625 and 0.60 both achieved 9,243 correct, while lowering the order to 0.60 reduced cross-entropy from 0.2090745 to 0.2090570; 0.5875 conservatively continues that verified trend.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the fixed 38-unit flattened bottleneck with eight input-dependent spatial attention pools and a 228-unit fusion layer will exceed 9,243 correct predictions by retaining multiple discriminative part descriptors while learning where to gather them.
change: Replace the parameter-dominant flattened classifier with learned positional attention pooling, restore the strongest verified 0.60-order flip ensemble, and restore the verified 12.5% cosine learning-rate floor; the resulting model has 249,342 learned parameters.
mechanism: Multi-query positional part-attention pooling
evidence_used: The best design reached 9,243 correct with the 0.60-order ensemble, but all available designs share a 179,256-parameter flatten-to-38 head. The failed global-average shortcut pooled the same shallow features into one first-order descriptor; eight learned, position-aware pools instead preserve complementary regions and fuse them through a substantially wider learned representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249342, "training_seconds": 78.29844929184765, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.2308595317840576, "validation_score": 9166.406220195797}

RECENT RESULT
hypothesis: A 0.5875-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below 0.209057.
change: Lower only the inference-time probability power-mean order from 0.60 to 0.5875.
mechanism: Fine-grained sub-arithmetic probability power-mean flip ensemble
evidence_used: Lowering the order from 0.625 to 0.60 retained 9,243 correct predictions and reduced cross-entropy from 0.2090745 to 0.2090570; the prior 0.5875 verification timed out and therefore provided no contradictory accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 81.74250462488271, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2090484359741211, "validation_score": 9243.413548361772}

RECENT RESULT
hypothesis: A 0.575-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below the verified 0.5875-order result of 0.2090484.
change: Lower only the inference-time probability power-mean order from 0.625 to 0.575.
mechanism: 0.575-order probability power-mean flip ensemble
evidence_used: Orders 0.625, 0.60, and 0.5875 all achieved 9,243 correct while progressively lowering cross-entropy; 0.575 continues this trend by a conservative 0.0125 step.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.86250283382833, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.2090399383544922, "validation_score": 9242.413551268357}

RECENT RESULT
hypothesis: A 0.58125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090484.
change: Replace arithmetic probability averaging with a 0.58125-order generalized mean, leaving training unchanged.
mechanism: Boundary-seeking 0.58125-order probability power-mean flip ensemble
evidence_used: Order 0.5875 achieved 9,243 correct at 0.2090484 cross-entropy, while 0.575 reduced cross-entropy to 0.2090399 but lost one correct prediction; their midpoint probes the accuracy boundary while seeking better tie-break calibration.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.63287791609764, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.2090441707611084, "validation_score": 9242.41354982067}

RECENT RESULT
hypothesis: A 0.584375-order power mean with the verified 12.5% learning-rate floor will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090484.
change: Restore the strongest verified training schedule and replace logit averaging with a 0.584375-order generalized mean of original and horizontally flipped probabilities.
mechanism: Boundary-search probability power-mean flip ensemble
evidence_used: Order 0.5875 achieved 9,243 correct at 0.2090484 cross-entropy, while 0.58125 achieved 9,242 correct at the lower 0.2090442; their midpoint directly probes the accuracy boundary for a better tie-break score.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.71922179101966, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904630241394043, "validation_score": 9243.413549091545}

RECENT RESULT
hypothesis: A 0.5828125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090463.
change: Lower only the inference-time probability power-mean order from 0.584375 to 0.5828125.
mechanism: Binary-search calibration of the flip-ensemble power mean
evidence_used: Order 0.584375 achieved 9,243 correct with 0.2090463 cross-entropy, while 0.58125 achieved 9,242 correct with lower 0.2090442 cross-entropy; their midpoint is the most informative next probe of the accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.26420804182999, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.20904523696899413, "validation_score": 9242.413549455978}

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
