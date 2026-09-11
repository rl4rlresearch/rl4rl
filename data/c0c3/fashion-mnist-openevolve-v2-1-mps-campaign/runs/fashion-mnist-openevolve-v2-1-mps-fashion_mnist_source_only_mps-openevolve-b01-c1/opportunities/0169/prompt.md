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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.779890707926825, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798606872558, "validation_score": 9267.411384494533}
prior_hypothesis: A 9.83251953125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215408002, yielding a strict validation-score improvement.

## Recent verification evidence

RECENT RESULT
hypothesis: A 9.875% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215411, yielding a strict validation-score improvement.
change: Interpolate 90.125% calibrated arithmetic log-probabilities with 9.875% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.75% blend retained 9,267 correct and improved cross-entropy to 0.215411, while 10% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 58.15853725001216, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2154065544128418, "validation_score": 9266.411384979112}

RECENT RESULT
hypothesis: A 9.8125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215411, yielding a strict validation-score improvement.
change: Interpolate 90.1875% calibrated arithmetic log-probabilities with 9.8125% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.75% blend retained 9,267 correct, whereas 9.875% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 42.97136574983597, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2154086585998535, "validation_score": 9267.411384266898}

RECENT RESULT
hypothesis: A 9.84375% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215409, yielding a strict validation-score improvement.
change: Interpolate 90.15625% calibrated arithmetic log-probabilities with 9.84375% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.8125% blend retained 9,267 correct, whereas 9.875% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.6113464999944, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540760498046874, "validation_score": 9266.411384623521}

RECENT RESULT
hypothesis: A direct class-specific route that factorizes channel evidence and learned spatial templates will exceed 9,267 correct predictions by preserving garment layout while avoiding exclusive reliance on the shared 64-dimensional flattened bottleneck.
change: Retain the existing classifier and add four learned part maps and spatial templates per class, contributing additive logits through a bounded learned scale; total parameters become 237,715.
mechanism: Additive rank-four class-specific spatial-part evidence head
evidence_used: Position-free pooling regressed to 9,253/9,249, showing layout is load-bearing, while widening the flattened head regressed to 9,221 and attention timed out. This tests structured, class-specific spatial evidence without discarding the successful dense head or adding expensive input-dependent attention.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237715, "training_seconds": 69.45442333281972, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.215401220703125, "validation_score": 9248.411386784448}

RECENT RESULT
hypothesis: A 9.828125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215409, yielding a strict validation-score improvement.
change: Interpolate 90.171875% calibrated arithmetic log-probabilities with 9.828125% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.8125% blend retained 9,267 correct, whereas 9.84375% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 73.00214608409442, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2154081325531006, "validation_score": 9267.411384444951}

RECENT RESULT
hypothesis: A 9.8359375% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215408133, yielding a strict validation-score improvement.
change: Interpolate 90.1640625% calibrated arithmetic log-probabilities with 9.8359375% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.828125% blend retained 9,267 correct, whereas 9.84375% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.14545087493025, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540786476135254, "validation_score": 9266.411384535591}

RECENT RESULT
hypothesis: A 9.83203125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215408133, yielding a strict validation-score improvement.
change: Interpolate 90.16796875% calibrated arithmetic log-probabilities with 9.83203125% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.828125% blend retained 9,267 correct, whereas 9.8359375% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 42.18529541697353, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2154080017089844, "validation_score": 9267.411384489238}

RECENT RESULT
hypothesis: A 9.833984375% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215408002, yielding a strict validation-score improvement.
change: Interpolate 90.166015625% calibrated arithmetic log-probabilities with 9.833984375% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83203125% blend retained 9,267 correct, whereas 9.8359375% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 48.051219042157754, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2154079345703125, "validation_score": 9266.411384511965}

RECENT RESULT
hypothesis: A 9.8330078125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215408002, yielding a strict validation-score improvement.
change: Interpolate 90.1669921875% calibrated arithmetic log-probabilities with 9.8330078125% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83203125% blend retained 9,267 correct, whereas 9.833984375% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 76.07868620892987, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540796775817872, "validation_score": 9266.41138450073}

RECENT RESULT
hypothesis: A 9.83251953125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215408002, yielding a strict validation-score improvement.
change: Interpolate 90.16748046875% calibrated arithmetic log-probabilities with 9.83251953125% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83203125% blend retained 9,267 correct, whereas 9.8330078125% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.779890707926825, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798606872558, "validation_score": 9267.411384494533}

RECENT RESULT
hypothesis: A 9.832763671875% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215407986, yielding a strict validation-score improvement.
change: Interpolate 90.167236328125% calibrated arithmetic log-probabilities with 9.832763671875% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83251953125% blend retained 9,267 correct, whereas 9.8330078125% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 41.06665354198776, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2154079734802246, "validation_score": 9266.411384498793}

RECENT RESULT
hypothesis: A 9.8326416015625% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215407986, yielding a strict validation-score improvement.
change: Interpolate 90.1673583984375% calibrated arithmetic log-probabilities with 9.8326416015625% normalized geometric logits during validation inference.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83251953125% blend retained 9,267 correct, whereas 9.832763671875% lost one correct prediction; their midpoint is the most informative refinement of the observed boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 54.10228850017302, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540797843933104, "validation_score": 9266.411384497114}



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
