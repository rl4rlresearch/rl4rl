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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 53.91276849992573, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9328.419650108654}
prior_hypothesis: A fusion coefficient of 0.0872955322265625 will preserve 9,328 correct predictions while reducing cross-entropy below the current 0.191468821.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.06298958393745, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687774658203, "validation_score": 9328.419650107042}
prior_hypothesis: A fusion coefficient of 0.0873046875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.12923370790668, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9328.419650108654}
prior_hypothesis: A fusion coefficient of 0.0872937679290771484375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687713623047.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.97299470915459, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687713623047, "validation_score": 9328.419650109192}
prior_hypothesis: A fusion coefficient of 0.08729381561279296875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687744140625.

## Recent verification evidence

RECENT RESULT
hypothesis: A fusion coefficient of 0.0872955322265625 will preserve 9,328 correct predictions while reducing cross-entropy below the current 0.191468821.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket; retain temperature 0.800713 and all training settings.
mechanism: Ultra-fine bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.08729248046875 produced 9,327 correct, while 0.087298583984375 produced 9,328 correct at 0.191468777 cross-entropy; their midpoint is the most informative remaining boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 53.91276849992573, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9328.419650108654}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08729400634765625 will preserve 9,328 correct predictions while reducing validation cross-entropy below 0.1914687728881836.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Ultra-fine bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.08729248046875 produced 9,327 correct, while 0.0872955322265625 produced 9,328 correct; their midpoint is the most informative next boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.54648870788515, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687744140625, "validation_score": 9328.419650108117}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087293243408203125 will preserve 9,328 correct predictions while reducing validation cross-entropy below 0.1914687728881836.
change: Use the midpoint of the tightest verified 9,327/9,328 fusion-coefficient bracket and retain the best verified temperature of 0.800713.
mechanism: Ultra-fine flip-fusion boundary bisection
evidence_used: Coefficient 0.08729248046875 produced 9,327 correct, while 0.08729400634765625 produced 9,328; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.06709100003354, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146877212524413, "validation_score": 9327.419650108923}

RECENT RESULT
hypothesis: A fusion coefficient of 0.0872936248779296875 will preserve 9,328 correct predictions while reducing validation cross-entropy below 0.1914687728881836.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket; retain all training settings and temperature 0.800713.
mechanism: Ultra-fine flip-fusion boundary bisection
evidence_used: Coefficient 0.087293243408203125 produced 9,327 correct with cross-entropy 0.19146877212524413, while 0.08729400634765625 produced 9,328 correct; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 70.41709516686387, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146877365112305, "validation_score": 9327.419650108384}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08729381561279296875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687744140625.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Ultra-fine flip-fusion boundary bisection
evidence_used: Coefficient 0.0872936248779296875 produced 9,327 correct, while 0.08729400634765625 produced 9,328; their midpoint is the most informative remaining boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.97299470915459, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687713623047, "validation_score": 9328.419650109192}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087293720245361328125 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687713623047.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Ultra-fine flip-fusion boundary bisection
evidence_used: Coefficient 0.0872936248779296875 produced 9,327 correct, while 0.08729381561279296875 produced 9,328 correct with the best observed cross-entropy; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.35504358285107, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687744140625, "validation_score": 9327.419650108117}

RECENT RESULT
hypothesis: A fusion coefficient of 0.0872937679290771484375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687713623047.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket and use temperature 0.800713.
mechanism: Ultra-fine flip-fusion boundary bisection
evidence_used: Coefficient 0.087293720245361328125 produced 9,327 correct, while 0.08729381561279296875 produced 9,328 correct with the best observed cross-entropy; their midpoint is the most informative next boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.12923370790668, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9328.419650108654}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08729374408721923828125 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687713623047.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket, retaining all other settings.
mechanism: Ultra-fine flip-fusion boundary bisection
evidence_used: Coefficient 0.087293720245361328125 produced 9,327 correct, while 0.0872937679290771484375 produced 9,328; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 59.12717179208994, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9327.419650108654}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087293756008148193359375 will preserve 9,328 correct predictions and may reduce cross-entropy below 0.1914687713623047.
change: Replace the evaluation-only confidence-adaptive flip-fusion coefficient with the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Ultra-fine flip-fusion decision-boundary bisection
evidence_used: Coefficient 0.08729374408721923828125 produced 9,327 correct, while 0.0872937679290771484375 produced 9,328; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.9387844169978, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9327.419650108654}

RECENT RESULT
hypothesis: A fusion coefficient of 0.0872937619686126708984375 will preserve 9,328 correct predictions and may lower cross-entropy below 0.1914687713623047.
change: Replace the evaluation-only confidence-adaptive flip-fusion coefficient with the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Ultra-fine flip-fusion decision-boundary bisection
evidence_used: Coefficient 0.087293756008148193359375 produced 9,327 correct, while 0.0872937679290771484375 produced 9,328; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 54.58396566705778, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9327.419650108654}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08729376494884490966796875 will preserve 9,328 correct predictions and may lower cross-entropy below 0.1914687713623047.
change: Replace the evaluation-only confidence-adaptive flip-fusion coefficient with the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Ultra-fine flip-fusion decision-boundary bisection
evidence_used: Coefficient 0.0872937619686126708984375 produced 9,327 correct, while 0.0872937679290771484375 produced 9,328; their midpoint is the most informative remaining boundary test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A fusion coefficient of 0.08729376494884490966796875 will preserve 9,328 correct predictions and may lower cross-entropy below 0.1914687713623047.
change: Replace the evaluation-only confidence-adaptive flip-fusion coefficient with the midpoint of the tightest verified 9,327/9,328 bracket.
mechanism: Retry unresolved flip-fusion boundary midpoint
evidence_used: 0.0872937619686126708984375 produced 9,327 correct, while 0.0872937679290771484375 produced 9,328; their midpoint’s prior verification timed out, leaving the most informative boundary test unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 79.63564995815977, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9328.419650108654}



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
