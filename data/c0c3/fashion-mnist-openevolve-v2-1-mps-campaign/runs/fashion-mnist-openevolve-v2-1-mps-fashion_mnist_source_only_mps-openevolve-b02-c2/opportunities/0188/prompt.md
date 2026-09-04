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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.33100033388473, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19147203903198243, "validation_score": 9328.419648958281}
prior_hypothesis: Temperature 0.8007 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.06298958393745, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687774658203, "validation_score": 9328.419650107042}
prior_hypothesis: A fusion coefficient of 0.0873046875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 53.91276849992573, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9328.419650108654}
prior_hypothesis: A fusion coefficient of 0.0872955322265625 will preserve 9,328 correct predictions while reducing cross-entropy below the current 0.191468821.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.54648870788515, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687744140625, "validation_score": 9328.419650108117}
prior_hypothesis: A fusion coefficient of 0.08729400634765625 will preserve 9,328 correct predictions while reducing validation cross-entropy below 0.1914687728881836.

## Recent verification evidence

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.084375 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between the verified 0.08125 and 0.0875 designs; retain temperature 0.800713 and all training settings.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.08125 produced 9,327 correct with 0.191467437 cross-entropy, while 0.0875 produced 9,328 correct with 0.191468821; bisection is the most informative search for the smallest coefficient preserving the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.02939291717485, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914681121826172, "validation_score": 9327.419650341362}

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.0859375 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Reduce the evaluation-only flip-fusion coefficient from 0.0875 to the midpoint between 0.084375 and 0.0875; retain all training settings and temperature 0.800713.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.084375 produced 9,327 correct at 0.191468112 cross-entropy, while 0.0875 produced 9,328 correct at 0.191468821; their midpoint is the most informative next test for the smallest coefficient preserving the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 68.05107733304612, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146846160888673, "validation_score": 9327.41965021829}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08671875 will preserve 9,328 correct predictions while lowering cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.0859375 and 0.0875; retain temperature 0.800713 and all training settings.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.0859375 yielded 9,327 correct at 0.191468462 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.77383895893581, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146864013671874, "validation_score": 9327.41965015541}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087109375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.08671875 and 0.0875, and use the best verified temperature of 0.800713.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.08671875 yielded 9,327 correct at 0.191468640 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 62.650059666950256, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146873474121093, "validation_score": 9327.41965012209}

RECENT RESULT
hypothesis: A fusion coefficient of 0.0873046875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.087109375 and 0.0875, retaining temperature 0.800713 and all training settings.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.087109375 yielded 9,327 correct at 0.191468735 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.06298958393745, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687774658203, "validation_score": 9328.419650107042}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08720703125 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.087109375 and 0.0873046875, retaining all training settings and temperature 0.800713.
mechanism: Precision bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.087109375 yielded 9,327 correct at 0.191468735 cross-entropy, while 0.0873046875 yielded 9,328 correct at 0.191468777; their midpoint is the most informative next test for the smallest coefficient retaining the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.25742333289236, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687530517578, "validation_score": 9327.41965011564}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087255859375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.08720703125 and 0.0873046875, retaining temperature 0.800713 and all training settings.
mechanism: Precision bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.08720703125 yielded 9,327 correct, while 0.0873046875 yielded 9,328 correct at 0.191468777 cross-entropy; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.29992450005375, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687629699707, "validation_score": 9327.419650112148}

RECENT RESULT
hypothesis: A fusion coefficient of 0.0872802734375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.087255859375 and 0.0873046875, retaining temperature 0.800713 and all training settings.
mechanism: Precision bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.087255859375 yielded 9,327 correct, while 0.0873046875 yielded 9,328 correct at 0.191468777 cross-entropy; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.36375833302736, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146877059936523, "validation_score": 9327.41965010946}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08729248046875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.0872802734375 and 0.0873046875, while using the best verified temperature of 0.800713.
mechanism: Precision bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.0872802734375 yielded 9,327 correct, whereas 0.0873046875 yielded 9,328 correct at 0.191468777 cross-entropy; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.45046583283693, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687728881836, "validation_score": 9327.419650108654}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087298583984375 will preserve 9,328 correct predictions while reducing validation cross-entropy below 0.1914687774658203.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between the largest verified 9,327 coefficient and the smallest verified 9,328 coefficient, retaining temperature 0.800713 and all training settings.
mechanism: Ultra-fine bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.08729248046875 yielded 9,327 correct, while 0.0873046875 yielded 9,328 correct at 0.1914687774658203 cross-entropy; their midpoint is the most informative remaining boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.00860383315012, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687774658203, "validation_score": 9328.419650107042}

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
