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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.716946167172864, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2374291809082031, "validation_score": 9192.404063527605}
prior_hypothesis: A diagonal weight of 0.7162109375 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374292724609375.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.64090537489392, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2374292724609375, "validation_score": 9192.40406349771}
prior_hypothesis: A diagonal weight of 0.71640625 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23742967 achieved at 0.7171875.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.08528629108332, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742913818359376, "validation_score": 9192.404063541557}
prior_hypothesis: A diagonal weight of 0.7161365509033203125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742914199829102.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.37880758289248, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742913589477538, "validation_score": 9192.404063542304}
prior_hypothesis: A diagonal weight of 0.7161376953125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742915229797362.

## Recent verification evidence

RECENT RESULT
hypothesis: A diagonal weight of 0.71611328125 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374291809082031.
change: Bisect the diagonal-view weight between 0.716015625 and 0.7162109375 and adjust ensemble normalization to 15.72890625.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7162109375 achieved 9,192 correct, while 0.716015625 achieved 9,191 with slightly lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 62.803086541825905, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742913360595702, "validation_score": 9191.404063543052}

RECENT RESULT
hypothesis: A diagonal weight of 0.716162109375 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374291809082031.
change: Bisect the diagonal-view weight between 0.71611328125 and 0.7162109375 and normalize the ensemble by its exact total weight, 15.729296875.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7162109375 achieved 9,192 correct, while 0.71611328125 achieved 9,191 with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 74.13424341683276, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742915229797362, "validation_score": 9192.404063536947}

RECENT RESULT
hypothesis: A diagonal weight of 0.7161376953125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742915229797362.
change: Bisect the diagonal-view weight between 0.71611328125 and 0.716162109375 and normalize by the exact total ensemble weight, 15.7291015625.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.716162109375 achieved 9,192 correct, while 0.71611328125 achieved 9,191 with lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.37880758289248, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742913589477538, "validation_score": 9192.404063542304}

RECENT RESULT
hypothesis: A diagonal weight of 0.71612548828125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.
change: Bisect the diagonal-view weight between the failed 0.71611328125 and successful 0.7161376953125 settings, and normalize by the exact total ensemble weight of 15.72900390625.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7161376953125 achieved 9,192 correct with 0.23742913589477538 cross-entropy, while 0.71611328125 achieved 9,191 correct with slightly lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.78178787510842, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.2374291389465332, "validation_score": 9191.404063541308}

RECENT RESULT
hypothesis: A diagonal weight of 0.716131591796875 will retain 9,192 correct predictions while reducing cross-entropy below 0.23742913589477538.
change: Bisect between the failed 0.71612548828125 and successful 0.7161376953125 diagonal weights, using the exact total ensemble weight of 15.729052734375.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7161376953125 achieved 9,192 correct, while 0.71612548828125 achieved 9,191; their midpoint is the most informative remaining probe of the accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.41020429204218, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742913551330566, "validation_score": 9191.404063542428}

RECENT RESULT
hypothesis: A diagonal weight of 0.7161346435546875 will retain 9,192 correct predictions while reducing cross-entropy below 0.23742913589477538.
change: Bisect between the failed 0.716131591796875 and successful 0.7161376953125 diagonal weights, normalizing by the exact total ensemble weight of 15.7290771484375.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7161376953125 achieved 9,192 correct, whereas 0.716131591796875 achieved 9,191 with marginally lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A diagonal weight of 0.7161346435546875 will retain 9,192 correct predictions while reducing cross-entropy below 0.23742913589477538.
change: Bisect between the highest confirmed failing weight, 0.716131591796875, and the lowest confirmed successful weight, 0.7161376953125, with exact ensemble normalization.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: The two confirmed endpoints bracket the accuracy transition; the previous verification of this midpoint timed out and therefore supplied no evidence about its prediction count or cross-entropy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.90754104102962, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742913932800294, "validation_score": 9191.404063541184}

RECENT RESULT
hypothesis: A diagonal weight of 0.71613616943359375 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.
change: Bisect between the highest confirmed failing diagonal weight, 0.7161346435546875, and the lowest confirmed successful weight, 0.7161376953125, using the exact total ensemble weight of 15.72908935546875.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.7161346435546875 produced 9,191 correct, while 0.7161376953125 produced 9,192; their midpoint is the most informative remaining probe of the accuracy transition.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A diagonal weight of 0.71613616943359375 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.
change: Retry the unresolved midpoint between the highest confirmed failing and lowest confirmed successful diagonal weights, with exact ensemble normalization.
mechanism: Accuracy-boundary diagonal-weight bisection retry
evidence_used: Weight 0.7161346435546875 produced 9,191 correct and 0.7161376953125 produced 9,192; their midpoint is still the most informative probe because its only verification timed out and supplied no performance evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.18098683306016, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742914276123048, "validation_score": 9191.404063540062}

RECENT RESULT
hypothesis: A diagonal weight of 0.716136932373046875 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.
change: Bisect between the highest confirmed failing weight, 0.71613616943359375, and the lowest confirmed successful weight, 0.7161376953125, and normalize by the exact total ensemble weight.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: The two confirmed endpoints differ in validation correctness, so their midpoint is the most informative remaining probe of the accuracy transition.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.869725291850045, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742914199829102, "validation_score": 9192.40406354031}

RECENT RESULT
hypothesis: A diagonal weight of 0.7161365509033203125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.
change: Bisect between the highest confirmed failing diagonal weight and the lowest confirmed successful weight, with exact ensemble normalization.
mechanism: Accuracy-boundary diagonal-weight bisection
evidence_used: Weight 0.71613616943359375 produced 9,191 correct, whereas 0.716136932373046875 produced 9,192; their midpoint is the most informative remaining probe of the accuracy transition.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A diagonal weight of 0.7161365509033203125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742914199829102.
change: Retry the unresolved midpoint between the highest confirmed failing and lowest confirmed successful diagonal weights, with exact ensemble normalization.
mechanism: Accuracy-boundary diagonal-weight bisection retry
evidence_used: Weight 0.71613616943359375 produced 9,191 correct, while 0.716136932373046875 produced 9,192; the midpoint’s only verification timed out, so it remains the most informative accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.08528629108332, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742913818359376, "validation_score": 9192.404063541557}



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
