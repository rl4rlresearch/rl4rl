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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 68.40571224992163, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861835823059083, "validation_score": 9279.413695519843}
prior_hypothesis: Momentum 0.006225 will retain 9,279 correct predictions while lowering cross-entropy below 0.208620245.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.4407377501484, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861784286499024, "validation_score": 9279.413695696247}
prior_hypothesis: Momentum 0.00623125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208618358.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 75.907009416027, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20862859497070313, "validation_score": 9279.41369201596}
prior_hypothesis: Raising momentum from 0.006 to 0.0061 will retain 9,279 correct predictions while slightly lowering cross-entropy, thereby improving validation_score.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 60.10115800006315, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2085987159729004, "validation_score": 9278.41370224326}
prior_hypothesis: Setting all batch-normalization momenta to 0.0065 will exceed 9,279 correct predictions by interpolating between the best verified momentum of 0.006 and the nearby 0.007 result.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting all four batch-normalization momenta to 0.007 will exceed 9,279 correct predictions by reducing statistic staleness relative to 0.006 while retaining substantially more smoothing than the weaker 0.01 setting.
change: Increase every batch-normalization momentum from 0.005 to 0.007 while preserving the architecture, optimization, loss, schedule, and evaluation calibration.
mechanism: Intermediate-horizon batch-normalization statistics
evidence_used: Momentum 0.006 achieved the best result at 9,279 correct, improving over 9,278 at 0.005, while 0.01 achieved 9,277; 0.007 is the closest unexplored point in the supported higher-momentum direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.957012499915436, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20857204895019532, "validation_score": 9278.41371137156}

RECENT RESULT
hypothesis: Setting all four batch-normalization momenta to 0.006 will reproduce the verified 9,279 correct predictions and exceed the current 9,273.
change: Apply momentum=0.006 to every batch-normalization layer while preserving architecture, optimization, loss, schedule, and evaluation calibration.
mechanism: Near-optimal batch-normalization averaging horizon
evidence_used: Reference Design 2 achieved the best available result—9,279 correct—with this isolated setting; neighboring momenta 0.005 and 0.007 each reached only 9,278.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 73.42164154187776, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20863759803771972, "validation_score": 9279.413688934394}

RECENT RESULT
hypothesis: Setting all batch-normalization momenta to 0.0065 will exceed 9,279 correct predictions by interpolating between the best verified momentum of 0.006 and the nearby 0.007 result.
change: Apply momentum=0.0065 to all four batch-normalization layers and adopt the best reference design’s 1.17817 evaluation calibration.
mechanism: Fine-grained batch-normalization averaging horizon
evidence_used: Momentum 0.006 achieved 9,279 correct, while both neighboring tested settings—0.005 and 0.007—achieved 9,278; 0.0065 is the closest unexplored setting on the less failure-prone higher-momentum side.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 60.10115800006315, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2085987159729004, "validation_score": 9278.41370224326}

RECENT RESULT
hypothesis: Raising momentum from 0.006 to 0.0061 will retain 9,279 correct predictions while slightly lowering cross-entropy, thereby improving validation_score.
change: Set all four batch-normalization momenta to 0.0061 while preserving architecture, optimization, loss, and calibration.
mechanism: Conservative batch-normalization horizon refinement
evidence_used: Momentum 0.006 produced the best correctness, while 0.0065 produced lower cross-entropy but lost one correct prediction; a smaller upward step tests whether the cross-entropy benefit begins before that accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 75.907009416027, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20862859497070313, "validation_score": 9279.41369201596}

RECENT RESULT
hypothesis: Raising all four batch-normalization momenta from 0.006 to 0.0062 will retain 9,279 correct predictions while further lowering validation cross-entropy.
change: Set every batch-normalization layer’s momentum to 0.0062 while preserving architecture, optimization, loss, schedule, and evaluation calibration.
mechanism: Fine-grained batch-normalization horizon refinement
evidence_used: Momentum 0.0061 retained the best 9,279-correct result and reduced cross-entropy from 0.208637598 at 0.006 to 0.208628595; momentum 0.0065 lowered cross-entropy further but lost one correct prediction, motivating the next small upward step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 73.1613966668956, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20862024459838868, "validation_score": 9279.413694874163}

RECENT RESULT
hypothesis: Setting all four batch-normalization momenta to 0.0063 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208620245.
change: Raise every batch-normalization momentum from 0.007 to 0.0063 while preserving architecture, optimization, loss, schedule, and evaluation calibration.
mechanism: Fine-grained batch-normalization horizon refinement
evidence_used: Momentum increases from 0.006 to 0.0061 and 0.0062 retained 9,279 correct while monotonically reducing cross-entropy; 0.0065 lost one correct, making 0.0063 the next informative point toward that boundary.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 83.82144808303565, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861266021728517, "validation_score": 9278.413697470214}

RECENT RESULT
hypothesis: Momentum 0.00625 will retain 9,279 correct predictions while lowering cross-entropy below the 0.0062 result of 0.208620245.
change: Set all four batch-normalization momenta from 0.0065 to 0.00625, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization refinement
evidence_used: Momentum 0.0062 achieved 9,279 correct, while 0.0063 lowered cross-entropy to 0.208612660 but lost one correct prediction; their midpoint directly probes the apparent accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 70.1951156668365, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2086163013458252, "validation_score": 9278.413696223892}

RECENT RESULT
hypothesis: Momentum 0.006225 will retain 9,279 correct predictions while lowering cross-entropy below 0.208620245.
change: Set all four batch-normalization momenta from 0.0062 to 0.006225, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization refinement
evidence_used: Momentum 0.0062 retained 9,279 correct, whereas 0.00625 lowered cross-entropy but lost one prediction; their midpoint most directly probes the narrow accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 68.40571224992163, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861835823059083, "validation_score": 9279.413695519843}

RECENT RESULT
hypothesis: Momentum 0.0062375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208618358.
change: Set all four batch-normalization momenta from 0.0061 to 0.0062375 while preserving every other setting.
mechanism: Accuracy-boundary batch-normalization refinement
evidence_used: Momentum 0.006225 retained 9,279 correct, whereas 0.00625 lowered cross-entropy but lost one prediction; their midpoint directly probes the narrow accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 69.13970995787531, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861732482910156, "validation_score": 9278.413695873563}

RECENT RESULT
hypothesis: Momentum 0.00623125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208618358.
change: Set all four batch-normalization momenta to the midpoint between the successful 0.006225 setting and the one-error 0.0062375 setting, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization refinement
evidence_used: Momentum 0.006225 achieved 9,279 correct, while 0.0062375 lowered cross-entropy but lost one prediction; bisecting this narrow boundary is the most informative next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.4407377501484, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861784286499024, "validation_score": 9279.413695696247}

RECENT RESULT
hypothesis: Momentum 0.006234375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617843.
change: Set all four batch-normalization momenta to the midpoint between 0.00623125 and 0.0062375, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization refinement
evidence_used: Momentum 0.00623125 achieved 9,279 correct, while 0.0062375 reduced cross-entropy but lost one prediction; bisecting this narrow boundary is the most informative next test.
result: the patch search text did not match the source

RECENT RESULT
hypothesis: Momentum 0.006234375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617843.
change: Set all four batch-normalization momenta to 0.006234375, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization refinement
evidence_used: Momentum 0.00623125 achieved 9,279 correct, while 0.0062375 lowered cross-entropy but lost one prediction; the prior midpoint attempt was malformed and therefore left this boundary point untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 81.30879979208112, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861767044067384, "validation_score": 9278.413695755266}



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
