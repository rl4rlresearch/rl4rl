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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.957012499915436, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20857204895019532, "validation_score": 9278.41371137156}
prior_hypothesis: Setting all four batch-normalization momenta to 0.007 will exceed 9,279 correct predictions by reducing statistic staleness relative to 0.006 while retaining substantially more smoothing than the weaker 0.01 setting.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 75.907009416027, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20862859497070313, "validation_score": 9279.41369201596}
prior_hypothesis: Raising momentum from 0.006 to 0.0061 will retain 9,279 correct predictions while slightly lowering cross-entropy, thereby improving validation_score.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 73.1613966668956, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20862024459838868, "validation_score": 9279.413694874163}
prior_hypothesis: Raising all four batch-normalization momenta from 0.006 to 0.0062 will retain 9,279 correct predictions while further lowering validation cross-entropy.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 60.10115800006315, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2085987159729004, "validation_score": 9278.41370224326}
prior_hypothesis: Setting all batch-normalization momenta to 0.0065 will exceed 9,279 correct predictions by interpolating between the best verified momentum of 0.006 and the nearby 0.007 result.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting all four batch-normalization momenta to 0.01 will reproduce the verified 9,277-correct result, improving over the current 9,273 correct predictions without changing training optimization or capacity.
change: Add momentum=0.01 to every batch-normalization layer while preserving the architecture, loss, schedule, and evaluation calibration.
mechanism: Longer-horizon batch-normalization statistics
evidence_used: Reference Design 1 is otherwise identical and achieved 9,277 correct with momentum 0.01, compared with 9,273 for the current default-momentum design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting all four batch-normalization momenta to 0.01 and evaluation calibration to 1.17817 will reproduce the verified 9,277-correct result, exceeding the current 9,273 correct predictions.
change: Apply momentum=0.01 to every batch-normalization layer and use the calibration from the best qualified reference design.
mechanism: Longer-horizon batch-normalization statistics
evidence_used: Reference Design 2 used the identical architecture and training procedure with these settings and achieved 9,277 correct at 0.208538673 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing all batch-normalization momenta from 0.01 to 0.005 will exceed 9,277 correct predictions by further stabilizing evaluation statistics without altering training-mode normalization, optimization, or capacity.
change: Set every batch-normalization layer’s momentum to 0.005 while preserving the verified architecture, schedule, loss, and evaluation calibration.
mechanism: Extended-horizon batch-normalization statistics
evidence_used: Lowering momentum from the default 0.1 to 0.02 improved correctness from 9,273 to 9,276, and lowering it again to 0.01 improved correctness to 9,277; testing 0.005 is the most direct continuation of that isolated trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.780083833960816, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.208764559173584, "validation_score": 9278.413645483071}

RECENT RESULT
hypothesis: Reducing all four batch-normalization momenta from 0.01 to 0.005 will exceed 9,277 correct predictions by stabilizing evaluation statistics without altering training-mode normalization, optimization, or capacity.
change: Set every batch-normalization layer’s momentum to 0.005 while preserving the verified architecture, loss, schedule, and evaluation calibration.
mechanism: Extended-horizon batch-normalization statistics
evidence_used: The otherwise equivalent reference design with momentum 0.005 achieved 9,278 correct, improving on 9,277 at momentum 0.01 and continuing the observed gains from progressively lower momentum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.92440258292481, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.208764559173584, "validation_score": 9278.413645483071}

RECENT RESULT
hypothesis: Halving all batch-normalization momenta from 0.005 to 0.0025 will exceed 9,278 correct predictions by further stabilizing evaluation statistics without changing training-mode normalization, capacity, or optimization.
change: Set momentum=0.0025 on all four batch-normalization layers while preserving the verified architecture, schedule, loss, and evaluation calibration.
mechanism: Extended-horizon batch-normalization statistics
evidence_used: Successive momentum reductions improved correctness from 9,273 at 0.1 to 9,276 at 0.02, 9,277 at 0.01, and 9,278 at 0.005; another halving is the most direct test of that isolated trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.146878332830966, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21352931747436524, "validation_score": 9237.412021360176}

RECENT RESULT
hypothesis: Setting batch-normalization momentum to 0.004 will exceed 9,278 correct predictions by modestly extending the successful 0.005 averaging horizon without the severe statistic staleness observed at 0.0025.
change: Set all four batch-normalization layers to momentum 0.004 and retain the best reference design’s 1.17817 evaluation calibration.
mechanism: Near-optimal extended-horizon batch-normalization statistics
evidence_used: Momentum 0.005 achieved the best result of 9,278 correct, while 0.01 reached 9,277 and 0.0025 regressed to 9,237; 0.004 is a conservative unexplored point between the best setting and the lower-momentum failure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 60.328738583019, "validation_accuracy": 0.9269, "validation_correct": 9269, "validation_cross_entropy": 0.2089984043121338, "validation_score": 9269.413565475536}

RECENT RESULT
hypothesis: Increasing all batch-normalization momenta from 0.005 to 0.006 will exceed 9,278 correct predictions by reducing statistic staleness while retaining most of the noise suppression responsible for the current best result.
change: Set all four batch-normalization layers to momentum 0.006; preserve the architecture, optimizer, schedule, loss, and evaluation calibration.
mechanism: Near-optimal batch-normalization averaging horizon
evidence_used: Momentum 0.005 achieved 9,278 correct, while 0.01 achieved 9,277; reductions below 0.005 regressed sharply to 9,269 at 0.004 and 9,237 at 0.0025. Testing 0.006 explores the safer, higher-momentum side of the observed optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.83994604181498, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20863759803771972, "validation_score": 9279.413688934394}

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
