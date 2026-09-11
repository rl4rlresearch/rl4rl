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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 50.433580124983564, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436275100708008, "validation_score": 9359.422167954517}
prior_hypothesis: Increasing the accepted translation blend from 13.535% to 13.5375% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627540588379.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.5075% to 13.51% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436280479431152.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.75375% to 6.755%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.5075% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 57.71390941715799, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436280059814453, "validation_score": 9359.422167936842}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.51% to 13.5125% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436280059814453.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.755% to 6.75625%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.51% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 73.73643025010824, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436279525756835, "validation_score": 9359.422167938745}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.5125% to 13.515% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436279525756835.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.75625% to 6.7575%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.5125% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 51.49050441686995, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436279106140135, "validation_score": 9359.42216794024}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.515% to 13.5175% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436279106140135.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7575% to 6.75875%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.515% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.92546937498264, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436278419494628, "validation_score": 9359.422167942688}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.5175% to 13.52% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436278419494628.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.75875% to 6.76%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.5175% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 62.543648708146065, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843627815246582, "validation_score": 9359.42216794364}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.52% to 13.5225% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627815246582.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76% to 6.76125%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.52% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 48.419926000060514, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436277389526368, "validation_score": 9359.422167946359}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.5225% to 13.525% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436277389526368.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76125% to 6.7625%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.5225% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 61.33567599998787, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843627700805664, "validation_score": 9359.42216794772}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.525% to 13.5275% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627700805664.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7625% to 6.76375%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.525% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.37949812505394, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436276741027832, "validation_score": 9359.42216794867}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.5275% to 13.53% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436276741027832.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76375% to 6.765%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.5275% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.19093449995853, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843627639770508, "validation_score": 9359.422167949895}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.53% to 13.5325% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627639770508.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.765% to 6.76625%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.53% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 51.53483720798977, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843627586364746, "validation_score": 9359.422167951798}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.5325% to 13.535% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627586364746.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76625% to 6.7675%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.5325% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 51.90062920819037, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843627540588379, "validation_score": 9359.42216795343}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.535% to 13.5375% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627540588379.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7675% to 6.76875%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.535% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 50.433580124983564, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436275100708008, "validation_score": 9359.422167954517}



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
