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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 60.10115800006315, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2085987159729004, "validation_score": 9278.41370224326}
prior_hypothesis: Setting all batch-normalization momenta to 0.0065 will exceed 9,279 correct predictions by interpolating between the best verified momentum of 0.006 and the nearby 0.007 result.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 66.71724349982105, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177345275879, "validation_score": 9279.413695733328}
prior_hypothesis: Momentum 0.006233599853515625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 83.27274833293632, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861771774291993, "validation_score": 9279.413695739075}
prior_hypothesis: Momentum 0.00623359375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617760.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 70.53509458294138, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861774139404296, "validation_score": 9279.413695730978}
prior_hypothesis: Momentum 0.0062336025238037109375 will retain 9,279 correct predictions while reducing cross-entropy below 0.20861771774291993.

## Recent verification evidence

RECENT RESULT
hypothesis: Momentum 0.006233984375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between 0.00623359375 and 0.006234375, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.006234375 lost one prediction despite lower cross-entropy; bisecting this remaining boundary is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 48.93965408299118, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861759490966797, "validation_score": 9278.413695781119}

RECENT RESULT
hypothesis: Momentum 0.0062337890625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between the successful 0.00623359375 setting and the one-error 0.006233984375 setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.006233984375 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.65566462487914, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.208617573928833, "validation_score": 9278.4136957883}

RECENT RESULT
hypothesis: Momentum 0.00623369140625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between the successful 0.00623359375 setting and the one-error 0.0062337890625 setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.0062337890625 lost one prediction despite lower cross-entropy; bisecting this remaining boundary is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.81741962512024, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861763763427735, "validation_score": 9278.413695766494}

RECENT RESULT
hypothesis: Momentum 0.006233642578125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.00623369140625 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 60.13562887487933, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861768188476562, "validation_score": 9278.413695751347}

RECENT RESULT
hypothesis: Momentum 0.0062336181640625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.006233642578125 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Momentum 0.0062336181640625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the previously untested midpoint, preserving every other setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.006233642578125 lost one prediction with lower cross-entropy; the prior midpoint patch was malformed, so 0.0062336181640625 remains the most informative boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.876728374976665, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861759033203126, "validation_score": 9278.413695782685}

RECENT RESULT
hypothesis: Momentum 0.00623360595703125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.0062336181640625 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.87527345912531, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2086175521850586, "validation_score": 9278.413695795743}

RECENT RESULT
hypothesis: Momentum 0.006233599853515625 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.208617718.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623359375 achieved 9,279 correct, while 0.00623360595703125 lost one prediction despite lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 66.71724349982105, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2086177345275879, "validation_score": 9279.413695733328}

RECENT RESULT
hypothesis: Momentum 0.0062336029052734375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.006233599853515625 retained 9,279 correct, while 0.00623360595703125 lost one prediction with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 68.94491833285429, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20861754760742188, "validation_score": 9278.41369579731}

RECENT RESULT
hypothesis: Momentum 0.00623360137939453125 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.006233599853515625 retained 9,279 correct, while 0.0062336029052734375 lost one prediction with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 69.99181479192339, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861773147583007, "validation_score": 9279.413695734374}

RECENT RESULT
hypothesis: Momentum 0.006233602142333984375 will retain 9,279 correct predictions while lowering validation cross-entropy below 0.20861773147583007.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.00623360137939453125 retained 9,279 correct, while 0.0062336029052734375 lost one prediction with lower cross-entropy; their untested midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 50.38677345798351, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861774139404296, "validation_score": 9279.413695730978}

RECENT RESULT
hypothesis: Momentum 0.0062336025238037109375 will retain 9,279 correct predictions while reducing cross-entropy below 0.20861771774291993.
change: Set all four batch-normalization momenta to the midpoint between the highest successful setting and the nearest one-error setting.
mechanism: Accuracy-boundary batch-normalization bisection
evidence_used: Momentum 0.006233602142333984375 retained 9,279 correct, while 0.0062336029052734375 produced 9,278; their untested midpoint is the most informative remaining boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 70.53509458294138, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20861774139404296, "validation_score": 9279.413695730978}



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
