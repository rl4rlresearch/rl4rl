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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.780083833960816, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.208764559173584, "validation_score": 9278.413645483071}
prior_hypothesis: Reducing all batch-normalization momenta from 0.01 to 0.005 will exceed 9,277 correct predictions by further stabilizing evaluation statistics without altering training-mode normalization, optimization, or capacity.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.92440258292481, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.208764559173584, "validation_score": 9278.413645483071}
prior_hypothesis: Reducing all four batch-normalization momenta from 0.01 to 0.005 will exceed 9,277 correct predictions by stabilizing evaluation statistics without altering training-mode normalization, optimization, or capacity.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.86811845796183, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20851774215698243, "validation_score": 9273.413729962382}
prior_hypothesis: Scaling evaluation logits by 1.17817 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.2085177444458008.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 71.25856295810081, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177448272705, "validation_score": 9273.413729961467}
prior_hypothesis: Scaling evaluation logits by 1.17810 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517744.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging original and horizontally reflected evaluation logits will exceed 9,273 correct predictions by reducing orientation-specific errors while preserving the best verified training procedure.
change: During evaluation only, average logits from each image and its horizontal reflection before applying the verified 1.17817 calibration.
mechanism: Horizontal-reflection test-time self-ensemble
evidence_used: Training-time translation reduced correctness from 9,273 to 9,121, while architectural, loss, and regularization changes also regressed; an evaluation-only ensemble can change erroneous decisions without disturbing the best learned model.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 69.08042062493041, "validation_accuracy": 0.837, "validation_correct": 8370, "validation_cross_entropy": 0.3817786445617676, "validation_score": 8370.361852458762}

RECENT RESULT
hypothesis: Reducing batch-normalization momentum to 0.02 will exceed 9,273 correct predictions by producing less noisy evaluation statistics while leaving training-mode activations and learned parameters unchanged.
change: Use momentum 0.02 in every batch-normalization layer and retain the best verified 1.17817 evaluation-logit calibration.
mechanism: Longer-horizon batch-normalization population estimates
evidence_used: The unchanged model reached 9,273 correct, while architectural, augmentation, loss, and test-time reflection changes regressed; adjusting only batch-normalization running-statistic estimation preserves its successful optimization trajectory while enabling different validation decisions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 70.6279664998874, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.20861263732910157, "validation_score": 9276.413697478049}

RECENT RESULT
hypothesis: Setting every batch-normalization momentum to 0.02 and using the verified 1.17817 evaluation calibration will exceed the current 9,273 correct predictions, reproducing the observed improvement to 9,276.
change: Change all four batch-normalization layers from default momentum to 0.02 and refine the evaluation-only logit multiplier from 1.17810 to 1.17817.
mechanism: Longer-horizon batch-normalization statistics
evidence_used: Reference Design 2 changed these settings on the same 248,426-parameter architecture and achieved 9,276 correct versus the current design’s 9,273.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring unaugmented training and setting all four batch-normalization momenta to 0.02 will reproduce the verified 9,276-correct result and outperform the current 9,121-correct implementation.
change: Remove the harmful random-translation augmentation while applying momentum 0.02 to every batch-normalization layer; retain the verified architecture, optimizer, schedule, and 1.17817 evaluation calibration.
mechanism: Longer-horizon batch-normalization statistics without translation augmentation
evidence_used: Reference Design 1 achieved 9,276 correct with momentum 0.02 and unaugmented training, whereas the current translation augmentation reduced correctness to 9,121; the previous attempt to restore these settings could not be verified.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 63.25571879092604, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.20861263732910157, "validation_score": 9276.413697478049}

RECENT RESULT
hypothesis: Reducing all batch-normalization momenta from 0.02 to 0.01 will exceed 9,276 correct predictions by further reducing noise in evaluation statistics without changing training-mode normalization, capacity, or optimization.
change: Set every batch-normalization layer’s momentum to 0.01 while preserving the verified architecture, training procedure, and evaluation calibration.
mechanism: Longer-horizon batch-normalization statistics
evidence_used: Lowering momentum from the default 0.1 to 0.02 improved correctness from 9,273 to 9,276; extending that isolated trend is the most informative next test after other architecture, augmentation, loss, and test-time ensemble changes regressed.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 79.80127229215577, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20853867263793946, "validation_score": 9277.41372279706}

RECENT RESULT
hypothesis: Reducing all four batch-normalization momenta from 0.02 to 0.01 will reproduce the verified 9,277-correct result and exceed the current 9,276 correct predictions without changing optimization or capacity.
change: Set every batch-normalization layer’s momentum to 0.01 while preserving the architecture, training schedule, loss, and evaluation calibration.
mechanism: Longer-horizon batch-normalization statistics
evidence_used: Reference Design 1 made this isolated momentum change on the same 248,426-parameter procedure and achieved 9,277 correct versus 9,276 at momentum 0.02.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 64.04201279184781, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20853867263793946, "validation_score": 9277.41372279706}

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
