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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 63.25571879092604, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.20861263732910157, "validation_score": 9276.413697478049}
prior_hypothesis: Restoring unaugmented training and setting all four batch-normalization momenta to 0.02 will reproduce the verified 9,276-correct result and outperform the current 9,121-correct implementation.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 79.80127229215577, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20853867263793946, "validation_score": 9277.41372279706}
prior_hypothesis: Reducing all batch-normalization momenta from 0.02 to 0.01 will exceed 9,276 correct predictions by further reducing noise in evaluation statistics without changing training-mode normalization, capacity, or optimization.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.86811845796183, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20851774215698243, "validation_score": 9273.413729962382}
prior_hypothesis: Scaling evaluation logits by 1.17817 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.2085177444458008.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 71.25856295810081, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177448272705, "validation_score": 9273.413729961467}
prior_hypothesis: Scaling evaluation logits by 1.17810 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517744.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.209594.
change: Increase the evaluation-only positive logit multiplier from 1.1 to 1.2 without changing training.
mechanism: Stronger evaluation-only logit sharpening
evidence_used: Scaling logits by 1.1 preserved the best 9,273 correct predictions and reduced cross-entropy from 0.215099 to 0.209594, motivating one further sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.813546834047884, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20859080200195312, "validation_score": 9273.413704952223}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208591.
change: Apply a fixed 1.17 multiplier to logits only during evaluation, leaving training and learned parameters unchanged.
mechanism: Evaluation-only temperature calibration
evidence_used: Multipliers of 1.1 and 1.2 preserved 9,273 correct while reducing cross-entropy from 0.215099 to 0.209594 and 0.208591; these measurements suggest the calibration optimum lies near 1.17.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.78242424991913, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20852822303771973, "validation_score": 9273.413726374336}

RECENT RESULT
hypothesis: Restoring the verified 1e-4 cosine endpoint and scaling evaluation logits by 1.179 will preserve 9,273 correct predictions while reducing cross-entropy below 0.208528.
change: Restore the best training schedule and apply a fixed 1.179 logit multiplier only during evaluation.
mechanism: Quadratically refined evaluation-only temperature calibration
evidence_used: Evaluation scales 1.1, 1.17, and 1.2 preserved 9,273 correct with cross-entropies 0.209594, 0.208528, and 0.208591; quadratic interpolation places the minimum near 1.179.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.5536045411136, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20851787033081054, "validation_score": 9273.413729918502}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17823 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517870.
change: Refine the evaluation-only logit multiplier from 1.179 to the quadratic-fit optimum of approximately 1.17823.
mechanism: Fine-grained evaluation temperature calibration
evidence_used: Scales 1.17, 1.179, and 1.2 preserved 9,273 correct with cross-entropies 0.208528223, 0.208517870, and 0.208590802; quadratic interpolation of these measurements places the minimum near 1.17823.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.80794441699982, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177444458008, "validation_score": 9273.413729961598}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17810 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517744.
change: Refine the evaluation-only positive logit multiplier from 1.17 to 1.17810 without changing training or learned parameters.
mechanism: Fine-grained evaluation temperature calibration
evidence_used: Scales 1.17, 1.17823, and 1.179 preserved 9,273 correct with cross-entropies 0.208528223, 0.208517744, and 0.208517870; local quadratic interpolation places the minimum near 1.17810.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 71.25856295810081, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177448272705, "validation_score": 9273.413729961467}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17817 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.2085177444458008.
change: Replace the evaluation-only 1.2 logit multiplier with 1.17817, leaving training and learned parameters unchanged.
mechanism: Locally optimized evaluation temperature calibration
evidence_used: Scales 1.17810, 1.17823, and 1.179 preserved 9,273 correct with cross-entropies 0.2085177448272705, 0.2085177444458008, and 0.20851787033081054; local quadratic interpolation places the minimum near 1.17817.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.86811845796183, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20851774215698243, "validation_score": 9273.413729962382}

RECENT RESULT
hypothesis: Training the verified 9,273-correct pairwise-refinement model with independent zero-padded translations of up to two pixels will exceed 9,273 correct predictions by learning modest positional invariance without changing model capacity or the successful optimization schedule.
change: Restore the best 248,426-parameter design and its 1.17817 evaluation calibration, then apply a random 28×28 crop from images padded by two pixels during training.
mechanism: Per-example random translation augmentation
evidence_used: Reference Design 2 achieved the best validation result at 9,273 correct; architectural refinements, EMA, alternate pooling, loss annealing, and dropout removal all regressed, motivating an isolated input-regularization change while preserving every verified model and optimizer setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 41.50152979185805, "validation_accuracy": 0.9121, "validation_correct": 9121, "validation_cross_entropy": 0.23574685134887696, "validation_score": 9121.404613614393}

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
