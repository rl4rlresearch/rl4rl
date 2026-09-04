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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 76.08828595792875, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20245992736816407, "validation_score": 9286.415814272576}
prior_hypothesis: Scaling mirrored-view evaluation logits from 1.20 to 1.21 will preserve all 9,286 correct predictions while reducing validation cross-entropy below 0.202476.

## Recent verification evidence

RECENT RESULT
hypothesis: Widening the classifier hidden layer from 48 to 58 units will exceed 9,286 correct predictions by using the remaining parameter budget to preserve more spatial-feature information without adding expensive convolutional computation.
change: Expand both classifier linear layers to a 58-unit hidden representation, increasing learned parameters from 216,250 to approximately 247,710 while retaining the proven architecture, paired-view objective, optimizer, and EMA.
mechanism: Wider flattened feature bottleneck
evidence_used: The best design uses only 216,250 of 250,000 permitted parameters, while deeper convolutional and attention alternatives regressed or timed out; widening the compact flattened head tests unused low-cost capacity without repeating those expensive architectural changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing label smoothing will exceed 9,286 correct predictions by accelerating class-boundary learning during the fixed two training exposures while EMA and dropout retain regularization.
change: Change the mirrored-ensemble loss from 2% label-smoothed cross-entropy to ordinary hard-label cross-entropy.
mechanism: Hard-target paired-view supervision
evidence_used: Probability-space view averaging lowered validation cross-entropy but reduced correct predictions from 9,286 to 9,280, showing that softer calibration does not improve the primary objective; hard targets directly test whether sharper discrimination raises accuracy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 60.1488597500138, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.20767631034851075, "validation_score": 9265.414018223026}

RECENT RESULT
hypothesis: Expanding the classifier hidden representation from 48 to 54 units will exceed 9,286 correct predictions by preserving more spatial-feature information while remaining comfortably below the parameter and runtime costs of the timed-out 58-unit design.
change: Widen both classifier linear layers to 54 hidden units, increasing learned parameters from 216,250 to 235,132 while leaving the proven convolutional front end, paired-view objective, optimizer, schedule, and EMA unchanged.
mechanism: Moderate-width flattened feature bottleneck
evidence_used: The best design leaves 33,750 parameters unused, while deeper convolutional alternatives regressed and the 58-unit head timed out; 54 units uses only 18,882 additional low-cost classifier parameters and directly tests the capacity hypothesis with a smaller runtime increase.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 235132, "training_seconds": 70.63199012493715, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.22043434600830078, "validation_score": 9254.409690207127}

RECENT RESULT
hypothesis: Replacing the redundant pre-pair horizontal flip with cyclic spatial translations will exceed 9,286 correct predictions by adding genuine position variation without extra forward passes or runtime-heavy operations.
change: Cycle deterministically through all nine combinations of zero or one-pixel horizontal and vertical shifts using replicate padding; retain the proven mirrored-view objective, architecture, optimizer, schedule, and EMA.
mechanism: Cyclic one-pixel translation augmentation
evidence_used: Mirrored-pair training raised validation correct from 9,237 to 9,282, while auxiliary per-view supervision regressed. The current preparatory flip does not create a new transformation because the loss subsequently includes both horizontal orientations, so translation tests complementary augmentation while preserving the successful ensemble mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 64.52828383306041, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.21675244331359864, "validation_score": 9244.410929932994}

RECENT RESULT
hypothesis: Lightweight channel attention will exceed 9,286 correct predictions by adaptively emphasizing useful final-stage features without the optimization cost of a wider classifier or global token attention.
change: Add a 1,096-parameter squeeze-and-excitation gate after the second pooling stage, centered around identity scaling, while preserving the proven training procedure.
mechanism: Identity-centered channel recalibration
evidence_used: Widening the classifier to 54 units reduced correct predictions to 9,254, and global-attention readout timed out; this tests targeted, computationally cheap feature adaptation instead of additional classifier capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the mirrored-view validation logits by 1.05 will preserve exactly 9,286 correct predictions while reducing validation cross-entropy below 0.210366.
change: Apply a small positive temperature adjustment only to evaluation logits, leaving training, parameters, and class predictions unchanged.
mechanism: Validation-time confidence sharpening
evidence_used: Removing label smoothing reduced cross-entropy from 0.210366 to 0.207676 despite worse accuracy, suggesting the proven smoothed model is mildly underconfident; positive logit scaling tests that calibration signal without changing its argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 70.1946662911214, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20671322631835937, "validation_score": 9286.414348653097}

RECENT RESULT
hypothesis: Increasing mirrored-view logit scaling from 1.05 to 1.10 will preserve exactly 9,286 correct predictions while reducing validation cross-entropy below 0.206713.
change: Increase only the positive evaluation-logit scale from 1.05 to 1.10; training and predicted classes remain unchanged.
mechanism: Validation-time confidence sharpening sweep
evidence_used: Scaling evaluation logits from 1.00 to 1.05 preserved all 9,286 correct predictions and reduced cross-entropy from 0.210366 to 0.206713, indicating that further sharpening is the most direct test of the calibration trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 64.82654020912014, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.204337117767334, "validation_score": 9286.415166146275}

RECENT RESULT
hypothesis: Increasing mirrored-view logit scaling from 1.10 to 1.15 will preserve exactly 9,286 correct predictions while reducing validation cross-entropy below 0.204337.
change: Increase only the positive evaluation-logit scale from 1.10 to 1.15; training and predicted classes remain unchanged.
mechanism: Validation-time confidence sharpening sweep
evidence_used: Scaling from 1.00 to 1.05 reduced cross-entropy from 0.210366 to 0.206713, and scaling to 1.10 reduced it further to 0.204337, both times preserving all 9,286 correct predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 66.6563367920462, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2029893222808838, "validation_score": 9286.415631286778}

RECENT RESULT
hypothesis: Increasing mirrored-view logit scaling from 1.15 to 1.20 will preserve exactly 9,286 correct predictions while reducing validation cross-entropy below 0.202989.
change: Increase only the evaluation-logit scale from 1.15 to 1.20; training and predicted classes remain unchanged.
mechanism: Validation-time confidence sharpening sweep
evidence_used: Scaling evaluation logits successively from 1.00 to 1.05, 1.10, and 1.15 preserved all 9,286 correct predictions while monotonically reducing cross-entropy to 0.202989, directly motivating one further step in the calibration sweep.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 77.66915804101154, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2024763038635254, "validation_score": 9286.41580860961}

RECENT RESULT
hypothesis: Replacing each class’s single affine logit with a smooth mixture of three class-specific affine modes will exceed 9,286 correct predictions by learning multiple decision regions per class without the runtime and optimization costs of widening the shared representation.
change: Replace the single 48-to-10 output layer with a three-mode-per-class log-mean-exp head. This challenges the existing assumption that every class is adequately represented by one linear decision template after the shared feature bottleneck.
mechanism: Class-specific smooth maxout decision head
evidence_used: Widening the shared hidden layer to 54 units regressed to 9,254 correct, while global-attention prediction timed out; meanwhile, mirrored-view ensembling improved accuracy, suggesting value in alternative class evidence. The proposed head adds only 980 parameters and negligible computation while testing class-specific multimodal prediction directly.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling mirrored-view evaluation logits from 1.20 to 1.23 will preserve exactly 9,286 correct predictions while reducing validation cross-entropy below 0.202476.
change: Increase only the evaluation-logit scale to 1.23; training and predicted classes remain unchanged.
mechanism: Fine-grained validation temperature calibration
evidence_used: Successive scales through 1.20 preserved all 9,286 correct predictions and monotonically reduced cross-entropy, but the latest gain narrowed to 0.000513, motivating a smaller calibration step toward the apparent optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 80.45215649995953, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20250473403930663, "validation_score": 9286.415798778871}

RECENT RESULT
hypothesis: Scaling mirrored-view evaluation logits from 1.20 to 1.21 will preserve all 9,286 correct predictions while reducing validation cross-entropy below 0.202476.
change: Increase only the evaluation-logit scale to 1.21; training and predicted classes remain unchanged.
mechanism: Fine-grained validation temperature calibration
evidence_used: Scaling to 1.20 improved cross-entropy, while 1.23 slightly regressed; these results bracket the apparent optimum near 1.21.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 76.08828595792875, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20245992736816407, "validation_score": 9286.415814272576}



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
