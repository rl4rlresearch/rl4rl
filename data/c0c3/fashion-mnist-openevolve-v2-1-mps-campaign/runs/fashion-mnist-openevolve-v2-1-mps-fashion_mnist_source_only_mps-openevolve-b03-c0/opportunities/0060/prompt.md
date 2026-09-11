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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 65.27861908287741, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20046691131591796, "validation_score": 9319.416504607738}
prior_hypothesis: Dividing evaluation logits by 0.77 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20059863.

## Recent verification evidence

RECENT RESULT
hypothesis: An evaluation temperature of 0.78 will preserve all 9,319 argmax-correct predictions while reducing validation cross-entropy below 0.20153608.
change: Reduce the evaluation-only logit temperature from 0.80 to 0.78 without changing training or learned parameters.
mechanism: Evaluation-only temperature calibration
evidence_used: Temperatures from 0.95 through 0.80 preserved exactly 9,319 correct predictions while progressively lowering cross-entropy; 0.78 is an untested intermediate between the verified 0.80 design and the timed-out 0.77 attempt.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An evaluation temperature of 0.79 will preserve all 9,319 argmax-correct predictions while reducing validation cross-entropy below 0.20153608.
change: Reduce the evaluation-only logit temperature from 0.80 to 0.79, leaving training and learned parameters unchanged.
mechanism: Evaluation-only temperature calibration
evidence_used: Temperatures from 0.95 through 0.80 preserved exactly 9,319 correct predictions while progressively lowering cross-entropy; 0.79 is the smallest untested step beyond the verified 0.80 design, while lower-temperature attempts only timed out and provided no contrary calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging predictions for the original image and one-pixel left/right translations will exceed 9,319 correct predictions by reducing sensitivity to horizontal placement.
change: During evaluation only, average logits from the original image and two replicate-padded horizontal translations before applying the verified 0.80 temperature.
mechanism: Three-view horizontal translation ensemble
evidence_used: Temperature scaling improved cross-entropy but cannot change argmax accuracy; training-time translation augmentation timed out without negative accuracy evidence, so a lightweight evaluation-only ensemble tests positional robustness without altering training.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.85044179204851, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.20633014755249024, "validation_score": 9300.414480232475}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.04 to 0 over training will exceed 9,319 correct predictions by retaining early regularization while allowing late updates to strengthen hard-label decision margins.
change: Replace fixed label smoothing with a linear schedule based on examples-training progress; preserve the verified evaluation temperature and all other training behavior.
mechanism: Linearly annealed label smoothing
evidence_used: Evaluation sharpening from temperature 0.95 through 0.80 repeatedly reduced cross-entropy without changing argmax predictions, indicating underconfident logits; fixed 0.04 smoothing suppresses confidence through the final update, motivating a late transition toward hard-label training.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.04 to 0.02 will exceed 9,319 correct predictions by preserving regularization while allowing stronger class margins during the limited two-pass exposure.
change: Use fixed 0.02 label smoothing; retain the verified 0.80 evaluation temperature and all other training behavior.
mechanism: Reduced fixed label smoothing
evidence_used: Evaluation sharpening from 0.95 through 0.80 consistently lowered cross-entropy without changing 9,319 argmax predictions, indicating underconfident logits; the annealed-smoothing attempt timed out, so this isolates a lower smoothing strength without adding scheduling overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Starting AdamW at the schedule’s 0.2× warmup rate instead of taking one anomalous full-rate update will exceed 9,319 correct predictions by stabilizing early feature learning.
change: Initialize the optimizer at 4.0e-4; retain the existing warmup, cosine schedule, loss, architecture, and verified 0.80 evaluation temperature.
mechanism: Warmup-consistent optimizer initialization
evidence_used: Temperature calibration repeatedly preserved exactly 9,319 correct predictions, so further gains require changing training. The current optimizer takes its first step at 2.0e-3 before the scheduler reduces it to approximately 4.0e-4, contradicting the intended warmup.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Randomly flipping each training image horizontally with probability 0.5 will exceed 9,319 correct predictions by teaching left-right invariance without the harmful boundary shifts observed with translation ensembling.
change: Apply independent per-image horizontal flips in `prepare_training_batch`; preserve the architecture, optimizer, loss, schedule, and verified 0.80 evaluation temperature.
mechanism: Stochastic horizontal-reflection augmentation
evidence_used: Temperature calibration repeatedly left all 9,319 argmax predictions unchanged, so accuracy gains require altered training; evaluation-time horizontal translations reduced accuracy to 9,300, motivating a semantic reflection augmentation that avoids translated padding artifacts.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding input-dependent channel reweighting after the final convolutional stage will exceed 9,319 correct predictions while retaining the verified training recipe and negligible computational overhead.
change: Add a three-parameter channel-attention module initialized as an identity mapping, keeping the model below the 250,000-parameter ceiling.
mechanism: Identity-initialized efficient channel attention
evidence_used: Temperature scaling preserved exactly 9,319 predictions and translation ensembling reduced accuracy, so further accuracy gains require a discriminative model change; lightweight channel attention adds adaptive capacity without costly augmentation or materially longer training.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.04 to 0.03 will exceed 9,319 correct predictions by strengthening decision margins while retaining most of the baseline’s regularization.
change: Use fixed 0.03 label smoothing while preserving the verified architecture, optimizer, schedule, and 0.80 evaluation temperature.
mechanism: Moderately reduced label smoothing
evidence_used: Evaluation temperatures from 0.95 through 0.80 preserved exactly 9,319 predictions while progressively lowering cross-entropy, indicating underconfident logits; the 0.02 smoothing attempt timed out without contrary accuracy evidence, motivating a more conservative intermediate reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging calibrated probabilities for each validation image and its horizontal reflection will exceed 9,319 correct predictions by adding reflection invariance without the padding artifacts that caused the translation ensemble to lose 19 correct predictions.
change: During evaluation only, combine the original and horizontally reflected predictions in probability space while retaining the verified 0.80 temperature; training remains unchanged.
mechanism: Two-view horizontal-reflection probability ensemble
evidence_used: The three-view translation ensemble fell from 9,319 to 9,300 correct, implicating shifted-boundary artifacts, while the reflection-training attempt timed out without testing its accuracy; an evaluation-only reflection ensemble isolates the proposed semantic invariance without altering training time or learned parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.775 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20153608.
change: Reduce the evaluation-only temperature from 0.80 to 0.775 without changing training, runtime, or learned parameters.
mechanism: Evaluation-only temperature calibration
evidence_used: Temperatures from 0.95 through 0.80 preserved exactly 9,319 correct predictions while progressively lowering cross-entropy; lower-temperature attempts timed out without contrary calibration evidence, so 0.775 tests the midpoint of the most promising unverified interval.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 75.71691079088487, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20059862747192383, "validation_score": 9319.416458913545}

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.77 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.20059863.
change: Reduce the evaluation-only temperature from 0.775 to 0.77 without changing training, parameters, or argmax predictions.
mechanism: Evaluation-only temperature calibration
evidence_used: Sharpening from 0.95 through 0.775 consistently preserved 9,319 correct predictions while lowering cross-entropy; the prior 0.77 verification timed out and supplied no contrary calibration evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 65.27861908287741, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20046691131591796, "validation_score": 9319.416504607738}



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
