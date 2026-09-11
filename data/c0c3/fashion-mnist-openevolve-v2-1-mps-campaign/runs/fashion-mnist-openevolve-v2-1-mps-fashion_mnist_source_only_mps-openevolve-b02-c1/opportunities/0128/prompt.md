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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 78.30724849994294, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19236609153747558, "validation_score": 9311.419334299717}
prior_hypothesis: Replacing probability averaging with weighted logit averaging will suppress transformation-specific confidence outliers and achieve at least 9,311 correct validation predictions without increasing parameters or runtime.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the tail-average blend from 0.5 to 0.75 will achieve at least 9,312 correct predictions by moving farther toward the better-generalizing averaged parameters.
change: Strengthen only the final interpolation toward the existing tail parameter average, without changing computation, parameters, BatchNorm buffers, or TTA.
mechanism: Stronger tail-checkpoint interpolation
evidence_used: A 0.5 tail-average blend produced 9,311 correct, while reducing it to 0.25 regressed to 9,307; this directional result motivates testing a stronger blend.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using unit temperature will preserve the weighted-logit ensemble’s 9,311 correct predictions while reducing validation cross-entropy below 0.192366.
change: Remove the inherited 0.912 sharpening factor from evaluation logits without changing training, parameters, model-forward count, or class argmaxes.
mechanism: Logit-ensemble temperature recalibration
evidence_used: Weighted logit averaging achieved the current best 9,311 correct; its 0.912 temperature came from probability aggregation, while the prior unit-temperature verification timed out and therefore did not test this computation-neutral recalibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 75.26128700003028, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.192871427154541, "validation_score": 9311.419156657304}

RECENT RESULT
hypothesis: Increasing the tail-average blend from 0.5 to 0.75 will achieve at least 9,312 correct predictions by moving farther toward the better-generalizing averaged parameters.
change: Strengthen only the final interpolation toward the existing tail parameter average, without changing computation, parameters, BatchNorm buffers, or TTA.
mechanism: Stronger tail-checkpoint interpolation
evidence_used: A 0.5 blend achieved 9,311 correct while 0.25 regressed to 9,307; the previous 0.75 verification timed out, so this directionally motivated setting remains untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 82.14168354100548, "validation_accuracy": 0.9306, "validation_correct": 9306, "validation_cross_entropy": 0.19295940551757812, "validation_score": 9306.419125745342}

RECENT RESULT
hypothesis: Independently sampling horizontal and vertical one-pixel shifts while preserving the current per-axis displacement probabilities will exceed 9,311 correct predictions by exposing the model to diagonal translations without increasing augmentation strength or computation materially.
change: Replace mutually exclusive cardinal translations with independent per-axis sampling; each axis remains centered with probability 2/3 and shifted each direction with probability 1/6.
mechanism: Marginal-preserving two-axis translation augmentation
evidence_used: Weighted transformed-view aggregation improved the best result to 9,311 correct, showing translation handling affects borderline decisions, while recent architectural additions repeatedly timed out; this tests broader spatial invariance through a computation-neutral training-only change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the single linear class readout with three input-gated experts per class will exceed 9,311 correct predictions by learning image-dependent decision surfaces while retaining the proven convolutional representation, TTA, and runtime profile at 249,585 parameters.
change: Replace the 147-unit single-head classifier with a 139-unit normalized hidden layer whose class logits are independently blended from three learned experts using per-image, per-class gates.
mechanism: Class-conditional mixture-of-experts readout
evidence_used: The normalized dense head and weighted-logit TTA reached the best 9,311 correct, but fixed alternative pooling regressed and full-resolution token interaction timed out. This challenges the remaining load-bearing assumption that one fixed linear readout is sufficient, using conditional computation with negligible additional runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharpening weighted-logit TTA from temperature 0.912 to 0.900 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.192366.
change: Lower only the evaluation temperature; training, parameters, runtime, and class argmaxes remain unchanged.
mechanism: Argmax-preserving logit-ensemble recalibration
evidence_used: Temperature 0.912 achieved the same 9,311 correct predictions as temperature 1.0 but lowered cross-entropy from 0.192871 to 0.192366, motivating a small additional sharpening step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics over the same tail checkpoints as the parameters will exceed 9,311 correct predictions by eliminating parameter/statistic misalignment.
change: Retain the verified 0.5 tail-parameter blend while also averaging and blending all floating-point model buffers, including BatchNorm running means and variances.
mechanism: Tail-aligned BatchNorm buffer averaging
evidence_used: The 0.5 tail blend achieved 9,311 correct versus 9,307 at 0.25 and 9,306 at 0.75, showing the existing midpoint is useful; prior buffer-alignment attempts timed out and therefore left this computation-light correction untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing each centered view’s weight from 2 to 3 will exceed 9,311 correct predictions by reducing the influence of padding-affected shifted views while preserving transformation ensembling.
change: Give the original and horizontally flipped centered views weight 3, and renormalize the ensemble’s temperature-scaled denominator from 12 to 14.
mechanism: Center-prior weighted logit TTA
evidence_used: Weighted-logit TTA improved the best result from 9,310 to 9,311 correct, demonstrating that aggregation changes borderline decisions; center reweighting tests view reliability without altering training, parameters, or forward count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.900 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.192366.
change: Lower only the evaluation temperature from 0.912 to 0.900.
mechanism: Argmax-preserving logit-ensemble recalibration
evidence_used: Temperature 0.912 matched unit temperature’s 9,311 correct predictions while lowering cross-entropy from 0.192871 to 0.192366; the prior 0.900 attempt timed out and therefore did not test the predicted further sharpening benefit.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.900 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Lower only the evaluation-time ensemble temperature from 0.912 to 0.900.
mechanism: Conservative logit-ensemble temperature sharpening
evidence_used: Temperature 0.912 matched unit temperature’s 9,311 correct predictions while improving cross-entropy from 0.192871 to 0.192366; prior 0.900 attempts timed out and therefore provide no contradictory metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.900 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Lower only the evaluation-time ensemble temperature from 0.912 to 0.900.
mechanism: Conservative logit-ensemble temperature sharpening
evidence_used: Temperature 0.912 preserved unit temperature’s 9,311 correct predictions while improving cross-entropy from 0.192871 to 0.192366; prior 0.900 attempts timed out and therefore provide no contradictory metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.900 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Lower only the evaluation-time ensemble temperature from 0.912 to 0.900.
mechanism: Conservative logit-ensemble temperature sharpening
evidence_used: Temperature 0.912 preserved unit temperature’s 9,311 correct predictions while improving cross-entropy from 0.192871 to 0.192366; prior 0.900 attempts timed out and produced no contradictory metric evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 70.49403779092245, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1924720932006836, "validation_score": 9311.419297024098}



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
