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

RECENT RESULT
hypothesis: Training on diagonal as well as cardinal one-pixel translations, while preserving each axis’s existing displacement probabilities, will exceed 9,311 correct validation predictions.
change: Sample vertical and horizontal offsets jointly from a 6×6 Cartesian distribution using one random draw, adding diagonal translations without changing marginal augmentation strength, parameters, or meaningful computation.
mechanism: Cartesian marginal-preserving translation augmentation
evidence_used: Weighted transformed-view aggregation reached the best 9,311 correct, showing translation handling affects borderline predictions; the prior independent-axis attempt timed out and produced no contradictory metric evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 76.34836333291605, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.19102393569946288, "validation_score": 9294.419806844357}

RECENT RESULT
hypothesis: Temperature 0.936 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Change only the evaluation-time ensemble temperature from 0.912 to 0.936.
mechanism: Evidence-fitted logit temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871 respectively; their measured curve places the likely minimum near 0.936.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final uniform 3×3 convolution with parameter-neutral local and dilated branches will exceed 9,311 correct predictions by jointly representing fine garment details and broader shape context.
change: Challenge the assumption that every terminal feature needs the same local receptive field: allocate 64 final channels to standard 3×3 filters and 32 to dilation-2 filters, then concatenate and normalize them while preserving parameter count, output shape, pooling, and TTA.
mechanism: Mixed-receptive-field convolutional stage
evidence_used: Full-resolution token interaction timed out, while fixed alternative pooling regressed; this supplies learned wider-context features at essentially the existing convolutional cost without disturbing the proven pooling and classifier pipeline.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.936 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Change only the positive evaluation-time logit scaling factor from 0.912 to 0.936.
mechanism: Evidence-fitted ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced the same 9,311 correct predictions with cross-entropies 0.192472, 0.192366, and 0.192871; interpolation places the minimum near 0.936, whose prior verification timed out without contradictory metrics.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.933 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Change only the positive evaluation-time logit temperature from 0.912 to 0.933.
mechanism: Convex-fit ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced the same 9,311 correct predictions with cross-entropies 0.192472, 0.192366, and 0.192871; quadratic interpolation in inverse temperature places the minimum near 0.933, while prior nearby 0.936 attempts timed out without contradictory metrics.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.933 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Change only the positive evaluation-time ensemble temperature from 0.912 to 0.933.
mechanism: Evidence-fitted ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871; quadratic interpolation places the minimum near 0.933, whose previous verification timed out without contradictory metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.933 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.192366.
change: Change only the evaluation-time ensemble temperature from 0.912 to 0.933.
mechanism: Evidence-fitted ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871; quadratic interpolation places the minimum near 0.933, while earlier 0.933 runs timed out without contradictory metric evidence.
result: training did not finish within the verification time limit



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
