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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 75.6753031250555, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1922903751373291, "validation_score": 9311.41936092954}
prior_hypothesis: Temperature 0.930 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.1923047371.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Temperature 0.933 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.192366.
change: Change only the evaluation-time ensemble temperature from 0.912 to 0.933.
mechanism: Evidence-fitted ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871; quadratic interpolation places the minimum near 0.933, while prior 0.933 verifications timed out without contradictory metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.924 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.
change: Raise only the evaluation-time ensemble temperature from 0.912 to 0.924.
mechanism: Bracketed logit-ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871; their fitted minimum lies near 0.933–0.936, so 0.924 is a conservative untested step toward that minimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 75.10231937514618, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19230473709106446, "validation_score": 9311.419355878112}

RECENT RESULT
hypothesis: Temperature 0.934 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.1923047371.
change: Raise only the evaluation-time ensemble temperature from 0.924 to 0.934.
mechanism: Local quadratic ensemble-temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 0.924 successively reduced cross-entropy while preserving 9,311 correct predictions; their local curvature places the minimum near 0.934, and nearby 0.933–0.936 attempts timed out without contradictory metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.930 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.1923047371.
change: Raise only the evaluation-time ensemble temperature from 0.924 to 0.930.
mechanism: Conservative local temperature calibration
evidence_used: Temperatures 0.900, 0.912, and 0.924 successively improved cross-entropy while preserving exactly 9,311 correct predictions; 0.930 is a conservative step toward the fitted minimum near 0.934, whose attempted verifications timed out without contradictory metric evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 75.6753031250555, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1922903751373291, "validation_score": 9311.41936092954}

RECENT RESULT
hypothesis: Temperature 0.934 will preserve exactly 9,311 argmax predictions while reducing validation cross-entropy below 0.1922903751.
change: Raise only the positive evaluation-time ensemble temperature from 0.930 to 0.934.
mechanism: One-dimensional ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, 0.924, and 0.930 progressively reduced cross-entropy without changing the 9,311 correct predictions; the measured curve places its minimum near 0.934, while prior nearby attempts timed out without contradictory metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Content-dependent attention among the nine terminal spatial tokens will exceed 9,311 correct predictions by resolving shape-part relationships that the fixed flattened bottleneck cannot model.
change: Reallocate classifier capacity from the 147-unit flattening layer to four-head self-attention over the 3×3 feature grid, followed by a smaller positional classifier; retain the established convolutional trunk, training procedure, and TTA.
mechanism: Terminal spatial-relation attention
evidence_used: Full-resolution token interaction timed out, suggesting interaction cost was load-bearing rather than disproving relational modeling; applying attention only after reduction to nine tokens tests the mechanism cheaply. Translation-aware aggregation achieved the best 9,311 correct, further indicating that spatial organization affects borderline predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.932 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922903751.
change: Raise only the evaluation-time ensemble temperature from 0.930 to 0.932.
mechanism: Bracketed ensemble temperature calibration
evidence_used: Temperatures 0.900, 0.912, 0.924, and 0.930 progressively reduced cross-entropy with unchanged argmax predictions; 0.932 is a conservative step toward the fitted minimum near 0.933–0.934, whose attempted verifications timed out without contradictory metric evidence.
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
