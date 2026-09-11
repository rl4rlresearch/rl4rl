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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 48.64446824998595, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820373535156, "validation_score": 9348.42099299992}
prior_hypothesis: Multiplying by the nearest float32 reciprocal of the best temperature will preserve all 9,348 predictions while reducing validation cross-entropy below 0.18766820449829102.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.40722645795904, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656127929689, "validation_score": 9348.420979403752}
prior_hypothesis: A 0.25-ULP interpolation will preserve all 9,348 predictions while lowering validation cross-entropy below 0.18770656051635742 by crossing more float32 rounding thresholds while remaining inside the quadratic estimate’s improving interval.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.354036750039086, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820755004882, "validation_score": 9348.420992998568}
prior_hypothesis: Averaging each native/reflected pair before applying the 1.174-order crop power mean will exceed 9,348 correct predictions by suppressing orientation-specific confidence outliers while retaining the beneficial emphasis on crop-specific evidence.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.98722320818342, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820449829102, "validation_score": 9348.420992999649}
prior_hypothesis: Interpolating 3/22 of the way from the best verified temperature toward its sharper neighbor will preserve all 9,348 predictions while reducing cross-entropy below 0.18766820449829102.

## Recent verification evidence

RECENT RESULT
hypothesis: Realizing the fitted 3/22-step sharper temperature blend in float64 will preserve all 9,348 predictions while reducing validation cross-entropy below 0.18766820449829102.
change: Replace the single float32 temperature division with float64 interpolation between the best temperature and its adjacent sharper float32 temperature.
mechanism: Float64 sub-ULP temperature interpolation
evidence_used: The two adjacent temperatures were worse on opposite sides of the current optimum, placing the quadratic minimum 3/22 toward the sharper neighbor; the prior float32 interpolation tied the current result, consistent with its sub-ULP adjustment being lost to float32 rounding.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing arithmetic native/reflected probability averaging with normalized geometric averaging will exceed 9,348 correct predictions by suppressing class evidence that is strong in only one orientation while preserving the verified crop weighting and power-mean fusion.
change: Average each crop’s native and reflected logits before softmax, then retain the existing 1.174-order crop fusion and calibrated temperature.
mechanism: Log-domain orientation consensus
evidence_used: Hierarchical orientation-before-crop consensus retained 9,348 correct while improving cross-entropy over flat fusion, indicating that resolving orientation disagreement before crop aggregation is beneficial; architecture changes, diagonal crops, and altered training translations all reduced correctness.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.887719250051305, "validation_accuracy": 0.9345, "validation_correct": 9345, "validation_cross_entropy": 0.18768878173828124, "validation_score": 9345.420985705758}

RECENT RESULT
hypothesis: Mixing arithmetic native/reflected ensemble cross-entropy into training will exceed 9,348 correct predictions by aligning optimization with the verified evaluation consensus.
change: Retain the architecture and training procedure, but allocate 25% of the loss to the label-smoothed cross-entropy of the averaged native/reflected probabilities.
mechanism: Arithmetic orientation-ensemble training objective
evidence_used: Arithmetic orientation consensus achieved 9,348 correct, while geometric log-domain consensus fell to 9,345; the current per-orientation loss optimizes log probabilities separately rather than the arithmetic consensus used by the best evaluation rule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.741251749917865, "validation_accuracy": 0.9335, "validation_correct": 9335, "validation_cross_entropy": 0.18703881072998046, "validation_score": 9335.421216219285}

RECENT RESULT
hypothesis: Restoring the verified 1.174-order hierarchical fusion will preserve 9,348 correct predictions while reducing validation cross-entropy from 0.18770656127929689 to approximately 0.18766820449829102.
change: Average native/reflected probabilities within each crop, combine crop consensuses with the verified weighted power mean, and use the best verified temperature.
mechanism: Hierarchical orientation consensus with crop power-mean fusion
evidence_used: Reference Design 1 achieved the highest available validation score at the same 9,348 correct predictions, while architecture, augmentation, geometric-consensus, and ensemble-loss changes all reduced correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Multiplying by the nearest float32 reciprocal of the best temperature will preserve all 9,348 predictions while reducing validation cross-entropy below 0.18766820449829102.
change: Replace temperature division with an equivalent float32 reciprocal multiplication whose rounding realizes a small sharpening close to the fitted 3/22-ULP optimum.
mechanism: Reciprocal-quantized sub-ULP temperature sharpening
evidence_used: Adjacent temperatures placed the fitted optimum 3/22 ULP toward the sharper value, but float32 logit interpolation tied the current score; reciprocal multiplication provides a distinct float32 rounding path near that optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 48.64446824998595, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820373535156, "validation_score": 9348.42099299992}

RECENT RESULT
hypothesis: Parallel 3×3, 5×5, and 7×7 refinement branches will exceed 9,348 correct predictions by capturing local textures and whole-garment structure without discarding the 7×7 spatial layout.
change: Replace the single-scale depthwise refinement with three nonlinear scale-specific branches concatenated before channel mixing, reduce the dense bottleneck from 56 to 52 units to remain below 250,000 parameters, and retain the best verified reciprocal temperature scaling.
mechanism: Nonlinear multi-scale spatial refinement
evidence_used: Learned strided aggregation fell to 9,334 and global pooling to 9,085, showing that spatial compression is harmful; this alternative reallocates capacity to multi-scale feature extraction while preserving the successful full 7×7 classifier layout.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247917, "training_seconds": 73.92683366592973, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.18695523109436035, "validation_score": 9321.42124587929}

RECENT RESULT
hypothesis: Multiplying by the nearest float32 reciprocal will preserve all 9,348 correct predictions while reducing validation cross-entropy below the current 0.18766820755004882.
change: Replace temperature division with the best verified float32 reciprocal multiplication.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 achieved the highest available score, preserving 9,348 correct while lowering cross-entropy to 0.18766820373535156.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing flat view averaging with the best verified hierarchical 1.174-order fusion and reciprocal temperature scaling will preserve 9,348 correct predictions while reducing validation cross-entropy from 0.18770656127929689 to approximately 0.18766820373535156.
change: Average native/reflected probabilities within each crop, fuse the five crop consensuses using the verified weighted power mean, renormalize, and apply the best verified float32 reciprocal temperature multiplier.
mechanism: Hierarchical orientation consensus with reciprocal-calibrated crop power-mean fusion
evidence_used: Reference Design 1 achieved the highest available validation score, retaining 9,348 correct predictions while lowering cross-entropy to 0.18766820373535156; flat fusion in the current design has the same correctness but worse cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying the verified 1.174-order power mean within native/reflected pairs will exceed 9,348 correct predictions by emphasizing reliable orientation-specific evidence rather than suppressing it.
change: Replace arithmetic orientation averaging with a 1.174-order generalized mean while retaining the verified crop fusion, weights, training, and reciprocal calibration.
mechanism: Matched-order orientation power fusion
evidence_used: Geometric orientation consensus reduced correctness to 9,345 versus 9,348 for arithmetic consensus, while super-arithmetic 1.174-order crop fusion produced the best verified design; this motivates testing the same evidence-emphasizing direction across orientations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.12118783290498, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18768614044189452, "validation_score": 9348.420986641988}

RECENT RESULT
hypothesis: Using the verified float32 reciprocal multiplier will preserve all 9,348 correct predictions while reducing validation cross-entropy from 0.18766820449829102 to approximately 0.18766820373535156.
change: Replace sub-ULP interpolation between temperature divisions with the single reciprocal multiplication from the highest-scoring verified design.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 achieved the best available validation score, retaining 9,348 correct predictions and improving cross-entropy to 0.18766820373535156; the current interpolation has identical correctness but slightly higher cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing temperature division with the verified float32 reciprocal multiplier will preserve 9,348 correct predictions and reduce validation cross-entropy to approximately 0.18766820373535156.
change: Apply the calibration using the exact reciprocal-multiplication path from the highest-scoring qualified design.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 used identical training and fusion, preserved 9,348 correct predictions, and achieved the best available validation score by lowering cross-entropy versus the current division.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact highest-scoring verified evaluation path will preserve 9,348 correct predictions and reduce validation cross-entropy to approximately 0.18766820373535156.
change: Average native/reflected probabilities within each crop, combine crop consensuses using the verified weighted 1.174-order power mean, renormalize, and apply the verified float32 reciprocal temperature multiplier.
mechanism: Hierarchical crop power-mean fusion with reciprocal temperature calibration
evidence_used: Reference Design 1 achieved the best available validation_score, retaining 9,348 correct predictions while lowering cross-entropy from the current 0.18770656127929689 to 0.18766820373535156.
result: the implementation could not be verified



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
