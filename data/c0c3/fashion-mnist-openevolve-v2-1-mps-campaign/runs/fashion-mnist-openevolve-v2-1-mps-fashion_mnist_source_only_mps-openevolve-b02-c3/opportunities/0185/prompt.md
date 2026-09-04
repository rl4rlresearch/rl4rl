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

RECENT RESULT
hypothesis: Using orientation power 1.087—halfway between arithmetic averaging and the accuracy-preserving 1.174 endpoint—will exceed 9,348 correct predictions if beneficial decision-boundary crossings occur before offsetting regressions.
change: Replace arithmetic native/reflected probability averaging with a normalized 1.087-order power mean while retaining the verified crop fusion, weights, and reciprocal calibration.
mechanism: Half-strength super-arithmetic orientation fusion
evidence_used: Arithmetic orientation fusion and 1.174-order orientation fusion both achieved 9,348 correct, while the latter only modestly worsened cross-entropy; testing their midpoint is an informative probe for differently ordered discrete prediction changes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 43.98414979223162, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18767453384399413, "validation_score": 9348.420990756096}

RECENT RESULT
hypothesis: Replacing interpolated temperature division with the verified float32 reciprocal multiplier will preserve 9,348 correct predictions and lower cross-entropy from 0.18766820449829102 to approximately 0.18766820373535156.
change: Keep the verified hierarchical 1.174-order fusion unchanged and apply calibration through the exact reciprocal-multiplication path from Reference Design 2.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 used the same architecture, training, and fusion, achieving the best available validation_score with 9,348 correct and cross-entropy 0.18766820373535156.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing temperature division with the verified float32 reciprocal multiplier will preserve all 9,348 correct predictions while reducing validation cross-entropy from 0.18766820755004882 to approximately 0.18766820373535156.
change: Keep training and hierarchical crop fusion unchanged, but apply final calibration through the exact multiplication path of the highest-scoring qualified design.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 used the same architecture, training, and fusion, retained 9,348 correct predictions, and achieved the best available validation_score with cross-entropy 0.18766820373535156.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding content-dependent all-to-all interactions across the preserved 7×7 feature grid will exceed 9,348 correct predictions by modeling relationships between distant garment regions that fixed local refinement cannot capture.
change: Retain the successful local refinement and full-grid classifier, insert a residual four-head attention mixer with learned spatial positions, and reduce the dense bottleneck from 56 to 49 units to keep 247,820 learned parameters.
mechanism: Position-aware nonlocal spatial attention
evidence_used: Global pooling fell to 9,085 because it discarded spatial layout, while parallel local refinements reached only 9,321; this motivates a mechanism that preserves every spatial position but adds genuinely nonlocal, image-dependent aggregation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247820, "training_seconds": 66.47797862510197, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.18509335021972656, "validation_score": 9325.421907691834}

RECENT RESULT
hypothesis: Realizing the fitted 3/22-ULP sharpening in float64 will preserve all 9,348 predictions while reducing cross-entropy below 0.18766820373535156.
change: Replace the quantized float32 reciprocal multiplier with double-precision interpolation between the best temperature and its sharper neighbor.
mechanism: Float64 sub-ULP temperature calibration
evidence_used: Reference Design 3 located the optimum 3/22 ULP toward the sharper temperature, while the current reciprocal multiplication improved cross-entropy by approximating that otherwise-unrepresentable scale.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying the exact verified float32 reciprocal multiplier will preserve all 9,348 correct predictions while reducing validation cross-entropy from 0.18766820449829102 to approximately 0.18766820373535156.
change: Replace interpolated temperature division with the single reciprocal-multiplication expression from the highest-scoring qualified design.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 used identical training and fusion, retained 9,348 correct predictions, and achieved the best available validation_score with cross-entropy 0.18766820373535156.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying the verified float32 reciprocal multiplier will preserve all 9,348 correct predictions while reducing validation cross-entropy from 0.18766820755004882 to approximately 0.18766820373535156.
change: Replace final temperature division with the exact reciprocal-multiplication expression from the highest-scoring qualified design.
mechanism: Reciprocal-quantized temperature calibration
evidence_used: Reference Design 2 uses otherwise identical training and fusion, preserves 9,348 correct predictions, and achieves the best available validation_score with cross-entropy 0.18766820373535156.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact qualified Reference Design 1 evaluation path will preserve 9,348 correct predictions and reduce validation cross-entropy to 0.18766820373535156.
change: Average native/reflected probabilities per crop, fuse crop consensuses with the verified weighted 1.174-order power mean, renormalize, and apply the verified reciprocal temperature multiplier.
mechanism: Hierarchical crop power-mean fusion with reciprocal temperature calibration
evidence_used: Reference Design 1 is the highest-scoring qualified implementation, improving validation_score from 9348.420979403752 to 9348.42099299992 with identical training and parameter count; prior unverifiable attempts do not contradict its verified result.
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
