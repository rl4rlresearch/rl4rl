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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.354036750039086, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820755004882, "validation_score": 9348.420992998568}
prior_hypothesis: Averaging each native/reflected pair before applying the 1.174-order crop power mean will exceed 9,348 correct predictions by suppressing orientation-specific confidence outliers while retaining the beneficial emphasis on crop-specific evidence.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.40722645795904, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656127929689, "validation_score": 9348.420979403752}
prior_hypothesis: A 0.25-ULP interpolation will preserve all 9,348 predictions while lowering validation cross-entropy below 0.18770656051635742 by crossing more float32 rounding thresholds while remaining inside the quadratic estimate’s improving interval.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 48.64446824998595, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820373535156, "validation_score": 9348.42099299992}
prior_hypothesis: Multiplying by the nearest float32 reciprocal of the best temperature will preserve all 9,348 predictions while reducing validation cross-entropy below 0.18766820449829102.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.98722320818342, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820449829102, "validation_score": 9348.420992999649}
prior_hypothesis: Interpolating 3/22 of the way from the best verified temperature toward its sharper neighbor will preserve all 9,348 predictions while reducing cross-entropy below 0.18766820449829102.

## Recent verification evidence

RECENT RESULT
hypothesis: Decreasing the best hierarchical-fusion temperature by one additional float32 ULP will preserve all 9,348 argmax predictions while reducing validation cross-entropy below 0.1876682071685791.
change: Restore the verified native/reflected consensus followed by a 1.174-order crop power mean, then use the preceding float32 temperature, 0.753169953823089599609375.
mechanism: Hierarchical orientation consensus with continued one-ULP sharpening
evidence_used: Hierarchical 1.174-order fusion improved cross-entropy from 0.18770656051635742 to 0.18766820755004882 at unchanged correctness, and its first one-ULP temperature decrease further improved it to 0.1876682071685791 without changing any predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.315159915946424, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820449829102, "validation_score": 9348.420992999649}

RECENT RESULT
hypothesis: Replacing the unstructured 7×7 flattened bottleneck with a shared strided convolution and a 4×4 spatial head will exceed 9,348 correct predictions by learning higher-level local motifs while retaining the layout evidence that global pooling lost.
change: Reallocate the dense head’s parameters into a 64→96 learned spatial downsampling stage and a 78-unit 4×4 classifier, totaling 249,803 learned parameters; retain the best verified hierarchical orientation/crop fusion and temperature.
mechanism: Learned strided motif aggregation with coarse spatial layout
evidence_used: Global pooling fell to 9,085 correct, showing that spatial layout is load-bearing, while repeated TTA calibration changes plateaued at 9,348. The old design assumes an unstructured dense layer should learn all final spatial composition; this patch instead imposes translation-shared motif formation without discarding position.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249803, "training_seconds": 66.4323424580507, "validation_accuracy": 0.9334, "validation_correct": 9334, "validation_cross_entropy": 0.1845059196472168, "validation_score": 9334.422116928}

RECENT RESULT
hypothesis: Decreasing the positive evaluation temperature by one additional float32 ULP will preserve all 9,348 argmax predictions while reducing validation cross-entropy below 0.18766820449829102.
change: Retain the verified training procedure and hierarchical 1.174-order fusion, changing only the evaluation temperature from 0.753169953823089599609375 to the preceding float32 value, 0.75316989421844482421875.
mechanism: Continued one-ULP hierarchical-fusion sharpening
evidence_used: Two consecutive one-ULP temperature decreases preserved 9,348 correct predictions and improved cross-entropy from 0.18766820755004882 to 0.1876682071685791 and then 0.18766820449829102, motivating one more local sharpening probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.598051792010665, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820602416992, "validation_score": 9348.420992999108}

RECENT RESULT
hypothesis: Interpolating 3/22 of the way from the best verified temperature toward its sharper neighbor will preserve all 9,348 predictions while reducing cross-entropy below 0.18766820449829102.
change: Restore the best verified temperature and blend its logits with those from the adjacent lower float32 temperature using the three-point quadratic optimum.
mechanism: Sub-ULP quadratic temperature interpolation
evidence_used: The best temperature produced 0.18766820449829102 cross-entropy; its immediately higher and lower neighbors were worse by 2.67028808e-9 and 1.52587890e-9 respectively, placing the fitted minimum 3/22 of one ULP toward the lower neighbor.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.98722320818342, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820449829102, "validation_score": 9348.420992999649}

RECENT RESULT
hypothesis: Adding weak, factorized diagonal crop views will exceed 9,348 correct predictions by recovering images requiring simultaneous horizontal and vertical alignment while preserving the dominant verified axial consensus.
change: Extend evaluation TTA from five axial crops to the full 3×3 translation grid, assign each diagonal the separable product of its axial weights divided by the center weight, and restore the best verified temperature.
mechanism: Separable full-grid translation marginalization
evidence_used: Hierarchical orientation-then-crop fusion reached 9,348 correct, whereas changing the learned spatial architecture reduced correctness; this motivates preserving the model and fusion while filling the remaining diagonal gap in its translation ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.04738162504509, "validation_accuracy": 0.9335, "validation_correct": 9335, "validation_cross_entropy": 0.18776871452331542, "validation_score": 9335.420957374854}

RECENT RESULT
hypothesis: Training on axial translations with probabilities proportional to the best verified crop weights will exceed 9,348 correct predictions by emphasizing useful vertical invariance while reducing weaker horizontal perturbations.
change: Restore hierarchical orientation-then-crop power-mean evaluation at the best verified temperature, and align training’s axial translation distribution with its crop weights.
mechanism: Evaluation-aligned anisotropic translation augmentation
evidence_used: Hierarchical fusion preserved 9,348 correct while improving cross-entropy to 0.18766820449829102; its optimized weights favor vertical over horizontal crops, while adding diagonal views reduced correctness to 9,335.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 53.619205333990976, "validation_accuracy": 0.9337, "validation_correct": 9337, "validation_cross_entropy": 0.1892971694946289, "validation_score": 9337.420416370967}

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
