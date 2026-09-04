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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.3754118331708, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656051635742, "validation_score": 9348.420979404023}
prior_hypothesis: Raising the positive temperature by one float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656127929689.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.28536408301443, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18768770637512208, "validation_score": 9348.420986086929}
prior_hypothesis: A 1.125-order power mean will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18770656051635742 by modestly favoring crop-specific high-confidence evidence.

## Recent verification evidence

RECENT RESULT
hypothesis: Sampling training translations in the 9:4:4:2:2 ratio implied by the best TTA weights will exceed 9,348 correct predictions by improving invariance to the more valuable vertical shifts while reducing exposure to less reliable horizontal shifts.
change: Replace the current 3:1:1:1:1 translation distribution with a 21-way discrete approximation of the verified evaluation weights, leaving architecture, runtime, and example accounting unchanged.
mechanism: Evaluation-aligned anisotropic translation sampling
evidence_used: Axial translation TTA raised correctness from 9,344 to 9,348, and its best weights scale almost exactly to 9:4:4:2:2; diagonal translations subsequently reduced correctness to 9,335, showing that matching the useful translation directions is consequential.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 52.56239566579461, "validation_accuracy": 0.9338, "validation_correct": 9338, "validation_cross_entropy": 0.18753530426025392, "validation_score": 9338.421040114097}

RECENT RESULT
hypothesis: Sampling translations in a 9:2:2:4:4 center/up/down/left/right ratio will exceed 9,348 correct predictions by reversing the vertical oversampling that previously reduced correctness.
change: Keep the center-translation probability unchanged while transferring training exposure from vertical shifts to horizontal shifts; architecture, evaluation ensemble, and calibration remain unchanged.
mechanism: Reverse-anisotropic translation sampling
evidence_used: Changing the original scaled 9:3:3:3:3 distribution to 9:4:4:2:2 reduced validation correctness from 9,348 to 9,338. Because the center probability was identical, this supplies directional evidence that favoring vertical over horizontal shifts is harmful and motivates testing the symmetric reverse allocation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 65.03316524997354, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1865619785308838, "validation_score": 9322.421385489377}

RECENT RESULT
hypothesis: Blending 12.5% geometric consensus into the successful axial probability ensemble will exceed 9,348 correct predictions by suppressing isolated high-confidence crop errors while preserving the proven view set.
change: Retain the five axial crops and equal reflection weighting, blend their arithmetic and geometric probability ensembles, and restore the best verified 0.136 adjacent-temperature interpolation.
mechanism: Arithmetic–geometric TTA consensus blending
evidence_used: Axial translation TTA improved correctness from 9,344 to 9,348, while adding diagonal views reduced it to 9,335; changing how the proven axial views are fused is therefore a more informative accuracy probe than adding views or continuing temperature-only refinements.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.16849345783703, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18773939895629882, "validation_score": 9348.420967764847}

RECENT RESULT
hypothesis: A 1.125-order power mean will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18770656051635742 by modestly favoring crop-specific high-confidence evidence.
change: Restore equal native/reflected weighting and the best verified temperature, then replace arithmetic probability averaging with a normalized 1.125-order weighted power mean over the proven ten axial views.
mechanism: Mild higher-order power-mean TTA fusion
evidence_used: Blending 12.5% toward geometric consensus preserved correctness but worsened cross-entropy to 0.18773939895629882; this supplies directional evidence that moving the fusion rule oppositely, above the arithmetic mean, is the most informative local test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.28536408301443, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18768770637512208, "validation_score": 9348.420986086929}

RECENT RESULT
hypothesis: Raising the fusion order from 1.125 to 1.25 will preserve all 9,348 correct predictions while reducing validation cross-entropy below 0.18768770637512208.
change: Increase only the axial-view probability power-mean order, retaining the proven crops, weights, training procedure, and temperature.
mechanism: Quadratically guided higher-order power-mean TTA
evidence_used: Moving below arithmetic fusion worsened cross-entropy to 0.18773939895629882, while moving equally far above it improved cross-entropy to 0.18768770637512208; those opposing probes imply the local optimum remains above 1.125 and approximately near 1.23.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.646500959061086, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18768998832702638, "validation_score": 9348.420985278073}

RECENT RESULT
hypothesis: Combining fixed-position shape evidence with centered channel-covariance evidence will exceed 9,348 correct predictions by distinguishing classes with similar silhouettes but different feature co-occurrence patterns.
change: Replace the single 56-unit flattened bottleneck with a 43-unit spatial branch plus a zero-initialized second-order covariance branch, retaining full 7×7 layout and fitting 249,940 learned parameters.
mechanism: Hybrid spatial–covariance classification head
evidence_used: Global pooling fell to 9,085 correct, establishing that layout must remain available; axial TTA reached 9,348 by improving alignment, but subsequent fusion and calibration changes did not improve correctness. This challenges the shared assumption that class predictions need only first-order features at fixed positions while preserving the proven spatial pathway.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A fusion order of 1.174 will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18768770637512208.
change: Replace arithmetic axial-view averaging and temperature interpolation with a normalized 1.174-order probability power mean using Reference Design 3’s verified temperature.
mechanism: Quadratically interpolated power-mean TTA fusion
evidence_used: Orders 1.0, 1.125, and 1.25 produced cross-entropies of 0.1877065605, 0.1876877064, and 0.1876899883 with identical correctness; quadratic interpolation places the local minimum near 1.174.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.889032542007044, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1876861358642578, "validation_score": 9348.42098664361}

RECENT RESULT
hypothesis: Adding a 0.10-weight Jensen–Shannon penalty between paired native and reflected predictions to the verified 1.174-order fusion design will exceed 9,348 correct predictions by reducing orientation-sensitive errors.
change: Restore Reference Design 2’s best verified power-mean evaluation and augment the existing paired supervised loss with symmetric reflection-consistency regularization.
mechanism: Reflection-pair Jensen–Shannon consistency regularization
evidence_used: The 1.174-order fusion achieved the best verified cross-entropy with 9,348 correct, while preferring either orientation was not an improvement; this motivates teaching the model itself to agree across the equally weighted reflection pair.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.42085550003685, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.18897107925415038, "validation_score": 9321.420531675432}

RECENT RESULT
hypothesis: Averaging each native/reflected pair before applying the 1.174-order crop power mean will exceed 9,348 correct predictions by suppressing orientation-specific confidence outliers while retaining the beneficial emphasis on crop-specific evidence.
change: Replace the flat power mean over all ten views with arithmetic fusion within each reflection pair followed by the verified higher-order fusion across the five axial crops.
mechanism: Hierarchical orientation-then-crop power-mean fusion
evidence_used: The 1.174-order flat power mean achieved the best cross-entropy with 9,348 correct, while favoring either orientation and reflection-consistency training both failed; this suggests equal orientation consensus should be preserved while isolating the useful higher-order weighting to crop disagreement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.354036750039086, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18766820755004882, "validation_score": 9348.420992998568}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
