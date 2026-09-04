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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 64.70731712505221, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21543648796081544, "validation_score": 9267.411374847597}
prior_hypothesis: Increasing the geometric component from 8% to 9% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215472, exceeding validation_score 9267.411363.

## Recent verification evidence

RECENT RESULT
hypothesis: Blending 10% equal hard votes into the successful arithmetic probability ensemble will exceed 9,266 correct predictions by reducing residual sensitivity to differing confidence magnitudes across transformed views.
change: Keep training unchanged and add a small one-hot vote component to each validation-view probability vector before the existing spatial aggregation.
mechanism: Soft-probability and plurality-vote TTA hybrid
evidence_used: Arithmetic probability pooling improved validation correct from 9,265 to 9,266, while aligning the training loss to probability pooling regressed to 9,229; this motivates refining only inference aggregation toward confidence-independent voting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.20351720904, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.21567442932128905, "validation_score": 9264.411294330077}

RECENT RESULT
hypothesis: Blending 1% of the prior geometric logit ensemble into arithmetic probability pooling will retain 9,266 correct predictions while lowering cross-entropy below 0.215801, exceeding validation_score 9266.411252.
change: Accumulate validation-view logits alongside probabilities and interpolate 99% of the calibrated arithmetic log-probabilities with 1% of the geometric ensemble’s normalized logits.
mechanism: Near-arithmetic log-opinion pooling
evidence_used: Arithmetic pooling improved correct predictions from 9,265 to 9,266 but worsened cross-entropy from 0.211944 to 0.215801; a small interpolation toward the better-calibrated geometric endpoint is likely to improve the tie-breaker without crossing enough decision boundaries to lose the accuracy gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 62.29763370892033, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2157552589416504, "validation_score": 9266.41126698513}

RECENT RESULT
hypothesis: Increasing the geometric component from 1% to 2% will retain 9,266 correct predictions while lowering cross-entropy below 0.215755, exceeding validation_score 9266.411267.
change: Interpolate 98% calibrated arithmetic log-probabilities with 2% normalized geometric logits during validation inference.
mechanism: Two-percent geometric log-opinion blend
evidence_used: Adding a 1% geometric component preserved the arithmetic ensemble’s 9,266 correct predictions and improved cross-entropy from 0.215801 to 0.215755; the geometric endpoint’s substantially lower 0.211944 cross-entropy motivates one more conservative step in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 70.40756420791149, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21571103439331055, "validation_score": 9267.411281946}

RECENT RESULT
hypothesis: Increasing the geometric component from 2% to 3% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215711, thereby exceeding validation_score 9267.411282.
change: Interpolate 97% calibrated arithmetic log-probabilities with 3% normalized geometric logits during validation inference.
mechanism: Three-percent geometric log-opinion blend
evidence_used: Moving from 1% to 2% geometric blending improved validation correct from 9,266 to 9,267 and reduced cross-entropy from 0.215755 to 0.215711; the geometric endpoint also has substantially lower cross-entropy, motivating one further conservative step.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.86444654199295, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2156680419921875, "validation_score": 9267.41129649109}

RECENT RESULT
hypothesis: Increasing the geometric component from 3% to 4% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215668, exceeding validation_score 9267.411296.
change: Interpolate 96% calibrated arithmetic log-probabilities with 4% normalized geometric logits during validation inference.
mechanism: Four-percent geometric log-opinion blend
evidence_used: Increasing the geometric blend from 1% to 2% raised correct predictions from 9,266 to 9,267, and increasing it to 3% retained 9,267 while reducing cross-entropy from 0.215711 to 0.215668; another one-point step directly tests the established local trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 63.652752083027735, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21562630386352538, "validation_score": 9267.411310612819}

RECENT RESULT
hypothesis: Increasing the geometric component from 4% to 5% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215626, exceeding validation_score 9267.411311.
change: Interpolate 95% calibrated arithmetic log-probabilities with 5% normalized geometric logits during validation inference.
mechanism: Five-percent geometric log-opinion blend
evidence_used: Each increase from 1% through 4% geometric blending preserved or improved validation correct while monotonically reducing cross-entropy; the 4% blend achieved the current best result, so one further one-point step is the clearest continuation of the established local trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 48.84712812490761, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21558580894470214, "validation_score": 9267.411324314846}

RECENT RESULT
hypothesis: Increasing the geometric component from 5% to 6% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215586, exceeding validation_score 9267.411324.
change: Interpolate 94% calibrated arithmetic log-probabilities with 6% normalized geometric logits during validation inference.
mechanism: Six-percent geometric log-opinion blend
evidence_used: Every increase from 1% through 5% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 5% result makes another one-point step the most direct test of this trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 76.09206066606566, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21554657669067384, "validation_score": 9267.411337590504}

RECENT RESULT
hypothesis: Replacing the pixel-aligned flattened head with shared local part refinement plus coarse spatial-average and regional-maximum summaries will exceed 9,267 correct predictions by improving shift tolerance without discarding garment layout.
change: Add a lightweight pointwise/depthwise residual part encoder and classify concatenated 4×4 average-pooled and 2×2 max-pooled features; the resulting model has approximately 249,178 learned parameters.
mechanism: Coordinate-aware multi-scale part aggregation
evidence_used: Position-free pooling regressed to 9,253 and 9,249, showing layout is essential, while widening the static flattened head regressed to 9,221 and attention timed out. This tests content-derived local parts and multi-scale aggregation while preserving explicit spatial regions at modest computational cost.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the geometric component from 6% to 7% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215547, exceeding validation_score 9267.411338.
change: Interpolate 93% calibrated arithmetic log-probabilities with 7% normalized geometric logits during validation inference.
mechanism: Seven-percent geometric log-opinion blend
evidence_used: Every increase from 1% through 6% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 6% result makes another one-point step the clearest test of the established trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 40.551128040999174, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21550860137939454, "validation_score": 9267.411350441644}

RECENT RESULT
hypothesis: Increasing the geometric component from 7% to 8% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215509, exceeding validation_score 9267.411350.
change: Interpolate 92% calibrated arithmetic log-probabilities with 8% normalized geometric logits during validation inference.
mechanism: Eight-percent geometric log-opinion blend
evidence_used: Every increase from 1% through 7% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 7% result makes another one-point step the most informative continuation of this established local trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 61.702546207932755, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21547190666198732, "validation_score": 9267.411362860186}

RECENT RESULT
hypothesis: Increasing the geometric component from 8% to 9% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215472, exceeding validation_score 9267.411363.
change: Interpolate 91% calibrated arithmetic log-probabilities with 9% normalized geometric logits during validation inference.
mechanism: Nine-percent geometric log-opinion blend
evidence_used: Every increase from 1% through 8% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 8% result makes another one-point step the most informative continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 64.70731712505221, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21543648796081544, "validation_score": 9267.411374847597}

RECENT RESULT
hypothesis: Increasing the geometric component from 9% to 10% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215436, exceeding validation_score 9267.411375.
change: Interpolate 90% calibrated arithmetic log-probabilities with 10% normalized geometric logits during validation inference.
mechanism: Ten-percent geometric log-opinion blend
evidence_used: Every increase from 1% through 9% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 9% result makes another one-point step the most informative continuation of the established local trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 57.5416045000311, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540235862731932, "validation_score": 9266.411386399286}



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
