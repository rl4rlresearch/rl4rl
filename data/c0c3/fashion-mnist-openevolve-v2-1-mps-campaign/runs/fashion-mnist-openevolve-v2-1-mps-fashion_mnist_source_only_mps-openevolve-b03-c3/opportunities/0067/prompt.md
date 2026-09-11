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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}
prior_hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 35.00231925002299, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21682710800170898, "validation_score": 9240.410904718272}
prior_hypothesis: Scaling evaluation logits by 1.285 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216927 by compensating for the greater confidence induced by the new 0.04→0 smoothing schedule.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.942004709038883, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}
prior_hypothesis: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.103174792137, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758494262695313, "validation_score": 9240.410648967883}
prior_hypothesis: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.

## Recent verification evidence

RECENT RESULT
hypothesis: Ending with pure flip-averaged supervision while preserving the current mean ensemble weight will exceed 9,240 correct predictions by aligning late optimization with flip-averaged evaluation without forcing view agreement.
change: Adopt the best verified smoothing and calibration, and widen the ensemble-weight schedule from 0.55→0.95 to 0.50→1.00.
mechanism: Late flip-ensemble objective alignment
evidence_used: The 0.04→0 smoothing schedule retained 9,240 correct and achieved the best cross-entropy, while explicit flip-consistency reduced correctness to 9,234; optimizing the evaluated average permits complementary views without penalizing disagreement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 40.80001358385198, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.2169219982147217, "validation_score": 9239.410872677734}

RECENT RESULT
hypothesis: Adding a zero-initialized nonlinear residual transformation to the 48-dimensional classifier representation, atop the verified 0.04→0 smoothing schedule, will exceed 9,240 correct predictions without destabilizing the successful initial optimization path.
change: Insert a lightweight residual MLP block in the classifier, adopt the best verified label-smoothing schedule, and use the verified 1.296352 evaluation-logit calibration; total learned parameters become 244,994.
mechanism: Zero-initialized residual classifier refinement
evidence_used: The 0.04→0 smoothing design retained 9,240 correct and achieved the best available cross-entropy, while spatial gating reduced correctness and attention exceeded the time limit. A zero-initialized residual head adds inexpensive global nonlinear capacity while initially preserving the proven classifier exactly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 244994, "training_seconds": 51.77934629190713, "validation_accuracy": 0.9231, "validation_correct": 9231, "validation_cross_entropy": 0.22194619674682617, "validation_score": 9231.409183318652}

RECENT RESULT
hypothesis: Reducing initial smoothing from 0.04 to 0.03 while retaining the gradual zero endpoint will exceed 9,240 correct predictions by easing early underconfidence without prematurely removing regularization.
change: Change only the cosine label-smoothing schedule from 0.04→0 to 0.03→0.
mechanism: Lower-amplitude full-horizon label-smoothing anneal
evidence_used: The 0.04→0 schedule achieved the best score, while 0.05→0 fell to 9,237 correct; the unsuccessful half-horizon anneal indicates smoothing should still decay across the full training run.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 43.101726291002706, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.2168480842590332, "validation_score": 9236.41089763502}

RECENT RESULT
hypothesis: A 0.04→0.01 smoothing schedule will exceed 9,240 correct predictions by combining the empirically optimal 0.04 initial smoothing with modest late regularization that avoids fully hard-label updates.
change: Change only the cosine label-smoothing schedule from 0.05→0.01 to 0.04→0.01.
mechanism: Mild terminal label smoothing
evidence_used: The 0.04→0 schedule achieved the best score with 9,240 correct, while both 0.03→0 and 0.05→0 reduced correctness; the current 0.05→0.01 schedule also reached 9,240, making 0.04→0.01 the informative missing combination.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Narrowing the ensemble-weight schedule from 0.55→0.95 to 0.60→0.90 atop the best verified smoothing schedule will exceed 9,240 correct predictions by balancing evaluation alignment with robust individual-view classification.
change: Adopt the verified 1.296352 calibration and 0.04→0 smoothing, then reduce the amplitude of the flip-ensemble weighting schedule.
mechanism: Narrowed flip-ensemble curriculum
evidence_used: The verified 0.04→0 design achieved the best score with 9,240 correct, while widening ensemble weighting to 0.50→1.00 lost one correct prediction; testing the opposite direction is the most direct remaining comparison.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 56.07909670798108, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.2170023998260498, "validation_score": 9239.410845533313}

RECENT RESULT
hypothesis: Replacing exact-grid flattening with multiscale regional mean/max pooling and a wider nonlinear bottleneck will exceed 9,240 correct predictions by learning shape evidence that is less sensitive to residual translations.
change: Challenge the assumption that every 7×7 location needs an independent dense weight: summarize features over 1×1, 2×2, and 4×4 regions using average and maximum evidence, expand the bottleneck from 48 to 56 units, and retain the best verified smoothing and calibration.
mechanism: Dual-statistic spatial pyramid classifier
evidence_used: The successful design relies heavily on translation augmentation and 25-shift evaluation, indicating exact feature alignment is a nuisance. Unlike the failed multiplicative coordinate gating, this pyramid preserves regional layout and feature content; unlike self-attention, it adds negligible computational cost. Reference Design 1 establishes 0.04→0 smoothing as the strongest verified training objective.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A second depthwise-separable 7×7-feature refinement block will exceed 9,240 correct predictions by extending spatial processing to full-image garment structure while remaining below the parameter ceiling.
change: Add a second residual depthwise/pointwise convolutional block, increasing learned parameters from 242,546 to 249,986 while preserving the verified optimizer, augmentation, smoothing, and calibration.
mechanism: Second spatial residual refinement block
evidence_used: The zero-initialized classifier MLP fell to 9,231 correct and self-attention timed out, motivating a computationally cheaper allocation of the remaining 7,454-parameter budget to spatial feature refinement instead of additional global-head complexity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249986, "training_seconds": 42.65292058279738, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.21865935707092285, "validation_score": 9232.410286924807}

RECENT RESULT
hypothesis: Replacing non-overlapping 2×2 pooling with overlapping 3×3 stride-2 pooling will exceed 9,240 correct predictions by reducing one-pixel pooling-phase sensitivity without adding parameters, while the verified 0.04→0 smoothing schedule preserves the strongest training objective.
change: Use overlapping pooling at both downsampling stages and adopt the best verified full-horizon label-smoothing anneal; retain the proven architecture, optimizer, TTA, and calibration otherwise.
mechanism: Overlapping phase-stable spatial pooling
evidence_used: The classifier requires translation augmentation and 25-shift evaluation, indicating residual shift sensitivity, while an added spatial refinement block fell to 9,232 correct. This motivates parameter-free downsampling stability instead of more capacity; Reference Design 1 establishes 0.04→0 smoothing as the strongest verified objective at 9,240 correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 43.782836084021255, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22527412300109864, "validation_score": 9208.408071949463}

RECENT RESULT
hypothesis: Replacing the best verified cosine 0.04→0 smoothing anneal with a full-horizon linear anneal of identical endpoints and mean will exceed 9,240 correct predictions by retaining more regularization during later updates without weakening the empirically optimal initial smoothing.
change: Adopt the verified 1.296352 calibration and replace the current 0.05→0.01 cosine smoothing with a linear 0.04→0 schedule; architecture, optimizer, learning-rate schedule, and ensemble weighting remain unchanged.
mechanism: Mean-preserving linear label-smoothing anneal
evidence_used: Cosine 0.04→0 achieved the best verified score at 9,240 correct, while ending smoothing halfway fell to 9,233 and changing the initial amplitude to 0.03 or 0.05 also lost correctness; a full-horizon linear schedule isolates decay curvature while preserving the successful endpoints and average smoothing.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 37.57991841691546, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21694990501403807, "validation_score": 9239.410863255702}

RECENT RESULT
hypothesis: EMA over the final quarter of training, combined with the best verified 0.04→0 smoothing schedule, will exceed 9,240 correct predictions by reducing sensitivity to individual late updates while retaining the proven nonzero learning-rate floor.
change: Adopt the verified 1.296352 calibration and cosine 0.04→0 smoothing, then average learned parameters with decay 0.95 during the final quarter and install the averaged weights after the last optimizer step.
mechanism: Late-trajectory exponential weight averaging
evidence_used: The 0.04→0 design achieved the best verified score with 9,240 correct, while zero and doubled terminal learning-rate floors fell to 9,219 and 9,205; late optimization is therefore consequential, motivating trajectory averaging without changing the successful floor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 50.22502345801331, "validation_accuracy": 0.9225, "validation_correct": 9225, "validation_cross_entropy": 0.2172421531677246, "validation_score": 9225.410764611379}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.285 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216927 by compensating for the greater confidence induced by the new 0.04→0 smoothing schedule.
change: Reduce only the evaluation-logit multiplier from 1.296352 to 1.285.
mechanism: Downward logit-temperature recalibration
evidence_used: The 1.296352 optimum was measured with the reference 0.05→0.01 smoothing schedule, whereas the current 0.04→0 schedule lowers smoothing by 0.01 throughout training and improved cross-entropy to 0.216927 without changing correctness; a modest downward recalibration tests that confidence shift while mathematically preserving every predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 35.00231925002299, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21682710800170898, "validation_score": 9240.410904718272}

RECENT RESULT
hypothesis: Averaging per-shift class probabilities instead of logits will exceed 9,240 correct predictions by limiting the influence of overconfident errors from individual translated views.
change: Adopt the best verified 0.04→0 smoothing schedule and 1.285 calibration, while changing only the 25-shift aggregation from a weighted logit average to a weighted posterior mixture; flip pairs remain logit-averaged.
mechanism: Posterior-mixture shift ensembling
evidence_used: Reference Design 1 achieved the best verified score with 9,240 correct using 0.04→0 smoothing and 1.285 calibration. Added spatial refinement and phase-stable pooling reduced correctness, while the design’s extensive shift evaluation indicates residual view sensitivity, motivating a robust aggregation change without altering learned capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 39.350304041057825, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21693955612182617, "validation_score": 9237.410866749697}



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
