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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.103174792137, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758494262695313, "validation_score": 9240.410648967883}
prior_hypothesis: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.27387729194015, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}
prior_hypothesis: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.21758331069946288.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}
prior_hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 29.321368207922205, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21786421813964843, "validation_score": 9240.410554799584}
prior_hypothesis: Increasing evaluation-logit scaling from 1.20 to 1.25 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.218881.

## Recent verification evidence

RECENT RESULT
hypothesis: A cosine label-smoothing schedule from 0.045 to 0.015 with the proven 55%–95% flip-ensemble curriculum will exceed 9,240 correct predictions by retaining progressive regularization while avoiding the harmful extremes of the 0.06→0.00 and 0.06→0.01 schedules.
change: Restore the strongest 55%–95% ensemble-loss curriculum and replace static 0.03 smoothing with a narrower cosine decay centered at 0.03.
mechanism: Intermediate-width target-smoothing curriculum
evidence_used: Static 0.03 smoothing with progressive flip supervision reached 9,237 correct, 0.05→0.01 improved this to 9,240, and the wider 0.06→0.00 schedule fell to 9,232; testing amplitude 0.015 brackets the successful region between static and amplitude 0.02 smoothing while avoiding the failed wider regime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 42.225944832898676, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.23561809120178223, "validation_score": 9235.404655777995}

RECENT RESULT
hypothesis: Restoring the 9,240-correct smoothing and flip curricula while scaling evaluation logits by 1.05 will retain 9,240 correct predictions and reduce cross-entropy below 0.234234, strictly improving validation_score.
change: Restore Reference Design 1’s 0.05→0.01 label-smoothing and 55%→95% ensemble-weight schedules, then apply a modest confidence correction to the averaged validation logits.
mechanism: Flip-orbit curriculum with confidence-calibrated translation ensembling
evidence_used: Reference Design 1 achieved the best accuracy at 9,240 correct; the 0.06→0.00 result produced the lowest cross-entropy (0.231344), indicating that sharper confidence can improve the tie-breaker, while constant positive logit scaling preserves every argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 33.20328233297914, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2281807456970215, "validation_score": 9240.407106203018}

RECENT RESULT
hypothesis: Increasing evaluation-logit scaling from 1.05 to 1.10 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.228181.
change: Increase the positive evaluation-only logit multiplier to 1.10 without changing training or class decisions.
mechanism: Stronger post-ensemble confidence calibration
evidence_used: Scaling logits by 1.05 preserved 9,240 correct predictions and reduced cross-entropy from 0.234234 to 0.228181; this motivates a second equal calibration step in the same beneficial direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 30.37924558413215, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2238097038269043, "validation_score": 9240.408560251186}

RECENT RESULT
hypothesis: Adding a translation-tolerant prediction branch over channel-wise mean and maximum activations will exceed 9,240 correct predictions by complementing the existing absolute-position classifier with global shape and feature-presence evidence, while retaining the proven local representation.
change: Preserve the strongest backbone, training curricula, and spatial classifier; add a 5,802-parameter global-moment head, blend its logits residually, and retain the proven 1.10 evaluation calibration. Total learned parameters become 248,348.
mechanism: Residual global-moment evidence pooling
evidence_used: Optimization and loss-schedule refinements have plateaued at 9,240 correct, while the strong reliance on translation augmentation and 25-view evaluation indicates that the flattened 7×7 classifier’s position sensitivity is load-bearing. The unverified multiscale replacement discarded that strong route entirely; this patch cleanly tests translation-tolerant pooled evidence as a complementary learned mechanism. Reference Design 1 also shows that 1.10 logit scaling preserves accuracy while lowering cross-entropy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 248348, "training_seconds": 38.54326529218815, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.22531021614074706, "validation_score": 9227.408059929163}

RECENT RESULT
hypothesis: Restoring the 9,240-correct smoothing curriculum and increasing evaluation-logit scaling to 1.15 will preserve all argmax predictions while reducing validation cross-entropy below 0.223810.
change: Restore the proven 0.05→0.01 cosine label-smoothing schedule and apply a 1.15 evaluation-only multiplier to translation-and-flip-ensembled logits.
mechanism: Continued post-ensemble confidence calibration
evidence_used: Scaling the same 9,240-correct model from 1.00 to 1.05 and then 1.10 monotonically reduced cross-entropy from 0.234234 to 0.228181 and 0.223810 without changing any class decisions.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.55396529100835, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2207953384399414, "validation_score": 9240.409569060641}

RECENT RESULT
hypothesis: Restoring the 9,240-correct smoothing and flip curricula while increasing evaluation-logit scaling to 1.20 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.220795.
change: Restore Reference Design 3’s 0.05→0.01 label-smoothing and 55%→95% flip-ensemble schedules, then increase the positive evaluation-only logit multiplier from 1.15 to 1.20.
mechanism: Extended post-ensemble confidence calibration
evidence_used: For the same 9,240-correct training recipe, evaluation scaling of 1.00, 1.05, 1.10, and 1.15 monotonically reduced cross-entropy from 0.234234 to 0.228181, 0.223810, and 0.220795 without changing any argmax prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.48900929209776, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21888097610473634, "validation_score": 9240.410212325733}

RECENT RESULT
hypothesis: Increasing evaluation-logit scaling from 1.20 to 1.25 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.218881.
change: Increase the positive evaluation-only logit multiplier to 1.25 without changing training or predicted classes.
mechanism: Incremental post-ensemble confidence calibration
evidence_used: Scaling the same 9,240-correct model from 1.00 through 1.20 monotonically reduced cross-entropy at every 0.05 increment, with the latest change improving it from 0.220795 to 0.218881 while preserving every argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 29.321368207922205, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21786421813964843, "validation_score": 9240.410554799584}

RECENT RESULT
hypothesis: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.
change: Restore the best verified 1.25 calibration and extend it by one proven 0.05 increment to 1.30, without changing training or predicted classes.
mechanism: Incremental post-ensemble confidence calibration
evidence_used: Scaling the same 9,240-correct model from 1.00 through 1.25 monotonically reduced cross-entropy at every 0.05 increment; the latest step improved it from 0.218881 to 0.217864 while preserving every argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.103174792137, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758494262695313, "validation_score": 9240.410648967883}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.
change: Restore the proven training recipe and change its evaluation-only logit multiplier from 1.10 to 1.295.
mechanism: Fine-grained post-ensemble confidence calibration
evidence_used: Cross-entropy decreased monotonically from 0.218881 at 1.20 to 0.217864 at 1.25 and 0.217585 at 1.30 while all 9,240 predictions remained unchanged; the sharply diminishing gains indicate an optimum near 1.30, motivating a fine-grained interpolation rather than another 0.05 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2964 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217583535.
change: Apply a 1.2964 evaluation-only multiplier to the translation-and-flip-ensembled logits.
mechanism: Local quadratic confidence calibration
evidence_used: Cross-entropy improved from 0.217864218 at scale 1.25 to 0.217583535 at 1.295, then worsened slightly to 0.217584943 at 1.30; local quadratic interpolation places the calibration minimum near 1.2964.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.54287579189986, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758331069946288, "validation_score": 9240.410649518277}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.21758331069946288.
change: Move the evaluation-only multiplier from 1.2964 to the local quadratic optimum inferred from neighboring verified scales.
mechanism: Second-order confidence calibration refinement
evidence_used: Cross-entropy was 0.21758353462219238 at 1.295, 0.21758331069946288 at 1.2964, and 0.21758494262695313 at 1.30; quadratic interpolation places the minimum near 1.296352.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.27387729194015, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.217583535 to approximately 0.217583305.
change: Replace the evaluation-only logit multiplier with the best verified local-quadratic calibration.
mechanism: Second-order confidence calibration refinement
evidence_used: Reference Design 1 used the identical training recipe with scale 1.296352 and achieved the best available score, 9,240.410649520078, improving on the current 1.295 scale while preserving all predictions.
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
