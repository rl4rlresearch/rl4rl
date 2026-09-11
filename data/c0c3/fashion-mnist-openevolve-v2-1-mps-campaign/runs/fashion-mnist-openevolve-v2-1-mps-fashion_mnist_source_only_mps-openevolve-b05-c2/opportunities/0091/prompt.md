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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 67.63246566615999, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.222062410736084, "validation_score": 9192.409144406707}
prior_hypothesis: A slope of 0.01705 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220624153137207.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.7834930839017, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2220299663543701, "validation_score": 9192.409155269319}
prior_hypothesis: Adding a −0.001 batch-mean-centered quadratic margin term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22206240844726563.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.1875938330777, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22200976676940917, "validation_score": 9192.409162032576}
prior_hypothesis: Doubling the verified quadratic coefficient to −0.002 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220299663543701.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.38871358381584, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22211728744506837, "validation_score": 9192.409126034903}
prior_hypothesis: Redistributing the proven average sharpening toward higher-margin predictions will preserve all 9,192 argmax decisions while reducing validation cross-entropy below 0.22237433319091796.

## Recent verification evidence

RECENT RESULT
hypothesis: Collapsing ensemble normalization and the best verified sharpening coefficient into one float32 multiplication will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Apply one combined scale directly to the accumulated ensemble logits, then max-center the calibrated result.
mechanism: Single-rounding fused ensemble calibration
evidence_used: Reference Design 1 has the best observed cross-entropy using separate normalization and multiplication, while post-scale centering with separate operations was only slightly worse; eliminating the intermediate division rounding is the most informative untested equivalent path.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.56899600010365, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2223743350982666, "validation_score": 9192.409040001612}

RECENT RESULT
hypothesis: Expressing the midpoint sharpening as `x + alpha*x` will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796 through a distinct float32 rounding path.
change: Replace the current denominator-based multiplication with residual-form sharpening whose effective coefficient is the exact midpoint between the two best adjacent coefficients.
mechanism: Residual-form sub-ULP sharpening
evidence_used: The adjacent coefficients 1.0978670120239258 and 1.0978671312332153 tied for the best observed cross-entropy, while exterior coefficients and a multi-operation midpoint blend were worse; residual-form addition tests the sub-ULP midpoint without float64 output or the blend’s extra rounded interpolation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.54304054193199, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433547973634, "validation_score": 9192.409040001485}

RECENT RESULT
hypothesis: Redistributing the proven average sharpening toward higher-margin predictions will preserve all 9,192 argmax decisions while reducing validation cross-entropy below 0.22237433319091796.
change: Replace uniform evaluation-time sharpening with a positive, batch-mean-centered scale that slightly sharpens high-margin samples and softens low-margin samples.
mechanism: Margin-adaptive logit calibration
evidence_used: Numerous adjacent-float and arithmetic-order probes preserved 9,192 correct predictions but failed to beat the current cross-entropy, indicating that uniform calibration is exhausted and motivating an argmax-preserving, confidence-dependent calibration.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.38871358381584, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22211728744506837, "validation_score": 9192.409126034903}

RECENT RESULT
hypothesis: Doubling the verified margin-adaptation slope from 0.01 to 0.02 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22211728744506837.
change: Replace uniform evaluation-time sharpening with positive, batch-mean-centered scaling that more strongly sharpens high-margin predictions and softens low-margin predictions.
mechanism: Stronger margin-adaptive logit calibration
evidence_used: Reference Design 3’s slope-0.01 margin adaptation preserved 9,192 correct predictions while improving cross-entropy from 0.22237433319091796 to 0.22211728744506837; testing a larger slope is the most direct probe of whether the observed gain continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 61.57555366703309, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22207223510742188, "validation_score": 9192.40914111755}

RECENT RESULT
hypothesis: A slope of 0.0175 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22207223510742188, because the gains from slopes 0.01 and 0.02 imply a shallow optimum near 0.017.
change: Replace uniform fused calibration with max-centered ensemble normalization and batch-mean-centered margin scaling at slope 0.0175.
mechanism: Quadratically tuned margin-adaptive calibration
evidence_used: Margin adaptation improved cross-entropy from roughly 0.22237433 at slope 0 to 0.22211729 at 0.01 and 0.22207224 at 0.02 without changing correct predictions; quadratic interpolation of those three observations places the estimated optimum near 0.0171.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.84990404197015, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22206263122558595, "validation_score": 9192.409144332887}

RECENT RESULT
hypothesis: A slope of 0.0170 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22206263122558595.
change: Replace residual-form uniform sharpening with max-centered, batch-mean-centered margin scaling at slope 0.0170.
mechanism: Locally optimized margin-adaptive calibration
evidence_used: Slopes 0.01, 0.0175, and 0.02 produced cross-entropies 0.2221172874, 0.2220626312, and 0.2220722351 while preserving 9,192 correct predictions; local quadratic interpolation places the minimum near 0.01702.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 62.855690917000175, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2220624153137207, "validation_score": 9192.409144405176}

RECENT RESULT
hypothesis: A slope of 0.01705 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220624153137207.
change: Increase the batch-mean-centered confidence-adaptation slope from 0.017 to 0.01705.
mechanism: Fine-grained margin-slope calibration
evidence_used: Slopes 0.017, 0.0175, and 0.02 yielded cross-entropies 0.2220624153, 0.2220626312, and 0.2220722351; local quadratic interpolation places the minimum near 0.01705–0.01706.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 67.63246566615999, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.222062410736084, "validation_score": 9192.409144406707}

RECENT RESULT
hypothesis: A slope of 0.017065 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.222062410736084.
change: Change the batch-mean-centered confidence-adaptation slope from 0.0175 to 0.017065.
mechanism: Local quadratic margin-slope refinement
evidence_used: Slopes 0.017, 0.01705, and 0.0175 produced cross-entropies 0.2220624153137207, 0.222062410736084, and 0.22206263122558595 with identical accuracy; local quadratic interpolation places the minimum near 0.017064.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.53074033395387, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22206240844726563, "validation_score": 9192.409144407473}

RECENT RESULT
hypothesis: A slope of 0.01706436 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22206240844726563.
change: Replace the current 0.02 confidence-adaptation slope with the quadratic-interpolated local optimum.
mechanism: Quadratic-vertex margin calibration
evidence_used: Slopes 0.017, 0.01705, and 0.0175 yielded cross-entropies 0.2220624153137207, 0.222062410736084, and 0.22206263122558595; their quadratic vertex is approximately 0.01706436, immediately beside the best verified 0.017065 result.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.88432687497698, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22206240882873535, "validation_score": 9192.409144407346}

RECENT RESULT
hypothesis: Setting the confidence-adaptation slope to 0.017065 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22211728744506837.
change: Increase the evaluation-time, batch-mean-centered margin-scaling slope from 0.01 to 0.017065.
mechanism: Best-verified margin-slope calibration
evidence_used: Reference Design 1 used slope 0.017065 and achieved the best observed score, with 9,192 correct predictions and cross-entropy 0.22206240844726563.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a −0.001 batch-mean-centered quadratic margin term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22206240844726563.
change: Extend the best linear confidence scaling with a zero-mean concave correction that further softens low-margin predictions while tapering sharpening for extreme high margins.
mechanism: Concave margin-adaptive calibration
evidence_used: Linear margin adaptation reduced cross-entropy from 0.2223743351 to 0.2220624084 without changing accuracy, while increasingly fine slope adjustments have nearly plateaued; this motivates testing curvature in the order-preserving calibration function.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.7834930839017, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2220299663543701, "validation_score": 9192.409155269319}

RECENT RESULT
hypothesis: Doubling the verified quadratic coefficient to −0.002 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220299663543701.
change: Extend the best linear calibration with a stronger batch-mean-centered concave quadratic margin correction.
mechanism: Stronger concave margin-adaptive calibration
evidence_used: Adding a −0.001 centered quadratic term improved cross-entropy from 0.22206240844726563 to 0.2220299663543701 without changing accuracy; doubling that first successful curvature step directly tests whether the gain continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.1875938330777, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22200976676940917, "validation_score": 9192.409162032576}



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
