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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.54304054193199, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433547973634, "validation_score": 9192.409040001485}
prior_hypothesis: Expressing the midpoint sharpening as `x + alpha*x` will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796 through a distinct float32 rounding path.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.84990404197015, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22206263122558595, "validation_score": 9192.409144332887}
prior_hypothesis: A slope of 0.0175 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22207223510742188, because the gains from slopes 0.01 and 0.02 imply a shallow optimum near 0.017.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 61.57555366703309, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22207223510742188, "validation_score": 9192.40914111755}
prior_hypothesis: Doubling the verified margin-adaptation slope from 0.01 to 0.02 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22211728744506837.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.38871358381584, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22211728744506837, "validation_score": 9192.409126034903}
prior_hypothesis: Redistributing the proven average sharpening toward higher-margin predictions will preserve all 9,192 argmax decisions while reducing validation cross-entropy below 0.22237433319091796.

## Recent verification evidence

RECENT RESULT
hypothesis: Direct division by the reciprocal of the best verified float32 sharpening coefficient will preserve all 9,192 predictions while producing validation cross-entropy below 0.22237433319091796 through a distinct float32 rounding path.
change: Restore the best verified calibration target but realize it with tensor division instead of multiplication, leaving training, ensembling, centering, and class ordering unchanged.
mechanism: Reciprocal-division temperature calibration
evidence_used: Reference Design 1 achieved the highest observed score with coefficient 1.0978671312332153; adjacent multiplication coefficients and alternative centering paths were worse, motivating an untested arithmetic realization of the same temperature.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.61569770798087, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433547973634, "validation_score": 9192.409040001485}

RECENT RESULT
hypothesis: Increasing the reciprocal-division temperature by one float32 ULP will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796 through a previously untested division-rounding path.
change: Replace rational multiplication with division by the next representable float32 temperature above the reciprocal of the best verified sharpening coefficient.
mechanism: Directed adjacent-float reciprocal calibration
evidence_used: Reciprocal division preserved 9,192 correct predictions, while the lower-sharpening multiplication probe degraded less than the higher-sharpening probe; this motivates moving the division temperature one ULP toward lower sharpening.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.487110999878496, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433471679688, "validation_score": 9192.409040001741}

RECENT RESULT
hypothesis: Moving the reciprocal-division temperature one additional float32 ULP upward will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Replace multiplication calibration with division by the float32 temperature two ULPs above the reciprocal of the best verified sharpening coefficient.
mechanism: Continued adjacent-float reciprocal temperature sweep
evidence_used: The first upward reciprocal-temperature ULP improved cross-entropy from 0.22237433547973634 to 0.22237433471679688 while preserving 9,192 correct predictions, making one further step in that direction the most informative remaining probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.040810415986925, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433547973634, "validation_score": 9192.409040001485}

RECENT RESULT
hypothesis: Equal interpolation between the two adjacent multiplication coefficients tied at the best verified cross-entropy will preserve all 9,192 predictions while producing validation cross-entropy below 0.22237433319091796.
change: Replace reciprocal division with an elementwise midpoint blend of logits produced by Reference Designs 3 and 1.
mechanism: Adjacent-float logit interpolation
evidence_used: The rational 10500/9564 coefficient and its immediately larger float32 neighbor both achieved the best observed cross-entropy, while coefficients outside that pair were worse, indicating that an effective midpoint is the most informative remaining calibration probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.614102999912575, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433624267577, "validation_score": 9192.40904000123}

RECENT RESULT
hypothesis: Applying the exact midpoint between the two best adjacent float32 sharpening coefficients in float64 will preserve 9,192 predictions and reduce cross-entropy below 0.22237433319091796.
change: Cast centered evaluation logits to float64 and sharpen once using the exact midpoint coefficient, avoiding the extra float32 rounding introduced by the prior elementwise blend.
mechanism: Float64 sub-ULP temperature interpolation
evidence_used: The adjacent multiplication coefficients 1.0978670120239258 and 1.0978671312332153 tied for the best cross-entropy, while both exterior coefficients were worse; the prior float32 midpoint blend added multiple rounded operations and was slightly worse, leaving a single high-precision midpoint multiplication untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying the exact midpoint between the two best adjacent sharpening coefficients in float64, then restoring float32 output, will preserve all 9,192 predictions and reduce cross-entropy below 0.22237433319091796.
change: Replace rational float32 multiplication with one high-precision midpoint multiplication followed by an explicit cast back to the original logit dtype.
mechanism: Float64 sub-ULP calibration with float32 output
evidence_used: The adjacent coefficients 1.0978670120239258 and 1.0978671312332153 tied for the best cross-entropy while both exterior probes were worse; the prior float64 midpoint was unverified, so explicitly restoring float32 output tests the midpoint without changing the model’s output interface.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Centering the accumulated logits before normalization will preserve all 9,192 predictions while reducing validation cross-entropy below 0.22237433319091796 by avoiding subtraction between separately rounded quotients.
change: Move order-preserving max-centering from normalized logits to the raw ensemble sum, retaining the best verified float32 sharpening coefficient.
mechanism: Pre-division raw-ensemble max-centering
evidence_used: The current calibration has the best observed cross-entropy, while post-scale and mean-centering variants were worse; centering before division is an untested rounding path that preserves the same mathematical logits and class ordering.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.363380500115454, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433738708495, "validation_score": 9192.409040000846}

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
