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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.361273041926324, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392585754394, "validation_score": 9192.409184079088}
prior_hypothesis: A heptadecic coefficient of +0.00000000000002409365 will preserve all 9,192 argmax predictions and may reduce validation cross-entropy below 0.2219439224243164.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.45751720806584, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392471313476, "validation_score": 9192.409184079472}
prior_hypothesis: A heptadecic coefficient of +0.0000000000000238886 will preserve all 9,192 argmax predictions and may reduce validation cross-entropy below 0.2219439224243164.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 77.6402839999646, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219439224243164, "validation_score": 9192.409184080238}
prior_hypothesis: A heptadecic coefficient of +0.000000000000024162 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439224243164.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.84086508303881, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219439224243164, "validation_score": 9192.409184080238}
prior_hypothesis: A heptadecic coefficient of +0.0000000000000240253 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the heptadecic coefficient to +0.00000000000001165494349433176 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194397964477539.
change: Restore the best pentadecic coefficient, derive the centered seventeenth-power margin feature, and double the best verified heptadecic coefficient.
mechanism: One-sided centered-heptadecic calibration continuation
evidence_used: Successive heptadecic coefficients of 0, +0.00000000000000291373587358294, and +0.00000000000000582747174716588 reduced cross-entropy from 0.22194401245117187 to 0.2219439956665039 and then 0.22194397964477539 at identical accuracy, showing continued benefit with only slight diminishing returns.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 40.80079400003888, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219439598083496, "validation_score": 9192.40918406772}

RECENT RESULT
hypothesis: Doubling the heptadecic coefficient to +0.00000000000002330988698866352 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439598083496.
change: Double the centered seventeenth-power confidence coefficient while leaving the learned model and all lower-order calibration terms unchanged.
mechanism: One-sided centered-heptadecic calibration continuation
evidence_used: Successive heptadecic coefficients of 0, +0.00000000000000291373587358294, +0.00000000000000582747174716588, and +0.00000000000001165494349433176 reduced cross-entropy at identical accuracy, with the latest coefficient producing the best verified score.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.54852104187012, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392585754394, "validation_score": 9192.409184079088}

RECENT RESULT
hypothesis: Doubling the best verified heptadecic coefficient to +0.00000000000004661977397732704 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.
change: Increase only the centered seventeenth-power confidence coefficient, leaving training and lower-order calibration unchanged.
mechanism: One-sided centered-heptadecic calibration continuation
evidence_used: Successive coefficients of 0, +0.00000000000000291373587358294, +0.00000000000000582747174716588, +0.00000000000001165494349433176, and +0.00000000000002330988698866352 consistently reduced cross-entropy at identical accuracy, so another doubling is the most informative continuation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.10163479205221, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440860748291, "validation_score": 9192.409184025439}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.00000000000002268621107894823 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.
change: Move only the centered seventeenth-power confidence coefficient to the quadratic minimum fitted from the closest verified bracket.
mechanism: Bracketed quadratic heptadecic confidence calibration
evidence_used: Coefficient multiples 4×, 8×, and 16× produced cross-entropies 0.2219439598083496, 0.22194392585754394, and 0.2219440860748291 at identical accuracy; quadratic interpolation places the minimum near 7.78595× the original coefficient.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 49.74300741683692, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392852783204, "validation_score": 9192.409184078195}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.0000000000000275912039129 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.
change: Add the centered seventeenth-power margin feature and apply the quadratic minimum estimated from the nearest verified lower, current-best, and upper coefficients.
mechanism: Bracketed quadratic heptadecic confidence calibration
evidence_used: Coefficients +0.00000000000002268621107894823, +0.00000000000002330988698866352, and +0.00000000000004661977397732704 produced cross-entropies 0.22194392852783204, 0.22194392585754394, and 0.2219440860748291 with identical accuracy; quadratic interpolation estimates the minimum near +0.0000000000000275912039129.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.11453233403154, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392776489258, "validation_score": 9192.40918407845}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.0000000000000252194 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.
change: Move only the centered seventeenth-power confidence coefficient to the quadratic vertex estimated from the closest verified points surrounding the current best.
mechanism: Tight-bracket quadratic heptadecic confidence calibration
evidence_used: Coefficients +0.00000000000002268621107894823, +0.00000000000002330988698866352, and +0.0000000000000275912039129 produced cross-entropies 0.22194392852783204, 0.22194392585754394, and 0.22194392776489258 at identical accuracy; quadratic interpolation places the minimum near +0.0000000000000252194.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.2335014580749, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392776489258, "validation_score": 9192.40918407845}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.0000000000000240253 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.
change: Move only the centered seventeenth-power confidence coefficient to the quadratic vertex fitted from the three closest distinct verified coefficients.
mechanism: Tight-bracket refitted heptadecic confidence calibration
evidence_used: Coefficients +0.00000000000002268621107894823, +0.00000000000002330988698866352, and +0.0000000000000252194 produced cross-entropies 0.22194392852783204, 0.22194392585754394, and 0.22194392776489258 at identical accuracy; quadratic interpolation places the minimum near +0.0000000000000240253.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.84086508303881, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219439224243164, "validation_score": 9192.409184080238}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.000000000000024162 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439224243164.
change: Move only the centered seventeenth-power confidence coefficient to the quadratic minimum fitted from the closest verified coefficients surrounding the current best.
mechanism: Tight-bracket quadratic heptadecic calibration
evidence_used: Coefficients +0.00000000000002330988698866352, +0.0000000000000240253, and +0.0000000000000252194 produced cross-entropies 0.22194392585754394, 0.2219439224243164, and 0.22194392776489258 at identical accuracy; quadratic interpolation places the minimum near +0.000000000000024162.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 77.6402839999646, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219439224243164, "validation_score": 9192.409184080238}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.00000000000002409365 will preserve all 9,192 argmax predictions and may reduce validation cross-entropy below 0.2219439224243164.
change: Move only the centered seventeenth-power confidence coefficient to the midpoint of the two co-best verified coefficients.
mechanism: Co-best midpoint heptadecic confidence calibration
evidence_used: Coefficients +0.0000000000000240253 and +0.000000000000024162 both achieved the best observed cross-entropy, 0.2219439224243164, at identical accuracy; their midpoint is the most focused probe of the unresolved optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.361273041926324, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392585754394, "validation_score": 9192.409184079088}

RECENT RESULT
hypothesis: A conservative positive nineteenth-power coefficient of +0.0000000000000030031625 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439224243164.
change: Add centered eighteenth- and nineteenth-power margin features and a positive nonadecic confidence-calibration term equal to one-eighth of the best heptadecic coefficient.
mechanism: Centered-nonadecic confidence calibration
evidence_used: Positive pentadecic calibration improved cross-entropy, successive positive heptadecic probes improved it further, and both tested hexadecic directions were worse; this motivates probing the next odd-order basis while retaining the clamped positive scale that preserves argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 69.07929624989629, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194614448547365, "validation_score": 9192.409183336153}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.0000000000000242987 will preserve all 9,192 argmax predictions and may reduce validation cross-entropy below 0.2219439224243164.
change: Move only the centered seventeenth-power confidence coefficient one co-best interval above the current value.
mechanism: Upper-neighbor heptadecic calibration probe
evidence_used: Coefficients +0.0000000000000240253 and +0.000000000000024162 achieved the best observed cross-entropy at identical accuracy; probing the equally spaced upper neighbor tests whether the optimum extends beyond the current coefficient before the known worse +0.0000000000000252194 result.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.57923174998723, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392471313476, "validation_score": 9192.409184079472}

RECENT RESULT
hypothesis: A heptadecic coefficient of +0.0000000000000238886 will preserve all 9,192 argmax predictions and may reduce validation cross-entropy below 0.2219439224243164.
change: Move only the centered seventeenth-power confidence coefficient to one co-best interval below +0.0000000000000240253.
mechanism: Lower-neighbor heptadecic calibration probe
evidence_used: Coefficients +0.0000000000000240253 and +0.000000000000024162 tied for the best observed cross-entropy, while the equally spaced upper neighbor +0.0000000000000242987 was worse; probing the corresponding lower neighbor tests the unresolved lower side of the optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.45751720806584, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392471313476, "validation_score": 9192.409184079472}



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
