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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.419849042082205, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194401245117187, "validation_score": 9192.409184050091}
prior_hypothesis: A pentadecic coefficient of +0.00000000000016472021446 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440155029297.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.28274358296767, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194397964477539, "validation_score": 9192.409184061078}
prior_hypothesis: Doubling the beneficial heptadecic coefficient to +0.00000000000000582747174716588 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439956665039.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.54852104187012, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392585754394, "validation_score": 9192.409184079088}
prior_hypothesis: Doubling the heptadecic coefficient to +0.00000000000002330988698866352 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439598083496.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 49.74300741683692, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194392852783204, "validation_score": 9192.409184078195}
prior_hypothesis: A heptadecic coefficient of +0.00000000000002268621107894823 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194392585754394.

## Recent verification evidence

RECENT RESULT
hypothesis: A pentadecic coefficient of +0.000000000000166798718 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440155029297.
change: Replace the current overshot pentadecic coefficient with the quadratic vertex estimated from the latest verified bracket.
mechanism: Three-point local quadratic pentadecic calibration
evidence_used: Coefficients +0.000000000000161987918, +0.000000000000162840358, and +0.000000000000171287193 produced cross-entropies 0.22194401741027833, 0.2219440155029297, and 0.22194401664733887 with identical accuracy; quadratic interpolation estimates the minimum near +0.000000000000166798718.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 40.079733999911696, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440158843994, "validation_score": 9192.409184048942}

RECENT RESULT
hypothesis: A pentadecic coefficient of +0.00000000000016472021446 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440155029297.
change: Move the centered-pentadecic confidence coefficient to the quadratic vertex estimated from the tightest verified bracket around the current optimum.
mechanism: Tight-bracket quadratic pentadecic calibration
evidence_used: Coefficients +0.000000000000161987918, +0.000000000000162840358, and +0.000000000000166798718 yielded cross-entropies 0.22194401741027833, 0.2219440155029297, and 0.2219440158843994 at identical accuracy; quadratic interpolation places the minimum near +0.00000000000016472021446.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.419849042082205, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194401245117187, "validation_score": 9192.409184050091}

RECENT RESULT
hypothesis: Adding a negative hexadecic coefficient of −0.00000000000002190778852318 to the best verified pentadecic design will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194401245117187.
change: Restore the best verified pentadecic coefficient, derive the centered sixteenth-power margin feature, and apply an initial opposite-sign coefficient equal to 0.133 times the pentadecic magnitude.
mechanism: Initial centered-hexadecic confidence calibration
evidence_used: The +0.00000000000016472021446 pentadecic coefficient achieved the best observed cross-entropy at unchanged accuracy; the preceding higher-order transition improved only after reversing sign, and 0.133 was an effective initial magnitude ratio.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.782891042064875, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194401321411134, "validation_score": 9192.409184049837}

RECENT RESULT
hypothesis: Adding a positive hexadecic coefficient of +0.00000000000002190778852318 to the best pentadecic design will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22194401245117187.
change: Restore the best verified pentadecic coefficient, derive the centered sixteenth-power margin feature, and test the opposite sign of the negative hexadecic probe.
mechanism: Sign-reversed centered-hexadecic confidence calibration
evidence_used: The best pentadecic design achieved 0.22194401245117187 cross-entropy; adding the negative hexadecic coefficient worsened it to 0.22194401321411134 at identical accuracy, so the equal-magnitude positive direction is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 68.08700474980287, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194404067993165, "validation_score": 9192.409184040638}

RECENT RESULT
hypothesis: A hexadecic coefficient of −0.000000000000010377373499 on the best pentadecic design will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194401245117187.
change: Restore the best verified pentadecic coefficient, derive the centered sixteenth-power margin feature, and apply the quadratic minimum estimated from the positive, zero, and negative hexadecic probes.
mechanism: Quadratic-vertex centered-hexadecic confidence calibration
evidence_used: Hexadecic coefficients −0.00000000000002190778852318, 0, and +0.00000000000002190778852318 produced cross-entropies 0.22194401321411134, 0.22194401245117187, and 0.22194404067993165 at identical accuracy; quadratic interpolation places the minimum near −0.000000000000010377373499.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.42211274988949, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194401473999023, "validation_score": 9192.409184049326}

RECENT RESULT
hypothesis: A pentadecic coefficient of +0.000000000000164761301515 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194401245117187.
change: Move the centered-pentadecic confidence coefficient to the quadratic vertex fitted from the closest verified points bracketing the current optimum.
mechanism: Tight-bracket refitted pentadecic confidence calibration
evidence_used: Coefficients +0.000000000000162840358, +0.00000000000016472021446, and +0.000000000000166798718 produced cross-entropies 0.2219440155029297, 0.22194401245117187, and 0.2219440158843994 at identical accuracy; quadratic interpolation estimates a slightly higher optimum near +0.000000000000164761301515.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 68.85891833319329, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440155029297, "validation_score": 9192.40918404907}

RECENT RESULT
hypothesis: A positive heptadecic coefficient of +0.00000000000000291373587358294 on the best verified pentadecic design will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194401245117187.
change: Restore the best pentadecic-only calibration, remove the unsuccessful hexadecic term, and add a bounded positive centered seventeenth-power margin feature.
mechanism: Centered-heptadecic confidence calibration
evidence_used: The pentadecic-only design achieved the best observed cross-entropy, while both tested hexadecic directions were worse; the positive pentadecic direction was beneficial, motivating a smaller same-sign probe in the next odd-order basis.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.42737854202278, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219439956665039, "validation_score": 9192.409184055712}

RECENT RESULT
hypothesis: Doubling the beneficial heptadecic coefficient to +0.00000000000000582747174716588 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219439956665039.
change: Remove the unsuccessful hexadecic calibration term, derive the centered seventeenth-power margin feature, and test twice the best verified positive heptadecic coefficient.
mechanism: One-sided centered-heptadecic calibration continuation
evidence_used: The pentadecic-only design achieved 0.22194401245117187 cross-entropy, while adding +0.00000000000000291373587358294 heptadecic calibration improved it to the best observed 0.2219439956665039 at identical accuracy; continuing in that demonstrated beneficial direction is the most informative next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.28274358296767, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194397964477539, "validation_score": 9192.409184061078}

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
