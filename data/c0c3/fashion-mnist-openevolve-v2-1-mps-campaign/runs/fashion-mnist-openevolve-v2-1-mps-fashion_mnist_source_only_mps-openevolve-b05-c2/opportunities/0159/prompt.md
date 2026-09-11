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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.20081895892508, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219441982269287, "validation_score": 9192.409183987882}
prior_hypothesis: A centered undecic coefficient of −0.0000000001178008529 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194422187805177.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.76188237499446, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194422187805177, "validation_score": 9192.409183979962}
prior_hypothesis: Adding a centered undecic coefficient of −0.0000000002283 to the best verified decic design will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194422760009766.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.426200416870415, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194406814575196, "validation_score": 9192.409184031441}
prior_hypothesis: Setting the centered duodecic coefficient to −0.0000000000421373651 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440757751465.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.19061870803125, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194410552978516, "validation_score": 9192.409184018923}
prior_hypothesis: Adding a centered tridecic coefficient of −0.0000000000056042695583 to the best verified duodecic design will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194406814575196.

## Recent verification evidence

RECENT RESULT
hypothesis: Doubling the centered nonic coefficient to −0.0000000202 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194476356506349.
change: Restore the best verified octic calibration, derive the centered-nonic margin feature, and continue one equal-sized nonic step beyond the successful −0.0000000101 probe.
mechanism: Directional nonic calibration continuation
evidence_used: Adding −0.0000000101 nonic calibration reduced cross-entropy from 0.22194539108276368 to 0.22194476356506349 with unchanged accuracy; the successful sextic, septic, and octic searches used an equal-step continuation as the next directional probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.28520108410157, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194598770141602, "validation_score": 9192.409183388654}

RECENT RESULT
hypothesis: A centered nonic coefficient of −0.00000000847284714 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194476356506349.
change: Restore the optimized septic and octic terms, add the centered-nonic feature, and set its coefficient to the quadratic optimum fitted from the three verified nonic probes.
mechanism: Quadratic-vertex nonic confidence calibration
evidence_used: Nonic coefficients 0, −0.0000000101, and −0.0000000202 produced cross-entropies 0.22194539108276368, 0.22194476356506349, and 0.22194598770141602 at identical accuracy; quadratic interpolation places the minimum near −0.00000000847284714.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.909013958182186, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194473266601564, "validation_score": 9192.409183808919}

RECENT RESULT
hypothesis: Adding a centered decic coefficient of −0.000000001127 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194473266601564.
change: Retain the optimized lower-order calibration and add a conservative centered tenth-power margin correction.
mechanism: Tenth-order margin confidence calibration
evidence_used: Each new sextic through nonic term improved cross-entropy at unchanged accuracy, with successful initial coefficients approximately 0.133 times the preceding optimized coefficient; applying that ratio to −0.00000000847284714 motivates −0.000000001127.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.63408099999651, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194429321289064, "validation_score": 9192.409183956075}

RECENT RESULT
hypothesis: Doubling the centered decic coefficient to −0.000000002254 while restoring the optimized nonic coefficient will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194429321289064.
change: Restore the best verified nonic coefficient, derive the centered tenth-power margin feature, and continue one equal-sized decic step beyond the successful −0.000000001127 probe.
mechanism: Directional decic confidence-calibration continuation
evidence_used: Adding a decic coefficient of −0.000000001127 reduced cross-entropy from 0.22194473266601564 to 0.22194429321289064 without changing accuracy; equal-step continuation was the informative next probe for each preceding polynomial calibration term.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.606263166991994, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194428329467775, "validation_score": 9192.409183959397}

RECENT RESULT
hypothesis: A centered decic coefficient of −0.000000001716523096 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194428329467775.
change: Restore the optimized nonic term and set the decic term to the quadratic optimum interpolated from the three verified decic coefficients.
mechanism: Quadratic-vertex decic confidence calibration
evidence_used: Decic coefficients 0, −0.000000001127, and −0.000000002254 produced cross-entropies 0.22194473266601564, 0.22194429321289064, and 0.22194428329467775 with identical accuracy; quadratic interpolation places the minimum near −0.000000001716523096.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.88432174990885, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194422760009766, "validation_score": 9192.409183978047}

RECENT RESULT
hypothesis: Adding a centered undecic coefficient of −0.0000000002283 to the best verified decic design will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194422760009766.
change: Restore the optimized nonic and decic coefficients, derive the centered eleventh-power margin feature, and apply a conservative undecic residual correction.
mechanism: Eleventh-order margin confidence calibration
evidence_used: The optimized decic coefficient −0.000000001716523096 achieved the best available cross-entropy at unchanged accuracy; successful initial higher-order coefficients have been approximately 0.133 times the preceding optimized coefficient, motivating an undecic probe near −0.0000000002283.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.76188237499446, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194422187805177, "validation_score": 9192.409183979962}

RECENT RESULT
hypothesis: Doubling the centered undecic coefficient to −0.0000000004566 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194422187805177.
change: Continue one equal-sized undecic step beyond the successful initial coefficient while retaining all optimized lower-order calibration terms.
mechanism: Directional undecic confidence-calibration continuation
evidence_used: Adding −0.0000000002283 improved cross-entropy from 0.22194422760009766 to 0.22194422187805177 without changing accuracy; an equal-step continuation is the established next probe for each higher-order term.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.149762499844655, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194457397460937, "validation_score": 9192.409183862059}

RECENT RESULT
hypothesis: A centered undecic coefficient of −0.0000000001178008529 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194422187805177.
change: Add the centered eleventh-power margin feature to the optimized decic design and set its coefficient to the quadratic optimum fitted from the three verified undecic probes.
mechanism: Quadratic-vertex undecic confidence calibration
evidence_used: Undecic coefficients 0, −0.0000000002283, and −0.0000000004566 produced cross-entropies 0.22194422760009766, 0.22194422187805177, and 0.22194457397460937 at identical accuracy; quadratic interpolation places the minimum near −0.0000000001178008529.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.20081895892508, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219441982269287, "validation_score": 9192.409183987882}

RECENT RESULT
hypothesis: Adding a centered duodecic coefficient of −0.00000000001566751344 to the best verified undecic design will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.2219441982269287.
change: Restore the optimized decic and undecic coefficients, derive the centered twelfth-power margin feature, and apply a conservative duodecic residual correction.
mechanism: Twelfth-order margin confidence calibration
evidence_used: The optimized undecic coefficient −0.0000000001178008529 produced the best verified cross-entropy at unchanged accuracy; successful initial higher-order coefficients have been approximately 0.133 times the preceding optimized coefficient, motivating the proposed duodecic coefficient.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.53865258395672, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219441188812256, "validation_score": 9192.409184014452}

RECENT RESULT
hypothesis: Doubling the centered duodecic coefficient to −0.00000000003133502688 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219441188812256.
change: Restore the optimized decic and undecic terms, add the centered twelfth-power margin feature, and continue one equal-sized duodecic step beyond the successful initial probe.
mechanism: Directional duodecic confidence-calibration continuation
evidence_used: Adding a duodecic coefficient of −0.00000000001566751344 reduced cross-entropy from 0.2219441982269287 to 0.2219441188812256 at identical accuracy; an equal-step continuation is the most informative next probe before quadratic interpolation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 61.61346745793708, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440757751465, "validation_score": 9192.409184028887}

RECENT RESULT
hypothesis: Setting the centered duodecic coefficient to −0.0000000000421373651 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440757751465.
change: Move the duodecic coefficient from the continued probe to the quadratic optimum interpolated from the three verified duodecic coefficients.
mechanism: Quadratic-vertex duodecic confidence calibration
evidence_used: Coefficients 0, −0.00000000001566751344, and −0.00000000003133502688 yielded cross-entropies 0.2219441982269287, 0.2219441188812256, and 0.2219440757751465 at identical accuracy, placing the fitted minimum near −0.0000000000421373651.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.426200416870415, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194406814575196, "validation_score": 9192.409184031441}

RECENT RESULT
hypothesis: Adding a centered tridecic coefficient of −0.0000000000056042695583 to the best verified duodecic design will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194406814575196.
change: Restore the optimized duodecic coefficient, derive the centered thirteenth-power margin feature, and apply a conservative tridecic residual correction.
mechanism: Thirteenth-order margin confidence calibration
evidence_used: The optimized duodecic coefficient −0.0000000000421373651 achieved the best verified cross-entropy, 0.22194406814575196, at unchanged accuracy; successful initial higher-order coefficients have been approximately 0.133 times the preceding optimized coefficient, motivating −0.0000000000056042695583.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.19061870803125, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194410552978516, "validation_score": 9192.409184018923}



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
