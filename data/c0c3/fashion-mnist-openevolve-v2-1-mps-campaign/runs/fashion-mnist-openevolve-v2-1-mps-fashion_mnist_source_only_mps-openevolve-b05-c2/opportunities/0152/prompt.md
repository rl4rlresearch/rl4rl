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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.28520108410157, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194598770141602, "validation_score": 9192.409183388654}
prior_hypothesis: Doubling the centered nonic coefficient to −0.0000000202 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194476356506349.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.88432174990885, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194422760009766, "validation_score": 9192.409183978047}
prior_hypothesis: A centered decic coefficient of −0.000000001716523096 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194428329467775.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.63408099999651, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194429321289064, "validation_score": 9192.409183956075}
prior_hypothesis: Adding a centered decic coefficient of −0.000000001127 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194473266601564.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.606263166991994, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194428329467775, "validation_score": 9192.409183959397}
prior_hypothesis: Doubling the centered decic coefficient to −0.000000002254 while restoring the optimized nonic coefficient will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194429321289064.

## Recent verification evidence

RECENT RESULT
hypothesis: A centered septic coefficient of −0.000000354 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219506046295166.
change: Restore the best verified quintic and sextic calibration, then add a conservative centered septic residual term.
mechanism: Seventh-order margin confidence calibration
evidence_used: Sextic refinement reduced cross-entropy to 0.2219506046295166 without changing accuracy; the similarly scaled transition from exhausted quintic calibration to a higher-order term previously produced a material improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.48878745897673, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194767913818358, "validation_score": 9192.409182822257}

RECENT RESULT
hypothesis: Doubling the centered septic coefficient to −0.000000708 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194767913818358.
change: Continue one equal-sized step beyond the verified −0.000000354 septic coefficient while retaining all lower-order calibration terms.
mechanism: Directional septic calibration continuation
evidence_used: Introducing −0.000000354 reduced cross-entropy from 0.2219506046295166 to 0.22194767913818358 with unchanged accuracy; the successful sextic sequence shows that an equal-step continuation is the most informative next directional probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.77858674991876, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219504325866699, "validation_score": 9192.409181900235}

RECENT RESULT
hypothesis: A centered septic coefficient of −0.000000359362196 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194767913818358.
change: Add the centered septic margin feature and set its coefficient to the quadratic optimum interpolated from the verified zero, −0.000000354, and −0.000000708 results.
mechanism: Quadratic-vertex septic confidence calibration
evidence_used: The septic coefficients 0, −0.000000354, and −0.000000708 produced cross-entropies 0.2219506046295166, 0.22194767913818358, and 0.2219504325866699 respectively, placing the fitted minimum slightly beyond the best verified coefficient.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.89732729201205, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194767456054687, "validation_score": 9192.40918282379}

RECENT RESULT
hypothesis: Adding a centered octic coefficient of −0.0000000477 to the best verified lower-order calibration will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194767456054687.
change: Restore the best verified sextic and septic coefficients, then add a conservative centered-octic residual term.
mechanism: Eighth-order margin confidence calibration
evidence_used: The optimized septic term reduced cross-entropy from 0.2219506046295166 to 0.22194767456054687 without changing accuracy; the successful initial sextic and septic coefficients were each approximately 0.133 times the preceding coefficient, motivating an octic probe of the same relative scale.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 49.02360504097305, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194570770263672, "validation_score": 9192.409183482416}

RECENT RESULT
hypothesis: Doubling the centered octic coefficient to −0.0000000954 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194570770263672.
change: Restore the best verified sextic and septic coefficients, add the centered-octic feature, and continue one equal-sized octic step beyond the successful −0.0000000477 probe.
mechanism: Directional octic calibration continuation
evidence_used: Adding −0.0000000477 octic calibration reduced cross-entropy from 0.22194767456054687 to 0.22194570770263672 without changing accuracy; an equal-step continuation is the most informative directional probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.13052662508562, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194553794860838, "validation_score": 9192.40918353926}

RECENT RESULT
hypothesis: An octic coefficient of −0.000000076055785 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22194553794860838.
change: Move the evaluation-time centered-octic coefficient from the continued probe to the quadratic optimum interpolated from the verified coefficients 0, −0.0000000477, and −0.0000000954.
mechanism: Quadratic-vertex octic confidence calibration
evidence_used: The three octic coefficients produced cross-entropies 0.22194767456054687, 0.22194570770263672, and 0.22194553794860838 at identical accuracy; their diminishing improvement places the fitted minimum near −0.000000076055785.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.68580429186113, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194539108276368, "validation_score": 9192.40918358844}

RECENT RESULT
hypothesis: Restoring the optimized octic coefficient and adding a centered nonic coefficient of −0.0000000101 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194539108276368.
change: Use the best verified octic coefficient, derive the centered-nonic margin feature, and apply a conservative nonic residual correction.
mechanism: Ninth-order margin confidence calibration
evidence_used: The optimized octic coefficient −0.000000076055785 achieved the best available cross-entropy, 0.22194539108276368. Successful initial sextic, septic, and octic coefficients were each about 0.133 times the preceding optimized coefficient, motivating a nonic probe near −0.0000000101.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.96359783294611, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194476356506349, "validation_score": 9192.40918379857}

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
