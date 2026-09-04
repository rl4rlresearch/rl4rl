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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.560465291840956, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194403190612794, "validation_score": 9192.409184043578}
prior_hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000963557349529824 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194403381347655.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.451280457898974, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194404106140136, "validation_score": 9192.40918404051}
prior_hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000535309638627680 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.221944047164917.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.09965983382426, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440341949463, "validation_score": 9192.40918404281}
prior_hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000749433494078752 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194403610229493.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 45.42974808393046, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194403610229493, "validation_score": 9192.409184042172}
prior_hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000642371566353216 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194404106140136.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a centered tetradecic coefficient of −0.000000000000107061927725536 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440673828125.
change: Derive the centered fourteenth-power margin feature and add a conservative residual calibration term while retaining every optimized lower-order coefficient.
mechanism: Fourteenth-order margin confidence calibration
evidence_used: The optimized tridecic coefficient −0.000000000000804976900192 produced the best verified cross-entropy at unchanged accuracy; the established initial higher-order probe ratio of approximately 0.133 motivates a tetradecic coefficient of −0.000000000000107061927725536.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.729267792077735, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194406433105468, "validation_score": 9192.409184032718}

RECENT RESULT
hypothesis: Doubling the centered tetradecic coefficient to −0.000000000000214123855451072 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194406433105468.
change: Restore the optimized tridecic term, derive the centered fourteenth-power margin feature, and continue one equal-sized tetradecic step beyond the successful initial probe.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: Adding a tetradecic coefficient of −0.000000000000107061927725536 reduced cross-entropy from 0.2219440673828125 to 0.22194406433105468 at identical accuracy; prior higher-order searches use an equal-step continuation after a successful initial probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 45.58789570815861, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194405670166015, "validation_score": 9192.409184035274}

RECENT RESULT
hypothesis: Setting the centered tetradecic coefficient to −0.000000000000321185783176608 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194405670166015.
change: Restore the optimized tridecic coefficient, derive the centered fourteenth-power margin feature, and advance the tetradecic coefficient by one more equal-sized negative step.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: Moving the tetradecic coefficient from −0.000000000000107061927725536 to −0.000000000000214123855451072 improved cross-entropy from 0.22194406433105468 to 0.22194405670166015 with unchanged accuracy, so another equal step is the most informative continuation probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 33.92788129206747, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440544128418, "validation_score": 9192.40918403604}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000428247710902144 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440544128418.
change: Restore the optimized tridecic coefficient, add the centered fourteenth-power margin feature, and continue the tetradecic coefficient by one equal-sized negative step.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: Moving the tetradecic coefficient from −0.000000000000214123855451072 to −0.000000000000321185783176608 reduced cross-entropy from 0.22194405670166015 to 0.2219440544128418 without changing accuracy, motivating one further continuation probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.713741834042594, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.221944047164917, "validation_score": 9192.409184038468}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000535309638627680 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.221944047164917.
change: Continue the tetradecic coefficient by one equal-sized negative step while retaining all optimized lower-order calibration terms.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: The last equal step, from −0.000000000000321185783176608 to −0.000000000000428247710902144, reduced cross-entropy from 0.2219440544128418 to 0.221944047164917 without changing accuracy.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.451280457898974, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194404106140136, "validation_score": 9192.40918404051}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000642371566353216 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194404106140136.
change: Continue the tetradecic coefficient by one equal-sized negative step while retaining all optimized lower-order calibration terms.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: Five consecutive equal negative tetradecic steps improved cross-entropy without changing accuracy; the latest step reached 0.22194404106140136 at coefficient −0.000000000000535309638627680.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 45.42974808393046, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194403610229493, "validation_score": 9192.409184042172}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000749433494078752 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194403610229493.
change: Continue the tetradecic coefficient by one equal-sized negative step while retaining all optimized lower-order calibration terms.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: Six consecutive equal negative tetradecic steps improved cross-entropy without changing accuracy; the latest verified step reached 0.22194403610229493 at coefficient −0.000000000000642371566353216.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.09965983382426, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219440341949463, "validation_score": 9192.40918404281}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000856495421804288 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2219440341949463.
change: Restore the best verified tetradecic coefficient and continue it by one equal-sized negative step while retaining all lower-order calibration terms.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: Seven consecutive equal negative tetradecic steps improved cross-entropy without changing accuracy; the latest reached 0.2219440341949463 at −0.000000000000749433494078752.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 65.74467341601849, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194403381347655, "validation_score": 9192.409184042937}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000000963557349529824 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194403381347655.
change: Continue the tetradecic coefficient by one equal-sized negative step while retaining every lower-order calibration term.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: The previous step from −0.000000000000749433494078752 to −0.000000000000856495421804288 reduced cross-entropy from 0.2219440341949463 to 0.22194403381347655 without changing accuracy, so one further equal step is the most informative continuation probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.560465291840956, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194403190612794, "validation_score": 9192.409184043578}

RECENT RESULT
hypothesis: Advancing the centered tetradecic coefficient to −0.000000000001070619277255360 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194403190612794.
change: Restore the best verified tetradecic coefficient and continue it by one equal-sized negative step while retaining every lower-order calibration term.
mechanism: Directional tetradecic confidence-calibration continuation
evidence_used: The latest equal step, from −0.000000000000856495421804288 to −0.000000000000963557349529824, reduced cross-entropy from 0.22194403381347655 to 0.22194403190612794 without changing accuracy, supporting one further continuation probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.21048862510361, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.221944034576416, "validation_score": 9192.409184042683}

RECENT RESULT
hypothesis: A centered tetradecic coefficient of −0.000000000000954635522219363 will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22194403190612794.
change: Move the tetradecic coefficient from the current continuation value to the quadratic minimum estimated from the three verified coefficients bracketing the optimum.
mechanism: Quadratic-vertex tetradecic confidence calibration
evidence_used: Coefficients −0.000000000000856495421804288, −0.000000000000963557349529824, and −0.000000000001070619277255360 yielded cross-entropies 0.22194403381347655, 0.22194403190612794, and 0.221944034576416 at identical accuracy; quadratic interpolation places the minimum near −0.000000000000954635522219363.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.0468390409369, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194403800964355, "validation_score": 9192.409184041533}

RECENT RESULT
hypothesis: Restoring the best verified tetradecic coefficient and adding a pentadecic coefficient of −0.000000000000128153127487466592 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22194403190612794.
change: Restore the best verified tetradecic calibration, derive the centered fifteenth-power margin feature, and apply an initial negative coefficient equal to 0.133 times the tetradecic magnitude.
mechanism: Initial centered-pentadecic confidence-calibration probe
evidence_used: The tetradecic coefficient −0.000000000000963557349529824 achieved the best verified cross-entropy at unchanged accuracy, while the previously successful initial higher-order probe used an approximately 0.133 coefficient ratio.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 73.77426483412273, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22194408493041992, "validation_score": 9192.40918402582}



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
