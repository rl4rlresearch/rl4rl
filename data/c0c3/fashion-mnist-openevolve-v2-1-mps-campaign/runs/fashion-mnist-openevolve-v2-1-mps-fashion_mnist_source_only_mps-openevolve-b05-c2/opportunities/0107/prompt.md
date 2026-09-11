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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.007795916870236, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22199077491760255, "validation_score": 9192.409168391663}
prior_hypothesis: Adding a small centered cubic correction will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22200197410583497.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 70.88258283305913, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053512573243, "validation_score": 9192.409171820358}
prior_hypothesis: A centered cubic coefficient of −0.00032515 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22198066902160646.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 50.1875938330777, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22200976676940917, "validation_score": 9192.409162032576}
prior_hypothesis: Doubling the verified quadratic coefficient to −0.002 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.2220299663543701.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 38.453026624862105, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198370323181152, "validation_score": 9192.409170759542}
prior_hypothesis: Doubling the successful centered cubic coefficient to −0.0002 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22199077491760255.

## Recent verification evidence

RECENT RESULT
hypothesis: Using −0.00311057 will preserve all 9,192 correct predictions and reduce validation cross-entropy below the current 0.22200976676940917.
change: Replace the evaluation-time centered quadratic coefficient −0.002 with the best verified coefficient, −0.00311057.
mechanism: Locally optimized quadratic confidence calibration
evidence_used: Reference Design 2 directly verified 9,192 correct predictions and 0.22200197639465333 cross-entropy with this coefficient, outperforming the current design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the centered quadratic coefficient to −0.00311057 will preserve all 9,192 correct predictions and reduce validation cross-entropy below the current 0.2220299663543701.
change: Replace the current −0.001 evaluation-time quadratic calibration coefficient with the best verified coefficient, −0.00311057.
mechanism: Best-verified quadratic confidence calibration
evidence_used: Reference Design 2 verified 9,192 correct predictions and 0.22200197639465333 cross-entropy at −0.00311057, outperforming the current design while leaving argmax predictions unchanged.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A centered quadratic coefficient of −0.00311336 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22200197639465333.
change: Move the evaluation-time quadratic calibration coefficient from −0.00311057 to the local optimum fitted using the newest verification result.
mechanism: Updated quadratic-vertex confidence calibration
evidence_used: Coefficients −0.003, −0.00311057, and −0.003125 yielded cross-entropies 0.22200205307006837, 0.22200197639465333, and 0.22200197715759276 with identical accuracy; quadratic interpolation places the updated minimum near −0.00311336.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.379762082826346, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22200197410583497, "validation_score": 9192.40916464179}

RECENT RESULT
hypothesis: A centered quadratic coefficient of −0.00311743 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22200197410583497.
change: Move the evaluation-time centered quadratic confidence-calibration coefficient from −0.003125 to the vertex fitted from the three closest verified coefficients.
mechanism: Updated local quadratic-vertex calibration
evidence_used: Coefficients −0.00311057, −0.00311336, and −0.003125 produced cross-entropies 0.22200197639465333, 0.22200197410583497, and 0.22200197715759276 with identical accuracy; quadratic interpolation places the minimum near −0.00311743.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.86822441709228, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2220019790649414, "validation_score": 9192.409164640128}

RECENT RESULT
hypothesis: Using the qualified −0.00311336 coefficient will preserve all 9,192 correct predictions and reduce validation cross-entropy from 0.22200976676940917 to approximately 0.22200197410583497.
change: Replace the current −0.002 centered quadratic confidence-calibration coefficient with −0.00311336.
mechanism: Best-qualified quadratic confidence calibration
evidence_used: Reference Design 2 verified 9,192 correct predictions and the best reported cross-entropy, 0.22200197410583497, using this exact coefficient.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing −0.001 with the qualified −0.00311336 coefficient will preserve all 9,192 correct predictions and reduce validation cross-entropy from 0.2220299663543701 to approximately 0.22200197410583497.
change: Strengthen the evaluation-time centered quadratic margin correction while leaving training and argmax predictions unchanged.
mechanism: Best-qualified quadratic confidence calibration
evidence_used: Reference Design 2 verified this exact coefficient with 9,192 correct predictions and the best qualified cross-entropy, 0.22200197410583497.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a small centered cubic correction will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22200197410583497.
change: Extend the optimized quadratic evaluation-time calibration with a −0.0001 centered cubic margin term.
mechanism: Third-order margin-adaptive confidence calibration
evidence_used: Centered quadratic calibration reduced cross-entropy from 0.22206240844726563 to 0.22200197410583497 without changing accuracy, while further quadratic-coefficient refinement did not improve; testing the next polynomial term is the most direct extension of the successful mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.007795916870236, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22199077491760255, "validation_score": 9192.409168391663}

RECENT RESULT
hypothesis: Doubling the successful centered cubic coefficient to −0.0002 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22199077491760255.
change: Use the best-qualified quadratic coefficient and strengthen its evaluation-time centered cubic correction from −0.0001 to −0.0002.
mechanism: Stronger third-order margin-adaptive confidence calibration
evidence_used: Adding a −0.0001 centered cubic term at quadratic coefficient −0.00311336 improved cross-entropy from 0.22200197410583497 to 0.22199077491760255 without changing the 9,192 correct predictions; the next equal coefficient step tests whether that gain continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 38.453026624862105, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198370323181152, "validation_score": 9192.409170759542}

RECENT RESULT
hypothesis: Using the qualified quadratic coefficient with a −0.0003 centered cubic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198370323181152.
change: Replace the current quadratic-only evaluation calibration with the best-qualified quadratic coefficient and the next equally spaced centered cubic correction.
mechanism: Third-step centered cubic confidence calibration
evidence_used: Cubic coefficients 0, −0.0001, and −0.0002 yielded cross-entropies 0.22200197410583497, 0.22199077491760255, and 0.22198370323181152 with identical accuracy; the continuing diminishing improvement motivates testing −0.0003.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A −0.0003 centered cubic term will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22198370323181152.
change: Replace the current weak quadratic-only calibration with the best-qualified quadratic coefficient and the next centered cubic coefficient.
mechanism: Third-step centered cubic confidence calibration
evidence_used: Cubic coefficients 0, −0.0001, and −0.0002 successively reduced cross-entropy with identical accuracy; the prior −0.0003 verification timed out without producing contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 72.66229966608807, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198066902160646, "validation_score": 9192.409171775524}

RECENT RESULT
hypothesis: A centered cubic coefficient of −0.00032515 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22198066902160646.
change: Move the evaluation-time centered cubic correction from −0.0003 to the interpolated local optimum, −0.00032515.
mechanism: Local cubic-coefficient vertex calibration
evidence_used: Cubic coefficients −0.0001, −0.0002, and −0.0003 produced cross-entropies 0.22199077491760255, 0.22198370323181152, and 0.22198066902160646 with identical accuracy; quadratic interpolation of these equally spaced results places the minimum near −0.00032515.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 70.88258283305913, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053512573243, "validation_score": 9192.409171820358}

RECENT RESULT
hypothesis: A centered cubic coefficient of −0.00032589 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22198053512573243.
change: Move the evaluation-time centered cubic correction from −0.0002 to the updated interpolated optimum, −0.00032589.
mechanism: Updated local cubic-coefficient vertex calibration
evidence_used: Cubic coefficients −0.0002, −0.0003, and −0.00032515 produced cross-entropies 0.22198370323181152, 0.22198066902160646, and 0.22198053512573243 with identical accuracy; quadratic interpolation of these closest measurements places the minimum near −0.00032589.
result: training did not finish within the verification time limit



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
