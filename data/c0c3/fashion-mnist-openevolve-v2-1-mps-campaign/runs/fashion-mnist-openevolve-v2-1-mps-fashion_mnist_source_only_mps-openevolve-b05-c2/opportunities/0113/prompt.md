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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.83735750010237, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053398132325, "validation_score": 9192.409171820742}
prior_hypothesis: A centered cubic coefficient of −0.00032589 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22198053512573243.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 70.88258283305913, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053512573243, "validation_score": 9192.409171820358}
prior_hypothesis: A centered cubic coefficient of −0.00032515 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22198066902160646.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.44304508389905, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198054237365722, "validation_score": 9192.409171817932}
prior_hypothesis: A centered cubic coefficient of −0.00033082 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 38.453026624862105, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198370323181152, "validation_score": 9192.409170759542}
prior_hypothesis: Doubling the successful centered cubic coefficient to −0.0002 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22199077491760255.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A centered cubic coefficient of −0.00032589 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22198053512573243.
change: Replace the current −0.0001 centered cubic correction with the interpolated optimum, −0.00032589.
mechanism: Updated local cubic-coefficient calibration
evidence_used: Coefficients −0.0002, −0.0003, and −0.00032515 produced cross-entropies 0.22198370323181152, 0.22198066902160646, and 0.22198053512573243 with identical accuracy; interpolation places the minimum near −0.00032589, and its previous verification only timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.83735750010237, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053398132325, "validation_score": 9192.409171820742}

RECENT RESULT
hypothesis: A centered cubic coefficient of −0.00033082 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.
change: Replace the current quadratic-only calibration with the qualified quadratic coefficient and the newly interpolated cubic optimum.
mechanism: Updated local cubic-coefficient vertex calibration
evidence_used: Cubic coefficients −0.0003, −0.00032515, and −0.00032589 produced cross-entropies 0.22198066902160646, 0.22198053512573243, and 0.22198053398132325 with identical accuracy; quadratic interpolation of these newest local measurements places the minimum near −0.00033082.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.44304508389905, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198054237365722, "validation_score": 9192.409171817932}

RECENT RESULT
hypothesis: A centered cubic coefficient of −0.00032687 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.
change: Move the evaluation-time centered cubic correction from −0.00032589 to the updated local optimum, −0.00032687.
mechanism: Refined local cubic-coefficient vertex calibration
evidence_used: Coefficients −0.00032515, −0.00032589, and −0.00033082 produced cross-entropies 0.22198053512573243, 0.22198053398132325, and 0.22198054237365722 with identical accuracy; quadratic interpolation of these nearest measurements places the minimum near −0.00032687.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.12213333300315, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053665161133, "validation_score": 9192.409171819847}

RECENT RESULT
hypothesis: A centered cubic coefficient of −0.0003258314 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.
change: Move the evaluation-time centered cubic correction from −0.00032515 to the quadratic vertex fitted around the best verified coefficient.
mechanism: Bracketed local cubic-coefficient calibration
evidence_used: Coefficients −0.00032515, −0.00032589, and −0.00032687 produced cross-entropies 0.22198053512573243, 0.22198053398132325, and 0.22198053665161133 with identical accuracy; local interpolation places the minimum near −0.0003258314.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.8940243748948, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22198053512573243, "validation_score": 9192.409171820358}

RECENT RESULT
hypothesis: Restoring the verified −0.00032589 centered cubic coefficient will preserve all 9,192 correct predictions and reduce validation cross-entropy from 0.22198054237365722 to approximately 0.22198053398132325.
change: Replace the current overshot cubic calibration coefficient with the best verified coefficient.
mechanism: Best-qualified cubic confidence calibration
evidence_used: Reference Design 2 achieved the highest reported validation score and lowest cross-entropy at −0.00032589; nearby coefficients −0.0003258314, −0.00032687, and −0.00033082 were all non-improvements.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the centered cubic coefficient to −0.00032589 will preserve all 9,192 correct predictions while reducing validation cross-entropy from 0.22198370323181152 to approximately 0.22198053398132325.
change: Replace the current −0.0002 evaluation-time cubic calibration coefficient with the best verified coefficient, −0.00032589.
mechanism: Best-qualified cubic confidence calibration
evidence_used: Reference Design 3 verified −0.00032589 with 9,192 correct predictions and the lowest observed cross-entropy, 0.22198053398132325; nearby coefficients were non-improvements, while the latest restoration attempt produced no contrary result because it could not be verified.
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
