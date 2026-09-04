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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 63.40453733410686, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22196999893188477, "validation_score": 9192.40917534836}
prior_hypothesis: Adding a −0.000034 centered quartic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.03433533292264, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195911331176757, "validation_score": 9192.409178993432}
prior_hypothesis: Doubling the successful centered quintic coefficient to −0.0000164664 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22195767936706542.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.33382433303632, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195767936706542, "validation_score": 9192.409179473596}
prior_hypothesis: Restoring the best verified quartic coefficient and adding a −0.0000082332 centered quintic term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22196439743041993.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.414428082993254, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22196439743041993, "validation_score": 9192.409177224026}
prior_hypothesis: A centered quartic coefficient of −0.000082332 will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22196440048217772.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Adding a −0.000034 centered quartic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22198053398132325.
change: Extend the best verified quadratic-and-cubic evaluation calibration with a centered fourth-order margin term.
mechanism: Centered quartic margin-adaptive confidence calibration
evidence_used: Successive centered polynomial extensions improved cross-entropy without changing accuracy; the optimal cubic coefficient is roughly one-tenth the quadratic coefficient, motivating a similarly scaled quartic probe after cubic-only refinement plateaued.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 63.40453733410686, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22196999893188477, "validation_score": 9192.40917534836}

RECENT RESULT
hypothesis: Doubling the successful centered quartic coefficient to −0.000068 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22196999893188477.
change: Use the best-qualified cubic coefficient and strengthen its evaluation-time centered quartic correction from −0.000034 to −0.000068.
mechanism: Second-step centered quartic confidence calibration
evidence_used: Adding a −0.000034 centered quartic term at cubic coefficient −0.00032589 reduced cross-entropy from 0.22198053398132325 to 0.22196999893188477 without changing the 9,192 correct predictions; the next equal coefficient step tests whether that improvement continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.837076583178714, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22196489181518556, "validation_score": 9192.409177058482}

RECENT RESULT
hypothesis: A centered quartic coefficient of −0.00008299 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22196489181518556.
change: Restore the best-qualified cubic coefficient and add the quartic coefficient estimated from the three verified quartic measurements.
mechanism: Interpolated quartic confidence calibration
evidence_used: Quartic coefficients 0, −0.000034, and −0.000068 produced cross-entropies 0.22198053398132325, 0.22196999893188477, and 0.22196489181518556 with identical accuracy; quadratic interpolation places the local optimum near −0.00008299.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.010114333126694, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22196440048217772, "validation_score": 9192.409177223004}

RECENT RESULT
hypothesis: A centered quartic coefficient of −0.000082332 will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22196440048217772.
change: Restore the best-qualified cubic coefficient and add the interpolated local-optimum quartic correction.
mechanism: Bracketed local quartic-coefficient calibration
evidence_used: Quartic coefficients −0.000034, −0.000068, and −0.00008299 yielded cross-entropies 0.22196999893188477, 0.22196489181518556, and 0.22196440048217772 with identical accuracy; quadratic interpolation places the minimum near −0.000082332.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 55.414428082993254, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22196439743041993, "validation_score": 9192.409177224026}

RECENT RESULT
hypothesis: A centered quartic coefficient of −0.0000817727 will preserve all 9,192 correct predictions while reducing validation cross-entropy below 0.22196439743041993.
change: Move the evaluation-time centered quartic correction from −0.000082332 to the quadratic vertex fitted from the nearest verified measurements.
mechanism: Refined local quartic-coefficient vertex calibration
evidence_used: Quartic coefficients −0.000068, −0.000082332, and −0.00008299 produced cross-entropies 0.22196489181518556, 0.22196439743041993, and 0.22196440048217772 with identical accuracy; quadratic interpolation places the minimum near −0.0000817727.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 56.498529750155285, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2219643997192383, "validation_score": 9192.40917722326}

RECENT RESULT
hypothesis: Restoring the best verified quartic coefficient and adding a −0.0000082332 centered quintic term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22196439743041993.
change: Restore the qualified −0.000082332 quartic coefficient and extend evaluation-time calibration with a quintic term one-tenth its magnitude.
mechanism: Centered quintic margin-adaptive confidence calibration
evidence_used: The −0.000082332 quartic design achieved the best verified cross-entropy with unchanged accuracy, and the preceding cubic-to-quartic extension improved cross-entropy after same-degree coefficient refinement plateaued.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.33382433303632, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195767936706542, "validation_score": 9192.409179473596}

RECENT RESULT
hypothesis: Doubling the successful centered quintic coefficient to −0.0000164664 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22195767936706542.
change: Restore the best verified quartic coefficient, compute the centered quintic margin term, and double its successful evaluation-time correction.
mechanism: Second-step centered quintic confidence calibration
evidence_used: Adding a −0.0000082332 quintic term at the best quartic coefficient reduced cross-entropy from 0.22196439743041993 to 0.22195767936706542 without changing accuracy; the analogous second-step quartic probe also improved cross-entropy substantially.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.03433533292264, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195911331176757, "validation_score": 9192.409178993432}



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
