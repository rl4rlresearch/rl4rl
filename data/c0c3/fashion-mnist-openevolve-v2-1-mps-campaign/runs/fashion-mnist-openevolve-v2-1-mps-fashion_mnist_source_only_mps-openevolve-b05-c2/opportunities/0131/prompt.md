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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.06535691698082, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726585388184, "validation_score": 9192.409179612063}
prior_hypothesis: A centered quintic coefficient of −0.000010901574 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195767936706542.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.6695165829733, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726432800292, "validation_score": 9192.409179612574}
prior_hypothesis: A centered quintic coefficient of −0.00001094982902 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.55261874990538, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726203918456, "validation_score": 9192.40917961334}
prior_hypothesis: A centered quintic coefficient of −0.00001092570151 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726585388184.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 40.07324216607958, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726737976074, "validation_score": 9192.409179611552}
prior_hypothesis: A centered quintic coefficient of −0.00001092871744875 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.

## Recent verification evidence

RECENT RESULT
hypothesis: Doubling the successful centered quintic coefficient to −0.0000164664 will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22195767936706542.
change: Restore the best verified quartic coefficient, compute the centered quintic margin term, and double its successful evaluation-time correction.
mechanism: Second-step centered quintic confidence calibration
evidence_used: Adding a −0.0000082332 quintic term at the best quartic coefficient reduced cross-entropy from 0.22196439743041993 to 0.22195767936706542 without changing accuracy; the analogous second-step quartic probe also improved cross-entropy substantially.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.03433533292264, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195911331176757, "validation_score": 9192.409178993432}

RECENT RESULT
hypothesis: A centered quintic coefficient of −0.000010901574 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195767936706542.
change: Restore the best verified quartic coefficient and add the quintic coefficient at the quadratic vertex fitted from the three verified quintic measurements.
mechanism: Interpolated quintic confidence calibration
evidence_used: Quintic coefficients 0, −0.0000082332, and −0.0000164664 produced cross-entropies 0.22196439743041993, 0.22195767936706542, and 0.22195911331176757 with identical accuracy; quadratic interpolation places the minimum near −0.000010901574.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 46.06535691698082, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726585388184, "validation_score": 9192.409179612063}

RECENT RESULT
hypothesis: Adding a −0.0000010901574 centered sextic term will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726585388184.
change: Extend the best verified evaluation-time polynomial calibration with a centered sixth-order margin term one-tenth the magnitude of the quintic coefficient.
mechanism: Centered sextic margin-adaptive confidence calibration
evidence_used: Adding the −0.0000082332 quintic term after quartic refinement improved cross-entropy without changing accuracy; the interpolated −0.000010901574 quintic coefficient improved it further, motivating the next-degree probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting the centered quintic coefficient to −0.000010901574 will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22195767936706542.
change: Replace the current quintic calibration coefficient with the best verified interpolated coefficient.
mechanism: Interpolated quintic confidence calibration
evidence_used: Reference Design 2 used −0.000010901574 and achieved 9,192 correct with cross-entropy 0.22195726585388184, strictly improving the current design at identical accuracy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the best verified quintic coefficient and adding a −0.0000010901574 centered sextic term will preserve all 9,192 argmax predictions while reducing validation cross-entropy below 0.22195726585388184.
change: Replace the overshot quintic coefficient with −0.000010901574 and extend evaluation-time confidence calibration with a centered sixth-order margin term.
mechanism: Centered sextic margin-adaptive confidence calibration
evidence_used: Reference Design 2 achieved the best verified score at −0.000010901574; the successful quartic-to-quintic extension supports probing the next polynomial degree, while the previous sextic attempt timed out without producing contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding the verified −0.000010901574 centered quintic correction will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22196439743041993 to approximately 0.22195726585388184.
change: Extend the current quartic evaluation-time calibration with the best verified centered quintic term.
mechanism: Interpolated quintic confidence calibration
evidence_used: Reference Design 3 achieved the best available score, with 9,192 correct predictions and cross-entropy 0.22195726585388184; prior verification failure provided no contrary metric result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Moving the centered quintic coefficient to −0.00001087744649 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726585388184.
change: Replace the current evaluation-time quintic coefficient with the quadratic vertex fitted from the three nearest verified quintic measurements.
mechanism: Refined local quintic-coefficient vertex calibration
evidence_used: Quintic coefficients −0.0000082332, −0.000010901574, and −0.0000164664 yielded cross-entropies 0.22195767936706542, 0.22195726585388184, and 0.22195911331176757 with identical accuracy; local quadratic interpolation places the minimum near −0.00001087744649.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 47.52425087499432, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195727043151856, "validation_score": 9192.409179610531}

RECENT RESULT
hypothesis: A centered quintic coefficient of −0.00001092570151 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726585388184.
change: Move the evaluation-time centered quintic coefficient to the untested point opposite the unsuccessful −0.00001087744649 refinement around the best verified coefficient.
mechanism: Symmetric local quintic-coefficient bracketing
evidence_used: The verified −0.000010901574 coefficient achieved the lowest cross-entropy, while the nearby −0.00001087744649 coefficient was slightly worse; reflecting that probe across the current best efficiently tests whether the local optimum lies in the opposite direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.55261874990538, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726203918456, "validation_score": 9192.40917961334}

RECENT RESULT
hypothesis: A centered quintic coefficient of −0.00001094982902 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.
change: Move the evaluation-time quintic coefficient to the untested point opposite −0.000010901574 around the best verified coefficient.
mechanism: Symmetric local quintic-coefficient bracketing
evidence_used: Moving from −0.000010901574 to −0.00001087744649 worsened cross-entropy, while the symmetric probe at −0.00001092570151 improved it to 0.22195726203918456 with unchanged accuracy; reflecting the former best across the new best provides the nearest informative continuation probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.6695165829733, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726432800292, "validation_score": 9192.409179612574}

RECENT RESULT
hypothesis: A centered quintic coefficient of −0.00001092871744875 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.
change: Add a centered quintic margin term at the local quadratic optimum while retaining the verified quartic calibration.
mechanism: Quadratic-vertex quintic confidence calibration
evidence_used: Coefficients −0.000010901574, −0.00001092570151, and −0.00001094982902 produced cross-entropies 0.22195726585388184, 0.22195726203918456, and 0.22195726432800292 with identical accuracy; quadratic interpolation places the minimum near −0.00001092871744875.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 40.07324216607958, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726737976074, "validation_score": 9192.409179611552}

RECENT RESULT
hypothesis: A centered quintic coefficient of −0.00001092268557125 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22195726203918456.
change: Reflect the unsuccessful −0.00001092871744875 probe across the best verified coefficient, −0.00001092570151, to test the opposite side at equal distance.
mechanism: Symmetric micro-bracketing of quintic confidence calibration
evidence_used: The current coefficient achieved the lowest verified cross-entropy, while the nearest tested coefficient, −0.00001092871744875, was worse with identical accuracy; the symmetric opposite probe is the most informative local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 64.5151190829929, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726318359374, "validation_score": 9192.409179612958}

RECENT RESULT
hypothesis: A centered quintic coefficient of −0.00001092472576511 will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22195726203918456.
change: Replace the current overshot quintic coefficient with the quadratic vertex fitted from the closest symmetric probes around the best verified coefficient.
mechanism: Symmetric micro-bracket quadratic interpolation
evidence_used: The best coefficient, −0.00001092570151, achieved cross-entropy 0.22195726203918456; equal-distance probes at −0.00001092268557125 and −0.00001092871744875 were both worse, and their asymmetric losses place the interpolated minimum slightly toward the less-negative probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 52.79805037518963, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22195726737976074, "validation_score": 9192.409179611552}



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
