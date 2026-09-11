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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 49.56092550000176, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19056305084228517, "validation_score": 9359.4199693579}
prior_hypothesis: Increasing the evaluation-only calibration factor from 1.02 to 1.04 will preserve all 9,359 argmax predictions while further reducing validation cross-entropy below 0.19208800277709961.

## Recent verification evidence

RECENT RESULT
hypothesis: A 3.01951582729816436767578125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower rate that tied it.
mechanism: Lower-side EMA calibration bisection
evidence_used: The current 3.01951587200164794921875% rate and 3.0195157825946807861328125% both achieved 9,359 correct with cross-entropy 0.1938579158782959, while the next-lower tested rate was worse; bisecting the remaining plateau is the most informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 61.01016987510957, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}

RECENT RESULT
hypothesis: A 3.019516050815582275390625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection retry
evidence_used: The current 3.01951587200164794921875% rate is best, while 3.0195162296295166015625% was slightly worse; their midpoint remains the most informative candidate because its previous verification timed out without producing performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.019515849649906158447265625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower rate that tied it.
mechanism: Lower-side EMA calibration plateau bisection
evidence_used: Rates of 3.01951587200164794921875% and 3.01951582729816436767578125% both achieved 9,359 correct with cross-entropy 0.1938579158782959; their untested midpoint is the most informative refinement within the observed plateau.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 72.74671962508, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}

RECENT RESULT
hypothesis: A 3.0195159614086151123046875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the twice-timed-out upper midpoint.
mechanism: Upper-side EMA calibration quarter-step
evidence_used: The current 3.01951587200164794921875% rate is best, while the nearest completed higher rate was worse; because the direct upper midpoint timed out twice without performance evidence, this untested quarter-step probes the unresolved upper interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 56.59036537492648, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791664123536, "validation_score": 9359.418810306512}

RECENT RESULT
hypothesis: A 3.01951591670513153076171875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest completed higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.01951587200164794921875% rate is best, while 3.0195159614086151123046875% preserved 9,359 correct but increased cross-entropy; their untested midpoint is the most informative refinement of the remaining upper interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.61121779191308, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791664123536, "validation_score": 9359.418810306512}

RECENT RESULT
hypothesis: A 3.019515894353389739990234375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest completed higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.01951587200164794921875% rate is best, while 3.01951591670513153076171875% preserved 9,359 correct but increased cross-entropy; their untested midpoint most directly refines the remaining upper interval.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.0195158831775188446044921875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the most recent timed-out upper midpoint.
mechanism: Upper-side EMA calibration quarter-step
evidence_used: The current 3.01951587200164794921875% rate is best; the nearest completed higher rates worsened cross-entropy, while 3.019515894353389739990234375% timed out without performance evidence, motivating a smaller untested upper step.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 62.70567391603254, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}

RECENT RESULT
hypothesis: A 3.01951587758958339691162109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher rate that tied it.
mechanism: Upper-side EMA calibration plateau bisection
evidence_used: The current 3.01951587200164794921875% rate and 3.0195158831775188446044921875% both achieved 9,359 correct with cross-entropy 0.1938579158782959; their untested midpoint is the most informative remaining refinement near the upper edge of the observed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.84737487509847, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}

RECENT RESULT
hypothesis: A 3.019515874795615673065185546875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher rate that tied it.
mechanism: Upper-side EMA calibration plateau bisection
evidence_used: The current 3.01951587200164794921875% rate and 3.01951587758958339691162109375% both achieved 9,359 correct with cross-entropy 0.1938579158782959; their untested midpoint most precisely probes the remaining upper-side plateau.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 66.76201008306816, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.02 will preserve all 9,359 argmax predictions while reducing validation cross-entropy by correcting underconfidence induced by 0.02 label smoothing.
change: Multiply the flip-ensemble logits by a positive 1.02 calibration factor during evaluation only.
mechanism: Accuracy-invariant evaluation logit sharpening
evidence_used: Numerous representationally tiny buffer-EMA refinements tied the current 9,359 correct and 0.1938579158782959 cross-entropy, while adjacent effective rates were worse; an argmax-preserving calibration change is therefore a more informative direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 65.60768875014037, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19208800277709961, "validation_score": 9359.419432121484}

RECENT RESULT
hypothesis: Increasing the evaluation-only calibration factor from 1.02 to 1.04 will preserve all 9,359 argmax predictions while further reducing validation cross-entropy below 0.19208800277709961.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Scaling evaluation logits by 1.02 preserved 9,359 correct predictions and improved cross-entropy from 0.1938579158782959 to 0.19208800277709961, providing directional evidence that the ensemble remains underconfident.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 49.56092550000176, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19056305084228517, "validation_score": 9359.4199693579}

RECENT RESULT
hypothesis: Increasing the evaluation-only logit scale from 1.04 to 1.06 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.19056305084228517.
change: Increase only the positive scale applied to flip-ensemble logits during evaluation.
mechanism: Accuracy-invariant evaluation logit sharpening sweep
evidence_used: Scaling from 1.02 to 1.04 preserved 9,359 correct predictions and improved cross-entropy from 0.19208800277709961 to 0.19056305084228517, indicating continued underconfidence.
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
