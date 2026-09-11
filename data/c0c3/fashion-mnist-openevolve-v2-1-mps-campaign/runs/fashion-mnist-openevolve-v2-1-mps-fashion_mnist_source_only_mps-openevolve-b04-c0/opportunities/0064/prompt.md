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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 78.47725679189898, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}
prior_hypothesis: A 3.01951587200164794921875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791625976562.

## Recent verification evidence

RECENT RESULT
hypothesis: A 3.019516468048095703125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.0195159912109375% rate remains best, while 3.01951694488525390625% produced worse cross-entropy with the same correct count; bisecting this tight upper-side bracket is the most informative remaining refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 61.06924070813693, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791702270508, "validation_score": 9359.418810306377}

RECENT RESULT
hypothesis: A 3.0195140838623046875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested lower, worse-performing rate.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current 3.0195159912109375% rate remains best, while 3.019512176513671875% produced worse cross-entropy with the same correct count; bisecting this remaining lower-side bracket is the most informative local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.69755275011994, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579200744629, "validation_score": 9359.418810305307}

RECENT RESULT
hypothesis: A 3.01951503753662109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested lower, worse-performing rate.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current 3.0195159912109375% rate remains best, while 3.0195140838623046875% produced worse cross-entropy with the same correct count; bisecting this tight lower-side bracket is the most informative remaining refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 74.84942083293572, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791854858397, "validation_score": 9359.418810305842}

RECENT RESULT
hypothesis: A 3.019515514373779296875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested lower, worse-performing rate.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current 3.0195159912109375% rate remains best, while 3.01951503753662109375% produced worse cross-entropy with the same correct count; bisecting this wider remaining lower-side bracket is the most informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 61.96755750011653, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791931152344, "validation_score": 9359.418810305575}

RECENT RESULT
hypothesis: A 3.0195162296295166015625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.0195159912109375% rate remains best, while 3.019516468048095703125% produced slightly worse cross-entropy with the same correct count; bisecting this upper-side bracket is the most informative remaining refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 69.2208105828613, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791625976562, "validation_score": 9359.418810306646}

RECENT RESULT
hypothesis: A 3.01951634883880615234375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791625976562.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.0195162296295166015625% rate achieved the best score, while 3.019516468048095703125% preserved 9,359 correct but slightly worsened cross-entropy; their midpoint is the most informative untested upper-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.53276379196905, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791702270508, "validation_score": 9359.418810306377}

RECENT RESULT
hypothesis: A 3.019516289234161376953125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791625976562.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.0195162296295166015625% rate achieved the best score, while 3.01951634883880615234375% preserved 9,359 correct but slightly worsened cross-entropy; their midpoint is the most informative untested refinement.
result: training did not finish within the verification time limit

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A 3.01951587200164794921875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791625976562.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower, worse-performing rate.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current 3.0195162296295166015625% rate achieved the best score, while 3.019515514373779296875% preserved 9,359 correct but worsened cross-entropy; their midpoint is the most informative untested lower-side refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 78.47725679189898, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}

RECENT RESULT
hypothesis: A 3.019516050815582275390625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Increase only the floating-buffer EMA update to the midpoint between the current best rate and the nearest higher, worse-performing rate.
mechanism: Upper-side EMA calibration-rate bisection
evidence_used: The current 3.01951587200164794921875% rate achieved the best score, while 3.0195162296295166015625% preserved 9,359 correct but had slightly higher cross-entropy; their midpoint is the most informative untested refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.019515693187713623046875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579158782959.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest tested lower, worse-performing rate.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current 3.01951587200164794921875% rate achieved the best score, while 3.019515514373779296875% preserved 9,359 correct but had higher cross-entropy; their midpoint is the most informative untested lower-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 77.85790295898914, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385792083740233, "validation_score": 9359.41881030504}

RECENT RESULT
hypothesis: A 3.0195157825946807861328125% buffer EMA update will preserve 9,359 correct predictions while lower validation cross-entropy below 0.1938579158782959.
change: Decrease only the floating-buffer EMA update to the midpoint between the current best rate and the nearest lower, worse-performing rate.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current 3.01951587200164794921875% rate achieved the best score, while 3.019515693187713623046875% preserved 9,359 correct but increased cross-entropy; their midpoint is the most informative untested lower-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.784591084113345, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938579158782959, "validation_score": 9359.418810306779}



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
