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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 73.93056012503803, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791664123536, "validation_score": 9359.418810306512}
prior_hypothesis: A 3.0195159912109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579266.

## Recent verification evidence

RECENT RESULT
hypothesis: A 3.019500732421875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579308.
change: Increase only the floating-buffer EMA update from 3.01947021484375% to 3.019500732421875%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: Successive increases through 3.01947021484375% preserved 9,359 correct while monotonically lowering cross-entropy; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 75.46704999986105, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385792655944825, "validation_score": 9359.418810303032}

RECENT RESULT
hypothesis: A 3.0195159912109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579266.
change: Increase only the floating-buffer EMA update from 3.019500732421875% to 3.0195159912109375%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: Successive increases through 3.019500732421875% preserved 9,359 correct while monotonically lowering cross-entropy; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 73.93056012503803, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791664123536, "validation_score": 9359.418810306512}

RECENT RESULT
hypothesis: A 3.01952362060546875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938579166.
change: Increase only the floating-buffer EMA update from 3.0195159912109375% to 3.01952362060546875%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: Successive increases through 3.0195159912109375% preserved 9,359 correct while monotonically lowering cross-entropy; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 80.53475133306347, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791778564454, "validation_score": 9359.41881030611}

RECENT RESULT
hypothesis: A 3.019519805908203125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Increase only the floating-buffer EMA update from 3.0195159912109375% to the midpoint before the tested 3.01952362060546875% endpoint, preserving all other behavior.
mechanism: Local EMA calibration-rate bisection
evidence_used: The current rate achieved the best completed cross-entropy, while 3.01952362060546875% retained 9,359 correct but produced slightly worse cross-entropy and exceeded the time limit; their midpoint is the most informative refinement of the newly bracketed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 75.33371983421966, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791969299315, "validation_score": 9359.418810305442}

RECENT RESULT
hypothesis: A 3.0195178985595703125% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Increase only the floating-buffer EMA update from 3.0195159912109375% to the midpoint before the worse-performing 3.019519805908203125% trial.
mechanism: Local EMA calibration-rate bisection
evidence_used: Cross-entropy improved monotonically through 3.0195159912109375%, but worsened at 3.019519805908203125% with the same correct count; bisecting this newly bracketed interval is the most informative local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 69.47366650006734, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791816711426, "validation_score": 9359.418810305977}

RECENT RESULT
hypothesis: A 3.01951694488525390625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Increase only the floating-buffer EMA update from 3.0195159912109375% to the midpoint before the worse-performing 3.0195178985595703125% trial.
mechanism: Local EMA calibration-rate bisection
evidence_used: The current rate produced the best cross-entropy, while both tested higher rates worsened it with the same correct count; bisecting the nearest bracket is the most informative local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 61.772786000045016, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791816711426, "validation_score": 9359.418810305977}

RECENT RESULT
hypothesis: A 3.01950836181640625% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Decrease only the floating-buffer EMA update from 3.0195159912109375% to the midpoint above the previously worse 3.019500732421875% rate.
mechanism: Local EMA calibration-rate bisection
evidence_used: The current rate is the best tested point, while both the nearest tested lower and higher rates produced worse cross-entropy with the same correct count; bisecting the wider lower-side interval is the most informative remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 67.62496629194357, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791931152344, "validation_score": 9359.418810305575}

RECENT RESULT
hypothesis: A 3.019512176513671875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.19385791664123536.
change: Decrease only the floating-buffer EMA update from 3.0195159912109375% to the midpoint above the worse-performing 3.01950836181640625% trial.
mechanism: Lower-side EMA calibration-rate bisection
evidence_used: The current rate remains the best tested point, while the nearest lower rate produced worse cross-entropy with the same correct count; bisecting that bracket is the most informative remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 74.3491319171153, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385791931152344, "validation_score": 9359.418810305575}

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
