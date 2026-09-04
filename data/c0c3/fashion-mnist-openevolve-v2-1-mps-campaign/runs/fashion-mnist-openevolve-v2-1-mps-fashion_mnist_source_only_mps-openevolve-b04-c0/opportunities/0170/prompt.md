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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.32966958289035, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436287307739258, "validation_score": 9359.422167911005}
prior_hypothesis: Increasing only the accepted translation blend from 13.465% to 13.4675% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436287574768068.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4375% to 13.44% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436292915344238.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72% to each vertical shift in the returned refined logits.
mechanism: Continuous refinement of accepted vertical-TTA predictions
evidence_used: Decoupling selection and refinement at 13.4375% improved cross-entropy without changing correctness, and prior interpolation placed the continuous blend optimum near 13.44%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.377534040948376, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436292495727538, "validation_score": 9359.422167892513}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.44% to 13.4425% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436292495727538.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72125% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Raising the decoupled accepted blend from 13.4375% to 13.44% preserved correctness and reduced cross-entropy, motivating a same-sized one-sided probe near the estimated 13.44% continuous optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 65.37283979100175, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436291847229003, "validation_score": 9359.422167894825}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4425% to 13.445% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436291847229003.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.7225% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Each increase from 13.4375% through 13.44% to 13.4425% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe is the most informative test of whether the continuous optimum lies higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 50.82235675002448, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436291732788085, "validation_score": 9359.422167895233}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.445% to 13.4475% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436291732788085.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72375% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Increasing the decoupled accepted blend from 13.4375% through 13.445% repeatedly preserved correctness and reduced cross-entropy; another equal-sized probe most directly tests whether the continuous optimum lies higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 56.83741812501103, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843629119873047, "validation_score": 9359.422167897137}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4475% to 13.45% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843629119873047.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.725% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.4475% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.832243542186916, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436290550231935, "validation_score": 9359.422167899447}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.45% to 13.4525% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436290550231935.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72625% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.45% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.734300333075225, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436290130615235, "validation_score": 9359.422167900944}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4525% to 13.455% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436290130615235.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.7275% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.4525% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 65.99390808306634, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436289710998535, "validation_score": 9359.42216790244}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.455% to 13.4575% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436289710998535.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.72875% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.455% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 56.977678375085816, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436289443969728, "validation_score": 9359.42216790339}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4575% to 13.46% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436289443969728.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.73% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.4575% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe most directly tests whether the continuous optimum lies higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.616094917058945, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436288795471192, "validation_score": 9359.422167905703}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.46% to 13.4625% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436288795471192.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.73125% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.46% preserved correctness and reduced cross-entropy; another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 46.60003958409652, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628791809082, "validation_score": 9359.42216790883}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4625% to 13.465% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628791809082.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.7325% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.4625% preserved correctness and reduced cross-entropy; the latest increase produced the largest recent reduction, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 52.64964299998246, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436287574768068, "validation_score": 9359.422167910054}

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.465% to 13.4675% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436287574768068.
change: Keep the verified 13.39739990234375% selection gate fixed while assigning 6.73375% to each vertical shift in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.465% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.32966958289035, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436287307739258, "validation_score": 9359.422167911005}



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
