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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 69.97773416689597, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628200531006, "validation_score": 9359.422167929906}
prior_hypothesis: Increasing the accepted translation blend from 13.495% to 13.4975% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628242492676.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing only the accepted translation blend from 13.4675% to 13.47% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436287307739258.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each vertical-shift weight from 6.73375% to 6.735% in the returned refined logits.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every accepted-blend increase from 13.4375% through 13.4675% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 54.85415495908819, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436286964416504, "validation_score": 9359.42216791223}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.47% to 13.4725% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436286964416504.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.735% to 6.73625%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.47% preserved correctness and reduced cross-entropy; an equal-sized one-sided probe most directly tests whether the optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 49.886843791930005, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436286544799804, "validation_score": 9359.422167913724}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.4725% to 13.475% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436286544799804.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.73625% to 6.7375%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.4725% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe most directly tests whether the optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 45.61027012509294, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628620147705, "validation_score": 9359.422167914949}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.475% to 13.4775% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628620147705.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7375% to 6.73875%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.475% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.103607250144705, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436285629272461, "validation_score": 9359.42216791699}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.4775% to 13.48% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436285629272461.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.73875% to 6.74%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.4775% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.42322062491439, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436284980773926, "validation_score": 9359.4221679193}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.48% to 13.4825% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436284980773926.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74% to 6.74125%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.48% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.84011862496845, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628448486328, "validation_score": 9359.422167921068}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.4825% to 13.485% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628448486328.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74125% to 6.7425%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.4825% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 46.458523584064096, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436283950805665, "validation_score": 9359.422167922972}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.485% to 13.4875% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436283950805665.
change: Keep the 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7425% to 6.74375%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.485% preserved correctness and reduced cross-entropy, motivating another equal-sized one-sided probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 44.68751979083754, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628360748291, "validation_score": 9359.422167924196}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.4875% to 13.49% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628360748291.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74375% to 6.745%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.4875% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 47.6567075420171, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436282958984376, "validation_score": 9359.422167926507}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.49% to 13.4925% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436282958984376.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.745% to 6.74625%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.49% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 48.926453499821946, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436282691955566, "validation_score": 9359.42216792746}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.4925% to 13.495% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436282691955566.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74625% to 6.7475%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.4925% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.48549804184586, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628242492676, "validation_score": 9359.422167928411}

RECENT RESULT
hypothesis: Increasing the accepted translation blend from 13.495% to 13.4975% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628242492676.
change: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7475% to 6.74875%.
mechanism: One-sided refinement of accepted vertical-TTA weight
evidence_used: Every tested increase from 13.4375% through 13.495% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe tests whether the continuous optimum remains higher.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 69.97773416689597, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843628200531006, "validation_score": 9359.422167929906}



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
