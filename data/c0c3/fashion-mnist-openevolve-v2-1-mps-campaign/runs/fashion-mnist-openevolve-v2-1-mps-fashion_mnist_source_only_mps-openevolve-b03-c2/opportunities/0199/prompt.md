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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.9978640419431, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436874084472657, "validation_score": 9257.411736553473}
prior_hypothesis: A 653/8192 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436881332397462.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 58.75192158296704, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436964797973632, "validation_score": 9257.411736245906}
prior_hypothesis: A 5/64 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214378804397583.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.16306820791215, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436870040893555, "validation_score": 9257.411736567183}
prior_hypothesis: A 5229/65536 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870956420898.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.24619591701776, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436870956420898, "validation_score": 9257.411736564078}
prior_hypothesis: A 1307/16384 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436874084472657.

## Recent verification evidence

RECENT RESULT
hypothesis: An 11/128 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.
change: Increase geometric fusion from 1/64 to 11/128 while retaining the trained model, established views, view weights, and compensated calibration.
mechanism: Final binary search of the geometric-fusion ranking boundary
evidence_used: A 5/64 blend preserved 9,257 correct with 0.21436964797973632 cross-entropy, while 3/32 lost one correct but reduced cross-entropy further; 11/128 is their exact midpoint and isolates the remaining ranking boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 49.529807541985065, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436521072387696, "validation_score": 9256.411737750377}

RECENT RESULT
hypothesis: A 21/256 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.
change: Replace arithmetic-only fusion with a 235/256 arithmetic and 21/256 center-weighted geometric probability blend, retaining the qualified compensated logit calibration.
mechanism: Fine-grained geometric-fusion boundary search
evidence_used: The 5/64 blend preserved 9,257 correct predictions with 0.21436964797973632 cross-entropy, while 11/128 lost one correct prediction but further reduced cross-entropy; 21/256 is their midpoint and most directly refines the ranking boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.43280037492514, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2143674201965332, "validation_score": 9256.411737001245}

RECENT RESULT
hypothesis: A 41/512 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.
change: Increase geometric fusion from 5/64 to 41/512, the midpoint between the best qualified blend and the nearest tested accuracy-reducing blend.
mechanism: Sub-percent geometric-fusion boundary search
evidence_used: The 5/64 blend retained 9,257 correct at 0.21436964797973632 cross-entropy, while 21/256 lost one correct but lowered cross-entropy; 41/512 bisects this unresolved ranking boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 54.942358458880335, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436853256225585, "validation_score": 9256.411736624092}

RECENT RESULT
hypothesis: An 81/1024 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.
change: Increase geometric fusion from 5/64 to 81/1024, the exact midpoint between the best qualified blend and the nearest tested accuracy-reducing blend.
mechanism: Sub-permille geometric-fusion boundary search
evidence_used: The 5/64 blend retained 9,257 correct at 0.21436964797973632 cross-entropy, while 41/512 lost one correct; 81/1024 bisects this remaining ranking boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 66.46290395781398, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436909408569335, "validation_score": 9257.411736433705}

RECENT RESULT
hypothesis: A 163/2048 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436909408569335.
change: Increase geometric fusion from the current 1/64 to 163/2048, the midpoint between the best qualified blend and the nearest tested accuracy-reducing blend.
mechanism: Fine-grained geometric-fusion boundary bisection
evidence_used: The 81/1024 blend retained 9,257 correct at 0.21436909408569335 cross-entropy, while 41/512 lost one correct; 163/2048 exactly bisects that unresolved boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 63.32095912494697, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436881332397462, "validation_score": 9257.411736528898}

RECENT RESULT
hypothesis: A 327/4096 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436881332397462.
change: Add compensated arithmetic–geometric probability fusion using the midpoint between the best qualified 163/2048 blend and the nearest accuracy-reducing 41/512 blend.
mechanism: Geometric-fusion boundary bisection
evidence_used: The 163/2048 blend retained 9,257 correct at 0.21436881332397462 cross-entropy, while 41/512 lost one correct; 327/4096 exactly bisects this remaining boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 43.44022570899688, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436867637634277, "validation_score": 9256.41173657533}

RECENT RESULT
hypothesis: A 653/8192 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436881332397462.
change: Increase geometric fusion from 163/2048 to 653/8192, the exact midpoint between the best qualified blend and the nearest accuracy-reducing blend.
mechanism: Geometric-fusion boundary bisection
evidence_used: The 163/2048 blend retained 9,257 correct at 0.21436881332397462 cross-entropy, while 327/4096 lost one correct; 653/8192 bisects this remaining ranking boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.9978640419431, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436874084472657, "validation_score": 9257.411736553473}

RECENT RESULT
hypothesis: A 1307/16384 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436874084472657.
change: Increase geometric fusion to the exact midpoint between the best qualified 653/8192 blend and the nearest accuracy-reducing 327/4096 blend, retaining all other settings.
mechanism: Final geometric-fusion boundary bisection
evidence_used: The 653/8192 blend retained 9,257 correct at 0.21436874084472657 cross-entropy, while 327/4096 lost one correct; 1307/16384 bisects the remaining boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.24619591701776, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436870956420898, "validation_score": 9257.411736564078}

RECENT RESULT
hypothesis: A 2615/32768 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870956420898.
change: Increase geometric fusion from 1307/16384 to 2615/32768, the sole midpoint between the best qualified blend and the nearest accuracy-reducing blend, while retaining all other settings.
mechanism: Terminal geometric-fusion boundary bisection
evidence_used: The 1307/16384 blend retained 9,257 correct at 0.21436870956420898 cross-entropy, while 327/4096 lost one correct; 2615/32768 exactly bisects that final unresolved interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 43.39278887491673, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436869087219237, "validation_score": 9256.411736570415}

RECENT RESULT
hypothesis: A 5229/65536 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870956420898.
change: Replace arithmetic-only TTA fusion with a 60307/65536 arithmetic and 5229/65536 center-weighted geometric probability blend, retaining the verified compensated logit calibration.
mechanism: Geometric-fusion boundary bisection
evidence_used: The 1307/16384 blend retained 9,257 correct at 0.21436870956420898 cross-entropy, while the adjacent tested 2615/32768 blend lost one correct; 5229/65536 exactly bisects that unresolved interval.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.16306820791215, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436870040893555, "validation_score": 9257.411736567183}

RECENT RESULT
hypothesis: A 10459/131072 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870040893555.
change: Increase geometric fusion from 5229/65536 to 10459/131072, the exact midpoint between the best qualified blend and the nearest accuracy-reducing blend, retaining all other settings.
mechanism: Ultra-fine geometric-fusion boundary bisection
evidence_used: The current 5229/65536 blend retained 9,257 correct at 0.21436870040893555 cross-entropy, while the adjacent tested 2615/32768 blend lost one correct; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.05514349997975, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436869735717773, "validation_score": 9256.411736568218}

RECENT RESULT
hypothesis: A 20917/262144 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870040893555.
change: Increase geometric fusion to the sole representable midpoint between the best qualified 5229/65536 blend and the accuracy-reducing 10459/131072 blend, retaining all other settings.
mechanism: Terminal geometric-fusion boundary bisection
evidence_used: The 5229/65536 blend retained 9,257 correct at 0.21436870040893555 cross-entropy, while 10459/131072 lost one correct; 20917/262144 exactly bisects their final unresolved interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 38.97066370793618, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436870040893555, "validation_score": 9256.411736567183}



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
