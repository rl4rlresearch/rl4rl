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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 66.83421045797877, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436300659179689, "validation_score": 9359.422167863415}
prior_hypothesis: A 13.39739990234375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300964355468 while remaining below the degradation observed at 13.39794921875%.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A 13.4375% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436306648254394.
change: Increase total translated-view weight from 13.3671875% to 13.4375%, assigning 6.71875% to each vertical shift.
mechanism: Local quadratic vertical-TTA weight refinement
evidence_used: Cross-entropy improved at 13.296875% and 13.3671875% but worsened at 14.2578125%; quadratic interpolation of these probes estimates a minimum near 13.44%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.52603424992412, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843700107574463, "validation_score": 9359.422165366785}

RECENT RESULT
hypothesis: A 13.40234375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436306648254394 without triggering the degradation observed at 13.4375%.
change: Increase total translated-view weight from 13.3671875% to 13.40234375%, assigning 6.701171875% to each vertical shift.
mechanism: Midpoint localization of the guarded vertical-TTA boundary
evidence_used: Cross-entropy improved through 13.3671875% but sharply worsened at 13.4375%, suggesting a discrete confidence-gating boundary; testing their midpoint most directly localizes that boundary while probing the still-improving direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.48365300009027, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.184370059967041, "validation_score": 9359.422165349244}

RECENT RESULT
hypothesis: A 13.384765625% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436306648254394 while remaining below the degradation boundary observed at 13.40234375%.
change: Increase total translated-view weight from 13.3671875% to 13.384765625%, assigning 6.6923828125% to each vertical shift.
mechanism: Bisection of the confidence-gated vertical-TTA boundary
evidence_used: The 13.3671875% blend achieved the best cross-entropy, while 13.40234375% sharply worsened it; their midpoint is the most informative probe for localizing the apparent discrete confidence-gating boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.90966329211369, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436303024291992, "validation_score": 9359.422167854984}

RECENT RESULT
hypothesis: A 13.3935546875% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436303024291992 while remaining below the degradation boundary observed at 13.40234375%.
change: Increase total translated-view weight from 13.384765625% to 13.3935546875%, assigning 6.69677734375% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.384765625% blend improved cross-entropy, whereas 13.40234375% sharply worsened it; their midpoint is the most informative next probe of the apparent discrete gating boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 45.35159166716039, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436301498413085, "validation_score": 9359.422167860423}

RECENT RESULT
hypothesis: A 13.39794921875% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436301498413085 while remaining below the degradation observed at 13.40234375%.
change: Increase total translated-view weight from 13.3935546875% to 13.39794921875%, assigning 6.698974609375% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.3935546875% blend achieved the best cross-entropy, while 13.40234375% sharply worsened it; their midpoint is the most informative probe of the apparent discrete confidence-gating boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.62491662497632, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843700668334961, "validation_score": 9359.422165346796}

RECENT RESULT
hypothesis: A 13.395751953125% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436301498413085 while remaining below the degradation observed at 13.39794921875%.
change: Increase total translated-view weight from 13.3935546875% to 13.395751953125%, assigning 6.6978759765625% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.3935546875% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 57.67322141700424, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436301002502442, "validation_score": 9359.42216786219}

RECENT RESULT
hypothesis: A 13.3968505859375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436301002502442 while remaining below the degradation observed at 13.39794921875%.
change: Increase total translated-view weight from 13.395751953125% to 13.3968505859375%, assigning 6.69842529296875% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.395751953125% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 68.08622562512755, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436300964355468, "validation_score": 9359.422167862327}

RECENT RESULT
hypothesis: A 13.39739990234375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300964355468 while remaining below the degradation observed at 13.39794921875%.
change: Increase total translated-view weight from 13.3968505859375% to 13.39739990234375%, assigning 6.698699951171875% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.3968505859375% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 66.83421045797877, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436300659179689, "validation_score": 9359.422167863415}

RECENT RESULT
hypothesis: A 13.397674560546875% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300659179689 while remaining below the degradation observed at 13.39794921875%.
change: Increase total translated-view weight from 13.39739990234375% to 13.397674560546875%, assigning 6.6988372802734375% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.39739990234375% blend achieved the best cross-entropy, while 13.39794921875% sharply worsened it; their midpoint is the most informative probe for localizing the apparent discrete confidence-gating boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 65.8554142920766, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18437007102966307, "validation_score": 9359.4221653453}

RECENT RESULT
hypothesis: A 13.3975372314453125% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300659179689 while remaining below the degradation observed at 13.397674560546875%.
change: Increase total translated-view weight from 13.39739990234375% to 13.3975372314453125%, assigning 6.69876861572265625% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.39739990234375% blend achieved the best cross-entropy, while 13.397674560546875% sharply worsened it; their midpoint is the most informative probe for localizing the apparent discrete confidence-gating boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.14564866689034, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843630069732666, "validation_score": 9359.422167863278}

RECENT RESULT
hypothesis: A 13.39746856689453125% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436300659179689 while remaining below the degradation observed at 13.3975372314453125%.
change: Increase total translated-view weight from 13.39739990234375% to 13.39746856689453125%, assigning 6.698734283447265625% to each vertical shift.
mechanism: Bisection refinement of the confidence-gated vertical-TTA boundary
evidence_used: The 13.39739990234375% blend achieved the best cross-entropy, while 13.3975372314453125% was slightly worse; their midpoint is the most informative next probe for localizing the apparent discrete confidence-gating boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.82992662489414, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843700668334961, "validation_score": 9359.422165346796}



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
