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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 68.54769050003961, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436306648254394, "validation_score": 9359.422167842067}
prior_hypothesis: A 13.3671875% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436321144104004.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the symmetric vertical-translation blend from 3.125% to 6.25% will preserve all 9,359 predictions through the existing argmax guard while reducing validation cross-entropy below 0.1844868465423584.
change: Double each translated-view weight from 1.5625% to 3.125%, reducing the canonical ensemble weight accordingly.
mechanism: Stronger confidence-gated vertical-translation TTA
evidence_used: The guarded 3.125% vertical blend preserved 9,359 correct predictions and improved cross-entropy from 0.18456672592163087 to 0.1844868465423584, supporting a stronger probe in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.53305041696876, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18442666282653808, "validation_score": 9359.42214517428}

RECENT RESULT
hypothesis: Increasing the symmetric vertical-translation blend from 6.25% to 12.5% will preserve all 9,359 predictions through the existing argmax guard and reduce validation cross-entropy below 0.18442666282653808.
change: Double each translated-view weight from 3.125% to 6.25%, reducing the horizontal-ensemble weight from 93.75% to 87.5%.
mechanism: Stronger confidence-gated vertical-translation TTA
evidence_used: Doubling the guarded vertical blend from 3.125% to 6.25% preserved 9,359 correct predictions and improved cross-entropy from 0.1844868465423584 to 0.18442666282653808, supporting another stronger probe along the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.89964574994519, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436553497314453, "validation_score": 9359.422166962171}

RECENT RESULT
hypothesis: Increasing the symmetric vertical-translation blend from 12.5% to 25% will preserve all 9,359 predictions through the existing argmax guard and reduce validation cross-entropy below 0.18436553497314453.
change: Double each translated-view weight from 6.25% to 12.5%, reducing the horizontal-ensemble weight from 87.5% to 75%.
mechanism: Stronger confidence-gated vertical-translation TTA
evidence_used: Doubling the guarded vertical blend from 6.25% to 12.5% preserved 9,359 correct predictions and improved cross-entropy from 0.18442666282653808 to 0.18436553497314453, with no observed saturation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.53085966594517, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1844776134490967, "validation_score": 9359.422127015592}

RECENT RESULT
hypothesis: A 14.2578125% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436553497314453.
change: Increase total translated-view weight from 12.5% to 14.2578125%, assigning 7.12890625% to each vertical shift.
mechanism: Quadratic-interpolated vertical-translation TTA
evidence_used: Cross-entropy improved from 0.18442666282653808 at 6.25% translation weight to 0.18436553497314453 at 12.5%, but worsened to 0.1844776134490967 at 25%; quadratic interpolation of these three probes estimates a minimum near 14.3%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.4312843750231, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843695240020752, "validation_score": 9359.422165540287}

RECENT RESULT
hypothesis: A 12.625% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436553497314453.
change: Increase total translated-view weight from 12.5% to 12.625%, assigning 6.3125% to each vertical shift.
mechanism: Local quadratic vertical-TTA interpolation
evidence_used: Cross-entropy improved through 12.5% translation weight but worsened at 14.2578125%; quadratic interpolation using those results and the 6.25% probe places the estimated minimum near 12.625%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.85662749991752, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1843650894165039, "validation_score": 9359.42216712099}

RECENT RESULT
hypothesis: A 13.0625% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.1843650894165039.
change: Increase total translated-view weight from 12.625% to 13.0625%, assigning 6.53125% to each vertical shift.
mechanism: Local quadratic vertical-TTA weight refinement
evidence_used: The 12.625% blend improved cross-entropy over 12.5%, while 14.2578125% was worse; quadratic interpolation of these probes estimates a local minimum near 13.06%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 47.85888308403082, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436376991271972, "validation_score": 9359.422167591329}

RECENT RESULT
hypothesis: A 13.15625% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436376991271972.
change: Increase total translated-view weight from 13.0625% to 13.15625%, assigning 6.578125% to each vertical shift.
mechanism: Local quadratic vertical-TTA interpolation
evidence_used: Cross-entropy improved at 12.625% and 13.0625% but worsened at 14.2578125%; quadratic interpolation of these three nearby probes estimates a minimum near 13.16%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 51.98858133307658, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436353340148925, "validation_score": 9359.422167675633}

RECENT RESULT
hypothesis: A 13.296875% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436353340148925.
change: Increase total translated-view weight from 13.15625% to 13.296875%, assigning 6.6484375% to each vertical shift.
mechanism: Quadratic-interpolated vertical-TTA weight refinement
evidence_used: Cross-entropy improved from 0.18436376991271972 at 13.0625% to 0.18436353340148925 at 13.15625%, but worsened to 0.1843695240020752 at 14.2578125%; quadratic interpolation of these three probes estimates a minimum near 13.30%.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 13.296875% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436353340148925.
change: Increase total translated-view weight from 13.15625% to 13.296875%, assigning 6.6484375% to each vertical shift.
mechanism: Retry of quadratic-interpolated vertical-TTA optimum
evidence_used: The measured 13.0625%, 13.15625%, and 14.2578125% probes estimate an optimum near 13.30%; the previous 13.296875% verification timed out without producing contrary metric evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 48.27087799995206, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436321144104004, "validation_score": 9359.422167790395}

RECENT RESULT
hypothesis: A 13.3671875% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436321144104004.
change: Increase total translated-view weight from 13.296875% to 13.3671875%, assigning 6.68359375% to each vertical shift.
mechanism: Local quadratic vertical-TTA weight refinement
evidence_used: Cross-entropy improved from 0.18436353340148925 at 13.15625% to 0.18436321144104004 at 13.296875%, while the 14.2578125% probe was worse; local quadratic interpolation places the estimated minimum near 13.37%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 68.54769050003961, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436306648254394, "validation_score": 9359.422167842067}

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
