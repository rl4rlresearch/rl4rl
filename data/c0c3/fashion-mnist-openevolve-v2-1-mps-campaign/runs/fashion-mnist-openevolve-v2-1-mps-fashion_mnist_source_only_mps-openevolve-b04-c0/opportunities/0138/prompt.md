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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.89964574994519, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18436553497314453, "validation_score": 9359.422166962171}
prior_hypothesis: Increasing the symmetric vertical-translation blend from 6.25% to 12.5% will preserve all 9,359 predictions through the existing argmax guard and reduce validation cross-entropy below 0.18442666282653808.

## Recent verification evidence

RECENT RESULT
hypothesis: The adjacent ensemble weights differ on the single validation example responsible for the 9,359-to-9,358 regression; strongly favoring the accuracy-preserving prediction only on such disagreements will retain 9,359 correct while lowering cross-entropy below 0.1846570999145508.
change: Keep refined logits for agreement cases, but add an 8-logit confidence boost to the stable ensemble’s predicted class when the two ensembles disagree.
mechanism: Disagreement-targeted confidence correction
evidence_used: The refined weighting lowered cross-entropy but lost exactly one correct prediction, while the existing argmax guard recovered that prediction without a measurable cross-entropy penalty, isolating disagreement cases as the most informative target.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.94484812486917, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456673736572266, "validation_score": 9359.422095255783}

RECENT RESULT
hypothesis: Increasing the disagreement-only boost from 8 to 16 will preserve all 9,359 predictions while reducing validation cross-entropy below 0.18456673736572266.
change: Double the confidence added to the stable ensemble’s class only when the stable and refined ensembles disagree.
mechanism: Disagreement-targeted confidence saturation
evidence_used: The 8-logit boost preserved 9,359 correct and reduced aggregate validation loss by approximately 0.904, consistent with one isolated disagreement where the stable prediction is correct; further confidence should monotonically reduce that example’s residual loss.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 77.08237754111178, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456672821044923, "validation_score": 9359.422095259046}

RECENT RESULT
hypothesis: Increasing the disagreement-only boost from 16 to 32 will preserve all 9,359 predictions while further reducing validation cross-entropy below 0.18456672821044923.
change: Double the confidence added to the stable ensemble’s predicted class only when the stable and refined ensembles disagree.
mechanism: Disagreement-targeted confidence saturation
evidence_used: Raising the boost from 8 to 16 preserved 9,359 correct predictions and reduced cross-entropy from 0.18456673736572266 to 0.18456672821044923, supporting another monotonic saturation probe on the isolated disagreement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 65.46542858309112, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456672821044923, "validation_score": 9359.422095259046}

RECENT RESULT
hypothesis: Extrapolating the successful canonical-view weight increment once more, while retaining the current logits wherever the extrapolated ensemble changes the stable prediction, will preserve 9,359 correct predictions and reduce cross-entropy below 0.18456672821044923.
change: Add a 53.807373046875% canonical-view ensemble and use it only on samples where its argmax still matches the accuracy-preserving stable ensemble; otherwise retain the current disagreement-corrected logits.
mechanism: Argmax-guarded TTA weight extrapolation
evidence_used: Moving from 53.8072967529296875% to 53.80733489990234375% reduced cross-entropy, and the disagreement guard plus boost preserved all 9,359 predictions. The 16-to-32 boost probe then saturated, motivating another guarded move along the previously beneficial weight direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 54.935135249979794, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456673126220702, "validation_score": 9359.422095257958}

RECENT RESULT
hypothesis: Shifting the refined canonical-view weight one-eighth of the prior step toward the lower-weight probe will preserve 9,359 correct predictions while reducing cross-entropy below 0.18456672821044923.
change: Change the guarded refined ensemble from 53.8073348999% to 53.8073301315% canonical-view weight, retaining the disagreement correction and calibration.
mechanism: Three-point TTA weight interpolation
evidence_used: Equal-sized probes around the current refined weight worsened cross-entropy by approximately 1.9e-9 below and 3.1e-9 above; this asymmetry estimates a local optimum slightly below the current weight.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 46.51033579092473, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456672744750977, "validation_score": 9359.422095259317}

RECENT RESULT
hypothesis: A half-sized continuation of the successful canonical-weight reduction will preserve 9,359 predictions through the existing argmax guard while reducing cross-entropy below 0.18456672744750977.
change: Lower the refined canonical-view weight by 0.00000002384185791015625 and increase its complementary flipped-view weight equally.
mechanism: Guarded half-step TTA weight refinement
evidence_used: Reducing the refined weight from 0.5380733489990234375 to 0.5380733013153076171875 improved cross-entropy from 0.18456672821044923 to 0.18456672744750977 without changing correctness, motivating a finer continuation in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.26668858411722, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456672592163087, "validation_score": 9359.42209525986}

RECENT RESULT
hypothesis: Halving the latest successful canonical-weight reduction will preserve 9,359 predictions through the existing guard while reducing cross-entropy below 0.18456672592163087.
change: Lower the refined canonical-view weight by 0.000000011920928955078125 and increase its flipped-view complement equally.
mechanism: Argmax-guarded quarter-step TTA weight refinement
evidence_used: The two successive guarded reductions to 0.5380733013153076171875 and 0.53807327747344970703125 each improved cross-entropy without changing correctness, motivating a finer continuation in the same direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 79.30593091598712, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18456672973632812, "validation_score": 9359.422095258502}

RECENT RESULT
hypothesis: A low-weight symmetric one-pixel vertical TTA, used only when it preserves the stable prediction and increases that class’s calibrated probability, will retain 9,359 correct predictions while reducing cross-entropy below 0.18456672592163087.
change: Add up/down translation logits during evaluation and selectively adopt their guarded 3.125% blend when it increases confidence in the unchanged stable class.
mechanism: Confidence-gated vertical-translation TTA
evidence_used: The horizontal TTA and argmax guard preserved accuracy through prior refinements, while the latest lower-weight probe worsened cross-entropy, indicating that further scalar weight tuning is locally exhausted and motivating an orthogonal guarded augmentation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 62.74010904203169, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1844868465423584, "validation_score": 9359.422123725104}

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
