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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 55.08009249996394, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246076278686523, "validation_score": 9290.415813983685}
prior_hypothesis: Setting each unshifted view’s weight to 1.06640625 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising each unshifted view’s weight to 1.0625 will retain 9,290 correct predictions while lowering validation cross-entropy below 0.20248969497680663.
change: Give the original and horizontally flipped original views 1.0625× weight, with shifted views remaining at 1×.
mechanism: Accuracy-preserving canonical-view micro-upweighting
evidence_used: Weight 1.5 lowered cross-entropy to 0.20231712341308594 but lost six correct predictions; this one-eighth step toward that setting tests whether some calibration benefit is available before accuracy changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 62.289370709098876, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246240730285645, "validation_score": 9290.41581341501}

RECENT RESULT
hypothesis: Raising each unshifted view’s weight to 1.125 will retain 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.125×, leaving shifted views and all other calibration unchanged.
mechanism: Progressive canonical-view micro-upweighting
evidence_used: Moving from 1.0× to 1.0625× preserved 9,290 correct and improved cross-entropy from 0.20248969497680663 to 0.20246240730285645, while 1.5× improved cross-entropy further but lost accuracy; 1.125× is the next informative step toward that tradeoff boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.76517183310352, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.2024369430541992, "validation_score": 9288.415822220773}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.09375 will retain 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.
change: Move the original and horizontally flipped original views’ TTA weight halfway from the current best 1.0625 toward the accuracy-losing 1.125 setting.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0625 preserved 9,290 correct with 0.20246240730285645 cross-entropy, while 1.125 reduced cross-entropy to 0.2024369430541992 but lost two correct predictions; their midpoint directly probes the accuracy-preserving boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.09375 will retain 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.09375× while leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection retry
evidence_used: Weight 1.0625 preserved 9,290 correct, whereas 1.125 lost two correct but improved cross-entropy; the midpoint probe timed out, so retrying it is the most informative unresolved boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 73.29154329188168, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20244945182800292, "validation_score": 9289.41581789508}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.078125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.078125×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0625 retained 9,290 correct, while 1.09375 lost one correct but lowered cross-entropy; bisecting these settings directly probes for further calibration improvement before the observed accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 49.22190308384597, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2024558723449707, "validation_score": 9289.41581567482}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0703125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.0703125×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0625 retained 9,290 correct with 0.20246240730285645 cross-entropy, while 1.078125 lost one correct but lowered cross-entropy to 0.2024558723449707; their midpoint is the most informative probe of the accuracy-preserving boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 65.08369504101574, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.202459122467041, "validation_score": 9289.415814550914}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.06640625 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.06640625×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0625 retained 9,290 correct, while 1.0703125 lost one correct but reduced cross-entropy; their midpoint is the most informative probe of the accuracy-preserving boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 55.08009249996394, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246076278686523, "validation_score": 9290.415813983685}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.068359375 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.068359375×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06640625 preserved 9,290 correct with 0.20246076278686523 cross-entropy, while 1.0703125 lost one correct but lowered cross-entropy; their midpoint is the most informative probe of the accuracy-preserving boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.59110891609453, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20245993690490724, "validation_score": 9289.415814269278}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0673828125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.0673828125×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06640625 preserved 9,290 correct, while 1.068359375 lost one correct but lowered cross-entropy; their midpoint is the most informative probe of the accuracy-preserving boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 66.88462441600859, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246035232543946, "validation_score": 9289.415814125625}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.06689453125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.06689453125×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06640625 preserved 9,290 correct, while 1.0673828125 lost one correct but reduced cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.06689453125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.06689453125×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection retry
evidence_used: Weight 1.06640625 retained 9,290 correct, while 1.0673828125 lost one correct but lowered cross-entropy; their midpoint remains unresolved because its verification timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 74.40238141594455, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246055564880372, "validation_score": 9289.415814055314}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.066650390625 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.066650390625×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06640625 preserved 9,290 correct, while 1.06689453125 lost one correct but lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 59.29134212504141, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2024606559753418, "validation_score": 9289.415814020622}



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
