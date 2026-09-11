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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.70906737493351, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246070098876953, "validation_score": 9290.415814005055}
prior_hypothesis: Setting each unshifted view’s weight to 1.06655120849609375 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070251464843.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0665283203125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.0665283203125×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06640625 preserved 9,290 correct, while 1.066650390625 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.43639120901935, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024607120513916, "validation_score": 9290.41581400123}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.06658935546875 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024607120513916.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0665283203125× to 1.06658935546875×, leaving all shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0665283203125 preserved 9,290 correct, while 1.066650390625 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.06658935546875 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024607120513916.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0665283203125× to 1.06658935546875×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection retry
evidence_used: Weight 1.0665283203125 preserved 9,290 correct, while 1.066650390625 lost one correct with slightly lower cross-entropy; their midpoint remains the most informative unresolved boundary probe because its only verification timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 59.954880917211995, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246068420410157, "validation_score": 9289.41581401086}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.066558837890625 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024607120513916.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0665283203125× to 1.066558837890625×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0665283203125 preserved 9,290 correct, while 1.06658935546875 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 60.3721764581278, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069374084472, "validation_score": 9289.415814007561}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0665435791015625 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024607120513916.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0665283203125× to 1.0665435791015625×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0665283203125 preserved 9,290 correct, while 1.066558837890625 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 52.605748041998595, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246070251464843, "validation_score": 9290.415814004527}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.06655120849609375 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070251464843.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.0665435791015625× to 1.06655120849609375×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.0665435791015625 preserved 9,290 correct, while 1.066558837890625 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.70906737493351, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246070098876953, "validation_score": 9290.415814005055}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0665550231933594 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070098876953.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06655120849609375× to 1.0665550231933594×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06655120849609375 preserved 9,290 correct, while 1.066558837890625 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.383112707873806, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069831848146, "validation_score": 9289.415814005979}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0665531158447266 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070098876953.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06655120849609375× to 1.0665531158447266×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06655120849609375 preserved 9,290 correct, while 1.0665550231933594 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.8843849170953, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069869995117, "validation_score": 9289.415814005846}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.0665521621704102 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070098876953.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06655120849609375× to 1.0665521621704102×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06655120849609375 preserved 9,290 correct, while 1.0665531158447266 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.87531654187478, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246070022583007, "validation_score": 9289.415814005319}

RECENT RESULT
hypothesis: Setting each unshifted view’s weight to 1.066551685333252 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070098876953.
change: Increase the original and horizontally flipped original views’ TTA weight from 1.06655120849609375× to 1.066551685333252×, leaving shifted views unchanged.
mechanism: Canonical-view weight boundary bisection
evidence_used: Weight 1.06655120849609375 preserved 9,290 correct, while 1.0665521621704102 lost one correct with lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 42.6169190001674, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}



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
