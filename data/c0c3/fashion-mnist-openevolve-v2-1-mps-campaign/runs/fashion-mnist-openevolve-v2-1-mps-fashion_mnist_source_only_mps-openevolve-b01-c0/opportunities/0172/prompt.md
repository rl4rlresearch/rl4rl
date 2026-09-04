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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 56.30717062507756, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}
prior_hypothesis: Setting each unshifted view’s weight to 1.0665515661239624 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246069984436035.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting the unshifted-view weight to 1.0665515884757042 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024606979370117.
change: Increase the original and horizontally flipped original views’ TTA weight to the midpoint between the highest tested accuracy-preserving value and the lowest tested value that lost one prediction.
mechanism: Log-domain boundary bisection
evidence_used: Weight 1.0665515810251236 preserved 9,290 correct, while 1.0665515959262848 produced 9,289 with lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 57.425102290930226, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: Setting the unshifted-view weight to 1.066551584750414 will preserve 9,290 correct predictions and may reduce cross-entropy below 0.2024606979370117.
change: Increase the original and horizontally flipped original views’ TTA weight to the midpoint between the highest tested accuracy-preserving value and the lowest tested value that lost one prediction.
mechanism: Log-domain boundary bisection
evidence_used: Weight 1.0665515810251236 preserved 9,290 correct, while 1.0665515884757042 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 66.90020741592161, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515866130591 will preserve 9,290 correct predictions and may reduce cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain boundary bisection
evidence_used: Weight 1.066551584750414 preserved 9,290 correct, while 1.0665515884757042 produced 9,289; their midpoint is the most informative remaining probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 72.28931245789863, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515875443816 will preserve 9,290 correct predictions and may reduce cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain boundary bisection
evidence_used: Weight 1.0665515866130591 preserved 9,290 correct, while 1.0665515884757042 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 54.315151875140145, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515870787203 will preserve 9,290 correct predictions and may reduce cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain boundary bisection
evidence_used: Weight 1.0665515866130591 preserved 9,290 correct, while 1.0665515875443816 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 51.41519212513231, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868458897 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.0665515866130591 preserved 9,290 correct, while 1.0665515870787203 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.71250320808031, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515867294744 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.0665515866130591 preserved 9,290 correct, while 1.0665515868458897 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 49.7319152909331, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.066551586787682 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.0665515867294744 preserved 9,290 correct, while 1.0665515868458897 produced 9,289; their midpoint is the most informative remaining probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 53.03832895797677, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868167859 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.066551586787682 preserved 9,290 correct, while 1.0665515868458897 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.860704583115876, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868313378 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.0665515868167859 preserved 9,290 correct, while 1.0665515868458897 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 68.6283289170824, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868386138 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.0665515868313378 preserved 9,290 correct, while 1.0665515868458897 produced 9,289; their midpoint is the most informative remaining boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868386138 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.
change: Set the original and horizontally flipped original views’ shared TTA weight to the unresolved midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary retry
evidence_used: Weight 1.0665515868313378 preserved 9,290 predictions, while 1.0665515868458897 produced 9,289. Their midpoint was not evaluated because verification timed out; otherwise identical runs completed in 49.7–72.3 seconds, so retrying it remains the most informative probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 62.17146491701715, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}



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
