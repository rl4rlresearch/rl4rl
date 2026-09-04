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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 57.86884120805189, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246065788269044, "validation_score": 9290.415814019962}
prior_hypothesis: Increasing the ensemble temperature to 0.717287428855896 will preserve all 9,290 correct predictions and reduce cross-entropy if the observed degradation at 0.717 indicates the ensemble is slightly overconfident.

## Recent verification evidence

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868322473 will preserve 9,290 correct predictions while probing for lower validation cross-entropy.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Log-domain accuracy-boundary bisection
evidence_used: Weight 1.0665515868313378 preserved 9,290 predictions, while 1.0665515868331568 produced 9,289; their midpoint is the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 59.3676905001048, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868317925 will preserve 9,290 correct predictions while testing the narrowest unresolved accuracy boundary.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868313378 preserved 9,290 correct predictions, while 1.0665515868322473 produced 9,289; their midpoint is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 66.7103769578971, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868315652 will preserve 9,290 correct predictions while resolving the narrowest remaining accuracy boundary.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868313378 preserved 9,290 correct predictions, while 1.0665515868317925 produced 9,289; their midpoint is the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 66.42079495894723, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868314515 will preserve 9,290 correct predictions while testing the narrowest unresolved accuracy boundary.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868313378 preserved 9,290 correct predictions, while 1.0665515868315652 produced 9,289; their midpoint is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 75.92771891714074, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868315084 will preserve 9,290 correct predictions while resolving the narrowest remaining accuracy boundary.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868314515 preserved 9,290 correct predictions, while 1.0665515868315652 produced 9,289; their midpoint is the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 47.21470291703008, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868315368 will preserve 9,290 correct predictions while resolving the narrowest remaining accuracy boundary.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868315084 preserved 9,290 correct predictions, while 1.0665515868315652 produced 9,289; their midpoint is the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 52.698312708875164, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.066551586831551 will preserve 9,290 correct predictions while resolving the narrowest remaining accuracy boundary.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868315368 preserved 9,290 correct predictions, while 1.0665515868315652 produced 9,289; their midpoint is the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 59.468972250120714, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868315439 will preserve 9,290 correct predictions while probing for lower validation cross-entropy.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest unresolved accuracy boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868315368 preserved 9,290 predictions, while 1.066551586831551 produced 9,289; their midpoint is the most informative remaining probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 65.87353554205038, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: An unshifted-view weight of 1.0665515868315475 will preserve 9,290 correct predictions while testing the narrowest unresolved boundary for lower validation cross-entropy.
change: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.
mechanism: Accuracy-boundary bisection
evidence_used: Weight 1.0665515868315439 preserved 9,290 correct predictions, while 1.066551586831551 produced 9,289; their midpoint is the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 77.1984787080437, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20246069946289064, "validation_score": 9289.415814005582}

RECENT RESULT
hypothesis: Lowering the ensemble temperature by one float32 ULP will preserve all 9,290 predictions while slightly reducing validation cross-entropy if the ensemble remains marginally underconfident.
change: Decrease only the final inference temperature from 0.717143714427948 to its adjacent lower float32 value.
mechanism: Argmax-invariant temperature refinement
evidence_used: Unshifted-view weighting has reached a quantized accuracy boundary without improving the reported cross-entropy; positive temperature scaling leaves argmax predictions unchanged, making it the safest remaining tie-breaker probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 46.94158320920542, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606979370117, "validation_score": 9290.41581400611}

RECENT RESULT
hypothesis: Lowering the ensemble temperature to 0.717 will preserve all 9,290 predictions while producing a measurable cross-entropy reduction if the ensemble remains underconfident.
change: Apply a larger, still conservative temperature decrement after the one-ULP change was hidden by metric quantization.
mechanism: Argmax-invariant temperature plateau escape
evidence_used: The adjacent-lower-float32 temperature preserved 9,290 correct predictions but left reported cross-entropy unchanged; positive temperature scaling preserves argmax, so a larger decrement is the safest informative tie-breaker probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 52.593087166082114, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246075401306152, "validation_score": 9290.41581398672}

RECENT RESULT
hypothesis: Increasing the ensemble temperature to 0.717287428855896 will preserve all 9,290 correct predictions and reduce cross-entropy if the observed degradation at 0.717 indicates the ensemble is slightly overconfident.
change: Increase the final inference temperature by the same magnitude as the unsuccessful decrease from 0.717143714427948 to 0.717.
mechanism: Argmax-invariant temperature direction reversal
evidence_used: Lowering the temperature to 0.717 preserved accuracy but worsened cross-entropy from 0.2024606979370117 to 0.20246075401306152; probing the opposite direction is the most informative argmax-safe response.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 57.86884120805189, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246065788269044, "validation_score": 9290.415814019962}



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
