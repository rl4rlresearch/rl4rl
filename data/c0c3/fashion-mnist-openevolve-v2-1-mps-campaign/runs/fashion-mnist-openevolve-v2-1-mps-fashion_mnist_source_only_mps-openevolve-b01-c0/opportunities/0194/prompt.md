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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 51.57202141592279, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246061096191406, "validation_score": 9290.415814036187}
prior_hypothesis: Setting the inference temperature to 0.717663049697876 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Raising the inference temperature to 0.717574857711792 will preserve all 9,290 argmax predictions and further reduce validation cross-entropy.
change: Move the final inference temperature to the estimated minimum of the three measured temperature–cross-entropy points.
mechanism: Quadratic temperature interpolation
evidence_used: Symmetric probes around 0.717143714427948 improved cross-entropy when increased to 0.717287428855896 and worsened it when decreased to 0.717; quadratic interpolation places the estimated optimum three probe increments above the center.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 49.66013358393684, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246061935424806, "validation_score": 9290.415814033286}

RECENT RESULT
hypothesis: Raising the inference temperature to 0.7176308860778809 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061935424806.
change: Move the final inference temperature to the nearest float32 value of the quadratic minimum estimated from the three latest informative temperature measurements.
mechanism: Local quadratic temperature refinement
evidence_used: Cross-entropy improved from 0.2024606979370117 at 0.717143714427948 to 0.20246065788269044 at 0.717287428855896 and 0.20246061935424806 at 0.717574857711792; fitting these points places the local minimum near 0.71763089, while positive temperature scaling preserves argmax predictions.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 79.83381745801307, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246061401367188, "validation_score": 9290.415814035132}

RECENT RESULT
hypothesis: Setting the inference temperature to 0.7176308860778809 will preserve all 9,290 correct predictions and reduce validation cross-entropy below 0.20246061935424806.
change: Replace only the final ensemble temperature with the empirically better value from the timed-out verification.
mechanism: Argmax-invariant temperature retry
evidence_used: The prior attempt at 0.7176308860778809 retained 9,290 correct predictions and achieved lower cross-entropy of 0.20246061401367188; its failure was solely the variable verification time limit.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 67.4197439160198, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246061401367188, "validation_score": 9290.415814035132}

RECENT RESULT
hypothesis: Increasing the inference temperature to 0.7180816164016724 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.
change: Replace only the final ensemble temperature with the float32 value nearest the minimum estimated from the three latest temperature measurements.
mechanism: Uneven-spacing quadratic temperature extrapolation
evidence_used: Cross-entropy improved monotonically from 0.20246065788269044 at 0.717287428855896 to 0.20246061935424806 at 0.717574857711792 and 0.20246061401367188 at 0.7176308860778809; fitting an unevenly spaced quadratic to these points estimates the minimum near 0.7180816, while positive temperature scaling preserves argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.32624862506054, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246067008972168, "validation_score": 9290.41581401574}

RECENT RESULT
hypothesis: Setting the inference temperature to 0.7177127828598023 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.
change: Replace only the final ensemble temperature with the nearest float32 value to the quadratic minimum bracketed by the latest temperature measurements.
mechanism: Bracketed quadratic temperature interpolation
evidence_used: Cross-entropy was higher at both 0.717574857711792 and 0.7180816164016724 than at 0.7176308860778809; fitting those three measurements places the minimum near 0.7177128, while positive temperature scaling preserves predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 45.82920425012708, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246061515808106, "validation_score": 9290.415814034735}

RECENT RESULT
hypothesis: Setting the inference temperature to 0.717663049697876 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.
change: Replace only the final inference temperature with the nearest float32 value to the minimum estimated from the tightest three-point bracket.
mechanism: Bracketed local quadratic temperature refinement
evidence_used: Cross-entropy is higher at both 0.717574857711792 and 0.7177127828598023 than at 0.7176308860778809; quadratic interpolation of those measurements estimates the minimum near 0.717663, while positive temperature scaling leaves predictions unchanged.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 51.57202141592279, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20246061096191406, "validation_score": 9290.415814036187}

RECENT RESULT
hypothesis: Setting the inference temperature to 0.7176686525344849 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.
change: Replace only the final inference temperature with the nearest float32 value to the quadratic minimum estimated from the tightest available bracket.
mechanism: Tight-bracket quadratic temperature refinement
evidence_used: Cross-entropy is higher at both 0.7176308860778809 and 0.7177127828598023 than at the current best 0.717663049697876; quadratic interpolation of these three measurements places the minimum near 0.71766865, while positive temperature scaling preserves argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 49.99554412509315, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024606170654297, "validation_score": 9290.415814034077}



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
