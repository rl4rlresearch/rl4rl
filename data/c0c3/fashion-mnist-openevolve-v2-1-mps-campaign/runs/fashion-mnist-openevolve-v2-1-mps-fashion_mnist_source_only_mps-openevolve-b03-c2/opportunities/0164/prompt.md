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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 56.10338895814493, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891174316408, "validation_score": 9257.411719543532}
prior_hypothesis: The next float32 center-view weight above 3.3960161209106445 will preserve 9,257 correct predictions and reduce cross-entropy below 0.21441891098022461.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 38.930886124959216, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441890525817872, "validation_score": 9257.411719545731}
prior_hypothesis: Lowering the positive logit scale to 1.35602 will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21441890335083008.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 54.941300415899605, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441890335083008, "validation_score": 9257.411719546377}
prior_hypothesis: Lowering the positive logit scale to 1.35603 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891059875487.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 59.910388958174735, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891059875487, "validation_score": 9257.411719543921}
prior_hypothesis: Restoring the best verified center-view weight and lowering the positive logit scale to 1.35604 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891098022461.

## Recent verification evidence

RECENT RESULT
hypothesis: A center-view weight of 3.396017551422119 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891098022461.
change: Move halfway between the accuracy-preserving 3.3960161209106445 weight and the accuracy-losing 3.39601898193359375 weight, updating ensemble normalization exactly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.3960161209106445 achieved 9,257 correct with the best verified cross-entropy of 0.21441891098022461, while 3.39601898193359375 achieved 9,256 correct; their midpoint is the most informative unresolved boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center-view weight of 3.396017551422119 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891098022461.
change: Retry the previously timed-out midpoint probe and update the ensemble normalization exactly.
mechanism: Retry unresolved upper-bound refinement
evidence_used: Weight 3.3960161209106445 achieved 9,257 correct, while 3.39601898193359375 achieved 9,256; the intervening 3.396017551422119 probe timed out and remains the most informative unresolved point.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.91678804112598, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891593933105, "validation_score": 9257.41171954211}

RECENT RESULT
hypothesis: A center-view weight of 3.3960182666778564 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891098022461.
change: Move halfway between the highest accuracy-preserving weight and the nearest accuracy-losing weight, updating ensemble normalization exactly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.396017551422119 preserved 9,257 correct, while 3.39601898193359375 produced 9,256; their midpoint is the most informative unresolved boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 53.02003925014287, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891708374022, "validation_score": 9257.411719541722}

RECENT RESULT
hypothesis: The next float32 center-view weight above 3.3960161209106445 will preserve 9,257 correct predictions and reduce cross-entropy below 0.21441891098022461.
change: Set the center-view weight to 3.3960163593292236 and update its exact ensemble normalization.
mechanism: Single-ULP local ensemble-weight refinement
evidence_used: Weight 3.3960161209106445 has the best verified 9,257-correct cross-entropy, while weights six and nine float32 steps higher still preserve 9,257 correct; probing its immediate neighbor is the finest unresolved local search.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 56.10338895814493, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891174316408, "validation_score": 9257.411719543532}

RECENT RESULT
hypothesis: The next float32 center-view weight below 3.3960161209106445 will preserve 9,257 correct predictions and reduce cross-entropy below 0.21441891098022461.
change: Set the center-view weight to 3.3960158824920654 and update its exact ensemble normalization.
mechanism: Single-ULP local ensemble-weight refinement
evidence_used: Weight 3.3960161209106445 has the best verified 9,257-correct cross-entropy, while its immediate higher float32 neighbor worsened cross-entropy; the immediate lower neighbor is the most informative unresolved local probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 53.850839250022545, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891098022461, "validation_score": 9257.41171954379}

RECENT RESULT
hypothesis: The next float32 center-view weight below 3.3960158824920654 will preserve 9,257 correct predictions and reduce validation cross-entropy below 0.21441891098022461.
change: Set the center-view weight to 3.3960156440734863 and update its exact ensemble normalization.
mechanism: Single-ULP downward ensemble-weight refinement
evidence_used: The immediate lower neighbor of the current weight tied its best 9,257-correct cross-entropy, while the immediate higher neighbor worsened it; continuing one float32 step downward is the finest unresolved local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 47.04526875005104, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891098022461, "validation_score": 9257.41171954379}

RECENT RESULT
hypothesis: The next float32 center-view weight below 3.3960156440734863 will preserve 9,257 correct predictions and reduce validation cross-entropy below 0.21441891098022461.
change: Set the center-view weight to 3.3960154056549072 and update the exact ensemble normalization.
mechanism: Single-ULP downward ensemble-weight refinement
evidence_used: The two consecutive lower float32 weights, 3.3960158824920654 and 3.3960156440734863, both preserved 9,257 correct with the best cross-entropy; continuing one ULP downward is the finest unresolved local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 76.80462695797905, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891174316408, "validation_score": 9257.411719543532}

RECENT RESULT
hypothesis: Reusing the best verified center-view weight and increasing the positive logit scale to 1.35606 will preserve 9,257 correct predictions while reducing cross-entropy below 0.21441891098022461.
change: Restore the best center-view weight and exact normalization, then make a small upward calibration probe of the final logit scale.
mechanism: Accuracy-invariant logit-temperature refinement
evidence_used: Weight 3.3960161209106445 achieved the best verified 9,257-correct cross-entropy, while adjacent center-weight probes tied or worsened it; positive logit scaling preserves class ordering and isolates the remaining calibration axis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 52.20727033400908, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891975402833, "validation_score": 9257.411719540816}

RECENT RESULT
hypothesis: Restoring the best verified center-view weight and lowering the positive logit scale to 1.35604 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891098022461.
change: Restore center weight 3.3960161209106445 with exact normalization and probe the opposite calibration direction from the unsuccessful 1.35606 scale.
mechanism: Symmetric logit-temperature refinement
evidence_used: Scale 1.35605 achieved the best verified cross-entropy, while increasing it to 1.35606 worsened cross-entropy without changing accuracy; an equally sized downward probe is the most informative unresolved calibration test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 59.910388958174735, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891059875487, "validation_score": 9257.411719543921}

RECENT RESULT
hypothesis: Lowering the positive logit scale from 1.35604 to 1.35603 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891059875487.
change: Keep the best verified center-view ensemble unchanged and decrease only its final logit scale by 0.00001.
mechanism: Downward logit-temperature refinement
evidence_used: Scale 1.35604 improved cross-entropy over 1.35605, while increasing to 1.35606 worsened it; the next equal-sized downward probe is the most informative unresolved calibration test and cannot change class ordering.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the positive logit scale to 1.35603 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891059875487.
change: Keep the best verified center-view ensemble unchanged and decrease only the final logit scale from 1.35605 to 1.35603.
mechanism: Retry unresolved downward logit-temperature refinement
evidence_used: Scale 1.35604 improved cross-entropy over both 1.35605 and 1.35606 without changing accuracy; the only 1.35603 verification timed out, leaving the next downward calibration probe unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 54.941300415899605, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441890335083008, "validation_score": 9257.411719546377}

RECENT RESULT
hypothesis: Lowering the positive logit scale to 1.35602 will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21441890335083008.
change: Restore the ensemble configuration from the best verified design and decrease only its accuracy-invariant final logit scale from 1.35603 to 1.35602.
mechanism: Downward logit-temperature refinement
evidence_used: Scale 1.35603 improved cross-entropy over 1.35604 while preserving 9,257 correct predictions; the next equal-sized downward probe is the most informative unresolved calibration test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 38.930886124959216, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441890525817872, "validation_score": 9257.411719545731}



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
