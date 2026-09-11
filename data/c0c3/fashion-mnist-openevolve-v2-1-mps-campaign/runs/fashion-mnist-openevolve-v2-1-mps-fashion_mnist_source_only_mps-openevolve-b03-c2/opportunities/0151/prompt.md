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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 67.94642099994235, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441883888244628, "validation_score": 9256.411719568234}
prior_hypothesis: A center-view weight of 3.3963623046875 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 78.94425720791332, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441887741088866, "validation_score": 9256.411719555172}
prior_hypothesis: A center-view weight of 3.39617919921875 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.985442999983206, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891708374022, "validation_score": 9257.411719541722}
prior_hypothesis: A center-view weight of 3.3960132598876953 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891479492187.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 60.95323512493633, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891479492187, "validation_score": 9257.411719542497}
prior_hypothesis: A center-view weight of 3.396007537841797 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.

## Recent verification evidence

RECENT RESULT
hypothesis: A center-view weight of 3.400390625 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.
change: Move halfway from the best verified weight, 3.39453125, toward the accuracy-losing 3.40625 weight; update normalization and use the verified 1.35605 calibration.
mechanism: Upper-bound refinement of center-view ensemble weighting
evidence_used: Weight 3.39453125 achieved the best result with 9,257 correct, while 3.40625 achieved 9,255 with slightly lower cross-entropy; their midpoint is the most informative probe of the upper accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 77.57328162482008, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21441797485351563, "validation_score": 9255.411719861162}

RECENT RESULT
hypothesis: A center-view weight of 3.3974609375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.
change: Move halfway from the best verified weight, 3.39453125, toward the accuracy-losing 3.400390625 weight, update normalization, and adopt the verified 1.35605 calibration.
mechanism: Upper-bound refinement of center-view ensemble weighting
evidence_used: Weight 3.39453125 achieved 9,257 correct, while 3.400390625 achieved 9,255 with lower cross-entropy; their midpoint is the most informative probe of the upper 9,257-correct boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 57.43195170792751, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441860122680664, "validation_score": 9256.411719648806}

RECENT RESULT
hypothesis: A center-view weight of 3.39599609375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.
change: Move halfway from the best verified weight, 3.39453125, toward the 3.3974609375 weight that lost one correct prediction, and update the ensemble normalization constant.
mechanism: Upper-bound refinement of center-view ensemble weighting
evidence_used: Weight 3.39453125 achieved the best result with 9,257 correct, while 3.3974609375 produced 9,256 correct with slightly lower cross-entropy; their midpoint is the most informative probe of the upper 9,257-correct boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 66.21664587501436, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441892204284668, "validation_score": 9257.411719540041}

RECENT RESULT
hypothesis: A center-view weight of 3.396728515625 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.3974609375 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct, while 3.3974609375 achieved 9,256 with lower cross-entropy; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 54.90483679110184, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441875915527345, "validation_score": 9256.411719595264}

RECENT RESULT
hypothesis: A center-view weight of 3.3963623046875 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.396728515625 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, while 3.396728515625 achieved 9,256 correct at the lower 0.2144187592; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 67.94642099994235, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441883888244628, "validation_score": 9256.411719568234}

RECENT RESULT
hypothesis: A center-view weight of 3.39617919921875 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.3963623046875 weight, update ensemble normalization, and use the verified 1.35605 calibration.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, whereas 3.3963623046875 achieved 9,256 correct at 0.2144188389; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 78.94425720791332, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441887741088866, "validation_score": 9256.411719555172}

RECENT RESULT
hypothesis: A center-view weight of 3.396087646484375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.39617919921875 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, while 3.39617919921875 achieved 9,256 correct at the lower 0.2144188774; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 75.49562166701071, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144188995361328, "validation_score": 9256.41171954767}

RECENT RESULT
hypothesis: A center-view weight of 3.3960418701171875 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.396087646484375 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, while 3.396087646484375 achieved 9,256 correct at the lower 0.2144188995; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 47.42738966597244, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144189094543457, "validation_score": 9256.411719544309}

RECENT RESULT
hypothesis: A center-view weight of 3.39601898193359375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.3960418701171875 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, while 3.3960418701171875 achieved 9,256 correct at the lower 0.2144189095; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 60.11205999995582, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144189125061035, "validation_score": 9256.411719543274}

RECENT RESULT
hypothesis: A center-view weight of 3.396007537841797 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.39601898193359375 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct at 0.2144189220 cross-entropy, while 3.39601898193359375 achieved 9,256 correct at the lower 0.2144189125; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center-view weight of 3.396007537841797 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Retry the previously timed-out boundary probe and update the ensemble normalization constant exactly.
mechanism: Retry unresolved upper-bound refinement
evidence_used: Weight 3.39599609375 achieved 9,257 correct, while 3.39601898193359375 achieved 9,256 with lower cross-entropy; the intervening 3.396007537841797 probe timed out and therefore remains the most informative unresolved point.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 60.95323512493633, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891479492187, "validation_score": 9257.411719542497}

RECENT RESULT
hypothesis: A center-view weight of 3.3960132598876953 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441891479492187.
change: Move halfway between the accuracy-preserving 3.396007537841797 weight and the accuracy-losing 3.39601898193359375 weight, updating the ensemble normalization exactly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.396007537841797 achieved 9,257 correct at 0.2144189148 cross-entropy, while 3.39601898193359375 achieved 9,256 correct at the slightly lower 0.2144189125; their midpoint is the most informative unresolved probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.985442999983206, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891708374022, "validation_score": 9257.411719541722}



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
