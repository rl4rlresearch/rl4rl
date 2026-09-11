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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.70478770881891, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2081323101043701, "validation_score": 9284.4138619552}
prior_hypothesis: Holding vertical weight at 0.10875 while reducing each horizontal radius-one weight to 0.07000 and transferring their combined mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132651.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.91189074981958, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20811298179626464, "validation_score": 9284.413868576477}
prior_hypothesis: Reducing the evaluation logit scale from 1.253 to 1.252 will preserve all 9,284 argmax predictions while lowering cross-entropy below 0.208131822.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.3833637079224, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813265075683593, "validation_score": 9284.413861838504}
prior_hypothesis: Reducing radius-one horizontal weight from 0.070625 to 0.0703125 while transferring its total mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132999.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 29.618943833047524, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813182220458984, "validation_score": 9284.413862122337}
prior_hypothesis: Horizontal weight 0.06953125 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while reducing cross-entropy below 0.208131979.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing each radius-one horizontal weight to 0.0696875 and transferring the combined mass to the center will retain 9,284 correct predictions while lowering cross-entropy below 0.208132310.
change: Advance one 0.0003125 step beyond Reference Design 3 along the validated horizontal-pruning direction, increasing the center weight to 0.363125 while preserving total ensemble mass and all training behavior.
mechanism: Continued center-compensated horizontal-vote pruning
evidence_used: Four consecutive center-compensated horizontal reductions retained 9,284 correct while monotonically lowering cross-entropy, most recently from 0.208132651 at weight 0.0703125 to 0.208132310 at 0.07000.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 56.234569374937564, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813197937011718, "validation_score": 9284.413862068497}

RECENT RESULT
hypothesis: Horizontal weight 0.069375 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions and reduce cross-entropy below 0.208131979.
change: Advance one 0.0003125 step beyond Reference Design 2, transferring the combined 0.000625 horizontal-vote mass to the center while preserving all training behavior.
mechanism: Continued center-compensated horizontal-vote pruning
evidence_used: Five consecutive horizontal-weight reductions with fixed vertical influence retained 9,284 correct and monotonically lowered cross-entropy; Reference Design 2 reached the best result at 0.0696875.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.907434874912724, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.2081316589355469, "validation_score": 9283.413862178266}

RECENT RESULT
hypothesis: Horizontal weight 0.06953125 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while reducing cross-entropy below 0.208131979.
change: Bisect the interval between the best 0.0696875 horizontal weight and the 9,283-correct 0.069375 setting, transferring the removed horizontal mass to the center view.
mechanism: Boundary-bisected center-compensated horizontal-vote pruning
evidence_used: Weight 0.0696875 achieved 9,284 correct at 0.208131979 cross-entropy, while 0.069375 lowered cross-entropy but lost one correct prediction; their midpoint is the most informative probe of the decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 29.618943833047524, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813182220458984, "validation_score": 9284.413862122337}

RECENT RESULT
hypothesis: Horizontal weight 0.069453125 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.
change: Move halfway from the best validated 0.06953125 horizontal weight toward the lower-cross-entropy but 9,283-correct 0.069375 setting, transferring the removed horizontal mass to the center view.
mechanism: Boundary-bisected center-compensated horizontal-vote pruning
evidence_used: Weight 0.06953125 achieved 9,284 correct at 0.208131822 cross-entropy, whereas 0.069375 reduced cross-entropy to 0.208131659 but lost one correct prediction; bisecting this interval is the closest evidence-backed probe of the boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 32.22072441712953, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813174133300782, "validation_score": 9283.41386215004}

RECENT RESULT
hypothesis: Horizontal weight 0.0694921875 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.
change: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the lower-cross-entropy but 9,283-correct 0.069453125 setting, transferring removed horizontal mass to the center view.
mechanism: Fine boundary-bisected horizontal-vote pruning
evidence_used: Weight 0.06953125 achieved 9,284 correct at 0.208131822 cross-entropy, while 0.069453125 achieved 9,283 correct at 0.208131741; their midpoint is the most informative unresolved boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.75013608299196, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813178100585938, "validation_score": 9283.41386213645}

RECENT RESULT
hypothesis: Horizontal weight 0.06951171875 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.
change: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the 9,283-correct 0.0694921875 setting, transferring the removed horizontal-vote mass to the center view.
mechanism: Ultra-fine boundary-bisected horizontal-vote pruning
evidence_used: The 0.06953125 design achieved 9,284 correct at 0.208131822 cross-entropy, while 0.0694921875 lowered cross-entropy to 0.208131781 but lost one correct prediction; their midpoint is the closest unresolved decision-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.249787458917126, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.2081317985534668, "validation_score": 9283.41386213044}

RECENT RESULT
hypothesis: Horizontal weight 0.069521484375 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.
change: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the 9,283-correct 0.06951171875 setting, transferring the removed horizontal-vote mass to the center view.
mechanism: Decision-boundary bisection with center-compensated horizontal-vote pruning
evidence_used: The current 0.06953125 weight achieved 9,284 correct at 0.208131822 cross-entropy, while 0.06951171875 reduced cross-entropy to 0.208131799 but lost one correct prediction; their midpoint is the closest unresolved boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.61616612505168, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813181076049805, "validation_score": 9283.413862126257}

RECENT RESULT
hypothesis: A residual 7×7 class-evidence head using log-mean-exp spatial pooling will exceed 9,284 correct predictions by learning translation-robust part presence, while preserving the validated classifier exactly at initialization.
change: Add a zero-initialized 810-parameter class-map branch before spatial aggregation, fuse its pooled logits with the layout-sensitive classifier, and use Reference Design 3’s strongest validated TTA weights.
mechanism: Zero-initialized class-specific local-evidence pooling
evidence_used: TTA boundary searches plateaued at 9,284 correct, while content-conditioned downsampling fell to 9,280; this retains all validated representation and pooling behavior but challenges the shared assumption that predictions must come solely from the flattened spatial head.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245962, "training_seconds": 38.824362584156916, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.20856401443481445, "validation_score": 9268.413714121907}

RECENT RESULT
hypothesis: Horizontal weight 0.0695263671875 with vertical weight fixed at 0.10875 will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.
change: Bisect the remaining interval between the 9,284-correct 0.06953125 setting and the 9,283-correct 0.069521484375 setting, transferring the removed horizontal-vote mass to the center view.
mechanism: Decision-boundary bisection with center-compensated horizontal-vote pruning
evidence_used: Weight 0.06953125 achieved 9,284 correct with 0.208131822 cross-entropy, while every tested lower weight through 0.069521484375 reduced cross-entropy but lost one prediction; their midpoint is the closest unresolved boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.31419170903973, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813181533813477, "validation_score": 9283.41386212469}

RECENT RESULT
hypothesis: With the best validated horizontal weight fixed at 0.06953125, reducing each radius-one vertical weight to 0.1084375 and transferring their combined mass to the center will retain 9,284 correct predictions while lowering cross-entropy below 0.208131822.
change: Restore Reference Design 2’s best horizontal weighting, then take one isolated 0.0003125 downward step in each vertical vote and add the removed 0.000625 total weight to the center view.
mechanism: Center-compensated vertical-vote pruning
evidence_used: Horizontal pruning has reached a sharp 9,284/9,283 decision boundary, while increased vertical influence was associated with regressions; an isolated downward vertical step is the closest untested, evidence-backed TTA direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.153311083093286, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813221740722657, "validation_score": 9284.413861986955}

RECENT RESULT
hypothesis: Increasing the evaluation logit scale from 1.253 to 1.254 will retain exactly 9,284 correct predictions while reducing validation cross-entropy below 0.208131822.
change: Apply a small positive temperature adjustment to the final evaluation logits without changing training or class rankings.
mechanism: Accuracy-preserving global logit recalibration
evidence_used: TTA-weight optimization has reached a sharp 9,284/9,283 boundary; a positive global scale preserves every non-tied argmax while directly probing the remaining cross-entropy tie-breaker.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 33.35787345794961, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20815124206542968, "validation_score": 9284.413855469904}

RECENT RESULT
hypothesis: Reducing the evaluation logit scale from 1.253 to 1.252 will preserve all 9,284 argmax predictions while lowering cross-entropy below 0.208131822.
change: Decrease only the final evaluation logit multiplier, leaving training and TTA weights unchanged.
mechanism: Accuracy-preserving reverse logit recalibration
evidence_used: Increasing the scale to 1.254 preserved 9,284 correct but worsened cross-entropy to 0.208151242, strongly motivating an equal-sized probe in the opposite direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.91189074981958, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20811298179626464, "validation_score": 9284.413868576477}



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
