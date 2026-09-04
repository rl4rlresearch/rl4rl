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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 43.480353417107835, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2144188995361328, "validation_score": 9257.41171954767}
prior_hypothesis: The next float32 logit scale below 1.35603 will preserve exactly 9,257 correct predictions and reduce validation cross-entropy below 0.21441890335083008.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.216582708992064, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.214408358001709, "validation_score": 9257.411723121555}
prior_hypothesis: Blending 1/64 geometric consensus into the established arithmetic ensemble will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.2144188995361328.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 58.75192158296704, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436964797973632, "validation_score": 9257.411736245906}
prior_hypothesis: A 5/64 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214378804397583.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.313613874837756, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.214378804397583, "validation_score": 9257.411733141413}
prior_hypothesis: A 1/16 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.2143981575012207.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the qualified compensated scale will preserve exactly 9,257 correct predictions while reducing validation cross-entropy from 0.21441891059875487 to approximately 0.2144188995361328.
change: Replace the inferior 1.35604 calibration with the best verified base scale plus its separately rounded sub-ULP residual.
mechanism: Best-verified compensated float32 logit calibration
evidence_used: Reference Design 1 achieved the highest available validation score, 9257.41171954767, with identical accuracy and lower cross-entropy than the current design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A half-sized compensated residual will preserve 9,257 correct predictions while producing validation cross-entropy below 0.2144188995361328.
change: Add a separately rounded 2.70930205e-9 residual multiplication to the best verified base logit scale.
mechanism: Midpoint compensated float32 logit calibration
evidence_used: The base scale and the 5.4186041e-9 compensated variant tied for the best cross-entropy, while the adjacent lower float32 scale was worse; testing their midpoint probes a distinct sub-ULP rounding pattern inside the winning calibration interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.66498887492344, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2144188995361328, "validation_score": 9257.41171954767}

RECENT RESULT
hypothesis: Adding diagonal translations at weight 1/3.3960161209106445 will alter class rankings constructively and increase validation_correct above the 9,257 plateau.
change: Extend the existing center-and-cardinal ensemble with four diagonal views, using the diagonal weight implied by a separable translation kernel while preserving existing center/cardinal weights.
mechanism: Separable 3×3 translation test-time ensemble
evidence_used: Multiple logit-calibration refinements preserved exactly 9,257 correct predictions and tied at 0.2144188995361328 cross-entropy, so further temperature probes are unlikely to improve the primary objective; the established weighted shift ensemble motivates testing its missing diagonal components.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 65.01236358308233, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.2154078140258789, "validation_score": 9247.411384552765}

RECENT RESULT
hypothesis: Restoring the qualified compensated scale will preserve 9,257 correct predictions and reduce validation cross-entropy from 0.21441890411376954 to approximately 0.2144188995361328.
change: Replace the inferior one-ULP-lower scale with the best verified base scale plus its separately rounded sub-ULP residual.
mechanism: Best-qualified compensated float32 logit calibration
evidence_used: Reference Design 1 achieved the highest available validation_score, 9257.41171954767, with identical accuracy and lower cross-entropy than the current design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the qualified compensated scale will preserve exactly 9,257 correct predictions and reduce validation cross-entropy from 0.21441891059875487 to approximately 0.2144188995361328.
change: Replace the current inferior calibration with the best verified base scale plus its separately rounded sub-ULP residual.
mechanism: Best-qualified compensated float32 logit calibration
evidence_used: Reference Design 1 achieved the highest available validation_score, 9257.41171954767, with identical accuracy and lower cross-entropy than the current design; the same compensated expression was previously valid and improved the objective.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Blending 1/8 weighted geometric consensus into the existing probability-average ensemble will suppress transformation-specific outlier predictions and increase validation_correct above 9,257.
change: Keep the trained model and proven center/cardinal views unchanged, but blend their current arithmetic probability fusion with a center-weighted geometric fusion before logit calibration.
mechanism: Robust arithmetic–geometric TTA fusion
evidence_used: Temperature refinements repeatedly plateaued at 9,257 correct, while adding diagonal views reduced accuracy to 9,247; this motivates changing how the established views are fused without introducing weaker views.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 60.137730250135064, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21433232460021973, "validation_score": 9256.411748900915}

RECENT RESULT
hypothesis: Blending 1/64 geometric consensus into the established arithmetic ensemble will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.2144188995361328.
change: Retain the proven views, weights, and compensated calibration, but add a conservative geometric-consensus component to the fused probabilities.
mechanism: Low-dose arithmetic–geometric TTA fusion
evidence_used: A 1/8 geometric blend reduced cross-entropy to 0.21433232460021973 but lost one correct prediction; a much smaller blend tests whether its calibration benefit can be retained before that ranking boundary is crossed.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 42.216582708992064, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.214408358001709, "validation_score": 9257.411723121555}

RECENT RESULT
hypothesis: Increasing the geometric-consensus blend from 1/64 to 1/32 will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214408358001709.
change: Restore the qualified arithmetic–geometric ensemble and compensated calibration, using a 1/32 geometric blend to probe the interval between the successful 1/64 blend and accuracy-reducing 1/8 blend.
mechanism: Moderate-dose arithmetic–geometric TTA fusion
evidence_used: The 1/64 blend preserved 9,257 correct and improved cross-entropy, while 1/8 lost only one prediction and improved cross-entropy further; 1/32 is a conservative increase toward that demonstrated calibration benefit.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 46.8521305420436, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2143981575012207, "validation_score": 9257.41172657988}

RECENT RESULT
hypothesis: A 1/16 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.2143981575012207.
change: Replace the arithmetic-only ensemble with a 15/16 arithmetic and 1/16 center-weighted geometric probability blend, retaining the best verified compensated logit calibration.
mechanism: Intermediate-dose arithmetic–geometric TTA fusion
evidence_used: Increasing the geometric blend from 1/64 to 1/32 preserved 9,257 correct predictions and improved cross-entropy, while 1/8 lost only one correct prediction and reduced cross-entropy further; 1/16 probes the midpoint of that ranking boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.313613874837756, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.214378804397583, "validation_score": 9257.411733141413}

RECENT RESULT
hypothesis: A 3/32 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214378804397583.
change: Increase the geometric fusion weight from 1/16 to 3/32, retaining all trained-model, view-weighting, and calibration settings.
mechanism: Boundary-seeking arithmetic–geometric TTA fusion
evidence_used: The 1/16 blend preserved 9,257 correct and improved cross-entropy, whereas 1/8 reduced cross-entropy further but lost one correct; 3/32 is the midpoint that probes the remaining ranking boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 62.134076208109036, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436085662841797, "validation_score": 9256.411739226665}

RECENT RESULT
hypothesis: A 5/64 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214378804397583.
change: Increase the geometric fusion weight from 1/32 to 5/64, retaining the trained model, established views, view weights, and compensated calibration.
mechanism: Binary search of the geometric-fusion ranking boundary
evidence_used: The 1/16 blend retained 9,257 correct with 0.214378804397583 cross-entropy, while 3/32 lost one correct but further reduced cross-entropy; 5/64 is their midpoint and directly probes the remaining ranking boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 58.75192158296704, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21436964797973632, "validation_score": 9257.411736245906}

RECENT RESULT
hypothesis: An 11/128 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.
change: Increase geometric fusion from 1/64 to 11/128 while retaining the trained model, established views, view weights, and compensated calibration.
mechanism: Final binary search of the geometric-fusion ranking boundary
evidence_used: A 5/64 blend preserved 9,257 correct with 0.21436964797973632 cross-entropy, while 3/32 lost one correct but reduced cross-entropy further; 11/128 is their exact midpoint and isolates the remaining ranking boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 49.529807541985065, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21436521072387696, "validation_score": 9256.411737750377}



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
