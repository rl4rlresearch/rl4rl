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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 59.910388958174735, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441891059875487, "validation_score": 9257.411719543921}
prior_hypothesis: Restoring the best verified center-view weight and lowering the positive logit scale to 1.35604 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891098022461.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 62.563669957919046, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2144188995361328, "validation_score": 9257.41171954767}
prior_hypothesis: Adding a float32 residual equivalent to 1/22 ULP above the best verified scale will preserve 9,257 correct predictions and reduce cross-entropy below 0.2144188995361328.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.88402358395979, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441890411376954, "validation_score": 9257.411719546119}
prior_hypothesis: The next float32 logit scale below 1.3560298681259155 will preserve exactly 9,257 correct predictions and reduce validation cross-entropy below 0.2144188995361328.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 43.480353417107835, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2144188995361328, "validation_score": 9257.41171954767}
prior_hypothesis: The next float32 logit scale below 1.35603 will preserve exactly 9,257 correct predictions and reduce validation cross-entropy below 0.21441890335083008.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A scale of 1.3560298735445196, evaluated in float64 before casting back, will preserve 9,257 predictions while reducing cross-entropy below 0.2144188995361328.
change: Restore the best scale neighborhood and realize the quadratic-interpolated optimum between adjacent float32 values using a float64 multiplication.
mechanism: Sub-ULP mixed-precision logit calibration
evidence_used: The best scale, 1.3560298681259155, beat both adjacent float32 values; their asymmetric degradations place the interpolated minimum about 1/22 ULP above it.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Computing the interpolated scale 1.3560298735445196 in float64 before returning float32 logits will preserve 9,257 correct predictions and reduce cross-entropy below 0.2144188995361328.
change: Restore the best verified calibration neighborhood and apply the quadratic-interpolated scale using explicit float64 multiplication followed by a float32 cast.
mechanism: Sub-ULP mixed-precision logit calibration
evidence_used: Scale 1.3560298681259155 achieved the best verified cross-entropy; both adjacent float32 scales were worse, with their asymmetric degradation placing the estimated minimum about 1/22 ULP above it. The prior mixed-precision attempt was not verified and supplied no evidence against the idea.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Retaining the quadratic-interpolated scale in float64 will preserve exactly 9,257 predictions and reduce validation cross-entropy below 0.2144188995361328.
change: Keep training and ensembling unchanged, but apply scale 1.3560298735445196 after converting aggregate logits to float64, avoiding float32 re-quantization.
mechanism: Float64 sub-ULP logit-temperature calibration
evidence_used: The current float32 scale achieved the best cross-entropy, while both adjacent float32 scales were worse; their asymmetric degradation estimates the minimum about 1/22 ULP above the current value.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the best verified float32 logit scale will preserve exactly 9,257 correct predictions and reduce validation cross-entropy from 0.21441890411376954 to approximately 0.2144188995361328.
change: Replace the inferior one-ULP-lower scale with the best verified scale, leaving training and ensembling unchanged.
mechanism: Best-verified calibration restoration
evidence_used: Scale 1.3560298681259155 achieved the best reported validation score, while its immediately lower float32 neighbor 1.356029748916626 produced worse cross-entropy with identical accuracy; the mixed-precision alternatives could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a float32 residual equivalent to 1/22 ULP above the best verified scale will preserve 9,257 correct predictions and reduce cross-entropy below 0.2144188995361328.
change: Restore the best verified scale and add the interpolated residual through a separate float32 multiplication, avoiding unsupported float64 logits.
mechanism: Compensated float32 sub-ULP logit calibration
evidence_used: Scale 1.3560298681259155 outperformed both adjacent float32 scales; their asymmetric losses estimate the optimum about 1/22 ULP higher, while prior float64 implementations could not be verified.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 62.563669957919046, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2144188995361328, "validation_score": 9257.41171954767}

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
