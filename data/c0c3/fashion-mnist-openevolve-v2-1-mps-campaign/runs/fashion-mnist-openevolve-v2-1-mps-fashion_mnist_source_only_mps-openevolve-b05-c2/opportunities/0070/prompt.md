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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 62.07162083289586, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433319091796, "validation_score": 9192.40904000225}
prior_hypothesis: The immediately larger float32 sharpening coefficient will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.38112712511793, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237594985961914, "validation_score": 9192.40903946127}
prior_hypothesis: Retrying the effective 76.74% denominator will retain 9,192 correct predictions while reducing validation cross-entropy below 0.2223825225830078.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.455218999879435, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433738708495, "validation_score": 9192.409040000846}
prior_hypothesis: An effective denominator of 76.511644% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 59.01604887493886, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433319091796, "validation_score": 9192.40904000225}
prior_hypothesis: An effective denominator of 76.512% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237594985961914.

## Recent verification evidence

RECENT RESULT
hypothesis: Using the proven 76.512% temperature while collapsing normalization and sharpening into one scaling operation will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Max-center the accumulated logits before applying one combined normalization-temperature scale, eliminating an intermediate tensor division and its rounding error.
mechanism: Single-rounding order-preserving logit calibration
evidence_used: The 76.512% design has the best observed cross-entropy; three finer denominator probes were worse, suggesting arithmetic precision is now more informative than further temperature interpolation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retrying the unresolved single-rounding implementation at the proven 76.512% temperature will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Max-center accumulated ensemble logits before applying normalization and temperature sharpening in one multiplication.
mechanism: Single-rounding order-preserving logit calibration
evidence_used: The 76.512% design produced the best observed cross-entropy; all finer temperature probes were worse, while the only single-rounding test timed out and remains unresolved.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified 76.512% effective denominator will preserve 9,192 correct predictions and reduce validation cross-entropy below the current 0.22237594985961914.
change: Replace the current 76.74% evaluation-logit denominator with the best verified 76.512% calibration while retaining max-centering and class ordering.
mechanism: Proven order-preserving temperature calibration
evidence_used: Reference Design 3 achieved 9,192 correct with the best observed cross-entropy, 0.22237433319091796; finer denominator probes were worse, and single-rounding variants repeatedly timed out.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying the proven 76.512% calibration in float64 will preserve all 9,192 predictions while reducing validation cross-entropy below 0.22237433319091796.
change: Cast accumulated ensemble logits to float64 before normalization, max-centering, and temperature scaling.
mechanism: Double-precision order-preserving logit calibration
evidence_used: The 76.512% denominator produced the best verified cross-entropy, while finer temperature probes were worse; this makes calibration arithmetic precision the most informative remaining variable.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the verified 76.512% effective denominator will preserve all 9,192 correct predictions and reduce validation cross-entropy from 0.22237433586120606 to approximately 0.22237433319091796.
change: Replace the current 76.512356% calibration with the exact rational scaling used by Reference Design 3.
mechanism: Proven order-preserving temperature restoration
evidence_used: Reference Design 3 achieved the best observed validation cross-entropy, 0.22237433319091796, with 9,192 correct predictions; probes immediately above and below 76.512% were both worse.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring Reference Design 3’s exact rational calibration will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22237433738708495 to approximately 0.22237433319091796.
change: Replace the current lower-side temperature counterprobe with the exact rational scaling of the best verified design.
mechanism: Proven rational temperature restoration
evidence_used: Reference Design 3 achieved the best observed validation cross-entropy, 0.22237433319091796, with the same 9,192 correct predictions; both adjacent denominator probes were worse.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact Reference Design 3 calibration will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22237594985961914 to approximately 0.22237433319091796.
change: Replace the current 76.74% effective denominator with the best verified 76.512% rational scaling, without changing class ordering.
mechanism: Proven rational temperature restoration
evidence_used: Reference Design 3 achieved the best observed result—9,192 correct with 0.22237433319091796 cross-entropy—and both immediately adjacent temperature probes were worse.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Moving the sharpening coefficient down by one float32 ULP will preserve all 9,192 predictions while reducing validation cross-entropy below 0.22237433319091796.
change: Replace the proven 76.512% rational scale with its immediately smaller representable float32 value, leaving class ordering and all training behavior unchanged.
mechanism: Adjacent-float temperature calibration
evidence_used: The 76.512% calibration is the best verified point; the symmetric lower-scale probe degraded less than the higher-scale probe, while prior decimal refinements skipped the immediately adjacent float32 coefficient.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.64198154211044, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433624267577, "validation_score": 9192.40904000123}

RECENT RESULT
hypothesis: The immediately larger float32 sharpening coefficient will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Replace the current calibration with the float32 value one ULP above Reference Design 3’s proven 10500/9564 scale.
mechanism: Symmetric adjacent-float temperature counterprobe
evidence_used: Reference Design 3 is the best verified calibration, while its immediately smaller float32 coefficient was valid but worse; testing the unmeasured opposite neighbor is the most informative remaining precision probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 62.07162083289586, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433319091796, "validation_score": 9192.40904000225}

RECENT RESULT
hypothesis: The float32 sharpening coefficient one ULP above Reference Design 2 will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Replace the current calibration with the next larger representable float32 coefficient after the best verified coefficient.
mechanism: Directed adjacent-float temperature sweep
evidence_used: The rational coefficient and its upper float32 neighbor tied for the best cross-entropy, while the lower neighbor was worse, motivating one further probe in the favorable direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 64.98730758298188, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2223743392944336, "validation_score": 9192.409040000208}

RECENT RESULT
hypothesis: The best-verified sharpening coefficient will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22237594985961914 to approximately 0.22237433319091796.
change: Replace the current 76.74%-denominator calibration with Reference Design 1’s explicit float32 coefficient.
mechanism: Proven float32 temperature calibration restoration
evidence_used: Reference Design 1 achieved the highest observed validation_score, 9192.40904000225; the next larger and immediately smaller float32 coefficients were both worse.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Moving max-centering after the proven 10500/9564 sharpening will preserve all 9,192 predictions while reducing validation cross-entropy below 0.22237433319091796 through a different float32 rounding path.
change: Retain the best verified calibration and separate normalization operations, but apply temperature scaling before subtracting each sample’s maximum logit.
mechanism: Post-scale order-preserving logit centering
evidence_used: The current calibration ties the best observed result, while both adjacent float32 coefficients were worse; prior combined-scaling attempts timed out, motivating an isolated arithmetic-order probe that leaves class ordering and training unchanged.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 57.44478087499738, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433395385742, "validation_score": 9192.409040001996}



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
