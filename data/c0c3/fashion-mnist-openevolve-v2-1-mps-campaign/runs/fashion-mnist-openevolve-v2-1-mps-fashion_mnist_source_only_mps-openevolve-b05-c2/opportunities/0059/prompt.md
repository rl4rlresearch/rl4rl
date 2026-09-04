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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.455218999879435, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433738708495, "validation_score": 9192.409040000846}
prior_hypothesis: An effective denominator of 76.511644% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.38112712511793, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237594985961914, "validation_score": 9192.40903946127}
prior_hypothesis: Retrying the effective 76.74% denominator will retain 9,192 correct predictions while reducing validation cross-entropy below 0.2223825225830078.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.97734533296898, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433586120606, "validation_score": 9192.409040001357}
prior_hypothesis: An effective denominator of 76.512356% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 59.01604887493886, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433319091796, "validation_score": 9192.40904000225}
prior_hypothesis: An effective denominator of 76.512% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237594985961914.

## Recent verification evidence

RECENT RESULT
hypothesis: Dividing the ensemble logits by 84% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.22917193641662598.
change: Increase post-hoc evaluation-logit sharpening from 8% to 16% while preserving ensemble weights and class ordering.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 0.5%, 1%, 2%, 4%, and 8% successively reduced cross-entropy while retaining all 9,192 correct predictions; doubling the latest successful calibration step is the most informative next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 43.87550191604532, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22403448944091797, "validation_score": 9192.408485221873}

RECENT RESULT
hypothesis: Dividing the ensemble logits by 68% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.22403448944091797.
change: Increase post-hoc evaluation-logit sharpening from 16% to 32% while preserving ensemble weights and class ordering.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 4%, 8%, and 16% successively reduced cross-entropy to 0.2329286087036133, 0.22917193641662598, and 0.22403448944091797 while retaining all 9,192 correct predictions; doubling the latest successful sharpening step is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.29174400004558, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.224783793258667, "validation_score": 9191.40823531692}

RECENT RESULT
hypothesis: Sharpening from 16% to 24% while anchoring class ordering to the proven 84%-denominator logits will retain 9,192 correct predictions and reduce cross-entropy below 0.22403448944091797.
change: Center the current logits on their per-image maximum, then apply the 21/19 scale corresponding to a 76%-of-normalization denominator.
mechanism: Order-locked midpoint temperature sharpening
evidence_used: The 84% denominator achieved 9,192 correct at 0.22403448944091797 cross-entropy, while 68% lost one prediction; 76% is the informative midpoint, and max-centering before positive scaling preserves the 84% logits’ argmax.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Anchoring predictions to the proven 84%-denominator logits and sharpening them to 76% will retain 9,192 correct predictions while reducing cross-entropy below 0.22403448944091797.
change: Retry the timed-out 76% calibration probe, max-centering the proven 84%-denominator logits before positive scaling to preserve their class ordering.
mechanism: Order-locked midpoint temperature sharpening
evidence_used: The 84% denominator achieved 9,192 correct with 0.22403448944091797 cross-entropy, while 68% lost one prediction; the previously attempted order-locked 76% midpoint timed out and therefore remains the most informative unresolved probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 72.8399857499171, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2223825225830078, "validation_score": 9192.409037261874}

RECENT RESULT
hypothesis: A 75.9%-of-normalization effective denominator will retain 9,192 correct predictions and reduce validation cross-entropy below 0.2223825225830078.
change: Anchor evaluation logits to the proven 84% denominator, max-center them to preserve class ordering, then sharpen to an effective 75.9% denominator.
mechanism: Quadratic-informed order-locked temperature refinement
evidence_used: Cross-entropy improved from 0.22403448944091797 at 84% to 0.2223825225830078 at 76%, but worsened to 0.224783793258667 at 68%; these measurements place the local calibration minimum near 75.9%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.42074770806357, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22238604049682617, "validation_score": 9192.409036084702}

RECENT RESULT
hypothesis: An effective denominator of 76.74% will retain 9,192 correct predictions while reducing validation cross-entropy below 0.2223825225830078.
change: Anchor logits to the proven 84% denominator, max-center them, then sharpen by 84/76.74 without changing class ordering.
mechanism: Quadratic-fit order-preserving temperature calibration
evidence_used: Cross-entropy was 0.224783793258667 at 68%, 0.2223825225830078 at 76%, and 0.22403448944091797 at 84%; a quadratic fit to these evenly spaced measurements places the minimum near 76.74%, while the 75.9% refinement was slightly worse.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retrying the effective 76.74% denominator will retain 9,192 correct predictions while reducing validation cross-entropy below 0.2223825225830078.
change: Keep the proven 84%-denominator logits and max-centering, but change the sharpening factor from 84/76 to 84/76.74.
mechanism: Order-preserving quadratic-fit temperature calibration
evidence_used: Cross-entropy at effective denominators of 68%, 76%, and 84% places the fitted minimum near 76.74%; its prior verification timed out, so it remains the most informative unresolved calibration probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.38112712511793, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237594985961914, "validation_score": 9192.40903946127}

RECENT RESULT
hypothesis: An effective denominator of 76.512% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237594985961914.
change: Replace the current 84/75.9 sharpening factor with 84/76.512 while retaining max-centering and class ordering.
mechanism: Local quadratic order-preserving temperature refinement
evidence_used: Cross-entropy was 0.22238604049682617 at 75.9%, 0.2223825225830078 at 76%, and 0.22237594985961914 at 76.74%; a local quadratic through these three nearby measurements places the estimated minimum near 76.512%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 59.01604887493886, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433319091796, "validation_score": 9192.40904000225}

RECENT RESULT
hypothesis: An effective denominator of 76.512356% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Max-center the proven 84%-denominator ensemble logits and sharpen them by 84/76.512356 without changing class ordering.
mechanism: Local quadratic order-preserving temperature refinement
evidence_used: Cross-entropies at effective denominators 76%, 76.512%, and 76.74% were 0.2223825225830078, 0.22237433319091796, and 0.22237594985961914; a quadratic interpolation places the local minimum near 76.512356%.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.97734533296898, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433586120606, "validation_score": 9192.409040001357}

RECENT RESULT
hypothesis: An effective denominator of 76.511644% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Restore the proven 84%-denominator ensemble logits, max-center them, and test a temperature equally far below 76.512% as the unsuccessful 76.512356% probe was above it.
mechanism: Symmetric order-preserving temperature counterprobe
evidence_used: The 76.512% denominator achieved the best observed cross-entropy, 0.22237433319091796, while 76.512356% was slightly worse; the symmetric lower-side probe is the most direct local refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 58.455218999879435, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22237433738708495, "validation_score": 9192.409040000846}

RECENT RESULT
hypothesis: An effective denominator of 76.512040% will preserve all 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Replace the current 76.512% effective denominator with the quadratic vertex estimated from the equally spaced 76.511644%, 76.512%, and 76.512356% probes.
mechanism: Three-point symmetric quadratic temperature refinement
evidence_used: The center denominator achieved 0.22237433319091796 cross-entropy, while equal-distance probes below and above it produced 0.22237433738708495 and 0.22237433586120606; their asymmetric degradation places the fitted minimum approximately 0.000040 percentage points above the center.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 68.56359237502329, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2223743392944336, "validation_score": 9192.409040000208}

RECENT RESULT
hypothesis: Using the proven 76.512% temperature while collapsing normalization and sharpening into one scaling operation will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.
change: Max-center the accumulated logits before applying one combined normalization-temperature scale, eliminating an intermediate tensor division and its rounding error.
mechanism: Single-rounding order-preserving logit calibration
evidence_used: The 76.512% design has the best observed cross-entropy; three finer denominator probes were worse, suggesting arithmetic precision is now more informative than further temperature interpolation.
result: training did not finish within the verification time limit



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
