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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.87751333299093, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420372009276, "validation_score": 9359.421982163683}
prior_hypothesis: An evaluation-only scale of 1.226016 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.

## Recent verification evidence

RECENT RESULT
hypothesis: An evaluation-only scale of 1.225 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18488877143859864.
change: Increase the positive flip-ensemble calibration factor from 1.22 to 1.225.
mechanism: Quadratic-refined evaluation logit calibration
evidence_used: Cross-entropy improvements shrank from 0.00019317 at 1.20 to 0.00008264 at 1.22; local quadratic interpolation places the minimum near 1.225.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 48.86259841709398, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488433990478514, "validation_score": 9359.421982115184}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.226 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488433990478514.
change: Increase only the positive flip-ensemble calibration factor from 1.225 to 1.226.
mechanism: Quadratic-refined evaluation logit calibration
evidence_used: Cross-entropy was 0.1849714141845703 at 1.20, 0.18488877143859864 at 1.22, and 0.18488433990478514 at 1.225; quadratic interpolation of these results places the local minimum near 1.226.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 74.10099791595712, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420639038086, "validation_score": 9359.421982162732}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.22603 will preserve all 9,359 argmax predictions and slightly reduce validation cross-entropy below 0.18488420639038086.
change: Refine only the positive flip-ensemble calibration factor from 1.226 to 1.22603.
mechanism: Fine-grained quadratic logit calibration
evidence_used: Cross-entropy improved from 0.18488877143859864 at 1.22 to 0.18488433990478514 at 1.225 and 0.18488420639038086 at 1.226; quadratic interpolation of these three measurements places the local minimum near 1.226032.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.97517937514931, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420486450197, "validation_score": 9359.421982163276}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.226032 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.
change: Refine only the positive flip-ensemble calibration factor from 1.22603 to 1.226032.
mechanism: Fine-grained quadratic logit calibration
evidence_used: Cross-entropy improved at every tested scale through 1.22603, while quadratic interpolation of the measurements at 1.22, 1.225, and 1.226 places the local minimum near 1.226032.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 77.51370766595937, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420715332032, "validation_score": 9359.421982162461}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.226016 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.
change: Refine only the positive flip-ensemble calibration factor from 1.22603 to 1.226016.
mechanism: Bracketed local logit-temperature refinement
evidence_used: Scale 1.22603 achieved the best observed cross-entropy, while 1.226032 regressed; a local quadratic fit using the measured results at 1.226, 1.22603, and 1.226032 places the minimum near 1.226016.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.87751333299093, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420372009276, "validation_score": 9359.421982163683}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.2260157 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420372009276.
change: Refine only the positive flip-ensemble calibration factor from 1.226016 to 1.2260157.
mechanism: Float32-aware bracketed logit calibration
evidence_used: Scale 1.226016 is the best observed point, while 1.226 and 1.22603 were worse; quadratic interpolation places the minimum near 1.22601568, motivating the nearest finer literal.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 56.628758458187804, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18488420600891114, "validation_score": 9359.421982162869}

RECENT RESULT
hypothesis: An evaluation-only scale of 1.2260162 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420372009276.
change: Increase the positive flip-ensemble calibration factor from 1.226016 to 1.2260162, targeting the next higher effective float32 multiplier.
mechanism: Float32-adjacent logit calibration
evidence_used: Scale 1.226016 is best observed; 1.2260157 regressed and 1.22603 was slightly worse, so testing the immediately adjacent higher float32 calibration value is the most informative remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 56.8242081659846, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1848842041015625, "validation_score": 9359.421982163547}

RECENT RESULT
hypothesis: The immediately lower float32 scale will preserve all 9,359 predictions while reducing cross-entropy below 0.18488420372009276.
change: Replace the evaluation scale with the exact float32 value immediately below 1.226016.
mechanism: Float32-adjacent logit calibration
evidence_used: The immediate higher float32 neighbor regressed, while the tested 1.2260157 lies roughly three float32 steps below the current best; the untested immediate lower neighbor completes the local bracket.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 45.65546633396298, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1848842041015625, "validation_score": 9359.421982163547}

RECENT RESULT
hypothesis: Averaging logits across the original image, its horizontal flip, and their four one-pixel translations will increase validation_correct above 9,359 by reducing pooling-phase sensitivity.
change: Preserve training and the best-known calibration scale, while expanding evaluation from two views to ten zero-padded views.
mechanism: One-pixel translation and flip test-time ensemble
evidence_used: Float32-adjacent scale trials bracketed 1.226016 without changing any predictions, so an accuracy-changing ensemble refinement is more informative than further temperature tuning.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.34917183313519, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18757213287353516, "validation_score": 9348.421027056933}

RECENT RESULT
hypothesis: Per-image one-pixel translations during the first 75% of training, followed by centered-image fine-tuning, will increase validation_correct above 9,359 without changing evaluation-time calibration.
change: Apply random replicate-padded ±1-pixel crops in `prepare_training_batch` for the first three quarters of training, while preserving centered batches for the final quarter.
mechanism: Early random translation augmentation with clean-data fine-tuning
evidence_used: Equal-weight translation test-time ensembling reduced validation_correct from 9,359 to 9,348, showing that shifted views are not currently modeled robustly; learning limited translation invariance before a clean-data finishing phase directly tests whether that sensitivity can be reduced without averaging shifted logits at evaluation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.16505929199047, "validation_accuracy": 0.9346, "validation_correct": 9346, "validation_cross_entropy": 0.1879907886505127, "validation_score": 9346.420878684226}

RECENT RESULT
hypothesis: Averaging original and flipped class probabilities instead of logits will increase validation_correct above 9,359 by allowing a confident correct view to outweigh an uncertain disagreeing view.
change: Keep training and the best calibration factor unchanged, but combine the two evaluation views in probability space.
mechanism: Arithmetic-probability flip ensembling
evidence_used: Float32 calibration refinements no longer changed predictions, while adding translated evaluation views reduced validation_correct to 9,348; this motivates changing how the two established original/flip views are aggregated without introducing weaker geometric views.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the original and flipped predictive probabilities will increase validation_correct above 9,359 by allowing a confidently correct view to outweigh an uncertain disagreeing view.
change: Combine the two established evaluation views in probability space, convert the average back to logits, and retain the best-known calibration scale.
mechanism: Arithmetic-probability flip ensembling
evidence_used: Equal-weight translated views reduced validation_correct to 9,348, while the prior probability-ensemble verification timed out without testing its accuracy; this retries the still-unmeasured aggregation change without introducing weaker geometric views.
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
