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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.3133676671423, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1846593635559082, "validation_score": 9359.422062252983}
prior_hypothesis: A 53.75% original / 46.25% flipped weighting will preserve all 9,359 correct predictions while lowering validation cross-entropy below 0.184717472076416.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Weighting the canonical validation image slightly more than its synthetic horizontal flip will increase validation_correct above 9,359 while retaining most of the flip ensemble’s benefit.
change: Change evaluation aggregation from an equal average to 55% original-image logits and 45% flipped-image logits, retaining the best-known calibration scale.
mechanism: Canonical-view-biased flip logit ensemble
evidence_used: Adding weaker translated evaluation views reduced validation_correct from 9,359 to 9,348, showing that equal weighting of transformed views can hurt; a modest canonical-view bias tests view reliability without adding computation or altering training.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 75.15047529200092, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.18461807746887207, "validation_score": 9358.422076962617}

RECENT RESULT
hypothesis: A 52.5% original / 47.5% flipped weighting will preserve all 9,359 correct predictions while lowering validation cross-entropy below 0.18488420372009276.
change: Bias evaluation logits halfway toward the tested 55% canonical-view weighting while leaving training and calibration unchanged.
mechanism: Half-step canonical-view-biased flip logit ensemble
evidence_used: The 55% original weighting lowered cross-entropy substantially to 0.18461807746887207 but lost one correct prediction; testing the midpoint determines whether part of that calibration gain occurs before the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 75.18545462493785, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.184717472076416, "validation_score": 9359.422041551496}

RECENT RESULT
hypothesis: A 53.75% original / 46.25% flipped weighting will preserve all 9,359 correct predictions while lowering validation cross-entropy below 0.184717472076416.
change: Move the evaluation ensemble halfway from the successful 52.5% canonical weighting toward the lower-cross-entropy 55% weighting, retaining the established calibration scale.
mechanism: Accuracy-constrained canonical-view weight refinement
evidence_used: The 52.5% weighting preserved 9,359 correct with 0.184717472076416 cross-entropy, while 55% reduced cross-entropy to 0.18461807746887207 but lost one correct prediction; their midpoint efficiently probes the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.3133676671423, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1846593635559082, "validation_score": 9359.422062252983}



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
