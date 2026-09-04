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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.49596275016665, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21420104522705077, "validation_score": 9253.41179341919}
prior_hypothesis: Increasing the positive ensemble scale from 1.20 to 1.25 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2160126.

## Recent verification evidence

RECENT RESULT
hypothesis: A low-cost bottleneck residual block plus a 54-unit classifier will increase validation_correct above 9,252 while remaining below the 250,000-parameter ceiling.
change: Add an identity-initialized 64→32→64 residual block at 7×7 resolution and widen the classifier bottleneck from 48 to 54 units, yielding 248,808 learned parameters.
mechanism: Identity-initialized post-pool residual refinement
evidence_used: The verified design reached 9,252 correct with 33,654 parameters unused; inference-only changes cannot improve its argmax, while previous training-side trials timed out without providing contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing per-image indexed crops with balanced batch-shared crops will finish verification while retaining at least 9,252 correct predictions because it preserves the 5:2:2:2:2 translation exposure distribution and removes costly advanced indexing from every training step.
change: Use the training-step index to cycle through the same thirteen translation outcomes, applying each batch’s translation with a contiguous slice while preserving independent horizontal flips and the verified model and ensemble.
mechanism: Stratified batch-shared translation
evidence_used: The verified design achieved 9,252 correct but took 76.9 training seconds; even single-view evaluation timed out later, pointing to training-path cost, while prior compute reductions left the per-example advanced-index augmentation unchanged.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.19069349998608, "validation_accuracy": 0.9218, "validation_correct": 9218, "validation_cross_entropy": 0.22453779983520508, "validation_score": 9218.408317325988}

RECENT RESULT
hypothesis: Preserving the verified per-image translation distribution while eliminating dense coordinate-grid indexing will recover 9,252 correct predictions and complete faster; scaling the unchanged ensemble to 1.20 will then lower cross-entropy without changing argmax predictions.
change: Replace coordinate-based crop gathering with disjoint masked assignments from five contiguous crop views, and increase inference calibration from 1.10 to 1.20.
mechanism: Sparse per-image crop assignment with calibrated ensemble sharpening
evidence_used: Batch-shared crops completed slightly faster but lost 34 correct predictions, showing that per-image crop diversity matters; the verified 1.05-to-1.10 scale increase preserved all 9,252 correct predictions while lowering cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Enforcing the 5:2:2:2:2 translation ratio within each batch while retaining randomized per-image offsets will exceed 9,252 correct predictions by reducing augmentation-induced gradient variance without sacrificing crop diversity.
change: Replace independent offset draws with shuffled 13-example strata containing five centered crops and two of each directional crop; any partial stratum is an unbiased random subset.
mechanism: Stratified per-image translation sampling
evidence_used: Batch-shared translations completed but lost 34 correct predictions, showing per-image crop diversity matters; this preserves that diversity and the verified translation distribution while improving batch-level balance.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.02833441714756, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.22161378936767578, "validation_score": 9232.409294659534}

RECENT RESULT
hypothesis: Weighting center translations 5:2 over each shifted translation will raise validation_correct above 9,252 by matching the verified training augmentation distribution while retaining flip averaging.
change: Replace equal averaging of the ten inference views with a normalized 5:5:2:2:2:2:2:2:2:2 probability mixture; preserve architecture, training, and 1.10 calibration.
mechanism: Augmentation-distribution-weighted test-time ensemble
evidence_used: The verified equal-weight ensemble reached 9,252 correct, while calibration-only changes cannot alter argmax; training samples center crops 5/13 of the time and each directional crop 2/13, providing a grounded weighting that favors the canonical view without discarding useful shifted predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing from 0.03 to zero over the final quarter of training will increase validation_correct above 9,252 by preserving early regularization while allowing low-learning-rate updates to sharpen class boundaries.
change: Replace constant label smoothing with a step-aware linear decay during the final 25% of optimizer steps; architecture, augmentation, optimizer, averaging, and inference remain unchanged.
mechanism: Late hard-label consolidation
evidence_used: The verified model’s cross-entropy improved when inference logits were sharpened from scale 1.05 to 1.10 without changing predictions, indicating lingering underconfidence; this compute-neutral training-side change tests whether removing smoothing late can also improve argmax decisions without worsening the recurring runtime constraint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the ten augmented-view logits directly will increase validation_correct above 9,252 by preventing uncertain views from disproportionately diluting confident class evidence, while slightly reducing evaluation work.
change: Preserve the verified architecture, training procedure, ten views, and 1.10 calibration, but replace arithmetic probability averaging with direct logit averaging.
mechanism: Logit-space test-time augmentation consensus
evidence_used: The equal-weight ten-view implementation is the strongest verified design at 9,252 correct, while training-side augmentation variants lost accuracy; changing only the ensemble aggregation isolates a compute-neutral source of argmax improvement and removes ten per-view softmax operations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 72.15950966696255, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.21981132049560548, "validation_score": 9251.409899458711}

RECENT RESULT
hypothesis: Blending 25% mean log-probability evidence into the verified probability ensemble will retain at least 9,252 correct predictions while lowering cross-entropy below 0.2235582.
change: Interpolate between probability averaging and the recently tested logit-equivalent geometric aggregation, retaining all ten views and the verified 1.10 calibration.
mechanism: Arithmetic–geometric ensemble interpolation
evidence_used: Direct logit averaging improved cross-entropy from 0.2235582 to 0.2198113 while losing only one correct prediction; a conservative interpolation should capture some calibration benefit without crossing that decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 71.2953763329424, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.22242485961914063, "validation_score": 9253.40902309542}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.10 to 1.20 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2224249.
change: Retain the verified arithmetic–geometric ensemble and change only its inference-time calibration scale.
mechanism: Post-ensemble calibration sharpening
evidence_used: The current 25% geometric blend achieved the best verified count of 9,253; positive scaling cannot change its argmax predictions, and the prior 1.05-to-1.10 increase reduced cross-entropy from 0.2287977 to 0.2235582 without changing predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 80.57814370794222, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21601255226135255, "validation_score": 9253.411179966088}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.20 to 1.30 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2160126.
change: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.
mechanism: Incremental post-ensemble calibration sharpening
evidence_used: Raising the scale from 1.10 to 1.20 preserved all 9,253 predictions and reduced cross-entropy from 0.2224249 to 0.2160126; another positive scale increase cannot change finite-logit argmax predictions and tests whether the ensemble remains underconfident.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.20 to 1.25 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2160126.
change: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.
mechanism: Conservative post-ensemble calibration sharpening
evidence_used: Raising the scale from 1.10 to 1.20 preserved all 9,253 predictions and reduced cross-entropy from 0.2224249 to 0.2160126; the 1.30 attempt timed out without producing contrary accuracy or calibration evidence, so 1.25 is a conservative continuation on the best verified design.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.49596275016665, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21420104522705077, "validation_score": 9253.41179341919}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.25 to 1.30 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2142011.
change: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.
mechanism: Post-ensemble calibration sharpening
evidence_used: Scaling from 1.20 to 1.25 preserved all 9,253 correct predictions and reduced cross-entropy from 0.2160126 to 0.2142010; the earlier 1.30 attempt timed out and produced no contrary calibration evidence.
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
