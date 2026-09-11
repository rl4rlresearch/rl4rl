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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.91783266700804, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.22355818634033203, "validation_score": 9252.40864423579}
prior_hypothesis: Pair-batching mirrored views will preserve the ten-view ensemble while reducing evaluation overhead enough to complete verification, and scaling its logits by 1.10 will preserve 9,252 correct predictions while lowering cross-entropy below 0.2287977.

## Recent verification evidence

RECENT RESULT
hypothesis: Folding evaluation-time batch normalization into adjacent convolutions and the classifier projection will prevent verification timeout while preserving 9,252 predictions, and a 1.20 ensemble scale will reduce cross-entropy below 0.2235582.
change: Fuse batch-normalization parameters into cached inference-only weights, retain the complete ten-view ensemble, and raise its calibration scale from 1.10 to 1.20.
mechanism: Batch-normalization folding for fast calibrated ten-view inference
evidence_used: The 1.10 ten-view design completed with 9,252 correct, while scaling from 1.05 to 1.10 lowered cross-entropy without changing predictions; subsequent equally expensive inference variants timed out, motivating an exactly equivalent but cheaper evaluation path.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing evaluation from ten augmented views to the center view and its mirror will complete verification reliably while retaining at least 9,200 correct predictions and comparable calibration.
change: Replace the five-pass ten-view ensemble with one pair-batched forward pass over the original images and their horizontal mirrors, retaining the verified 1.10 scale.
mechanism: Lean center-view flip ensemble
evidence_used: The ten-view pair-batched design reached 9,252 correct, but every subsequent design timed out, including a six-view variant; cutting evaluation workload from 100,000 to 20,000 image views directly addresses the recurring completion failure while preserving the training-aligned flip ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Eliminating test-time augmentation will finish within the verification limit while retaining at least 9,100 correct predictions, establishing whether evaluation overhead caused the repeated timeouts.
change: Replace the ten-view, five-pass evaluation ensemble with one calibrated center-view forward pass; training remains unchanged.
mechanism: Single-view inference fast path
evidence_used: The verified ten-view design reached 9,252 correct but took 76.9 training seconds, and every subsequent variant timed out—including the two-view center/flip ensemble—so halving that latest evaluation workload is the most direct completion-oriented test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the unchanged ten-view ensemble by 1.20 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2235582.
change: Increase only the positive inference-time ensemble scale from 1.10 to 1.20.
mechanism: Isolated post-ensemble calibration sharpening
evidence_used: Raising the scale from 1.05 to 1.10 preserved all 9,252 predictions and reduced cross-entropy from 0.2287977 to 0.2235582; prior 1.20 trials also changed evaluation mechanics, so this isolates calibration on the only verified-completing path.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing convolution widths to 24/48 while widening the classifier bottleneck to 80 will finish reliably by cutting convolutional work roughly 44%, while retaining at least 9,252 correct predictions through a larger spatial classification head.
change: Reallocate capacity from expensive feature-map convolutions to the classifier, producing a 226,002-parameter model while preserving the verified training procedure and ten-view ensemble.
mechanism: Compute-aware capacity reallocation
evidence_used: Single-view inference and batch-size 64 both still timed out, indicating evaluation overhead and optimizer-step count alone were not the primary issue; reducing training-time convolutional computation directly targets the remaining bottleneck.
result: training did not finish within the verification time limit

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
