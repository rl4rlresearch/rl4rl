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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 49.296941000036895, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1918064540863037, "validation_score": 9319.419531206839}
prior_hypothesis: Image-adaptive gating of the final residual refinement, while preserving the verified backbone, spatial head, training protocol, and calibration, will exceed 9,315 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 50.91195999993943, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19135721282958984, "validation_score": 9315.419689405173}
prior_hypothesis: Setting the ensemble temperature to 0.753 will preserve all 9,315 argmax predictions and reduce validation cross-entropy below 0.1913601.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248010, "training_seconds": 57.02238612505607, "validation_accuracy": 0.9291, "validation_correct": 9291, "validation_cross_entropy": 0.19672859649658203, "validation_score": 9291.417805675794}
prior_hypothesis: Replacing both hard max-pooling operations with pixel-unshuffle plus learned phase mixing, while retaining the best paired-view protocol, will exceed 9,315 correct predictions by preserving discriminative within-cell structure until the network learns what to discard.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 56.2051252911333, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19048782272338868, "validation_score": 9322.419995896184}
prior_hypothesis: Conditioning the successful refinement gate on both mean and peak channel responses will exceed 9,319 correct predictions by recognizing localized discriminative features that average pooling alone can dilute.

## Recent verification evidence

RECENT RESULT
hypothesis: Changing the paired-view probability ensemble temperature from 0.78 to 0.75 will preserve all 9,315 correct predictions while lowering validation cross-entropy to approximately 0.19136.
change: Sharpen only the final ensemble logits by changing the inference temperature to 0.75.
mechanism: Evidence-backed posterior temperature calibration
evidence_used: Reference Design 2 used the otherwise identical implementation at temperature 0.75 and achieved 9,315 correct with 0.1913601 cross-entropy, outperforming the current temperature-0.78 result of 0.1915621.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Removing explicit flip-consistency regularization and restoring temperature 0.75 will recover 9,315 correct predictions and approximately 0.19136 cross-entropy.
change: Revert the harmful KL consistency term to paired-view cross-entropy and sharpen the inference ensemble from 0.78 to 0.75.
mechanism: Qualified paired-view supervision with calibrated posterior ensembling
evidence_used: The consistency-regularized current design fell to 9,311 correct, while the otherwise matching Reference Design 1 achieved the best verified score with 9,315 correct and 0.1913601 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring Reference Design 2’s fixed max-pooling architecture and temperature 0.75 will recover 9,315 correct predictions with validation cross-entropy near 0.19136.
change: Replace both pixel-unshuffle projections with the verified 32/64-channel max-pooling backbone and sharpen the inference ensemble from temperature 0.78 to 0.75.
mechanism: Qualified max-pooling backbone with calibrated posterior ensembling
evidence_used: The current learned-downsampling design achieved 9,291 correct, while Reference Design 2’s otherwise matching max-pooling implementation achieved 9,315 correct and the best verified cross-entropy of 0.1913601.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Setting the ensemble temperature to 0.753 will preserve all 9,315 argmax predictions and reduce validation cross-entropy below 0.1913601.
change: Change only the positive inference temperature from 0.75 to 0.753.
mechanism: Quadratic-fit posterior temperature calibration
evidence_used: Temperatures 0.78, 0.76, and 0.75 preserved identical predictions while cross-entropy reached 0.1915621, 0.1913707, and 0.1913601; a local quadratic fit places the minimum near 0.753.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 50.91195999993943, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19135721282958984, "validation_score": 9315.419689405173}

RECENT RESULT
hypothesis: Temperature 0.7532 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1913572128.
change: Change only the positive inference temperature from 0.78 to 0.7532.
mechanism: Local quadratic posterior-temperature refinement
evidence_used: Temperatures 0.75, 0.753, and 0.76 produced cross-entropies 0.1913600662, 0.1913572128, and 0.1913706905 with identical predictions; quadratic interpolation places the minimum near 0.75315.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 61.61781333293766, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.1913572048187256, "validation_score": 9315.419689407994}

RECENT RESULT
hypothesis: Removing the harmful consistency penalty will restore 9,315 correct predictions, while temperature 0.75317 will preserve those argmax predictions and slightly reduce cross-entropy below 0.1913572048.
change: Restore pure paired-view cross-entropy and refine the inference temperature from 0.78 to the quadratic-interpolated optimum of 0.75317.
mechanism: Qualified paired-view supervision with locally optimized posterior calibration
evidence_used: Explicit flip consistency reduced correctness from 9,315 to 9,311; the otherwise matched paired-view design achieved 9,315, and temperatures 0.753 and 0.7532 gave successively lower cross-entropies with identical predictions, locating the minimum near 0.75317.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 57.66056666709483, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.1913572036743164, "validation_score": 9315.419689408398}

RECENT RESULT
hypothesis: Moving capacity from the flattened dense bottleneck into local and dilated residual reasoning, then classifying multi-scale average/max summaries, will exceed 9,315 correct predictions while retaining the coarse spatial layout that global pooling lost.
change: Restore the qualified max-pooling backbone, add two full-channel residual context blocks, and replace the location-by-location dense head with a 1×1/2×2/4×4 average-and-max spatial pyramid feeding class logits. The resulting model has 247,530 learned parameters.
mechanism: Deep residual context with dual-statistic spatial-pyramid readout
evidence_used: Global pooling fell to 9,085 correct, showing that spatial layout is load-bearing; shallow/deep bypass fusion reached only 9,243, and learned pixel-unshuffle downsampling reached 9,291. This motivates retaining the 9,315-correct max-pooling foundation while challenging the shared assumption that most capacity should reside in a flattened 56-unit head.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Image-adaptive gating of the final residual refinement, while preserving the verified backbone, spatial head, training protocol, and calibration, will exceed 9,315 correct predictions.
change: Add a neutral-at-initialization 64→7→64 squeeze-and-excitation gate to the refinement branch, increasing the model by 967 parameters to 249,881.
mechanism: Identity-initialized low-rank channel attention
evidence_used: Global pooling dropped to 9,085 correct, bypass fusion reached 9,243, and learned downsampling reached 9,291; this motivates preserving the 9,315-correct spatial architecture while using its remaining parameter budget for image-conditioned channel selection.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 49.296941000036895, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1918064540863037, "validation_score": 9319.419531206839}

RECENT RESULT
hypothesis: Conditioning the successful refinement gate on both mean and peak channel responses will exceed 9,319 correct predictions by recognizing localized discriminative features that average pooling alone can dilute.
change: Add the qualified identity-initialized 64→7→64 refinement gate, summarize each channel with an equal blend of global average and maximum activation, and use the reference gate’s calibrated temperature of 0.75317. The model has 249,881 learned parameters.
mechanism: Dual-statistic low-rank channel attention
evidence_used: Average-pooled channel gating improved the unchanged backbone from 9,315 to 9,319 correct, the best available result; enriching that proven gate with parameter-free peak statistics directly tests whether sparse feature presence can improve its image-conditioned refinement decisions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 56.2051252911333, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19048782272338868, "validation_score": 9322.419995896184}

RECENT RESULT
hypothesis: Learning a separate mean-versus-peak pooling mixture for each refinement channel will exceed Reference Design 3’s 9,322 correct predictions by preserving its successful initialization while adapting localized-feature sensitivity by channel.
change: Restore the qualified refinement gate, initialize it to Reference Design 3’s equal mean/max pooling, make that mixture learnable per channel, and use its calibrated temperature. The model has 249,945 learned parameters.
mechanism: Per-channel adaptive mean–peak refinement gating
evidence_used: Dual-statistic gating improved average-only gating from 9,319 to 9,322 correct; this motivates retaining both statistics while allowing channels representing diffuse and localized features to weight them differently.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249945, "training_seconds": 50.68952012504451, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.19127616195678712, "validation_score": 9300.419717959587}

RECENT RESULT
hypothesis: Replacing the successful gate’s single-pixel maximum with the mean of its four strongest spatial responses will exceed 9,322 correct predictions by retaining localized evidence while reducing sensitivity to isolated activation noise.
change: Restore Reference Design 3’s max-pooling backbone and identity-initialized refinement gate, but condition the gate on an equal blend of global mean and top-four spatial mean; retain its paired-view training, TTA, and calibrated temperature.
mechanism: Robust top-k dual-statistic channel gating
evidence_used: Reference Design 3’s mean/max gate achieved the best result at 9,322 correct, while unconstrained per-channel mean–peak mixing fell to 9,300; this motivates a parameter-free, robust refinement of the proven peak statistic rather than additional learned mixing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the peak statistic’s fixed contribution from 0.50 to 0.40 will exceed 9,322 correct predictions by retaining localized evidence while limiting sensitivity to isolated activation maxima.
change: Change the refinement-gate descriptor from an equal mean/max blend to a parameter-free 60% mean and 40% maximum blend.
mechanism: Fixed robust mean–peak channel gating
evidence_used: Equal mean/max gating achieved the best result at 9,322 correct, while average-only gating reached 9,319 and learnable per-channel mixing fell to 9,300; the timed-out top-four experiment further motivates this computationally free approximation to a less dominant peak statistic.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.10846025007777, "validation_accuracy": 0.9297, "validation_correct": 9297, "validation_cross_entropy": 0.1906832275390625, "validation_score": 9297.419926970026}



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
