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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 65.1542112918105, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19156210021972656, "validation_score": 9315.419617240183}
prior_hypothesis: Sharpening the paired-view posterior ensemble from temperature 0.80 to 0.78 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1919746.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 53.747452875133604, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19292894592285156, "validation_score": 9311.41913644707}
prior_hypothesis: Adding symmetric flip-consistency regularization to Reference Design 3’s qualified paired-view training will exceed 9,315 correct predictions by stabilizing the orientation-invariant decision boundary.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 50.91195999993943, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19135721282958984, "validation_score": 9315.419689405173}
prior_hypothesis: Setting the ensemble temperature to 0.753 will preserve all 9,315 argmax predictions and reduce validation cross-entropy below 0.1913601.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248010, "training_seconds": 57.02238612505607, "validation_accuracy": 0.9291, "validation_correct": 9291, "validation_cross_entropy": 0.19672859649658203, "validation_score": 9291.417805675794}
prior_hypothesis: Replacing both hard max-pooling operations with pixel-unshuffle plus learned phase mixing, while retaining the best paired-view protocol, will exceed 9,315 correct predictions by preserving discriminative within-cell structure until the network learns what to discard.

## Recent verification evidence

RECENT RESULT
hypothesis: Reproducing Reference Design 3’s paired flip supervision and 0.8-temperature probability ensemble will exceed the current 9,274 correct predictions; the identical qualified design achieved 9,315 correct.
change: Train each augmented image jointly with its horizontal flip and sharpen the probability-averaged inference logits using temperature 0.8.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 achieved the strongest verified result—9,315 correct and 0.1919746 cross-entropy—using the current architecture plus exactly these two changes.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing Reference Design 3 will exceed the current 9,273 correct predictions because the identical qualified implementation achieved 9,315 correct with 0.1919746 cross-entropy.
change: Train every augmented image jointly with its horizontal flip, then use weighted probability averaging across translated and flipped inference views with temperature 0.8.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 is the strongest verified design at 9,315 correct; paired flip supervision raised correctness from 9,274 to 9,315, and temperature 0.8 subsequently reduced cross-entropy while preserving all argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing Reference Design 3 will exceed the current 9,250 correct predictions; the identical qualified design achieved 9,315 correct with 0.1919746 cross-entropy.
change: Use a replicate-padded 7×7 refinement kernel, jointly supervise each augmented image and its horizontal flip, and apply 0.8-temperature probability averaging across translated and flipped inference views.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 is the strongest verified implementation: paired flip supervision raised correctness to 9,315, while temperature 0.8 preserved its argmax predictions and reduced cross-entropy to 0.1919746.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharpening the paired-view posterior ensemble from temperature 0.80 to 0.78 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1919746.
change: Change only the positive inference temperature from 0.80 to 0.78, leaving training and predicted classes unchanged.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: Sharpening the same paired-view ensemble from 0.90 to 0.80 preserved 9,315 correct predictions and reduced cross-entropy from 0.1971655 to 0.1919746, motivating a smaller further calibration step.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 65.1542112918105, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19156210021972656, "validation_score": 9315.419617240183}

RECENT RESULT
hypothesis: Replacing both hard max-pooling operations with pixel-unshuffle plus learned phase mixing, while retaining the best paired-view protocol, will exceed 9,315 correct predictions by preserving discriminative within-cell structure until the network learns what to discard.
change: Replace fixed max pooling with two space-to-depth projections that expose every 2×2 sample as channels before learned compression; retain the location-sensitive head and add Reference Design 3’s paired flip supervision and calibrated posterior ensemble. The model has 248,010 learned parameters.
mechanism: Learned polyphase-preserving downsampling
evidence_used: Global pooling fell to 9,085 correct and shallow/deep bypass fusion reached only 9,243, showing that spatial information is load-bearing and cannot be recovered merely at the head. The untested shared assumption is that early hard max pooling safely discards spatial phase; learned polyphase mixing challenges it while preserving the 9,315-correct training and inference foundation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248010, "training_seconds": 57.02238612505607, "validation_accuracy": 0.9291, "validation_correct": 9291, "validation_cross_entropy": 0.19672859649658203, "validation_score": 9291.417805675794}

RECENT RESULT
hypothesis: Adding symmetric flip-consistency regularization to Reference Design 3’s qualified paired-view training will exceed 9,315 correct predictions by stabilizing the orientation-invariant decision boundary.
change: Adopt probability-averaged translated/flip inference at temperature 0.78, jointly supervise horizontal view pairs, and penalize disagreement between their class distributions.
mechanism: Paired-view consistency regularization
evidence_used: Paired horizontal supervision raised correctness from 9,274 to 9,315, while temperature 0.78 preserved those predictions and achieved the best cross-entropy; explicit consistency directly strengthens the invariance responsible for that gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 53.747452875133604, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19292894592285156, "validation_score": 9311.41913644707}

RECENT RESULT
hypothesis: Reproducing Reference Design 3 with temperature 0.76 will preserve all 9,315 argmax predictions while lowering validation cross-entropy below 0.1915621.
change: Adopt the verified 7×7 replicate-padded refinement and paired flip supervision, use probability-averaged test-time augmentation, and sharpen its posterior from temperature 0.78 to 0.76.
mechanism: Fine-grained posterior temperature calibration
evidence_used: Temperatures 0.90, 0.80, and 0.78 preserved 9,315 correct predictions while successively reducing cross-entropy to 0.1971655, 0.1919746, and 0.1915621; the diminishing improvements indicate an optimum near 0.76.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 59.55104729090817, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19137069053649902, "validation_score": 9315.419684657321}

RECENT RESULT
hypothesis: Temperature 0.75 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1913707.
change: Sharpen only the final probability-ensemble logits from temperature 0.76 to 0.75.
mechanism: Quadratic-guided posterior temperature calibration
evidence_used: Temperatures 0.80, 0.78, and 0.76 preserved identical predictions while cross-entropy improved to 0.1919746, 0.1915621, and 0.1913707; those diminishing gains place the estimated minimum near 0.753.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 61.61502037500031, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19136006622314453, "validation_score": 9315.419688399985}

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
