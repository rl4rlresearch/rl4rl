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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 81.82675824989565, "validation_accuracy": 0.9341, "validation_correct": 9341, "validation_cross_entropy": 0.19476842727661134, "validation_score": 9341.418491139024}
prior_hypothesis: Blending 10% individual-view cross-entropy into the paired-flip ensemble objective will exceed 9,335 correct predictions by discouraging orientation-specific errors while retaining the validated ensemble predictor.

## Recent verification evidence

RECENT RESULT
hypothesis: A deeper 216,538-parameter CNN trained with 128-image batches, mild flip augmentation, and warmup-cosine AdamW will exceed the current 8,928 validation-correct result.
change: Replace the shallow network with four convolutional stages, halve batch size, add horizontal-flip augmentation, reduce label smoothing, and use a warmup-cosine learning-rate schedule with a nonzero floor.
mechanism: Capacity-efficient batch-normalized CNN with higher update density
evidence_used: The starting 105,866-parameter model reaches 89.28% after only 392 optimizer steps, leaving substantial parameter capacity and suggesting that richer spatial features plus twice as many updates can improve the fixed-exposure result.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216538, "training_seconds": 40.435976000037044, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.21250880165100097, "validation_score": 9283.412368140602}

RECENT RESULT
hypothesis: Halving the batch size from 128 to 64 will exceed 9,283 correct predictions by providing about 1,563 optimizer updates within the same 100,000-example exposure.
change: Change only the batch size, preserving the validated architecture, augmentation, loss, and warmup-cosine AdamW schedule.
mechanism: Higher optimizer-update density
evidence_used: The available design reached 92.83% while increasing update density from the earlier 392-step regime to 782 steps; testing 64-image batches directly extends the most promising observed mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216538, "training_seconds": 64.91070549981669, "validation_accuracy": 0.9272, "validation_correct": 9272, "validation_cross_entropy": 0.2075602508544922, "validation_score": 9272.414058014618}

RECENT RESULT
hypothesis: Adding random integer translations of up to two pixels while retaining batch size 128 will exceed 9,283 correct predictions by improving spatial robustness without reducing optimizer-update density.
change: Extend the validated horizontal-flip augmentation with per-image, border-padded random translations.
mechanism: Mild translation-invariant augmentation
evidence_used: The 216,538-parameter batch-128 design reached 92.83%, while batch 64 reduced correct predictions to 9,272; this motivates preserving the successful optimization regime and testing a targeted augmentation improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216538, "training_seconds": 35.033460832899436, "validation_accuracy": 0.9178, "validation_correct": 9178, "validation_cross_entropy": 0.2394177619934082, "validation_score": 9178.403415228773}

RECENT RESULT
hypothesis: Replacing the oversized 7×7 dense head with an 88-channel third convolutional stage and compact 3×3 spatial head will exceed 9,283 correct predictions while preserving the validated batch-128 optimization regime.
change: Add two batch-normalized 88-channel convolutions and a third pooling stage, then use a 64-unit classifier head; this reallocates parameters from dense layers to spatial feature extraction while remaining below the 250,000-parameter ceiling.
mechanism: Convolutional capacity reallocation with coarse spatial retention
evidence_used: The 216,538-parameter deeper CNN reached 9,283 correct, whereas increasing update density and adding translations did not improve it, motivating an architecture-only refinement that preserves the successful training and augmentation settings.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 43.200096665881574, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20879372520446776, "validation_score": 9290.413635502546}

RECENT RESULT
hypothesis: Averaging predictions from each validation image and its horizontal reflection will exceed 9,290 correct predictions by reducing orientation-sensitive errors in the model already trained with 50% horizontal flips.
change: Preserve training behavior while making evaluation average the original-view and flipped-view logits.
mechanism: Flip-invariant test-time logit ensembling
evidence_used: The current 237,818-parameter design reached 9,290 correct using horizontal-flip augmentation; evaluation currently uses only one orientation, so paired-view inference directly exploits the learned invariance without changing optimization or parameter count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 62.14365874999203, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.2046303867340088, "validation_score": 9302.415065073492}

RECENT RESULT
hypothesis: Training the same two-view averaged logits used during evaluation will exceed 9,302 correct predictions by directly optimizing the flip ensemble that already improved accuracy by 12 images.
change: Replace stochastic single-view flips with paired original/reflected views and compute cross-entropy on their averaged logits.
mechanism: Paired-flip ensemble training
evidence_used: Flip test-time ensembling improved validation_correct from 9,290 to 9,302 and reduced cross-entropy from 0.20879 to 0.20463, showing that the paired-view predictor is stronger than either evaluation orientation alone.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 52.95311883301474, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19535399169921874, "validation_score": 9322.418286134043}

RECENT RESULT
hypothesis: Replacing the final weights with a 0.99-decay EMA accumulated over the second half of training will exceed 9,322 correct predictions by reducing late-optimization variance without changing exposure, architecture, or predictions.
change: Add an optimizer post-step hook that tracks a parameter EMA after halfway and installs it exactly after the final optimizer step.
mechanism: Second-half exponential weight averaging
evidence_used: Paired-flip training raised validation_correct from 9,302 to 9,322 and lowered cross-entropy, while batch 64’s extra updates regressed accuracy; this motivates preserving the successful regime and stabilizing its final solution.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.442112624878064, "validation_accuracy": 0.9323, "validation_correct": 9323, "validation_cross_entropy": 0.19251141052246093, "validation_score": 9323.419283199799}

RECENT RESULT
hypothesis: Averaging BatchNorm running means and variances alongside the already successful parameter EMA will exceed 9,323 correct predictions by eliminating the evaluation mismatch between averaged weights and final-step normalization statistics.
change: Extend the second-half 0.99-decay EMA to floating-point model buffers and install those averaged buffers with the averaged parameters at the final step.
mechanism: EMA-consistent BatchNorm statistics
evidence_used: Parameter EMA improved validation_correct from 9,322 to 9,323 and cross-entropy from 0.19535 to 0.19251; the current implementation averages BatchNorm parameters but leaves their running statistics from the unaveraged final model, motivating a coherence fix without altering exposure or architecture.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.98462549992837, "validation_accuracy": 0.9335, "validation_correct": 9335, "validation_cross_entropy": 0.19444067611694335, "validation_score": 9335.418605971814}

RECENT RESULT
hypothesis: Blending 10% individual-view cross-entropy into the paired-flip ensemble objective will exceed 9,335 correct predictions by discouraging orientation-specific errors while retaining the validated ensemble predictor.
change: Preserve the architecture, schedule, and EMA behavior while training each original and reflected view with a small auxiliary loss alongside the averaged-logit loss.
mechanism: Auxiliary per-view supervision for flip ensembles
evidence_used: Paired-flip ensemble training improved validation_correct from 9,302 to 9,322, demonstrating that flip-paired optimization is effective; lightly supervising both constituent predictions is the most direct refinement of that successful mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 81.82675824989565, "validation_accuracy": 0.9341, "validation_correct": 9341, "validation_cross_entropy": 0.19476842727661134, "validation_score": 9341.418491139024}

RECENT RESULT
hypothesis: Increasing total individual-view loss weight from 10% to 20% will exceed 9,341 correct predictions by further reducing orientation-specific errors while keeping the flip-ensemble objective dominant.
change: Weight the ensemble loss at 80% and each constituent-view loss at 10%, preserving all other training and evaluation behavior.
mechanism: Stronger auxiliary per-view supervision
evidence_used: Adding 10% individual-view supervision improved validation_correct from 9,335 to 9,341; increasing that successful auxiliary signal is the most direct test of whether stronger constituent accuracy further improves the ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 66.54733533295803, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.1948249137878418, "validation_score": 9330.418471354447}



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
