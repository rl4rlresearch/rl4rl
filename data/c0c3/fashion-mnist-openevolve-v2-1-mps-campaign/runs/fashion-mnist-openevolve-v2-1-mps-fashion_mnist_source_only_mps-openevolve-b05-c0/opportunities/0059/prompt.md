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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 80.91814229195006, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.229253133392334, "validation_score": 9258.406751047785}
prior_hypothesis: Approximating the proven tail-EMA BatchNorm statistics from their midpoint and final values will retain at least 9,255 correct predictions without the runtime cost of updating buffer averages throughout training.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging BatchNorm statistics with the tail-EMA parameters will reach at least 9,255 correct predictions, while fused tensor updates will reduce the 84.95-second runtime enough to finish verification.
change: Retain the model on the optimizer, include floating-point buffers in EMA and restoration, and fuse each EMA update into one foreach operation.
mechanism: Fused BatchNorm-aware tail EMA
evidence_used: BatchNorm-buffer-aware EMA produced 9,255 correct versus the current 9,254, but took 84.95 seconds; fusing its many per-tensor EMA operations targets that observed runtime regression without changing the averaging schedule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics every 32 steps with a decay equivalent to the proven four-step EMA will retain at least 9,255 correct predictions while avoiding its observed 84.95-second runtime regression.
change: Expose floating-point model buffers to the optimizer, sparsely EMA them during the existing parameter-EMA tail, and restore both averaged parameters and statistics for evaluation.
mechanism: Sparse BatchNorm-statistics tail EMA
evidence_used: BatchNorm-buffer-aware EMA improved validation_correct from 9,254 to 9,255 but took 84.95 seconds; reducing buffer updates eightfold directly preserves that winning mechanism while targeting its runtime failure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final full-resolution convolution-plus-pooling pair with a stride-2 convolution will finish reliably while retaining at least 9,254 correct predictions by preserving all 3,126 batch-32 updates, learned parameter capacity, and the 7×7 classifier input.
change: Give the existing 48→48 convolution stride 2 and remove its following max-pooling operation, reducing that convolution’s spatial work by approximately fourfold without changing parameter count.
mechanism: Learned strided spatial downsampling
evidence_used: The 239,634-parameter batch-32 design achieved 9,254 correct but required 75.95 seconds, while numerous later runtime-neutral or more expensive variants timed out; prior batch-size evidence also shows preserving the batch-32 optimizer-step count is valuable for accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Moving the final spatial convolution to 7×7 resolution and widening it to 64 channels will finish reliably and exceed 9,254 correct predictions by preserving batch-32 optimization while increasing feature diversity and receptive field.
change: Move the second max-pool before the third convolution, widen that convolution from 48 to 64 channels, and resize the dense head to keep 249,499 learned parameters.
mechanism: Pre-pooling convolution with capacity reallocation
evidence_used: The 239,634-parameter batch-32 baseline achieved 9,254 correct but required 75.95 seconds, while the stride-2 attempt timed out; explicit max-pooling before a standard stride-1 convolution reduces its spatial workload fourfold, and reallocating the savings to 64 output channels retains near-ceiling capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the expensive final spatial convolution with a depthwise-separable block will finish reliably while exceeding 9,254 correct predictions by preserving 48-channel features and reallocating saved parameters to a wider classifier head.
change: Factor the 48→48 spatial convolution into depthwise 3×3 and pointwise 1×1 stages, then widen the classifier from 88 to 99 hidden units; the model remains below the 250,000-parameter ceiling.
mechanism: Full-width depthwise-separable feature refinement
evidence_used: The baseline reached 9,254 correct but required 75.95 seconds, and repeated variants timed out; unlike the inconclusive 24-channel bottleneck, this change substantially reduces spatial computation without narrowing intermediate features.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Channels-last convolution will offset the observed EMA runtime regression, allowing BatchNorm-buffer-aware tail EMA to finish verification while retaining at least its demonstrated 9,255 correct predictions.
change: Run the unchanged CNN in channels-last memory format and extend the existing tail EMA to average and restore floating-point BatchNorm buffers alongside model parameters.
mechanism: Channels-last acceleration for BatchNorm-aware tail EMA
evidence_used: BatchNorm-buffer-aware EMA improved validation_correct from 9,254 to 9,255 but required 84.95 seconds; repeated timeouts make a learning-neutral convolution layout optimization the most direct way to preserve that measured accuracy gain while reducing runtime.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will preserve all argmax predictions while reducing validation cross-entropy below 0.226035.
change: Increase the evaluation-only logit multiplier from 1.1 to 1.2 without changing training or runtime.
mechanism: Incremental evaluation-logit calibration
evidence_used: The 1.1 multiplier produced 9,254 correct with cross-entropy 0.226035, improving on the previously cited 0.241946; this motivates testing a further small increase.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Approximating the proven tail-EMA BatchNorm statistics from their midpoint and final values will retain at least 9,255 correct predictions without the runtime cost of updating buffer averages throughout training.
change: Attach the model to the optimizer, snapshot floating-point buffers when parameter EMA begins, and interpolate BatchNorm statistics to the EMA’s approximate 75%-through-tail position when restoring averaged parameters.
mechanism: Endpoint-interpolated BatchNorm statistics
evidence_used: Full BatchNorm-buffer EMA improved validation_correct from 9,254 to 9,255 but took 84.95 seconds; its parameter EMA effectively represents roughly 75% of the way through the tail, so one endpoint interpolation targets the same alignment with negligible per-step work.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 80.91814229195006, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.229253133392334, "validation_score": 9258.406751047785}

RECENT RESULT
hypothesis: Applying one-pixel translations to roughly half of training images will exceed 9,258 correct predictions by improving local shift invariance without changing the model, optimizer-step count, or evaluation path.
change: Add vectorized, training-only random crops from replicate-padded images, retaining the original position for half the batch.
mechanism: Mild per-image translation augmentation
evidence_used: The current batch-32 EMA design achieved 9,258 correct, while 0.05 label smoothing outperformed both hard targets and stronger 0.10 smoothing; this motivates another mild, parameter-free regularizer rather than a larger architectural or computational change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will retain exactly 9,258 correct predictions while reducing validation cross-entropy below 0.229253.
change: Increase the evaluation-only logit multiplier from 1.1 to 1.2 without affecting training, parameters, or predicted classes.
mechanism: Evaluation-only logit sharpening
evidence_used: A 1.1 multiplier previously reduced cross-entropy from 0.241946 to 0.226035 without changing argmax predictions; the prior 1.2 verification timed out but provided no contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will retain exactly 9,258 correct predictions while reducing validation cross-entropy below 0.229253.
change: Increase the inference-only logit multiplier from 1.1 to 1.2 without affecting training, parameters, or predicted classes.
mechanism: Evaluation-only logit sharpening
evidence_used: A 1.1 multiplier previously reduced cross-entropy from 0.241946 to 0.226035 without changing argmax predictions; the prior 1.2 attempt timed out and therefore provided no contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging original and horizontally reflected evaluation logits will exceed 9,258 correct predictions by reducing spatial-orientation sensitivity without adding training work.
change: Keep training unchanged, but evaluate each image together with its horizontal reflection and average their logits before applying the existing calibration.
mechanism: Two-view reflection logit ensemble
evidence_used: The current EMA design reached 9,258 correct, while the translation-augmentation attempt identified spatial invariance as promising but timed out; evaluation-only ensembling tests that mechanism without changing the 100,000-example training path.
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
