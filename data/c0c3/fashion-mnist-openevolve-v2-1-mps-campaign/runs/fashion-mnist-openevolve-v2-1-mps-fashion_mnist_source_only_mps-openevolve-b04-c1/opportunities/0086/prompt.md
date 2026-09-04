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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.22494162502699, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19341388931274414, "validation_score": 9328.418966131094}
prior_hypothesis: Holding label smoothing at 0.02 for the first half of training and annealing it to zero during EMA collection will exceed 9,323 correct predictions by removing late soft-target bias while retaining early regularization.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the second residual block’s receptive field with dilation will exceed 9,328 correct predictions by capturing broader garment structure while preserving local processing in the preceding layers.
change: Change the 14×14 residual convolution from standard dilation to dilation 2, keeping parameter count, optimizer, regularization, and computational complexity essentially unchanged.
mechanism: Compute-neutral dilated residual context expansion
evidence_used: The fused spatial classifier benefited from nonlinear spatial processing, while the proposed extra low-resolution context block timed out; dilation tests the same broader-context hypothesis without adding parameters or substantial compute.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the symmetry-redundant random flip with random two-pixel translations will exceed 9,328 correct predictions by improving positional robustness without changing model size or materially increasing convolutional compute.
change: Pad each training image by two pixels using replicated borders, then independently sample and extract a translated 28×28 crop for every example.
mechanism: Per-example translation augmentation
evidence_used: The current mirrored-view fusion already makes horizontal flips output-equivalent and achieved 9,328 correct; multiple subsequent regularization-timing changes failed to improve it, motivating an orthogonal augmentation that introduces genuinely new views.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 76.65227033291012, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22091196899414062, "validation_score": 9206.40952993557}

RECENT RESULT
hypothesis: A zero-initialized depthwise residual convolution over the fused 7×7 map will exceed 9,328 correct predictions by adding local nonlinear context at negligible cost while preserving the successful model at initialization.
change: Add a 3×3 depthwise residual refinement after mirrored-view fusion, increasing learned parameters by 576 to 225,018.
mechanism: Post-fusion depthwise spatial refinement
evidence_used: The nonlinear spatially fused classifier reached 9,328 correct, while a full low-resolution residual block timed out; refining only the already-fused batch with a depthwise convolution tests the same spatial-context idea with substantially less computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Evaluating both mirrored views together with live BatchNorm statistics will exceed 9,328 correct predictions by eliminating the mismatch between EMA parameters and non-averaged running statistics.
change: Disable BatchNorm running statistics and process paired views jointly during both training and evaluation, preserving training computation and parameter count.
mechanism: Paired-view batch-stat normalization
evidence_used: The 9,328-correct design copies final BatchNorm buffers onto EMA parameters, while prior attempts to improve normalization-state alignment timed out; this tests that unresolved mechanism without adding learned parameters or training compute.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 82.47981733293273, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19375095825195313, "validation_score": 9328.41884783132}

RECENT RESULT
hypothesis: Reducing the cosine schedule’s terminal learning-rate floor from 10% to 2% will exceed 9,328 correct predictions by reducing late parameter drift and improving alignment between EMA parameters and final BatchNorm statistics.
change: Preserve the successful architecture and regularization schedules while lowering the final learning rate from 2.1e-4 to 4.2e-5.
mechanism: Low-noise terminal EMA consolidation
evidence_used: Label-smoothing and dropout timing changes reduced accuracy, while faster EMA and BatchNorm-alignment experiments remained unresolved or tied; stronger terminal decay tests the same late-state consistency mechanism without added computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.4347715000622, "validation_accuracy": 0.9298, "validation_correct": 9298, "validation_cross_entropy": 0.19896078643798829, "validation_score": 9298.41702781747}

RECENT RESULT
hypothesis: Raising the cosine learning-rate floor from 10% to 15% will exceed 9,328 correct predictions by sustaining useful late hard-label updates; lowering the floor to 2% reduced accuracy by 30 images.
change: Increase the final learning rate from 2.1e-4 to 3.15e-4 while preserving the architecture, regularization schedules, EMA, and initial learning rate.
mechanism: Higher-energy terminal EMA consolidation
evidence_used: Reducing the terminal floor from 10% to 2% degraded validation correct from 9,328 to 9,298, providing directional evidence that stronger late optimization may improve the winning design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Classifying each orientation coherently before averaging its logits will exceed 9,328 correct predictions by preserving asymmetric spatial structure that early invariant/disagreement fusion destroys.
change: Remove hand-designed feature-level mirror fusion and apply the shared nonlinear classifier independently to both views, averaging only their final class logits.
mechanism: Shared-head late logit pooling across mirrored views
evidence_used: The feature-fusion baseline reached 9,328 correct, while the attention-based attempt to replace its static symmetric representation timed out; late logit pooling tests a genuinely different, exact-invariance mechanism while eliminating the 128→64 fusion convolution and avoiding attention’s computational cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 10% to 12.5% will exceed 9,328 correct predictions by sustaining useful late hard-label updates without substantially increasing parameter drift.
change: Increase the cosine schedule’s final learning rate from 2.1e-4 to 2.625e-4 while preserving architecture, regularization, EMA, and initial learning rate.
mechanism: Moderately elevated terminal cosine learning-rate floor
evidence_used: Lowering the floor from 10% to 2% reduced validation correct from 9,328 to 9,298; the 15% attempt was unresolved due to timeout, motivating a conservative intermediate increase.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding nonlinear local mean and contrast features with only 128 parameters will exceed 9,328 correct predictions by capturing garment texture and boundary context without the runtime cost that prevented depthwise and dilated spatial refinements from completing.
change: After mirrored-view fusion, add zero-initialized per-channel gates over nonlinear 3×3 local-mean and local-contrast maps, preserving the current classifier exactly at initialization.
mechanism: Zero-gated local mean-and-contrast refinement
evidence_used: The current fused classifier reached 9,328 correct, while both depthwise post-fusion refinement and dilated broader-context attempts timed out; this tests the same spatial-context hypothesis with negligible additional computation.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224570, "training_seconds": 83.95756775001064, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.19617273712158204, "validation_score": 9302.417999829358}

RECENT RESULT
hypothesis: Expanding the fused spatial classifier from 48 to 56 hidden units will exceed 9,328 correct predictions by using the remaining parameter budget to improve class separation without the runtime-heavy spatial operations that repeatedly timed out.
change: Widen the nonlinear classifier head to 56 units, increasing learned parameters from 224,442 to 249,618 while preserving all training, regularization, and EMA behavior.
mechanism: Low-cost spatial classifier width expansion
evidence_used: The 9,328-correct fused spatial classifier remains the strongest design, while added convolutions and local spatial refinements timed out; widening its existing dense head adds capacity with negligible additional convolutional compute.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249618, "training_seconds": 78.95353766693734, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.19473318519592286, "validation_score": 9300.41850348362}

RECENT RESULT
hypothesis: Increasing AdamW weight decay to 1e-2 will exceed 9,328 correct predictions by controlling late co-adaptation after dropout and label smoothing begin annealing away.
change: Raise AdamW’s weight decay from 2e-4 to 1e-2 while preserving the winning architecture, learning-rate schedule, EMA, and stochastic-regularization schedules.
mechanism: Persistent deterministic weight shrinkage
evidence_used: Expanding the classifier to 249,618 parameters reduced validation correct from 9,328 to 9,300, suggesting additional capacity is not beneficial; stronger persistent regularization tests that signal without adding runtime-heavy operations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 75.56032000016421, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.19485718688964843, "validation_score": 9290.418460051533}

RECENT RESULT
hypothesis: Eliminating AdamW weight decay will exceed 9,328 correct predictions by allowing better late hard-label fitting after dropout and label smoothing anneal away.
change: Set AdamW weight decay from 2e-4 to zero while preserving the architecture, learning-rate schedule, EMA, and stochastic regularization.
mechanism: Remove deterministic parameter shrinkage
evidence_used: Increasing weight decay to 1e-2 reduced validation correct from 9,328 to 9,290, providing directional evidence that persistent shrinkage conflicts with this short fixed-exposure training regime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.9665104590822, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.1946596031188965, "validation_score": 9300.418529260296}



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
