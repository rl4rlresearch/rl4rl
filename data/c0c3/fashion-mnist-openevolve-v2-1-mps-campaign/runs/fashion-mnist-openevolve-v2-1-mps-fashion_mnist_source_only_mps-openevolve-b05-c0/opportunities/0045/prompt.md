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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.94529608404264, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.22603530883789064, "validation_score": 9254.407818597389}
prior_hypothesis: Multiplying evaluation logits by 1.1 will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.241946, thereby increasing validation_score.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics alongside the proven tail parameter EMA will exceed 9,254 correct predictions by eliminating the mismatch between averaged weights and final-step normalization statistics.
change: Retain the model on the optimizer and apply the existing strided EMA to floating-point model buffers, restoring both averaged parameters and BatchNorm statistics before validation.
mechanism: BatchNorm-consistent parameter EMA
evidence_used: BatchNorm previously improved correct predictions from 9,170 to 9,210, and the parameter EMA improved the batch-32 result from 9,250 to 9,254; these results motivate making the two beneficial mechanisms internally consistent.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restricting tail EMA to the parameter-dominant classifier will exceed 9,254 correct predictions by retaining weight-averaging benefits while keeping convolution and BatchNorm parameters aligned with final running statistics.
change: Store the classifier parameters as the EMA targets and leave the entire feature extractor at its final trained state.
mechanism: Classifier-only tail EMA
evidence_used: Full-parameter EMA improved the batch-32 result from 9,250 to 9,254, while BatchNorm previously added 40 correct predictions and the classifier contains 207,954 of 239,634 parameters; this motivates concentrating EMA on the dense head without averaging parameters that determine BatchNorm feature statistics.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the cosine schedule’s peak learning rate by 10% will exceed 9,254 correct predictions by capturing part of the optimization benefit previously observed from increasing optimizer-step count, without increasing runtime.
change: Increase AdamW’s initial and scheduled base learning rate from 2.5e-3 to 2.75e-3 while preserving the proven architecture, loss, batch size, cosine decay, and strided EMA.
mechanism: Increased per-update optimization distance
evidence_used: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that greater finite-budget optimization progress is beneficial; a modest learning-rate increase tests that mechanism without the timeout risk of another batch-size reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Multiplying evaluation logits by 1.1 will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.241946, thereby increasing validation_score.
change: Apply a positive 1.1 logit scale only in evaluation mode, leaving training, parameters, and optimizer behavior unchanged.
mechanism: Inference-only logit sharpening
evidence_used: The current design already achieves 9,254 correct with 0.05 label smoothing; modest evaluation sharpening directly targets the cross-entropy tie-breaker, mathematically preserves predicted classes, and adds negligible runtime after several training-side changes timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.94529608404264, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.22603530883789064, "validation_score": 9254.407818597389}

RECENT RESULT
hypothesis: Increasing evaluation logit scaling from 1.1 to 1.2 will preserve all 9,254 argmax predictions while reducing validation cross-entropy below 0.226035.
change: Raise the positive evaluation-only logit multiplier from 1.1 to 1.2 without changing training or runtime-sensitive settings.
mechanism: Incremental inference-only logit sharpening
evidence_used: Scaling logits by 1.1 preserved 9,254 correct predictions and improved cross-entropy to 0.226035; a further modest increase directly tests whether the label-smoothed model remains underconfident.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing evaluation logit scaling from 1.1 to 1.15 will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.226035.
change: Raise only the evaluation-mode logit multiplier from 1.1 to 1.15.
mechanism: Conservative inference-only logit sharpening
evidence_used: Scaling logits by 1.1 preserved 9,254 correct predictions and improved cross-entropy to 0.226035; the 1.2 attempt produced no performance evidence because verification timed out, motivating a conservative intermediate scale with negligible added computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing evaluation logit scaling from 1.1 to 1.125 will preserve all 9,254 argmax predictions while lowering validation cross-entropy below 0.226035.
change: Raise only the evaluation-mode logit multiplier from 1.1 to 1.125.
mechanism: Conservative inference-only logit sharpening
evidence_used: Scaling logits by 1.1 preserved 9,254 correct predictions and improved cross-entropy to 0.226035; the larger 1.15 and 1.2 attempts yielded no performance evidence because verification timed out, motivating a smaller runtime-neutral refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the evaluation logit multiplier from 1.1 to 1.3 will preserve all argmax predictions while lowering validation cross-entropy below 0.226035.
change: Raise only the positive evaluation-mode logit multiplier to 1.3.
mechanism: Stronger inference-only logit sharpening
evidence_used: Scaling logits by 1.1 preserved 9,254 correct predictions and reduced cross-entropy to 0.226035; later scale attempts timed out without performance evidence, so a larger runtime-neutral step more informatively tests whether the label-smoothed model remains underconfident.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding 0.1 dropout to the parameter-dominant dense head will exceed 9,254 correct predictions by reducing hidden-feature co-adaptation without materially increasing runtime.
change: Insert dropout after the classifier’s hidden GELU while retaining the proven architecture, optimizer, EMA, label smoothing, batch size, and evaluation scaling.
mechanism: Mild classifier-head dropout
evidence_used: Label smoothing 0.05 outperformed both hard targets and 0.10 smoothing, showing mild regularization helps; the classifier contains 207,954 of 239,634 parameters, making its hidden layer the most targeted place for additional low-cost regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding an identity shortcut around the existing 48-channel convolution will exceed 9,254 correct predictions by improving finite-budget optimization without increasing parameter count or computational cost materially.
change: Wrap the third convolution and BatchNorm in a residual block, preserving the proven optimizer, EMA, loss, batch size, evaluation scaling, and 239,634-parameter budget.
mechanism: Single-convolution residual feature refinement
evidence_used: Accuracy improved monotonically as optimizer-step count increased from batch 128 to 64 to 32, indicating optimization within the fixed exposure is limiting; a parameter-neutral residual path directly improves gradient and feature propagation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Keeping the learning rate at 5% of its peak near the end of training will exceed 9,254 correct predictions by sustaining useful tail optimization while EMA limits late-update noise.
change: Change the cosine schedule from decay-to-zero to decay-to-5%-of-peak, preserving the model, peak learning rate, loss, EMA, batch size, and evaluation scaling.
mechanism: Nonzero cosine learning-rate floor
evidence_used: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that additional finite-budget optimization helps; a small learning-rate floor targets the under-optimized tail without raising the already-proven peak rate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the costly 48→48 spatial convolution with a 24-channel bottleneck and widening the dense head will finish reliably while exceeding 9,254 correct predictions by preserving batch-32 optimization and comparable parameter capacity.
change: Factor the third convolution into 48→24 pointwise reduction and 24→48 spatial expansion, then widen the classifier hidden layer from 88 to 92; estimated learned parameters remain below 250,000.
mechanism: Factorized spatial bottleneck with head-capacity reallocation
evidence_used: Batch-size reductions improved correct predictions from 9,210 to 9,235 to 9,250, favoring preservation of 3,126 updates, while the 75.95-second baseline and repeated subsequent timeouts motivate reducing per-example convolutional work.
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
