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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 76.45461608306505, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19714545440673828, "validation_score": 9286.417660191717}
prior_hypothesis: Interpolating the final parameters halfway toward a sparse mean of the last 20% of training will exceed 9,280 correct predictions while retaining some of the observed cross-entropy benefit of tail averaging.

## Recent verification evidence

RECENT RESULT
hypothesis: Interpolating the final parameters halfway toward a sparse mean of the last 20% of training will exceed 9,280 correct predictions while retaining some of the observed cross-entropy benefit of tail averaging.
change: Sample parameters every eight optimizer steps during the final 20% of training, maintain their online mean, and blend the final parameters 50:50 with that mean.
mechanism: Half-strength sparse tail parameter averaging
evidence_used: Sparse tail EMA lowered validation cross-entropy from 0.19808 to 0.19631 while losing only one correct prediction, suggesting useful late-trajectory smoothing; partial interpolation should preserve more of the final iterate’s decision boundaries than full averaging.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 76.45461608306505, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19714545440673828, "validation_score": 9286.417660191717}

RECENT RESULT
hypothesis: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by retaining most of the variance-reduction benefit while preserving more of the final iterate’s decision boundaries.
change: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.
mechanism: Reduced-strength tail interpolation
evidence_used: Half-strength interpolation improved validation_correct from the established 9,280 to 9,286, while full tail averaging previously lost a prediction; this motivates refining the averaging strength toward the final iterate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by preserving more of the final iterate’s decision boundaries while retaining useful late-training variance reduction.
change: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.
mechanism: Reduced-strength sparse tail interpolation
evidence_used: Half-strength interpolation improved validation_correct from 9,280 to 9,286, while full tail averaging previously lost a prediction; the prior 0.4 verification timed out and therefore did not test this accuracy hypothesis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending BatchNorm running statistics alongside the learned parameters will exceed 9,286 correct predictions by making the averaged weights and evaluation-time normalization state consistent.
change: Track a sparse tail mean of every BatchNorm running mean and variance, then blend those buffers 50:50 at the final step with the existing parameter blend.
mechanism: BatchNorm-statistic-aligned tail averaging
evidence_used: Half-strength parameter averaging improved validation_correct from 9,280 to 9,286, but the current implementation leaves BatchNorm running statistics at their noisy final values; aligning those statistics targets the remaining evaluation-state mismatch without changing training or parameter count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Softening the current tail-averaged ensemble with temperature 1.05 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Divide only the final evaluation ensemble logits by 1.05, leaving training, averaging, and class predictions unchanged.
mechanism: Post-ensemble temperature calibration
evidence_used: Half-strength tail averaging raised validation_correct to 9,286, while prior sparse averaging reduced cross-entropy to 0.19631; temperature scaling can improve the tie-break metric without changing argmax decisions, and the prior 1.05 attempt timed out without testing the hypothesis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a learned covariance-based scoring branch will exceed 9,286 correct predictions by distinguishing classes with similar first-order shapes through spatial feature co-occurrences, while preserving the proven convolutional representation and tail averaging.
change: Add a 32-channel projection that computes centered channel covariance over the final 3×3 feature map and blends its class logits with the existing flattening classifier through a learned scale.
mechanism: Low-rank second-order feature co-occurrence branch
evidence_used: Residual refinements and alternative pooling reduced accuracy, indicating that changing the convolutional backbone or spatial bottleneck is risky; the current design still assumes class predictions need only first-order flattened features, so a computation-light second-order branch tests a genuinely different and complementary representation within the parameter and runtime limits.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,286 correct predictions by reducing overfitting without materially impairing optimization during the fixed exposure.
change: Raise the existing classifier dropout probability to 0.20 while preserving the architecture, optimizer, augmentation, tail averaging, and runtime profile.
mechanism: Moderate classifier dropout regularization
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, directly indicating that weaker classifier regularization hurts; a modest increase tests the favorable direction with negligible computational overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by preserving more of the final iterate’s decision boundaries while retaining useful late-training variance reduction.
change: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.
mechanism: Reduced-strength sparse tail interpolation
evidence_used: Half-strength interpolation improved validation_correct from 9,280 to 9,286, while full averaging previously lost a prediction; both prior 0.4 verifications timed out and therefore left its accuracy effect unresolved.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,286 correct predictions by reducing overfitting during the fixed two-pass exposure without materially slowing optimization.
change: Raise the existing classifier dropout probability to 0.20 while preserving the proven architecture, augmentation, optimizer schedule, TTA, and tail averaging.
mechanism: Moderate classifier dropout regularization
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, showing that weaker classifier regularization hurts; the previous 0.20 attempt timed out, so its accuracy effect remains untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Label smoothing of 0.05 will exceed 9,286 correct predictions by adding output-level regularization without changing the proven architecture, augmentation, averaging, or runtime profile.
change: Train with 0.05 label smoothing instead of hard-label cross-entropy.
mechanism: Mild target-distribution smoothing
evidence_used: Reducing classifier dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating weaker regularization hurts; label smoothing tests complementary regularization with negligible computational overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by preserving more final-iterate decision boundaries while retaining late-training variance reduction.
change: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.
mechanism: Reduced-strength sparse tail interpolation
evidence_used: Half-strength interpolation improved validation_correct from 9,280 to 9,286, while full averaging lost a prediction; prior 0.4 attempts timed out and therefore did not test its accuracy effect.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging only the final 10% of training will exceed 9,286 correct predictions by reducing bias from earlier, higher-learning-rate parameters while retaining late-trajectory smoothing.
change: Start sparse tail averaging at 90% rather than 80%, preserving the sampling interval and 50:50 final blend.
mechanism: Narrow late-trajectory weight averaging
evidence_used: Half-strength averaging over the final 20% improved validation_correct from 9,280 to 9,286, while full averaging lost a prediction, suggesting smoothing helps but excessive trajectory bias hurts.
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
