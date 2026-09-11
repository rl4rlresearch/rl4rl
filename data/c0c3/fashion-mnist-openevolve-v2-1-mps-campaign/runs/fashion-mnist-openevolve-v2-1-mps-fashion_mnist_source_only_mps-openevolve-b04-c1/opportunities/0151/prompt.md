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
hypothesis: Averaging floating-point BatchNorm buffers at the same 0.02 rate as parameters will exceed 9,328 correct predictions by reducing the normalization mismatch between terminal running statistics and EMA-averaged weights.
change: During second-half parameter EMA, also EMA-average BatchNorm running means and variances while continuing to copy integer tracking buffers directly.
mechanism: EMA-aligned normalization buffers
evidence_used: The 9,328 baseline copies short-horizon BatchNorm statistics into its longer-horizon EMA model; the unverified low-momentum BatchNorm proposal identified this mismatch, and buffer averaging tests it directly without altering training dynamics or runtime materially.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 73.59538595797494, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19391162071228027, "validation_score": 9327.418791467748}

RECENT RESULT
hypothesis: Averaging predictions from intact original and mirrored feature maps will exceed 9,328 correct predictions by preserving within-view spatial and cross-channel relationships that are irreversibly discarded when symmetry is imposed before the nonlinear classifier.
change: Remove pre-classifier invariant/disagreement fusion and apply the shared nonlinear classifier independently to both views before averaging their logits; exact horizontal-flip invariance and the verified training schedule remain intact.
mechanism: Post-decision symmetric view averaging
evidence_used: The 9,294 per-view auxiliary result retained the existing lossy inference fusion, while disagreement-only fusion and an invariant residual skip also operated on fused features; none tested the load-bearing assumption that symmetry should be imposed before class prediction. Post-decision averaging instead preserves each complete representation and reduces computation by removing the 128-to-64 fusion convolution.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the parameter-EMA update rate from 0.02 to 0.04 will exceed 9,328 correct predictions by reducing lag between averaged weights and the terminal BatchNorm statistics while retaining meaningful late-step noise suppression.
change: Shorten the second-half parameter-averaging horizon from roughly 50 to 25 optimizer steps, while continuing to copy BatchNorm buffers directly.
mechanism: Terminal-stat-aligned parameter EMA
evidence_used: EMA-averaging BatchNorm buffers scored 9,327 versus the 9,328 baseline, suggesting terminal normalization statistics are preferable; moving averaged parameters closer to those statistics tests the complementary alignment direction without changing training dynamics or runtime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 55.37476908392273, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19435639991760253, "validation_score": 9326.418635509497}

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by stabilizing the final EMA window without changing the peak rate or decay timing.
change: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5.
mechanism: Low-noise terminal cosine annealing
evidence_used: Faster parameter EMA scored 9,326 and EMA-averaged BatchNorm buffers scored 9,327, while architecture and augmentation changes regressed or timed out; the runtime-neutral lower-floor hypothesis remains unresolved because its prior implementations could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Initializing both residual branches at 0.1 strength will exceed 9,328 correct predictions by improving early optimization stability while preserving gradient flow, architecture, inference, and runtime.
change: Initialize the final BatchNorm scale in each existing residual branch to 0.1 instead of the default 1.0.
mechanism: Small-gamma residual initialization
evidence_used: Dense-head widening reached only 9,300, auxiliary supervision reached 9,294, and added spatial refinement timed out; this tests optimization of the verified 224,442-parameter pathway without adding computation or changing its inference fusion.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the symmetry-null random flip with uniformly sampled ±2-pixel translations will exceed 9,328 correct predictions by teaching local shift robustness while preserving the verified architecture, loss, and inference path.
change: Replace random horizontal flipping with vectorized per-image crops from replicate-padded inputs.
mechanism: Training-only integer translation augmentation
evidence_used: The 9,328 baseline’s paired invariant/disagreement fusion makes its current random flip augmentation functionally redundant, while inference-fusion changes and auxiliary supervision regressed; this isolates a training-only source of genuinely new examples.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 62.864946000045165, "validation_accuracy": 0.9216, "validation_correct": 9216, "validation_cross_entropy": 0.21833572883605956, "validation_score": 9216.410395909901}

RECENT RESULT
hypothesis: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise; the opposite change to 0.04 reduced performance to 9,326.
change: Double the effective parameter-averaging horizon during the second half of training while leaving BatchNorm-buffer copying and all training dynamics unchanged.
mechanism: Longer-horizon terminal parameter EMA
evidence_used: Increasing the EMA rate to 0.04 scored 9,326 versus the 9,328 baseline, while averaging BatchNorm buffers scored 9,327; this motivates testing stronger parameter smoothing without altering normalization statistics.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise; increasing the rate to 0.04 reduced performance to 9,326.
change: Double the effective parameter-averaging horizon during the second half of training while retaining terminal BatchNorm buffers and all other training dynamics.
mechanism: Longer-horizon terminal parameter EMA
evidence_used: The verified 0.04 EMA scored 9,326 versus the 9,328 baseline, providing directional evidence that shorter averaging is harmful; the corresponding longer-horizon 0.01 setting remains unmeasured because prior verification attempts did not complete.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by stabilizing the final parameter-EMA window without reducing the peak learning rate.
change: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5 while preserving its peak and decay timing.
mechanism: Lower-noise terminal cosine annealing
evidence_used: Increasing the EMA update rate to 0.04 reduced correctness from 9,328 to 9,326, consistent with late-update noise being harmful; the complementary lower-floor intervention remains unmeasured because prior verification attempts did not complete.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 4,928-parameter near-identity spatial refinement branch will exceed 9,328 correct predictions by learning local part relationships without the runtime cost of the timed-out larger bottleneck.
change: Add a depthwise-separable residual block after view fusion, initialized at 0.1 output strength; total learned parameters become 229,370.
mechanism: Low-cost depthwise-separable spatial residual refinement
evidence_used: Dense-head widening reached 9,300 and global pooling reached 9,290, while the larger spatial bottleneck timed out; this tests the remaining spatial-processing hypothesis with substantially less computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the cosine learning-rate floor from 0.10 to 0.02 will exceed 9,328 correct predictions by reducing late-update noise during the final parameter-EMA window.
change: Lower the schedule’s terminal learning rate from 2.1e-4 to 4.2e-5 while preserving its peak rate, decay onset, architecture, and EMA behavior.
mechanism: Lower-noise terminal cosine annealing
evidence_used: Increasing the EMA update rate to 0.04 reduced correctness to 9,326, consistent with excessive sensitivity to late updates; the complementary lower-floor intervention remains unmeasured because its prior verification attempts did not complete.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing exact-position flattening with multiscale regional average and maximum statistics will exceed 9,328 correct predictions by adding local shift tolerance and activation-presence cues while retaining the coarse spatial layout that global pooling discarded.
change: Replace the 7×7 flat dense representation with 1×1, 2×2, and 4×4 average/max spatial-pyramid features, widening the hidden layer to 56 while remaining under the parameter ceiling.
mechanism: Multiscale average-max spatial pyramid prediction
evidence_used: Global pooling fell to 9,290, showing that spatial layout is load-bearing, while dense-head widening reached only 9,300 because it retained the same exact-position projection. This tests a different representation that preserves regional layout instead of either erasing it or memorizing every feature location independently.
result: the implementation could not be verified



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
