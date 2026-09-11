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
hypothesis: Conditioning spatial feature channels on each image’s global activation means and variances will exceed 9,328 correct predictions by adapting part and texture evidence to the image before spatial classification.
change: Add a lightweight, zero-initialized context gate that computes per-channel spatial moments and uses them to multiplicatively recalibrate the final convolutional features.
mechanism: Identity-initialized mean–variance channel routing
evidence_used: A direct global-average logit branch fell to 9,290 and dense-head widening reached 9,300, challenging the assumption that global summaries or more static classifier capacity should predict classes directly; using first- and second-order global context to route the successful spatial representation tests a distinct conditional mechanism at negligible computational cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the symmetry-redundant flips with one-pixel cardinal translations during the first half of training will exceed 9,328 correct predictions by improving alignment tolerance while retaining a fully clean second half.
change: Cycle whole batches through centered and four one-pixel translated views before EMA collection, using replicate padding and removing the redundant random flips.
mechanism: Annealed batch-shared cardinal translation
evidence_used: The model is exactly invariant to horizontal flips, while prior translation experiments timed out without resolving their accuracy benefit; this tests translation tolerance with a lightweight batch-shared operation that runs on only two-fifths of all training steps.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.37368162511848, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20073457717895507, "validation_score": 9288.416411761185}

RECENT RESULT
hypothesis: Training each mirrored feature view to remain independently class-discriminative, then decaying that auxiliary objective during EMA collection, will exceed 9,328 correct predictions without changing the verified inference pathway.
change: Add training-only classifier losses for the original and mirrored feature tensors, weighted at 0.10 through the first half and annealed to zero during the second half.
mechanism: Annealed per-view auxiliary supervision
evidence_used: Disagreement-only fusion fell to 9,290 and an invariant residual skip fell to 9,311, indicating that altering the successful inference fusion is harmful; auxiliary supervision instead strengthens its input features while preserving inference exactly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 77.3557915838901, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.19804727172851563, "validation_score": 9294.417345802456}

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by stabilizing the final parameter-EMA window while preserving nearly the same early optimization.
change: Retain the existing schedule shape and peak learning rate, but lower its cosine floor from 2.1e-4 to 4.2e-5.
mechanism: Lower-floor cosine stabilization
evidence_used: Architecture, fusion, augmentation, and late-loss changes failed to improve the 9,328 baseline; this isolates a runtime-neutral late-optimization change, while the near-baseline 9,325 late-margin result suggests final boundary refinement remains consequential.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by reducing late parameter noise during the successful EMA collection window while preserving the peak learning rate and early schedule.
change: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5 without changing its peak or decay timing.
mechanism: Lower-floor cosine stabilization
evidence_used: Architecture, fusion, augmentation, and loss changes did not improve 9,328 correct; the prior lower-floor implementation was not verified, so this runtime-neutral optimization hypothesis remains unresolved.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by reducing late parameter noise during EMA collection while preserving the peak learning rate and early optimization.
change: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5 without changing its peak, warm plateau, or decay timing.
mechanism: Lower-floor cosine stabilization
evidence_used: The 9,328 baseline survived architecture, augmentation, fusion, and late-loss changes; this runtime-neutral optimization hypothesis remains unresolved because both prior lower-floor implementations could not be verified.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A lightweight residual bottleneck over the fused 7×7 feature map will exceed 9,328 correct predictions by learning local part relationships that classifier widening and global pooling cannot represent, while its 0.1-scaled initialization preserves the verified pathway.
change: Add a 13,568-parameter convolutional bottleneck after view fusion, initialized as a small residual correction; total learned parameters become 238,010.
mechanism: Near-identity post-fusion spatial refinement
evidence_used: Dense-head widening reached only 9,300 and a global-average branch reached 9,290, indicating remaining capacity is better allocated to spatial feature processing while retaining the successful fusion and classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise while retaining the verified training trajectory and current BatchNorm-buffer handling.
change: Double the effective parameter-averaging horizon during the second half of training without changing architecture, loss, learning-rate schedule, runtime, or buffer copying.
mechanism: Longer-horizon terminal parameter EMA
evidence_used: The 0.02 parameter EMA produced the 9,328 baseline, while architecture, augmentation, and late-loss changes failed to improve it; the unresolved lower-learning-rate-floor experiments specifically motivate testing runtime-neutral late-update stabilization through stronger averaging.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing BatchNorm momentum to 0.02 will exceed 9,328 correct predictions by aligning the running-statistics horizon with the successful 0.02 parameter EMA and reducing terminal normalization noise.
change: Set all four BatchNorm layers to momentum 0.02 without changing architecture, learned parameters, loss, or runtime complexity.
mechanism: EMA-aligned BatchNorm statistics
evidence_used: The 9,328 baseline copies current BatchNorm buffers into the parameter-EMA model, so those statistics retain a much shorter effective horizon than its weights; architecture, fusion, augmentation, and late-loss changes failed to improve the baseline, motivating this isolated stabilization change.
result: the implementation could not be verified

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
