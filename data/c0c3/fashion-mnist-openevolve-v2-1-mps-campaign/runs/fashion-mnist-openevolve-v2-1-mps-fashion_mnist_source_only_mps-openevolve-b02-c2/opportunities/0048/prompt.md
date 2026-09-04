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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.78843287518248, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1926749450683594, "validation_score": 9307.419225709458}
prior_hypothesis: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.76099287485704, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19212254104614257, "validation_score": 9312.419419969663}
prior_hypothesis: Replacing smooth maximum attention with the strongest 2×2 activation average will exceed 9,322 correct predictions by capturing coherent four-activation evidence without top-k’s runtime cost or regional pooling’s forced spatial coverage.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing smooth maximum attention with the strongest 2×2 activation average will exceed 9,322 correct predictions by capturing coherent four-activation evidence without top-k’s runtime cost or regional pooling’s forced spatial coverage.
change: Replace temperature-controlled log-mean-exp saliency with a parameter-free maximum over local 2×2 average-pooled responses; preserve shared attention, paired-view training, and probability flip ensembling.
mechanism: Strongest-local-patch channel saliency
evidence_used: Arbitrary top-four averaging achieved the best result at 9,322 correct, while hard maxima reached 9,320 and four fixed regional maxima fell to 9,301; selecting the strongest local four-value patch tests robust multi-activation evidence without requiring irrelevant regions or expensive top-k selection.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.76099287485704, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19212254104614257, "validation_score": 9312.419419969663}

RECENT RESULT
hypothesis: Restoring the 9,320-correct hard-maximum attention and adding a mild centered-logit agreement penalty will exceed 9,322 correct predictions by improving orientation invariance without the harmful ensemble-aware objective or top-k runtime overhead.
change: Replace top-four channel saliency with global-maximum evidence and regularize the already-computed paired-orientation logits toward agreement while retaining ordinary cross-entropy.
mechanism: Lightweight flip-logit consistency on reliable maximum attention
evidence_used: Hard-maximum attention reached 9,320 correct with lower cross-entropy and faster training than the 9,322-correct top-four model; paired-view training and flip ensembling were beneficial, whereas directly optimizing ensemble likelihood reduced accuracy to 9,307.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.34299483313225, "validation_accuracy": 0.9296, "validation_correct": 9296, "validation_cross_entropy": 0.19279716911315917, "validation_score": 9296.41918275206}

RECENT RESULT
hypothesis: Giving global-average and global-maximum descriptors independent three-tap kernels will exceed 9,322 correct predictions by allowing complementary channel statistics to learn distinct interactions without changing the proven saliency computation.
change: Add a separately initialized three-parameter kernel for maximum-channel evidence while preserving paired-view training, flip ensembling, and all other behavior.
mechanism: Descriptor-specific channel-attention kernels
evidence_used: Hard-maximum attention reliably achieved 9,320 correct, while alternative saliency reductions produced only a two-correct gain or regressed; preserving maximum evidence and relaxing its shared-kernel constraint is a low-cost, previously untested refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249858, "training_seconds": 68.61705149989575, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.19352327423095703, "validation_score": 9300.418927733372}

RECENT RESULT
hypothesis: Restoring hard-maximum attention and adding identity-initialized per-channel scale and bias will exceed 9,322 correct predictions by preserving the successful shared descriptor kernel while allowing lightweight channel-specific calibration.
change: Restore the reliable global-average-plus-maximum descriptor and affine-calibrate its shared attention output with 128 learned parameters, keeping the model at 249,983 parameters.
mechanism: Per-channel affine calibration of shared channel attention
evidence_used: Shared average-plus-maximum attention reached 9,320 correct, whereas giving the two descriptors separate kernels fell to 9,300; this preserves beneficial kernel sharing and tests channel-specific flexibility along an orthogonal dimension.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249983, "training_seconds": 78.64067458407953, "validation_accuracy": 0.9308, "validation_correct": 9308, "validation_cross_entropy": 0.19216576232910157, "validation_score": 9308.419404763834}

RECENT RESULT
hypothesis: Restoring top-four channel saliency and plain paired-view cross-entropy while excluding normalization and bias vectors from weight decay will exceed 9,322 correct predictions by preserving the strongest verified architecture and avoiding unnecessary shrinkage of calibration parameters.
change: Use the verified top-four descriptor and paired-view loss, then apply AdamW decay only to multidimensional kernel and matrix parameters.
mechanism: Dimension-selective AdamW regularization
evidence_used: Top-four attention with plain paired-view cross-entropy achieved the best result at 9,322 correct, while ensemble-aware training fell to 9,307; selective decay is an orthogonal, parameter-neutral optimizer refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the redundant pre-pairing flip with ±2-pixel translations on half the examples will exceed 9,322 correct predictions by improving spatial robustness while retaining centered training views and the strongest top-four attention design.
change: Apply efficient per-example random crops from replicate-padded images; explicit paired-view training continues to provide both horizontal orientations.
mechanism: Conservative per-example translation augmentation
evidence_used: Top-four attention is the best verified architecture at 9,322 correct, while numerous attention and loss changes regressed or timed out. The current preparation flip adds no distinct orientation because training_loss already includes every image and its flip, motivating an orthogonal, low-overhead augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the redundant preparation flip with centered-or-cardinal one-pixel translations will exceed 9,322 correct predictions by improving shift robustness while retaining the faster hard-maximum attention model.
change: Use the unmodified batch on half the steps and a shared replicate-padded one-pixel translation on the remainder, avoiding costly per-example crop indexing.
mechanism: Low-overhead cyclic one-pixel translation augmentation
evidence_used: Hard-maximum attention reached 9,320 correct within the time limit, while per-example translations on the 9,322-correct top-four model timed out; the existing preparation flip is redundant because training_loss already evaluates both orientations.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.96869008406065, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.19687039031982423, "validation_score": 9278.417756178149}

RECENT RESULT
hypothesis: Averaging each channel’s two strongest activations will exceed 9,322 correct predictions by retaining top-four’s outlier robustness while preserving more of the peak emphasis associated with hard maximum’s lower cross-entropy.
change: Replace strongest-local-patch saliency with the mean of the top two spatial activations; preserve the architecture, paired-view loss, optimizer, and flip ensemble.
mechanism: Top-two channel saliency
evidence_used: Top-four saliency achieved 9,322 correct, hard maximum achieved 9,320 with lower cross-entropy, and strongest-local-patch averaging fell to 9,312; top-two saliency directly tests the unexplored intermediate between the two strongest verified reductions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the best verified top-four attention and paired-view cross-entropy while reducing classifier dropout from 0.10 to 0.05 will exceed 9,322 correct predictions by improving convergence under the fixed two-pass exposure budget without removing regularization entirely.
change: Replace hard-maximum saliency with top-four averaging, remove the harmful ensemble-aware loss, and halve classifier dropout.
mechanism: Reduced classifier dropout with top-four channel saliency
evidence_used: Top-four saliency with plain paired-view cross-entropy achieved the best result at 9,322 correct, while ensemble-aware training reduced accuracy to 9,307; recent attention, consistency, and augmentation changes failed to improve it, motivating an orthogonal, parameter-neutral adjustment to classifier regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the strongest verified top-four model’s late optimization trajectory will achieve at least 9,323 correct predictions by reducing checkpoint noise without altering its architecture or training objective.
change: Maintain an exponential moving average of learned parameters, update it every 16 optimizer steps with decay 0.90, and install the averaged weights after the final step.
mechanism: Sparse exponential weight averaging
evidence_used: Top-four attention produced the best result at 9,322 correct, while attention, augmentation, and consistency-loss changes generally regressed or timed out; weight averaging preserves that implementation and tests an orthogonal, low-overhead generalization refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing classifier dropout from 0.10 to 0.05 will exceed 9,322 correct predictions by improving convergence during the fixed two-pass exposure budget while retaining mild regularization.
change: Halve classifier dropout while preserving the reliable hard-maximum attention architecture, paired-view cross-entropy, optimizer, and flip ensemble.
mechanism: Reduced classifier dropout on fast maximum attention
evidence_used: Hard-maximum attention reached 9,320 correct in 75.3 seconds; the prior reduced-dropout trial used slower top-four saliency and timed out, so applying the parameter-neutral change to the faster verified backbone isolates its accuracy effect with better timing margin.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging paired-orientation logits with the 9,320-correct hard-maximum model will exceed 9,322 correct predictions by preventing one orientation’s overconfident probability from dominating the ensemble.
change: Restore global-maximum channel saliency and replace arithmetic probability averaging at validation with equal logit averaging.
mechanism: Geometric flip ensemble on fast maximum attention
evidence_used: Hard-maximum attention reached 9,320 correct with lower cross-entropy and faster training than the 9,322-correct top-four model; ensemble-aware training hurt accuracy, motivating an inference-only aggregation change that preserves ordinary paired-view cross-entropy.
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
