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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.56534891715273, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922137680053711, "validation_score": 9320.419387876082}
prior_hypothesis: Mildly favoring the more decisive horizontal view will finish within the reliable hard-maximum runtime and exceed 9,322 correct predictions by resolving flip disagreements more accurately.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 70.16063670790754, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.2018985595703125, "validation_score": 9328.416008485923}
prior_hypothesis: Applying 0.02 label smoothing to the reliable paired-view objective will exceed 9,322 correct predictions by reducing brittle class-boundary fitting, while the verified 0.10 confidence-adaptive fusion improves the cross-entropy tie-breaker.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging four regional maxima will finish within the reliable runtime envelope and exceed 9,322 correct predictions by approximating top-four saliency without a costly ranking operation.
change: Replace each channel’s single global maximum with the mean of a 2×2 adaptive-max grid, leaving training and evaluation otherwise unchanged.
mechanism: Spatially balanced four-region channel saliency
evidence_used: Global top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard maxima reliably reached 9,320; regional max pooling supplies four robust salient activations using the already-reliable pooling primitive instead of `topk`.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.54781024996191, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.1925997917175293, "validation_score": 9301.419252127556}

RECENT RESULT
hypothesis: Giving 25% of the training objective to the same arithmetic-probability flip ensemble used at validation will exceed 9,322 correct predictions by rewarding complementary paired views without forcing agreement.
change: Train with a convex combination of per-view NLL and paired probability-mixture NLL, then use equal probability-space flip fusion for evaluation.
mechanism: Paired probability-mixture supervision
evidence_used: Equal probability fusion preserved 9,320 correct while lowering cross-entropy to 0.192262, whereas directly forcing view agreement with Jensen–Shannon consistency regressed to 9,303; supervising the fused prediction targets validation behavior without imposing that harmful constraint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.62815520912409, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.19391542243957519, "validation_score": 9290.41879013421}

RECENT RESULT
hypothesis: Replacing global top-four ranking with the strongest contiguous 2×2 activation mean will finish within the reliable runtime envelope and exceed 9,322 correct predictions by preserving robust four-activation evidence with denser local gradients.
change: Pool each channel over overlapping 2×2 neighborhoods and use the maximum neighborhood mean as its salient descriptor, leaving all other training and evaluation behavior unchanged.
mechanism: Strongest local 2×2 mean channel saliency
evidence_used: Global top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard maxima reliably finished near 9,320; unlike the failed four-region design, overlapping local pooling does not force evidence from separate spatial regions and avoids ranking operations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.74580199993216, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19212254104614257, "validation_score": 9312.419419969663}

RECENT RESULT
hypothesis: Averaging maxima from four interleaved spatial lattices will exceed 9,322 correct predictions by approximating successful global top-four saliency without ranking cost or forced regional coverage, while completing within the reliable runtime envelope.
change: Replace each channel’s hard global maximum with the mean maximum across the four checkerboard-parity lattices; leave optimization, training, and confidence-adaptive flip fusion unchanged.
mechanism: Interleaved-lattice four-peak channel saliency
evidence_used: Global top-four saliency produced the best verified count of 9,322 but repeatedly timed out, whereas hard maximum reliably reached 9,320. Regional and local pooling regressed because they constrain where salient activations must occur; interleaved lattices let all four peaks come from anywhere while avoiding `topk`.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Globally softmax-weighting spatial activations will finish within the time limit and exceed 9,322 correct predictions by approximating top-four evidence without ranking or imposing spatial constraints.
change: Replace hard-max channel evidence with a temperature-0.5 softmax-weighted global descriptor and fuse the two bias-free attention convolutions into one equivalent call.
mechanism: Temperature-controlled soft top-four channel saliency
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly timed out; hard maximum reliably finished at 9,320, while regional and local approximations regressed because they constrained evidence spatially.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.69797208392993, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.19263459434509278, "validation_score": 9310.419239893234}

RECENT RESULT
hypothesis: Using top-four saliency only during validation will reach at least 9,322 correct predictions without the full-training timeout, while probability-space flip fusion will improve the cross-entropy tie-breaker.
change: Preserve the reliable hard-maximum training path, switch to exact top-four channel evidence only in evaluation mode, and average flip predictions in probability space.
mechanism: Evaluation-only top-four channel saliency
evidence_used: Exact top-four saliency produced the best verified count of 9,322 but repeatedly timed out when used during training; hard-max training reliably finished at 9,320, and probability fusion reduced its cross-entropy from 0.192650 to 0.192262 without reducing correct predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying a low-overhead EMA over the latter half of the reliable hard-maximum training trajectory will finish within the time limit and exceed 9,322 correct predictions by reducing late optimization noise.
change: Restore hard-maximum channel attention and maintain an EMA every four optimizer steps, installing the averaged parameters before validation.
mechanism: Sparse late exponential weight averaging
evidence_used: Hard-maximum attention repeatedly finished in 66.6–75.3 seconds at about 9,320 correct, while the 9,322-correct top-four design finished near the limit once and subsequently timed out; this trades two baseline predictions for dependable runtime headroom and a previously untested trajectory-level improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 79.67659595794976, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19206394424438478, "validation_score": 9312.419440586567}

RECENT RESULT
hypothesis: Averaging maxima from two interleaved checkerboard lattices will finish within the reliable runtime envelope and exceed 9,322 correct predictions by providing two robust salient gradients without `topk` or four separate lattice reductions.
change: Replace single hard-max channel evidence with the mean maxima of the two flattened checkerboard parities, and fuse the shared average/salient attention computation into one convolution call.
mechanism: Two-color checkerboard peak attention
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard-max attention reliably finished near 9,320; the four-lattice approximation also timed out, motivating a cheaper two-lattice compromise.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.32541258307174, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.19233039932250975, "validation_score": 9309.419346852419}

RECENT RESULT
hypothesis: Training each paired flip view under a deterministic 3×3 translation cycle will exceed 9,322 correct predictions while remaining within the hard-maximum model’s reliable runtime envelope.
change: Add zero-padded one-pixel translations before the existing random flip and paired-view supervision, covering all nine offsets without extra model parameters.
mechanism: Cyclic one-pixel translation augmentation
evidence_used: Hard-maximum attention reliably reaches 9,320, while numerous saliency replacements regressed or timed out; the current training path augments only horizontal orientation, making low-cost translation invariance the most informative untested axis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.76328500011005, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.2097536651611328, "validation_score": 9232.41330728263}

RECENT RESULT
hypothesis: Applying 0.02 label smoothing to the reliable paired-view objective will exceed 9,322 correct predictions by reducing brittle class-boundary fitting, while the verified 0.10 confidence-adaptive fusion improves the cross-entropy tie-breaker.
change: Add mild label smoothing without forcing flip agreement, and replace mean-logit evaluation with the best verified confidence-adaptive probability ensemble.
mechanism: Mild per-view target smoothing with confidence-adaptive flip fusion
evidence_used: Hard-max attention reliably reached 9,320 while attention approximations, translation augmentation, and paired-mixture supervision regressed or timed out; coefficient-0.10 probability fusion retained all 9,320 correct and improved cross-entropy from 0.192650 to 0.192214.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 70.16063670790754, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.2018985595703125, "validation_score": 9328.416008485923}

RECENT RESULT
hypothesis: Holding 0.02 smoothing through the first half of training and annealing it to zero will exceed 9,328 correct predictions by retaining the verified regularization benefit while allowing late hard-target boundary refinement; if correct counts tie, it will lower cross-entropy.
change: Replace constant 0.02 label smoothing with a schedule that remains at 0.02 for half the optimizer trajectory, then cosine-decays to zero.
mechanism: Late cosine decay of label smoothing
evidence_used: Fixed 0.02 smoothing improved the reliable hard-max design from 9,320 to 9,328 correct, but increased cross-entropy from about 0.1922 to 0.2019, motivating preservation of early smoothing with less late confidence suppression.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.50203983299434, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.18954639472961426, "validation_score": 9325.420328288343}

RECENT RESULT
hypothesis: Holding label smoothing at 0.025 throughout training will exceed 9,328 correct predictions by strengthening the regularization benefit observed at 0.02 without the late hard-target transition that reduced accuracy.
change: Restore the runtime-reliable hard-maximum channel descriptor and 0.10 confidence-adaptive flip fusion, then train paired views with constant 0.025 label smoothing.
mechanism: Slightly stronger constant label smoothing with reliable hard-max attention
evidence_used: Constant 0.02 smoothing improved hard-max attention from 9,320 to 9,328 correct, while annealing smoothing toward zero reduced the result to 9,325; this motivates a small upward search around the best verified constant setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 79.86398470797576, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.20622329635620118, "validation_score": 9307.414516948487}



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
