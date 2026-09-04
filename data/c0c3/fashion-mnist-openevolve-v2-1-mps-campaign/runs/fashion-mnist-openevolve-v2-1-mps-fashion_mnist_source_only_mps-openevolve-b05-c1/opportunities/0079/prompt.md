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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 75.34807645902038, "validation_accuracy": 0.9345, "validation_correct": 9345, "validation_cross_entropy": 0.18751400756835937, "validation_score": 9345.421047664966}
prior_hypothesis: A per-channel self-conditioned gate will exceed 9,344 correct predictions by retaining the beneficial full-feature-map recalibration signal while avoiding the runtime cost of a squeeze-and-excitation MLP.

## Recent verification evidence

RECENT RESULT
hypothesis: Conditioning each channel’s gate on its raw mean, standard deviation, and maximum will exceed 9,345 correct predictions by distinguishing diffuse from localized activations while preserving the successful pre-classifier modulation.
change: Extend the identity-initialized diagonal gate with per-channel standard-deviation and maximum coefficients, reusing reductions already computed for the statistics head and increasing parameters from 247,546 to 247,674.
mechanism: Multi-statistic diagonal channel recalibration
evidence_used: Raw-mean diagonal gating improved the best result to 9,345 correct, whereas BatchNorm-standardized means fell to 9,318; the existing raw standard-deviation and maximum statistics therefore provide the most direct low-cost additional context without the harmful normalization or an expensive cross-channel MLP.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A learned rank-one summary of all channel means will exceed 9,345 correct predictions by adding cross-channel context to the successful diagonal gate without the runtime cost of squeeze-and-excitation.
change: Extend the identity-initialized gate with a learned 64-to-1 context projection and per-channel response, while fusing the existing mean and variance reduction; parameters increase from 247,546 to 247,674.
mechanism: Rank-one cross-channel feature recalibration
evidence_used: Full squeeze-and-excitation and diagonal gating both reached 9,345 correct, but the cross-channel design exceeded the time limit; a rank-one interaction tests whether its missing cross-channel signal can improve the efficient diagonal design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Doubling the gate’s local sensitivity at its identity initialization will exceed 9,345 correct predictions by helping the successful raw-mean recalibration learn within only 1,042 optimizer steps.
change: Replace the `2·sigmoid` gate with an equally bounded, identity-initialized `1+tanh` gate, adding no parameters or reductions.
mechanism: High-gain identity-centered diagonal recalibration
evidence_used: Raw-mean diagonal gating produced the current best 9,345 correct, while standardized and more computationally elaborate conditioning failed; increasing the proven gate’s learning response is a low-cost test of whether it remains underfit.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 77.06420179200359, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1900846839904785, "validation_score": 9328.420138168927}

RECENT RESULT
hypothesis: Mixing channels into class-specific evidence maps before smooth-max spatial pooling will exceed 9,345 correct predictions by detecting co-located class features that channelwise statistics and the rank-30 flattened bottleneck cannot represent directly.
change: Add a zero-initialized 1×1 class-evidence head over the recalibrated feature map, aggregate each class map with normalized log-sum-exp pooling, and add those scores to the existing logits.
mechanism: Class-coherent spatial evidence pooling
evidence_used: The channelwise statistics bypass and diagonal gate improved the best result to 9,345, but further global-statistic conditioning failed or timed out. This challenges their shared pool-before-class-mixing assumption with a cheap class-mixing-before-pooling path while preserving the current model exactly at initialization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A zero-initialized 12-unit nonlinear branch over the normalized mean/std/max vector will exceed 9,345 correct predictions by learning cross-channel statistic interactions that the existing linear residual head and channelwise gate cannot represent.
change: Add a compact 192→12→10 GELU residual branch using the existing normalized statistics, raising learned parameters from 247,546 to 249,992 without new spatial reductions.
mechanism: Nonlinear cross-channel statistics interaction
evidence_used: The statistics bypass materially improved correctness, and full cross-channel recalibration reached 9,345 but was too slow; operating after the already-computed reductions tests cross-channel nonlinear context with negligible additional computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Centering each channel’s raw mean with the existing statistics BatchNorm running mean will exceed 9,345 correct predictions by separating image-dependent gate variation from static channel bias without the harmful variance normalization observed previously.
change: Drive the existing identity-initialized gate with running-mean-centered channel means, adding no parameters, reductions, or material runtime.
mechanism: Running-mean-centered diagonal channel recalibration
evidence_used: Raw-mean gating improved the best result to 9,345, while full BatchNorm standardization fell to 9,318; retaining raw scale while removing only the learned baseline is the most direct test of whether variance normalization caused that regression.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.07118250010535, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.18893792266845702, "validation_score": 9328.420543403037}

RECENT RESULT
hypothesis: Restricting the channel gate to the spatial classifier will exceed 9,345 correct predictions by preserving the successful feature-map modulation while preventing sample-dependent scaling from destabilizing the BatchNorm-normalized statistics branch.
change: Stop applying the learned channel gate to the mean/std/max statistics; retain it unchanged on the full feature map.
mechanism: Spatial-path-only diagonal recalibration
evidence_used: Raw-mean diagonal gating improved the best result from 9,344 to 9,345, while bottleneck-only gating tied at 9,344 and normalized gate conditioning regressed to 9,318. This directly tests whether the gain comes from pre-classifier spatial modulation while keeping the statistics residual on its established normalization distribution.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding learnable curvature to each normalized statistic will exceed 9,345 correct predictions while remaining within the time limit by capturing nonlinear distribution-shape evidence without an additional classifier branch.
change: Add 192 learned curvature coefficients that augment each normalized mean/std/max feature with a centered quadratic term before the existing statistics head, increasing parameters to 247,738.
mechanism: Identity-initialized diagonal quadratic statistics enrichment
evidence_used: The diagonal channel gate produced the best result of 9,345 correct, while the 12-unit nonlinear statistics branch timed out; a factorized quadratic transform tests nonlinear statistics at substantially lower computational cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247738, "training_seconds": 80.50659287511371, "validation_accuracy": 0.9338, "validation_correct": 9338, "validation_cross_entropy": 0.18618330001831054, "validation_score": 9338.421520012962}

RECENT RESULT
hypothesis: Averaging per-view log-probabilities will exceed 9,345 correct predictions by favoring class evidence consistent across translated and flipped views instead of allowing a single confident view to dominate the arithmetic probability mixture.
change: Replace arithmetic softmax averaging during evaluation with the same weighted ensemble over log-softmax outputs; training, parameters, view weights, and evaluation cost remain unchanged.
mechanism: Weighted geometric test-time augmentation ensemble
evidence_used: Raw-mean diagonal gating is the current best at 9,345 correct, while subsequent representation changes failed to improve or exceeded the time limit; refining the existing ten-view aggregation isolates an untested inference decision without disturbing the successful model.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final 64-channel residual block with a cheaper 48-channel bottleneck and widening the flattened classifier from 30 to 45 units will exceed 9,345 correct predictions by preserving more class-specific spatial information within the parameter and runtime limits.
change: Introduce a bottleneck residual block for the 7×7 feature stage and invest its parameter savings in a wider positional classifier, raising parameters from 247,546 to 247,997 while reducing convolutional computation.
mechanism: Compute-neutral spatial-to-classifier capacity reallocation
evidence_used: Direct statistics bypasses improved accuracy around the rank-30 flattened bottleneck, whereas additional positional heads timed out; reallocating capacity inside the existing path tests the same bottleneck limitation while substantially reducing final-stage convolution cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a per-channel bottom-versus-top activation contrast will exceed 9,345 correct predictions by supplying coarse spatial layout directly to the classifier without relying on the rank-30 flattened bottleneck.
change: Extend the normalized statistics residual from mean/std/max to mean/std/max/vertical-contrast, adding 768 parameters for a total of 248,314 while preserving the initial logits.
mechanism: Vertical-contrast statistics bypass
evidence_used: The statistics bypass improved prior results, while attempts to widen positional processing timed out; a single low-frequency spatial statistic targets the same lost spatial information with substantially less computation and parameter cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 248314, "training_seconds": 76.81869908282533, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.18973870925903322, "validation_score": 9331.420260344652}

RECENT RESULT
hypothesis: Reducing the batch size from 96 to 80 will exceed 9,345 correct predictions by providing 1,250 optimizer steps instead of 1,042 and modestly noisier gradients within the same fixed 100,000-example exposure.
change: Set the training batch size to 80 while preserving the architecture, optimizer, schedule shape, augmentation, and inference ensemble.
mechanism: Smaller-batch update densification
evidence_used: Multiple representation additions failed to improve the 9,345-correct result or exceeded the time limit, while every reported run used only 1,042 updates at batch size 96; increasing update density tests whether optimization exposure, rather than missing model capacity, is now the limiting factor.
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
