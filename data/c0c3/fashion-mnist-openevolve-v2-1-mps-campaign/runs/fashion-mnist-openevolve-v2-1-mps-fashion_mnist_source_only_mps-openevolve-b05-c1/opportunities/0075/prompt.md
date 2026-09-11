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
hypothesis: Recalibrating the 30-dimensional spatial embedding from normalized global channel means will exceed 9,344 correct predictions while finishing within the time limit by retaining input-conditioned multiplicative gating without backpropagating a gate across the full feature map.
change: Split the spatial classifier into projection and output layers, then add an identity-initialized 64→8→30 context gate that reuses the statistics head’s normalized means and modulates only the compact embedding; parameters increase to 248,208.
mechanism: Global-context bottleneck recalibration
evidence_used: Full-map squeeze-and-excitation reported 9,345 correct, the only representation change above the current 9,344, but took 79.6 seconds, and its narrower successor also timed out. Moving recalibration after the existing bottleneck removes the expensive feature-map multiplication while directly testing the promising global-context signal.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 248208, "training_seconds": 63.2225579158403, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.1879007553100586, "validation_score": 9344.42091058345}

RECENT RESULT
hypothesis: A zero-initialized classifier over 14×14 average-pooled input pixels will exceed 9,344 correct predictions by preserving coarse absolute shape and position information discarded by the trunk’s 30-dimensional bottleneck, with negligible runtime overhead.
change: Add a normalized class-specific residual head over 2×2 average-pooled input images, increasing learned parameters from 247,418 to 249,780.
mechanism: Coarse input-silhouette residual bypass
evidence_used: The normalized statistics bypass showed that direct class-specific paths around the bottleneck improve accuracy, while feature-map positional heads exceeded the time limit; pooling the single-channel input supplies positional evidence far more cheaply.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A per-channel self-conditioned gate will exceed 9,344 correct predictions by retaining the beneficial full-feature-map recalibration signal while avoiding the runtime cost of a squeeze-and-excitation MLP.
change: Add an identity-initialized, 128-parameter gate derived from each channel’s global mean and apply it analytically to both the feature map and its mean/std/max statistics.
mechanism: Diagonal global-context feature recalibration
evidence_used: Full-map squeeze-and-excitation reached 9,345 correct—the only representation change above 9,344—but exceeded the time limit, while bottleneck-only gating tied at 9,344. This suggests modulation must occur before the spatial classifier; a diagonal gate tests that signal with substantially less computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 75.34807645902038, "validation_accuracy": 0.9345, "validation_correct": 9345, "validation_cross_entropy": 0.18751400756835937, "validation_score": 9345.421047664966}

RECENT RESULT
hypothesis: Standardizing each channel’s global mean before self-gating will exceed 9,345 correct predictions by separating image-dependent variation from the gate’s static bias without adding learned parameters or material runtime.
change: Add a parameter-free BatchNorm context normalizer and use its standardized channel means to drive the existing identity-initialized gate.
mechanism: Standardized diagonal channel recalibration
evidence_used: Raw-mean diagonal gating improved the best result from 9,344 to 9,345, while bottleneck gating only tied and heavier full-map gates timed out; improving the successful pre-classifier gate’s conditioning is therefore the most direct low-cost refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 70.20112149999477, "validation_accuracy": 0.9318, "validation_correct": 9318, "validation_cross_entropy": 0.1872269500732422, "validation_score": 9318.421149469332}

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
