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
hypothesis: Averaging the late optimization trajectory with a warm-started 0.99-decay EMA will exceed 9,334 correct predictions by reducing final-weight noise without adding learned parameters or expensive forward passes.
change: Maintain an adaptive EMA of all learned parameters after optimizer steps and install the averaged weights after the final step.
mechanism: Warm-started exponential weight averaging
evidence_used: The global-statistics bypass reached 9,334 correct, while additional feature computations repeatedly exceeded the verification time limit; parameter averaging preserves the proven architecture and adds negligible training cost.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247418, "training_seconds": 73.0031390839722, "validation_accuracy": 0.9342, "validation_correct": 9342, "validation_cross_entropy": 0.18791906013488768, "validation_score": 9342.420904097577}

RECENT RESULT
hypothesis: Adding each channel’s standardized peak prominence will exceed 9,342 correct predictions by distinguishing localized activations from diffuse responses using statistics already computed by the successful residual head.
change: Extend the normalized statistics head from three to four features per channel with `(maximum - mean) / standard_deviation`, increasing learned parameters from 247,418 to 248,186 without another spatial reduction.
mechanism: Channel-wise peak-prominence statistic
evidence_used: The mean/std/max statistics bypass improved correctness from 9,314 to 9,334 and EMA raised it to 9,342, while additional spatial computations exceeded the time limit; this adds complementary nonlinear shape information through the proven bypass at negligible computational cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the fixed channel mixture with input-conditioned multiplicative gating will exceed 9,342 correct predictions because global context can suppress irrelevant feature channels before both the spatial classifier and statistics head compute logits.
change: Add a zero-initialized squeeze-and-excitation gate over the trunk’s global channel means and use it to recalibrate the feature map; the model begins exactly equivalent to the current design and has 249,546 learned parameters.
mechanism: Global-context channel recalibration
evidence_used: The global-statistics bypass improved correctness from 9,314 to 9,334, showing that pooled channel context contains useful information, while EMA raised the unchanged representation to 9,342. This challenges the shared assumption that global context should act only as an additive final-logit correction by instead using it to alter the image representation itself.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 249546, "training_seconds": 79.55539587489329, "validation_accuracy": 0.9345, "validation_correct": 9345, "validation_cross_entropy": 0.1889398380279541, "validation_score": 9345.420542725551}

RECENT RESULT
hypothesis: Averaging BatchNorm running means and variances with the same trajectory as the learned parameters will exceed 9,342 correct predictions by eliminating the evaluation mismatch between EMA weights and final-step normalization statistics.
change: Extend the existing EMA to track and install every floating-point model buffer alongside learned parameters, without changing architecture, parameter count, or forward-pass cost.
mechanism: BatchNorm-statistics-aware exponential weight averaging
evidence_used: Parameter-only EMA improved correctness from 9,334 to 9,342, establishing that trajectory averaging helps; the current implementation leaves BatchNorm buffers at their final-step values, making matched normalization statistics the most direct low-cost refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247418, "training_seconds": 73.57850220892578, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.1871829517364502, "validation_score": 9344.421165077605}

RECENT RESULT
hypothesis: An eight-dimensional squeeze-and-excitation gate will exceed 9,344 correct predictions while remaining within the time limit by retaining the promising global-context recalibration signal with roughly half the added parameters and a fused variance/mean reduction.
change: Add an identity-initialized low-rank channel gate, apply it to trunk features and their statistics, and compute variance and mean jointly.
mechanism: Compact global-context channel recalibration
evidence_used: The 16-dimensional channel-recalibration design reported 9,345 correct—the only tested representation change above the current 9,344—but took 79.6 seconds and timed out; a narrower gate directly tests that signal with less computation.
result: training did not finish within the verification time limit

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

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
