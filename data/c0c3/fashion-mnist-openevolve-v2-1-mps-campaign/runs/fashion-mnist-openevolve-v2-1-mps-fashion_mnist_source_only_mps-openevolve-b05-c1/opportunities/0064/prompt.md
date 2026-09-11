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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247418, "training_seconds": 73.57850220892578, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.1871829517364502, "validation_score": 9344.421165077605}
prior_hypothesis: Averaging BatchNorm running means and variances with the same trajectory as the learned parameters will exceed 9,342 correct predictions by eliminating the evaluation mismatch between EMA weights and final-step normalization statistics.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a zero-initialized class-specific head over normalized 2×2 pooled feature maps will exceed 9,334 correct predictions by preserving coarse positional evidence discarded by the shared 30-feature bottleneck.
change: Add a non-affine normalization and direct ten-class residual head over four fixed spatial regions per channel, increasing learned parameters from 247,418 to 249,988.
mechanism: Normalized coarse-spatial residual bypass
evidence_used: The normalized global-statistics bypass improved correctness from 9,314 to 9,334, while content-addressed pooling regressed. The prior coarse-spatial proposal was not verified, so it supplies no contrary accuracy evidence and merits a corrected direct test.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A normalized class-specific head over horizontal, vertical, and diagonal 2×2 feature contrasts will exceed 9,334 correct predictions by adding coarse positional evidence without redundantly relearning the global mean already used by the successful statistics head.
change: Add a zero-initialized residual classifier over three orthogonal contrasts from 2×2 average-pooled feature maps, bringing the model to 249,732 learned parameters.
mechanism: Orthogonal coarse-spatial contrast residual head
evidence_used: The global-statistics bypass improved correctness from 9,314 to 9,334, showing that class-specific paths around the 30-feature bottleneck help. The two coarse-spatial proposals were not verified and provide no contrary accuracy result; this compact formulation removes their redundant pooled-mean component and avoids operating at the parameter ceiling.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a zero-initialized class-specific head over horizontal, vertical, and diagonal 2×2 feature contrasts will exceed 9,334 correct predictions by preserving coarse positional evidence omitted by the successful global-statistics bypass.
change: Add an affine-normalized 192-feature contrast head computed from adaptive 2×2 pooled trunk features, keeping the model below the 250,000-parameter ceiling.
mechanism: Normalized coarse-spatial contrast residual head
evidence_used: The global-statistics residual head improved correctness from 9,314 to 9,334, showing that class-specific bypasses around the 30-feature bottleneck help; the earlier contrast proposal was not verified and therefore supplies no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding class-specific horizontal and vertical first moments will exceed 9,334 correct predictions by preserving coarse positional evidence omitted by the translation-invariant statistics head.
change: Add a zero-initialized residual classifier over normalized per-channel horizontal and vertical feature moments, increasing learned parameters from 247,418 to 248,964.
mechanism: Normalized spatial first-moment residual head
evidence_used: The global-statistics bypass improved correctness from 9,314 to 9,334, showing that direct class-specific paths around the bottleneck help; prior coarse-spatial proposals were never verified, so positional evidence remains an informative untested complement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Extending the successful global-statistics bypass with per-channel local variation will exceed 9,334 correct predictions by capturing feature-map edge and texture structure that mean, standard deviation, and maximum do not encode.
change: Add mean absolute horizontal/vertical feature differences as a fourth normalized statistic while retaining the same zero-initialized residual head design; learned parameters increase from 247,418 to 248,186.
mechanism: Channel-wise spatial total-variation statistic
evidence_used: The normalized global-statistics bypass improved correctness from 9,314 to 9,334, while more expensive positional bypasses could not be verified; this tests a compact complementary signal within the proven path.
result: training did not finish within the verification time limit

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
