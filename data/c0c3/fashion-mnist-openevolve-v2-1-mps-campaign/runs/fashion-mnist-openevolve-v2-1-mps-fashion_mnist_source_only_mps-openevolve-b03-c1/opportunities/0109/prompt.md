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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 66.76434204191901, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.23899501876831056, "validation_score": 9166.403552873438}
prior_hypothesis: Learning the existing three derivative kernels while constraining each to remain zero-mean will exceed 9,162 correct predictions by adapting edge and texture extraction to the dataset without adding the redundant gradient-magnitude channel that regressed to 9,114.

## Recent verification evidence

RECENT RESULT
hypothesis: Sampling training translations from the same center and cardinal ±1 distribution used by the successful validation ensemble will exceed 9,166 correct predictions by eliminating unsupported diagonal and ±2 crops.
change: Replace independent triangular x/y translations with center/cardinal ±1 translations sampled in the validation ensemble’s exact 3:2:2:2:2 proportions; retain horizontal flips and all other behavior.
mechanism: Validation-aligned cardinal translation sampling
evidence_used: Adding diagonal validation views reduced correctness from 9,166 to 9,155, while the current independent x/y augmentation frequently creates diagonal crops; this directly tests whether aligning training support with the established axis-only ensemble improves classification.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing label smoothing from 0.015 to 0.020 will exceed 9,166 correct predictions by strengthening useful target regularization without introducing mixup’s input-distribution distortion.
change: Use constant 0.020 label smoothing while preserving the architecture, augmentation, optimizer, schedule, EMA, and evaluation behavior.
mechanism: Modestly strengthened label smoothing
evidence_used: Annealing smoothing toward zero reduced correctness from 9,166 to 9,160, and weaker dropout reached only 9,141, indicating that reduced regularization is harmful; the small increase avoids the substantially stronger soft-target intervention that made mixup fall to 9,130.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final 2×2 pooling with 3×3 stride-2 pooling will exceed 9,166 correct predictions by preserving evidence from the bottom and right edges of the 7×7 feature map while retaining the same 3×3 classifier input.
change: Change only the third pooling layer to an overlapping 3×3, stride-2 max pool; parameter count and all training and evaluation behavior remain unchanged.
mechanism: Boundary-complete overlapping final pooling
evidence_used: Translation-ensemble refinements did not improve the 9,166-correct baseline. The current final 2×2 stride-2 pool maps 7×7 to 3×3 by discarding the last feature row and column, motivating a cheap architectural correction instead of further TTA expansion.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed max pooling with learned per-channel mixtures of max and average pooling will exceed 9,166 correct predictions by retaining distributed shape evidence and providing denser encoder gradients within the short fixed-exposure budget.
change: Add a lightweight mixed-pooling layer and replace all three fixed 2×2 max-pooling stages, adding 144 parameters for a 249,933-parameter model while preserving feature-map geometry and the existing classifier.
mechanism: Per-channel learned max–average downsampling
evidence_used: Head replacements and global-context classification either timed out or regressed to 9,128, while loss and TTA changes failed to beat 9,166. This challenges their shared assumption that fixed winner-take-all downsampling already provides an adequate representation, using a computationally inexpensive learned alternative initialized close to max pooling.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using broad ±2 translations during the first half of training, then matching the successful center/cardinal validation-view distribution during the second half, will exceed 9,166 correct predictions by retaining early invariance while removing unsupported diagonal and large-shift exposure during late fitting.
change: Preserve the existing augmentation initially, then switch halfway through training to center and cardinal ±1 crops sampled in the validation ensemble’s 3:2:2:2:2 proportions.
mechanism: Two-stage augmentation-support curriculum
evidence_used: Adding diagonal validation views reduced correctness from 9,166 to 9,155, indicating that diagonal shifts dilute useful predictions; a late alignment stage tests that signal without discarding the baseline’s broad early regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing both classifier dropout rates will exceed 9,166 correct predictions by reducing overfitting in the parameter-dominant flattened head without changing the successful representation, optimizer, or runtime profile.
change: Increase the pre-hidden dropout from 0.15 to 0.20 and the pre-logit dropout from 0.10 to 0.15.
mechanism: Stronger dense-head dropout regularization
evidence_used: Reducing dropout lowered correctness to 9,141, while architecture replacements and auxiliary heads failed or timed out; this directly tests the evidence-supported opposite direction with no added parameters or computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.98555345786735, "validation_accuracy": 0.9139, "validation_correct": 9139, "validation_cross_entropy": 0.241654695892334, "validation_score": 9139.402688446035}

RECENT RESULT
hypothesis: Reducing the centered-view weight from 3.0 to 2.5 will exceed 9,166 correct predictions by modestly strengthening the contribution of the successful cardinal-shift ensemble.
change: Decrease only the centered validation view’s weight while preserving training, cardinal views, horizontal flips, calibration, and parameter count.
mechanism: Translation-favoring logit-space ensemble
evidence_used: Increasing the centered-view weight from 3.0 to 4.0 reduced correctness from 9,166 to 9,164, directly motivating a conservative test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.89393425011076, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.2390326629638672, "validation_score": 9163.403540612726}

RECENT RESULT
hypothesis: Adding a trainable per-channel 3×3 residual refinement over the final feature map will exceed 9,166 correct predictions by learning local spatial context while preserving the successful representation at initialization and adding negligible runtime.
change: Add a zero-initialized depthwise residual layer after the final convolution and reduce the dense hidden width from 160 to 159, keeping the model under the parameter ceiling at 249,970 parameters.
mechanism: Zero-initialized depthwise spatial residual refinement
evidence_used: Learned zero-DC detail filters improved correctness to 9,166, supporting adaptive spatial filtering; prior heavier spatial-context attempts timed out, motivating a depthwise residual refinement with minimal computation and an identity initialization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging view probabilities instead of logits will exceed 9,166 correct predictions by preventing any overconfident shifted view from disproportionately suppressing the correct class.
change: Keep the established center/cardinal/flip views and weights, but combine their softmax probabilities and return calibrated log-probabilities as logits.
mechanism: Confidence-bounded probability-space test-time augmentation
evidence_used: Adjusting the center weight in either direction failed to improve the 9,166-correct ensemble, while adding diagonal views hurt; this motivates changing how the proven views are aggregated rather than changing their composition or weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending BatchNorm running statistics toward their bias-corrected EMA by the same factor as learned parameters will exceed 9,166 correct predictions by removing the state mismatch in the final averaged model.
change: Track EMAs of all BatchNorm running means and variances during training, then apply the existing final interpolation factor to those buffers alongside the parameters.
mechanism: EMA-consistent BatchNorm statistics
evidence_used: The 9,166-correct baseline remains unbeaten by classifier, pooling, loss, augmentation, and TTA changes; this preserves that implementation while addressing its currently unaveraged BatchNorm state at negligible computational cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 81.01280508283526, "validation_accuracy": 0.9164, "validation_correct": 9164, "validation_cross_entropy": 0.23914504623413085, "validation_score": 9164.403504013933}

RECENT RESULT
hypothesis: Excluding `detail_kernels` from the final EMA interpolation will exceed 9,166 correct predictions by preserving their most recently learned edge and texture adaptations while retaining EMA regularization for the rest of the model.
change: Continue tracking all parameter EMAs, but leave the learned zero-mean detail kernels at their final optimizer values when applying the end-of-training EMA blend.
mechanism: EMA-free late adaptation for learned detail filters
evidence_used: Learning the zero-DC detail kernels improved correctness from 9,162 to 9,166, while additional constraints reduced accuracy; this motivates allowing the successful parameterization to adapt fully without changing its architecture, learning rate, or runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing label smoothing linearly from 0.010 to 0.020 will exceed 9,166 correct predictions by concentrating regularization late in training while preserving the baseline’s average smoothing strength.
change: Replace constant 0.015 label smoothing with a linear 0.010-to-0.020 schedule over the fixed training exposure.
mechanism: Iso-mean increasing label-smoothing schedule
evidence_used: Annealing smoothing toward zero reduced correctness from 9,166 to 9,160, indicating that late-training smoothing is valuable; keeping the schedule’s mean at 0.015 avoids the stronger average regularization and distribution distortion associated with less successful alternatives.
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
