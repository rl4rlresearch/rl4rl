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
hypothesis: Annealing label smoothing from 0.015 to zero during the final quarter will exceed 9,166 correct predictions by retaining early regularization while sharpening decision boundaries before validation.
change: Preserve the existing loss for the first 75% of training, then linearly decay label smoothing to zero over the remaining optimizer steps.
mechanism: Late-stage label-smoothing annealing
evidence_used: Early mixup reduced correctness from 9,166 to 9,130, suggesting excess soft-target regularization is harmful, while reduced dropout reached only 9,141; this motivates removing target smoothing only late while preserving the successful structural regularization.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 74.53460283298045, "validation_accuracy": 0.916, "validation_correct": 9160, "validation_cross_entropy": 0.23391489028930665, "validation_score": 9160.40521433361}

RECENT RESULT
hypothesis: Averaging class probabilities across the existing ten validation views will exceed 9,166 correct predictions by limiting the influence of any single overconfident transformed view.
change: Replace weighted logit averaging during evaluation with weighted probability averaging, returning calibrated log-probabilities as valid logits; training remains unchanged.
mechanism: Probability-space test-time augmentation ensemble
evidence_used: The zero-DC adaptive-filter model reached 9,166 correct, while subsequent filter constraints, expanded support, and loss changes did not improve it, motivating an orthogonal refinement to the existing multi-view inference path.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 62.93076283391565, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.24157162399291993, "validation_score": 9163.40271538938}

RECENT RESULT
hypothesis: Replacing the parameter-dominant flattened MLP with residual 3×3 spatial reasoning and per-class attention pooling will exceed 9,166 correct predictions by learning where each class’s evidence occurs instead of encoding all locations through one unstructured dense representation.
change: Reallocate the dense classifier’s parameters to a zero-initialized residual spatial-context block and separate class-attention/evidence maps, retaining the successful adaptive detail filters, channel gate, training procedure, and a 248,581-parameter total.
mechanism: Residual spatial context with class-conditioned attention pooling
evidence_used: Adaptive detail filters improved correctness only from 9,162 to 9,166, while filter constraints, larger filters, loss changes, dropout reduction, and probability-space TTA did not improve it. This challenges the shared assumption that further refinement around the existing flattened classifier is sufficient and tests a genuinely different class-prediction mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding the four missing diagonal ±1-pixel views with separable center/cardinal/diagonal weights will exceed 9,166 correct predictions by averaging coupled horizontal–vertical translation errors that the current axis-only ensemble cannot cover.
change: Expand validation TTA from five to nine spatial views while retaining horizontal flips, logit-space aggregation, calibration scale, training behavior, and parameter count.
mechanism: Separable 3×3 translation-grid test-time ensemble
evidence_used: Probability-space aggregation reduced correctness from 9,166 to 9,163, supporting retention of weighted logit averaging; meanwhile, training independently augments both image axes but validation TTA covers only cardinal translations, motivating diagonal coverage instead of another aggregation change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 71.58531450014561, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.23966367874145508, "validation_score": 9155.403335201776}

RECENT RESULT
hypothesis: Increasing the unshifted view’s weight from 3 to 4 will exceed 9,166 correct predictions by retaining useful translation averaging while reducing dilution from less validation-aligned shifted views.
change: Preserve the existing ten-view logit ensemble but modestly increase the relative weight of the centered image and its horizontal flip.
mechanism: Center-anchored logit-space test-time aggregation
evidence_used: Adding diagonal shifted views reduced correctness from 9,166 to 9,155, while probability-space aggregation reached only 9,163; this supports keeping the established views and logit averaging while anchoring them more strongly to the centered input.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 82.71088891709223, "validation_accuracy": 0.9164, "validation_correct": 9164, "validation_cross_entropy": 0.23895806884765625, "validation_score": 9164.403564908751}

RECENT RESULT
hypothesis: Adding inexpensive learned smoothing and dilation paths over the final 3×3 feature maps will exceed 9,166 correct predictions by introducing nonlinear spatial context while preserving the verified model exactly at initialization.
change: Add 192 learned channel-wise mixing coefficients for local-average and local-maximum feature maps, increasing the parameter count from 249,789 to 249,981 with negligible computation.
mechanism: Zero-initialized per-channel spatial-statistic mixing
evidence_used: Filter, loss, dropout, and TTA refinements did not surpass 9,166, while the more extensive residual spatial-context design exceeded the verification time limit; this tests its architectural premise through a lightweight residual path without replacing the successful classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a class-specific pooled-context logit path will exceed 9,166 correct predictions by complementing the location-sensitive flattened classifier with translation-invariant average/maximum feature evidence.
change: Reduce the zero-initialized channel-gate bottleneck from 24 to 19 units to fund a 96-to-10 global-context classifier, initialized to preserve the current logits exactly; total parameters increase from 249,789 to 249,794.
mechanism: Zero-initialized global-context auxiliary classifier
evidence_used: The 9,166-correct adaptive-detail baseline remains unbeaten by filter, loss, and TTA refinements, while more extensive spatial-context designs timed out; this tests complementary contextual classification with negligible added computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249794, "training_seconds": 82.32098608301021, "validation_accuracy": 0.9128, "validation_correct": 9128, "validation_cross_entropy": 0.24052526321411133, "validation_score": 9128.403055072578}

RECENT RESULT
hypothesis: Replacing the 160-unit classifier layer with two 138-unit nonlinear layers will exceed 9,166 correct predictions by increasing decision depth while retaining the successful flattened representation, dropout strengths, and nearly identical parameter and compute budgets.
change: Reallocate the dense head into two GELU stages, reducing the model by 68 parameters to 249,721 without adding meaningful runtime.
mechanism: Compute-neutral factorized dense classifier
evidence_used: Filter, loss, and TTA refinements failed to surpass 9,166, while the spatial-attention replacement timed out and reduced dropout fell to 9,141; this motivates a lightweight head-capacity change that preserves the established representation and regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training the successful zero-DC 3×3 detail kernels at 1.5× the base learning rate will exceed 9,166 correct predictions by increasing their adaptation within the fixed exposure budget while preserving the verified architecture and initialization.
change: Place `detail_kernels` in a dedicated AdamW parameter group with a 1.5× learning-rate multiplier; retain all other optimization, regularization, model, and evaluation behavior.
mechanism: Selective learning-rate acceleration for adaptive detail filters
evidence_used: Learning the original unconstrained zero-DC kernels improved correctness from 9,162 to 9,166, whereas expanding them to 5×5 fell to 9,148 and constraining their energy fell to 9,153; this motivates faster optimization of the established useful parameterization rather than adding capacity or constraints.
result: training did not finish within the verification time limit

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
