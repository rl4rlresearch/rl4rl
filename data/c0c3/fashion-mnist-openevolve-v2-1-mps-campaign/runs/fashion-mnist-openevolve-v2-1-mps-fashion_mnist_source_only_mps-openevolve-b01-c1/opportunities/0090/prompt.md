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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 61.38587808399461, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21609233474731446, "validation_score": 9247.411152990373}
prior_hypothesis: Ramping flip-pair allocation from 25% early to 50% late will exceed 9,246 correct predictions by preserving individual-view feature learning early and emphasizing validation-aligned flip aggregation near convergence.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the ensemble-loss allocation from 55% early to 95% late will exceed 9,247 correct predictions by emphasizing individual-view feature learning early and the validation-aligned crop/flip ensemble near convergence.
change: Increase the cosine ensemble-weight amplitude from ±0.15 to ±0.20 while preserving its 0.75 mean and all other settings.
mechanism: Widened ensemble-supervision curriculum
evidence_used: The analogous 25%–50% pair-loss curriculum produced the current 9,247-correct best result, while direct flip-consistency supervision regressed to 9,241; this tests stronger temporal specialization using the already-successful supervised ensemble objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 65.12346683396026, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.21578074913024903, "validation_score": 9241.411258362461}

RECENT RESULT
hypothesis: A separable 2:3:2 central-crop distribution will exceed 9,247 correct predictions by capturing part of the binomial ensemble’s cross-entropy improvement without its excessive center concentration.
change: Sample central training offsets and weight central validation crops with a separable 2:3:2 kernel, preserving the proven 10% central allocation.
mechanism: Moderately center-weighted crop aggregation
evidence_used: The stronger 1:2:1 kernel improved cross-entropy from 0.21609 to 0.21532 but reduced correctness by one; an intermediate kernel directly tests whether weaker concentration preserves baseline decisions while gaining better-calibrated ones.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 54.43192683393136, "validation_accuracy": 0.9231, "validation_correct": 9231, "validation_cross_entropy": 0.21606991577148438, "validation_score": 9231.411160570224}

RECENT RESULT
hypothesis: Replacing fixed-coordinate flattening with translation-invariant channel means and covariances will exceed 9,247 correct predictions by capturing global shape and texture relationships while better matching the crop/flip ensemble.
change: Preserve the proven convolutional extractor and parameter count, but transform its 48×7×7 output into 48 channel means plus a 48×48 covariance matrix before the existing classifier.
mechanism: Global first/second-order feature-statistics decoder
evidence_used: The phase-preserving space-to-depth stem regressed to 9,191 and increasing flattened-head capacity regressed to 9,210, challenging the load-bearing assumption that more spatially indexed detail improves classification; second-order pooling instead models learned feature co-occurrences without adding capacity.
result: training did not finish within the verification time limit

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Applying the separable 1:2:1 central-crop weighting only during validation will exceed 9,247 correct predictions by retaining uniform central-crop training while capturing the stronger kernel’s demonstrated cross-entropy improvement.
change: Weight the central 3×3 validation logits with a separable 1:2:1 kernel and normalize by its total weight, leaving training sampling and all learned components unchanged.
mechanism: Evaluation-only center-weighted crop aggregation
evidence_used: Jointly center-weighting training and validation with 1:2:1 reached 9,246 correct but improved cross-entropy from 0.21609 to 0.21532; isolating the evaluation-side change tests whether the lost prediction came from biased training rather than the better-calibrated ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Deterministically cycling through every full and central crop offset will exceed 9,247 correct predictions by matching the uniform validation ensemble more precisely while retaining beneficial batch-correlated augmentation.
change: Replace random batch-wide crop draws with balanced, coprime cycles covering all 25 full offsets and all 9 central offsets.
mechanism: Low-discrepancy batch-shared crop cycling
evidence_used: Per-example translation sampling regressed to 9,219, indicating batch-shared offsets are beneficial; this preserves that structure and the proven uniform crop distribution while eliminating finite-run sampling imbalance.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 44.4235863329377, "validation_accuracy": 0.9226, "validation_correct": 9226, "validation_cross_entropy": 0.2158976577758789, "validation_score": 9226.411218819941}

RECENT RESULT
hypothesis: Averaging model parameters over the final half of training will exceed 9,247 correct predictions by reducing optimizer noise while preserving the proven architecture, augmentation, and supervision schedules.
change: Maintain a 0.99-decay EMA after training reaches 50% progress and install the averaged parameters after the final optimizer step for validation.
mechanism: Late-training parameter exponential moving average
evidence_used: Recent changes to crop sampling, crop weighting, pair-loss allocation, ensemble allocation, and flip consistency all failed to improve the 9,247-correct design, motivating stabilization of its existing solution rather than another change to its learned invariances or objectives.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 69.1146092088893, "validation_accuracy": 0.9225, "validation_correct": 9225, "validation_cross_entropy": 0.2195169174194336, "validation_score": 9225.40999841237}

RECENT RESULT
hypothesis: Averaging per-view class probabilities will exceed 9,247 correct predictions by preventing a single overconfident crop or flip from disproportionately steering the ensemble.
change: Replace validation-time logit averaging with weighted probability averaging, then convert the resulting distribution back to temperature-scaled logits.
mechanism: Arithmetic probability test-time aggregation
evidence_used: Changes to crop weights and supervision schedules repeatedly failed to improve the 9,247-correct design, while the fundamental aggregation rule remains untested; this isolates that rule without altering learned parameters or training.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 45.667048749979585, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.22146381072998048, "validation_score": 9243.40934491518}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,247 correct predictions by retaining early regularization while sharpening class boundaries as ensemble supervision strengthens late in training.
change: Apply a cosine-decayed label-smoothing value to every individual, pair, and ensemble cross-entropy term.
mechanism: Cosine target hardening
evidence_used: Added flip-consistency regularization reduced correctness from 9,247 to 9,241, while EMA reduced it to 9,225; reducing late-stage target regularization tests whether the established model instead benefits from sharper supervised convergence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a fifth spatial convolution while minimally shrinking the classifier will exceed 9,247 correct predictions by improving feature extraction within the parameter ceiling.
change: Add a 48-channel residual 3×3 refinement block before flattening and reduce the classifier width from 80 to 79, yielding approximately 248,909 learned parameters.
mechanism: Post-residual spatial refinement
evidence_used: Increasing flattened-head capacity regressed to 9,210 correct, indicating additional capacity is better tested in spatial feature learning than in the classifier; this change preserves the proven stem, augmentation, losses, and ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248909, "training_seconds": 56.7227943749167, "validation_accuracy": 0.9228, "validation_correct": 9228, "validation_cross_entropy": 0.21980519981384278, "validation_score": 9228.409901515484}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,247 correct predictions by preserving early regularization while sharpening class boundaries as ensemble supervision strengthens late in training.
change: Compute a cosine-decayed smoothing coefficient and apply it consistently to all individual, pair, and ensemble cross-entropy terms.
mechanism: Cosine target hardening
evidence_used: The prior target-hardening implementation could not be verified, leaving its hypothesis untested; meanwhile added consistency and EMA regularization regressed to 9,241 and 9,225 correct, respectively, motivating reduced late regularization in the established 9,247-correct design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Using batch size 96 with a proportionally moderated learning rate will exceed 9,247 correct predictions by providing 33% more optimizer updates and beneficial gradient noise without increasing total example exposure.
change: Reduce batch size from 128 to 96 and scale the initial and peak learning rates by approximately √(96/128).
mechanism: Smaller-batch, learning-rate-scaled optimization
evidence_used: Recent crop, ensemble, consistency, EMA, and capacity changes all regressed, motivating a shift from invariance and architecture tuning to the still-unexplored optimization granularity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 230442, "training_seconds": 66.7434008331038, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21325289039611817, "validation_score": 9240.412115234967}

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
