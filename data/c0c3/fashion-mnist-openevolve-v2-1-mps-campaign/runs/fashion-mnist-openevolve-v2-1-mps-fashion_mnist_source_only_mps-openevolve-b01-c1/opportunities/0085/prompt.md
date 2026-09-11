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
hypothesis: Expanding the cosine flip-pair allocation from 20% early to 55% late will exceed 9,247 correct predictions by strengthening individual-view learning early and validation-aligned flip aggregation near convergence while preserving the same mean loss allocation.
change: Increase the amplitude of the pair-loss curriculum from ±0.125 to ±0.175, leaving architecture, augmentation, optimizer, and all other objectives unchanged.
mechanism: Widened flip-pair supervision curriculum
evidence_used: The existing 25%-to-50% cosine pair curriculum improved fixed supervision from 9,246 to 9,247 correct, whereas later architecture, augmentation, averaging, and label-smoothing changes regressed; widening that successful curriculum is the most direct local test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 58.02698054094799, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21606791076660156, "validation_score": 9245.411161248128}

RECENT RESULT
hypothesis: Narrowing the cosine pair-loss allocation from 25%–50% to 30%–45% will exceed 9,247 correct predictions by retaining beneficial temporal specialization while avoiding the excessive endpoint contrast that reduced the widened 20%–55% schedule to 9,245.
change: Reduce the pair-loss curriculum amplitude from ±0.125 to ±0.075 while preserving its 0.375 mean and all other training settings.
mechanism: Narrowed flip-pair supervision curriculum
evidence_used: The 25%–50% curriculum improved fixed supervision from 9,246 to 9,247 correct, whereas widening it to 20%–55% regressed to 9,245; this motivates testing a smaller, centered amplitude.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 59.119725208031014, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.21618413543701173, "validation_score": 9243.411121955492}

RECENT RESULT
hypothesis: Concentrating the proven 10% central-crop allocation toward less-displaced views will exceed 9,247 correct predictions while preserving the locally optimal total central weight.
change: Weight the central 3×3 validation ensemble by a separable 1:2:1 kernel and sample central training offsets from the matching binomial distribution.
mechanism: Binomial center-weighted crop aggregation
evidence_used: Changing total central allocation from 10% to either 12.5% or 7.5% regressed to 9,239 and 9,238 correct; preserving 10% while refining its spatial distribution isolates whether central-view quality, rather than total allocation, can improve the ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 59.38379312492907, "validation_accuracy": 0.9246, "validation_correct": 9246, "validation_cross_entropy": 0.21532200698852538, "validation_score": 9246.411413598309}

RECENT RESULT
hypothesis: Reducing the batch size from 128 to 64 while halving the learning-rate scale will exceed 9,247 correct predictions by doubling optimizer updates and batch-shared crop draws without increasing fixed-example exposure or changing the established augmentation geometry.
change: Use batch size 64, a 1.5e-4 initial learning rate, and a 1.5e-3 scheduled peak, preserving the learning-rate trajectory per processed example.
mechanism: Smaller-batch, linearly scaled optimization
evidence_used: Per-example translation sampling regressed to 9,219, suggesting batch-correlated views are beneficial; smaller batches preserve that correlation while increasing crop-offset coverage and optimization granularity, an axis not tested by the recent architecture, regularization, or loss-schedule changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a modest late-ramped consistency penalty between each image and its flipped counterpart will exceed 9,247 correct predictions by reducing validation-ensemble disagreement that pair-label supervision does not directly penalize.
change: Preserve the proven architecture, augmentation, optimizer, and losses while adding an evaluation-aligned KL penalty that distills each flip pair toward its detached mean prediction, ramping from zero to 0.08.
mechanism: Late flip-consistency self-distillation
evidence_used: Ramping flip-pair allocation improved fixed supervision from 9,246 to 9,247 correct, while widening or narrowing that allocation regressed to 9,245 and 9,243; this tests complementary direct agreement within the same successful flip geometry.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 53.87174287484959, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.21633104782104493, "validation_score": 9241.411072298857}

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
