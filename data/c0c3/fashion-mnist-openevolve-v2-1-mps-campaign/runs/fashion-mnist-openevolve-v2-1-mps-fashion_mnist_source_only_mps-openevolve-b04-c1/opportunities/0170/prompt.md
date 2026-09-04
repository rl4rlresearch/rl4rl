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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.25880720792338, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19353691482543944, "validation_score": 9330.418922945566}
prior_hypothesis: Using a 0.04 EMA update rate for the classifier while retaining 0.02 for feature parameters will exceed 9,328 correct predictions by tracking the rapidly changing dropout- and smoothing-free terminal classifier without exposing the backbone to the weaker global 0.04 averaging regime.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the terminal learning-rate multiplier from 0.10 to 0.15 will exceed 9,328 correct predictions by preserving useful late learning during the fixed two-exposure budget.
change: Raise the cosine schedule’s terminal learning rate from 2.1e-4 to 3.15e-4 while preserving its peak, decay timing, EMA, and all other training dynamics.
mechanism: Moderately elevated terminal cosine floor
evidence_used: Lowering the terminal multiplier to 0.05 reduced correctness sharply from 9,328 to 9,302, indicating that stronger late annealing caused underfitting rather than beneficial stabilization and motivating a conservative test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 74.45238470798358, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.19542431182861328, "validation_score": 9309.418261528608}

RECENT RESULT
hypothesis: Adding BatchNorm and GELU after view fusion will exceed 9,328 correct predictions by enabling nonlinear per-location interactions between flip-invariant and flip-disagreement features without discarding spatial layout or materially increasing runtime.
change: Normalize and activate the 7×7 fused feature map before the existing position-sensitive classifier, adding only 128 learned parameters.
mechanism: Normalized nonlinear view fusion
evidence_used: Global pooling reduced correctness to 9,290 and dense-head widening reached only 9,300, indicating that preserved spatial structure and richer feature processing are more promising than removing layout or adding head capacity; the larger spatial refinement timed out, motivating this lightweight alternative.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224570, "training_seconds": 60.336675042053685, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19323929786682129, "validation_score": 9320.41902743305}

RECENT RESULT
hypothesis: Classifying each orientation independently before averaging logits will exceed 9,328 correct predictions by preserving coherent position-sensitive spatial configurations through the nonlinear classifier while retaining exact horizontal-flip invariance.
change: Remove feature-level invariant/disagreement fusion and apply the shared classifier separately to original and flipped feature maps, averaging their logits.
mechanism: Decision-level flip ensemble with nonlinear per-view classification
evidence_used: Global pooling fell to 9,290, showing spatial layout is load-bearing, while dense-head widening reached only 9,300, arguing against capacity as the main limitation. The current fusion mixes mirrored spatial locations before nonlinear classification; decision-level fusion tests whether preserving each view’s layout resolves that representational bottleneck.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 66.72217704099603, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.1997432300567627, "validation_score": 9278.416755841978}

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics at the proven 0.02 parameter-EMA rate will exceed 9,328 correct predictions by reducing the mismatch between smoothed parameters and terminal-model normalization statistics.
change: Apply EMA to floating-point buffers such as BatchNorm running means and variances while continuing to copy integer counters directly.
mechanism: EMA-aligned BatchNorm statistics
evidence_used: The 0.02 parameter EMA scored 9,328, outperforming rates of 0.015, 0.024, 0.03, and 0.04, but its BatchNorm buffers currently come from the terminal model rather than the same successful averaging horizon.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 68.03002141718753, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19391162071228027, "validation_score": 9327.418791467748}

RECENT RESULT
hypothesis: Content-adaptive channel recalibration before final pooling will exceed 9,328 correct predictions by enriching feature extraction while preserving the position-sensitive spatial representation.
change: Add a lightweight 64→8→64 global-context channel gate, initialized as an exact identity multiplier, after the second residual block.
mechanism: Identity-initialized squeeze-and-excitation channel gating
evidence_used: Dense-head widening reached only 9,300 and nonlinear post-fusion processing reached 9,320, suggesting additional classifier capacity is not the main limitation; unlike global pooling for classification, which fell to 9,290, this gate uses global context only to recalibrate channels while retaining the full 7×7 layout.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 225538, "training_seconds": 66.84141329210252, "validation_accuracy": 0.9272, "validation_correct": 9272, "validation_cross_entropy": 0.20214811477661132, "validation_score": 9272.415922126278}

RECENT RESULT
hypothesis: A low-cost residual 7×7 refinement block will exceed 9,328 correct predictions by expanding spatial context while preserving the position-sensitive representation.
change: Add a zero-initialized depthwise-separable residual block after the final pooling stage, increasing learned parameters by only 4,928.
mechanism: Identity-initialized depthwise spatial refinement
evidence_used: Global pooling fell to 9,290, showing spatial layout is essential, while a larger spatial refinement timed out; refining the retained 7×7 map with a lightweight block tests spatial feature extraction without the runtime and parameter costs of a full convolution.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a local second-order agreement channel while preserving the baseline fusion path at initialization will exceed 9,328 correct predictions by exposing symmetric co-activations that the existing linear 128→64 projection may discard.
change: Expand view fusion to accept the invariant mean, absolute disagreement, and elementwise product of the two orientation features; initialize the new branch to zero so initial behavior remains unchanged.
mechanism: Zero-initialized multiplicative view-agreement fusion
evidence_used: Generic post-fusion nonlinearity scored 9,320 and decision-level fusion scored 9,278, so this patch retains the successful position-sensitive classifier and existing fusion path while adding only a targeted, inexpensive interaction unavailable to the linear projection before channel compression.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 228538, "training_seconds": 82.49729895894416, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.19944511642456056, "validation_score": 9282.416859423707}

RECENT RESULT
hypothesis: Reducing batch size from 64 to 56 will exceed 9,328 correct predictions by providing roughly 14% more optimizer updates and mildly noisier gradients within the fixed 100,000-example exposure.
change: Use 56-example minibatches while preserving the proven architecture, learning-rate schedule, loss annealing, and parameter EMA.
mechanism: Higher-update stochastic optimization
evidence_used: Lowering the terminal learning-rate multiplier reduced correctness to 9,302, while raising it reduced correctness to 9,309, suggesting the current per-step schedule is well tuned; changing batch size tests additional optimization opportunities without perturbing that schedule or revisiting the tightly bracketed EMA settings.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 56, "examples_processed": 100000, "optimizer_steps": 1786, "parameters": 224442, "training_seconds": 73.348782582907, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.1994621150970459, "validation_score": 9281.416853516012}

RECENT RESULT
hypothesis: Using a 0.04 EMA update rate for the classifier while retaining 0.02 for feature parameters will exceed 9,328 correct predictions by tracking the rapidly changing dropout- and smoothing-free terminal classifier without exposing the backbone to the weaker global 0.04 averaging regime.
change: Shorten only the classifier’s EMA horizon from roughly 50 to 25 optimizer steps during second-half averaging.
mechanism: Layerwise terminal EMA horizons
evidence_used: Global EMA rates of 0.024, 0.03, and 0.04 scored 9,327, 9,325, and 9,326 versus 9,328 at 0.02; because dropout and label smoothing anneal specifically during this averaging window, the close 0.04 result motivates isolating its faster tracking to the classifier while preserving the proven backbone horizon.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.25880720792338, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19353691482543944, "validation_score": 9330.418922945566}

RECENT RESULT
hypothesis: Increasing only the classifier EMA update rate from 0.04 to 0.06 will exceed 9,330 correct predictions by further reducing head-parameter lag while retaining the proven 0.02 averaging horizon for feature extraction.
change: Shorten the classifier’s second-half EMA horizon from roughly 25 to 17 optimizer steps without changing training dynamics, backbone averaging, or BatchNorm-buffer handling.
mechanism: Faster terminal classifier EMA
evidence_used: A classifier-specific 0.04 rate with the backbone held at 0.02 improved correctness from 9,328 to 9,330, whereas changing the EMA rate globally was weaker; this motivates continuing in the successful classifier-only direction with a moderate increase.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 70.75498245796189, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19361486625671387, "validation_score": 9327.418895586956}

RECENT RESULT
hypothesis: A classifier EMA rate of 0.05 will exceed 9,330 correct predictions by balancing the reduced head-parameter lag achieved at 0.04 against the excessive responsiveness observed at 0.06.
change: Increase only the classifier’s second-half EMA update rate from 0.04 to 0.05, leaving feature-parameter EMA and all training dynamics unchanged.
mechanism: Midpoint classifier-only EMA horizon
evidence_used: Classifier-only EMA at 0.04 improved correctness to 9,330, while 0.06 fell to 9,327; testing their midpoint directly brackets the apparent local optimum without disturbing the proven 0.02 feature-parameter horizon.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 63.99281424982473, "validation_accuracy": 0.9329, "validation_correct": 9329, "validation_cross_entropy": 0.19357196693420411, "validation_score": 9329.418910642888}

RECENT RESULT
hypothesis: Applying the 0.04 EMA rate only to the final linear layer will exceed 9,330 correct predictions by tracking label-smoothing and dropout annealing at the logits while giving the classifier’s spatial feature projection the more stable 0.02 backbone horizon.
change: Restrict the faster classifier EMA from the entire classifier to `classifier.4`, leaving all other learned parameters at 0.02.
mechanism: Output-layer-specific EMA horizon
evidence_used: Raising the whole classifier’s EMA rate from 0.02 to 0.04 improved correctness from 9,328 to 9,330, while increasing it to 0.05 and 0.06 reduced correctness; the final linear layer is the component most directly exposed to the annealed dropout and label-smoothed objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 75.1602125831414, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1936003299713135, "validation_score": 9327.418900688484}

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
