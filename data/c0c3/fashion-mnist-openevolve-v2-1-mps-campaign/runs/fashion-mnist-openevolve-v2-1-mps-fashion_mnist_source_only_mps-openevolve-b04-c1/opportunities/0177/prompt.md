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

RECENT RESULT
hypothesis: Moving semantic refinement to 7×7 and adding a second full-channel residual block will exceed 9,330 correct predictions by giving each retained spatial feature near-global receptive context while preserving the position-sensitive classifier.
change: Move the existing 64-channel residual block after the final downsampling, add another 7×7 residual block, and fund it by reducing the dense hidden width from 48 to 36, lowering overall computation and remaining below the parameter ceiling.
mechanism: Compute-reallocated global-context residual stage
evidence_used: Dense-head widening reached only 9,300, suggesting parameters are better spent on feature extraction, while global pooling fell to 9,290, showing the 7×7 spatial layout must remain. Unlike the timed-out additive refinement, this design moves the expensive existing block from 14×14 to 7×7, so the deeper full-channel mechanism reduces net convolutional work.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 223670, "training_seconds": 65.44536720798351, "validation_accuracy": 0.9291, "validation_correct": 9291, "validation_cross_entropy": 0.20461907272338867, "validation_score": 9291.41506897186}

RECENT RESULT
hypothesis: Replacing the redundant random flip with one-pixel translations will exceed 9,330 correct predictions by improving spatial robustness while retaining exact flip invariance and the proven architecture.
change: Apply a random one-pixel horizontal and vertical translation to half of training examples using replicate padding; remove the output-equivalent random flip.
mechanism: Mild random translation augmentation
evidence_used: Multiple capacity and spatial-refinement changes underperformed, while the current symmetric fusion already makes horizontal flipping prediction-invariant, so the existing augmentation adds no meaningful diversity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying the 0.04 EMA rate only to `classifier.1` will exceed 9,330 correct predictions by retaining faster tracking for the spatial feature projection while avoiding the harmful output-layer-specific averaging change.
change: Use a 0.04 EMA update rate for the classifier’s first linear layer and 0.02 for every other parameter, including the final linear layer.
mechanism: Hidden-projection-specific EMA horizon
evidence_used: Faster EMA across both classifier layers reached 9,330, whereas applying it only to `classifier.4` fell to 9,327; this isolates the first linear projection as the likely source of the improvement and removes the output-layer change that was detrimental in isolation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using EMA rates of 0.04 for `classifier.1` and 0.03 for `classifier.4` will exceed 9,330 correct predictions by preserving faster tracking in the spatial projection while reducing the output layer’s harmful responsiveness.
change: Retain the proven 0.04 EMA rate for the first classifier linear layer, moderate the final linear layer to 0.03, and keep all feature parameters at 0.02.
mechanism: Split hidden/output classifier EMA horizons
evidence_used: Applying 0.04 to the whole classifier reached 9,330, while applying 0.04 only to the final layer fell to 9,327; the split rates directly test whether moderating the output layer retains the successful projection-side benefit.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 63.63371299998835, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1934666534423828, "validation_score": 9328.418947608261}

RECENT RESULT
hypothesis: Averaging predictions for the original image and one-pixel vertical shifts will exceed 9,330 correct predictions by reducing sensitivity to vertical alignment while preserving the position-sensitive 7×7 representation.
change: Keep training unchanged and add batched original/up/down test-time views, each retaining the model’s exact horizontal-flip fusion.
mechanism: Three-view vertical translation logit ensemble
evidence_used: Global pooling fell to 9,290, showing spatial layout should be retained, while training-time translation augmentation timed out; inference-only vertical translation directly tests positional robustness without increasing training work or learned parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 59.45096579100937, "validation_accuracy": 0.9323, "validation_correct": 9323, "validation_cross_entropy": 0.19534933166503907, "validation_score": 9323.418287764718}

RECENT RESULT
hypothesis: A classifier EMA rate of 0.035 will exceed 9,330 correct predictions by retaining most of the reduced head lag gained at 0.04 while adding stability suggested by the regression at 0.05 and 0.06.
change: Reduce the EMA update rate for all classifier parameters from 0.04 to 0.035 while retaining the proven 0.02 feature-parameter rate.
mechanism: Lower-side classifier EMA interpolation
evidence_used: Classifier-only EMA improved from 9,328 correct at 0.02 to 9,330 at 0.04, then declined to 9,329 at 0.05 and 9,327 at 0.06; testing 0.035 brackets the unexplored lower side of the apparent local optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 68.97597045800649, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19352028427124024, "validation_score": 9328.418928782852}

RECENT RESULT
hypothesis: Adding a lightweight residual 7×7 refinement branch will exceed 9,330 correct predictions by improving local spatial feature interactions while preserving the proven fusion and position-sensitive classifier at initialization.
change: Add an 8,352-parameter bottleneck convolutional branch after view fusion, zero-initialize its final projection, and apply it residually before classification.
mechanism: Zero-initialized bottleneck spatial refinement
evidence_used: Dense-head widening reached only 9,300 and replacing the established feature path with deeper 7×7 blocks reached 9,291; this instead preserves the 9,330 design exactly at initialization while spending unused parameter capacity on inexpensive spatial refinement.
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
