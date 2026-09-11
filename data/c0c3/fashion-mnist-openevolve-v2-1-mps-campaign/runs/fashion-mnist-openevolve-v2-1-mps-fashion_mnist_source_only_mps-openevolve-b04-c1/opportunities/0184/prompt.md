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

RECENT RESULT
hypothesis: Averaging BatchNorm running means and variances at the backbone’s 0.02 EMA rate will exceed 9,330 correct predictions by aligning normalization statistics with the averaged feature weights.
change: EMA floating-point BatchNorm buffers during second-half averaging while continuing to copy integer counters directly.
mechanism: EMA-aligned BatchNorm statistics
evidence_used: Classifier-specific parameter EMA produced the best 9,330-result, while architecture and translation changes underperformed or timed out; all reported EMA experiments retained final-step BatchNorm buffers, leaving weight/statistic alignment untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 77.51381649984978, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19407534255981446, "validation_score": 9327.418734046487}

RECENT RESULT
hypothesis: A cosine taper for dropout and label smoothing will exceed 9,330 correct predictions by preserving total regularization exposure while reducing late objective drift for the EMA classifier.
change: Replace the second-half linear regularization decay with a cosine decay that is stronger early and weaker near convergence.
mechanism: Front-loaded cosine regularization taper
evidence_used: Classifier-specific EMA at 0.04 improved correctness to 9,330, indicating sensitivity to the annealed terminal objective, while faster EMA rates regressed; reducing late regularization without further shortening the EMA horizon targets that lag.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 84.08936933311634, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19462583847045897, "validation_score": 9328.418541089519}

RECENT RESULT
hypothesis: A lightweight channel gate will exceed 9,330 correct predictions by adding global feature context while preserving the position-sensitive 7×7 representation and the proven model exactly at initialization.
change: Add a zero-initialized squeeze-and-excitation gate after view fusion, using global pooled context to rescale channels with negligible spatial computation.
mechanism: Identity-initialized global channel attention
evidence_used: Global pooling reduced correctness to 9,290, so spatial layout must remain; the zero-initialized spatial refinement timed out, motivating an identity-preserving global-context branch that pools only its gating signal and adds far less computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 226570, "training_seconds": 79.45981662487611, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.2012471221923828, "validation_score": 9309.416234087694}

RECENT RESULT
hypothesis: Classifying each orientation before averaging logits will exceed 9,330 correct predictions by preserving complete position-sensitive spatial representations through the nonlinear classifier instead of collapsing them during early coordinatewise fusion.
change: Remove invariant/disagreement feature fusion and apply the shared classifier independently to original and flipped feature maps, averaging their logits for exact horizontal-flip invariance.
mechanism: Late decision-level flip symmetrization
evidence_used: Global pooling fell to 9,290, demonstrating that spatial layout is important; the current early fusion partially destroys that layout before classification, whereas late symmetrization retains it while cleanly challenging the shared assumption that flip invariance must be constructed at the feature level.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 72.3720035830047, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.20301208763122558, "validation_score": 9285.415623421528}

RECENT RESULT
hypothesis: Deterministic one-pixel translations on half the training batches will exceed 9,330 correct predictions by adding spatial robustness without the runtime cost that prevented the prior per-example translation experiment from completing.
change: Replace the prediction-redundant random horizontal flip with evenly cycled up/down/left/right translations implemented by a single batchwise roll and replicated boundary repair.
mechanism: Compute-efficient batchwise translation augmentation
evidence_used: The earlier one-pixel translation design timed out rather than producing negative accuracy evidence, while horizontal flipping is already made exactly invariant by view fusion; this patch tests the same unresolved augmentation idea with substantially less indexing and padding overhead.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 84.22800141689368, "validation_accuracy": 0.9296, "validation_correct": 9296, "validation_cross_entropy": 0.20035354232788086, "validation_score": 9296.41654394507}

RECENT RESULT
hypothesis: Applying the proven 0.04 EMA rate to `view_fusion` as well as the classifier will exceed 9,330 correct predictions by reducing lag in the late-learned invariant/disagreement projection while retaining stable 0.02 averaging throughout the convolutional backbone.
change: Treat `view_fusion` as part of the prediction head for parameter averaging, changing its EMA rate from 0.02 to 0.04.
mechanism: Projection-stack EMA alignment
evidence_used: A 0.04 EMA across both classifier layers reached 9,330 correct, whereas applying 0.04 only to the output layer reached 9,327; this indicates that faster averaging is most useful in feature-projection layers, directly motivating the same treatment for the adjacent zero-initialized fusion projection.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 75.71716791717336, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.193691739654541, "validation_score": 9326.418868610202}

RECENT RESULT
hypothesis: A 0.0425 classifier EMA rate will exceed the 9,330-result by modestly reducing head lag without the instability observed at 0.05 and 0.06.
change: Increase only the classifier-parameter EMA rate from 0.04 to 0.0425 while retaining the 0.02 backbone rate.
mechanism: Upper-side classifier EMA interpolation
evidence_used: Classifier EMA peaked at 9,330 correct with 0.04, compared with 9,328 at 0.035, 9,329 at 0.05, and 9,327 at 0.06, motivating a narrow search immediately above the best observed rate.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 82.79745595809072, "validation_accuracy": 0.9329, "validation_correct": 9329, "validation_cross_entropy": 0.19354510536193847, "validation_score": 9329.418920070766}



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
