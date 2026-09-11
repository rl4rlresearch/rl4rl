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

RECENT RESULT
hypothesis: Removing label smoothing throughout the second half of training will exceed 9,330 correct predictions by eliminating target drift while the classifier EMA is accumulated, while retaining dropout’s proven linear taper.
change: Keep label smoothing at 0.02 before the EMA phase, then switch to hard-label cross-entropy at 50% progress; leave all other training behavior unchanged.
mechanism: Hard-target EMA phase
evidence_used: Faster classifier EMA improved correctness to 9,330, indicating late classifier lag, while jointly changing the dropout and smoothing taper fell to 9,328; isolating smoothing directly tests whether its moving second-half objective causes that lag.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Keeping BatchNorm affine parameters at their final values while EMA-averaging convolutional and classifier parameters will exceed 9,330 correct predictions by aligning normalization scales and offsets with the final BatchNorm running statistics.
change: Exclude all BatchNorm weight and bias parameters from EMA so they are copied directly each step, matching the existing treatment of BatchNorm buffers.
mechanism: Final-state BatchNorm affine/statistic alignment
evidence_used: EMA-averaging BatchNorm running statistics reduced correctness from 9,330 to 9,327, indicating final normalization statistics are preferable; retaining final BatchNorm affine parameters tests the complementary way to eliminate the current EMA-weight/final-statistic mismatch.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 80.52674845815636, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19359461517333984, "validation_score": 9328.418902694135}

RECENT RESULT
hypothesis: Lowering the cosine schedule floor from 10% to 5% will exceed 9,330 correct predictions by reducing late parameter drift while preserving enough updates for convergence.
change: Keep the optimizer, EMA rates, and schedule shape unchanged, but reduce the final learning rate from 2.1e-4 to 1.05e-4.
mechanism: Lower terminal learning-rate floor
evidence_used: Classifier EMA peaked at 0.04 and regressed at both slower and faster nearby rates, suggesting further EMA-rate tuning is unlikely to help; stabilizing the underlying late trajectory directly targets the remaining averaging lag.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 84.86875083297491, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.19555552558898925, "validation_score": 9302.418215623866}

RECENT RESULT
hypothesis: Raising only convolutional-backbone EMA rates from 0.02 to 0.025 will exceed 9,330 correct predictions by better aligning feature kernels with final BatchNorm statistics without disturbing the proven classifier or normalization averaging.
change: Use EMA rate 0.025 for the four spatial convolution weights, retain 0.04 for classifier parameters, and retain 0.02 for view fusion and BatchNorm affine parameters.
mechanism: Faster convolution-only EMA alignment
evidence_used: EMA-averaging BatchNorm buffers fell to 9,327 and using final BatchNorm affine parameters fell to 9,328, while classifier-specific faster EMA reached 9,330; this motivates a smaller, isolated acceleration of upstream convolution weights.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 64.12241741688922, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19375289802551268, "validation_score": 9326.418847150719}

RECENT RESULT
hypothesis: Using a 0.05 EMA rate only for the final linear layer while retaining 0.04 for the hidden classifier projection will exceed 9,330 correct predictions by reducing output-layer lag without weakening the averaging of learned spatial features.
change: Split classifier EMA rates: 0.05 for `classifier.4` parameters, 0.04 for the rest of the classifier, and 0.02 elsewhere.
mechanism: Layer-specific classifier EMA tracking
evidence_used: Applying 0.05 to the entire classifier reached 9,329 versus 9,330 at 0.04, while applying 0.04 only to the output layer reached 9,327; this suggests the hidden projection benefits from the proven 0.04 rate while leaving the rapidly adapting output layer’s preferred rate unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 69.8132424580399, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19358426361083986, "validation_score": 9330.418906327139}



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
