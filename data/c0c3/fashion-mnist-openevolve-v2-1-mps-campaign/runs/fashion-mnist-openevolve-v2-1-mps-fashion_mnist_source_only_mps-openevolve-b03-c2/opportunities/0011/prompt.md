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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 43.185732250101864, "validation_accuracy": 0.9112, "validation_correct": 9112, "validation_cross_entropy": 0.2538625297546387, "validation_score": 9112.398767798011}
prior_hypothesis: Reallocating the near-250K parameter budget from the dense classifier into wider convolutional features, adding a residual connection, and doubling update frequency will exceed the reference’s 9,105 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247538, "training_seconds": 22.307899000123143, "validation_accuracy": 0.9105, "validation_correct": 9105, "validation_cross_entropy": 0.25879054794311523, "validation_score": 9105.39720666859}
prior_hypothesis: Expanding convolutional capacity to 247,538 parameters, doubling optimizer updates via batch size 128, and using one augmented pass followed by clean fine-tuning will exceed 8,928 correct validation predictions.

REFERENCE DESIGN 3
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 81.30000304197893, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.24333785018920898, "validation_score": 9214.402143311188}
prior_hypothesis: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding convolutional capacity to 247,538 parameters, doubling optimizer updates via batch size 128, and using one augmented pass followed by clean fine-tuning will exceed 8,928 correct validation predictions.
change: Replace the shallow CNN with a four-convolution network, add translation/flip augmentation during the first half of training, reduce label smoothing, and use faster-adapting AdamW with nonzero-floor cosine decay.
mechanism: Near-capacity batch-normalized CNN with staged augmentation and denser optimization
evidence_used: The current 105,866-parameter model reaches 8,928 correct predictions in only 392 optimizer steps, leaving substantial room beneath the 250,000-parameter ceiling and for more frequent updates at the fixed exposure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247538, "training_seconds": 22.307899000123143, "validation_accuracy": 0.9105, "validation_correct": 9105, "validation_cross_entropy": 0.25879054794311523, "validation_score": 9105.39720666859}

RECENT RESULT
hypothesis: Reallocating the near-250K parameter budget from the dense classifier into wider convolutional features, adding a residual connection, and doubling update frequency will exceed the reference’s 9,105 correct predictions.
change: Use a 249,550-parameter 28/56/112-channel residual CNN, batch size 64, all 25 translation offsets during the augmented half, and the reference AdamW/loss/cosine recipe at a slightly reduced learning rate.
mechanism: Convolution-heavy residual CNN with denser optimization
evidence_used: The reference improved from 8,928 to 9,105 correct by increasing capacity and updates; its 247,538 parameters are split heavily into a 110K-parameter dense layer, motivating a more spatially efficient allocation and another update-density increase.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 43.185732250101864, "validation_accuracy": 0.9112, "validation_correct": 9112, "validation_cross_entropy": 0.2538625297546387, "validation_score": 9112.398767798011}

RECENT RESULT
hypothesis: Doubling update frequency and averaging predictions over trained flip and ±1-pixel translation symmetries will exceed the best prior result of 9,112 correct validation predictions.
change: Adopt the 249,550-parameter residual design, reduce batch size to 32 with a proportionally lower learning rate, and add probability-averaged ten-view inference while retaining staged augmentation.
mechanism: Dense-update residual CNN with transformation-marginalized inference
evidence_used: Reference Design 2 reached 9,112 correct using convolution-heavy residual features, batch size 64, and flip/translation augmentation; denser optimization and inference-time marginalization directly extend those successful mechanisms without increasing learned parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 70.82606091699563, "validation_accuracy": 0.9142, "validation_correct": 9142, "validation_cross_entropy": 0.2937472183227539, "validation_score": 9142.386474260906}

RECENT RESULT
hypothesis: Supervising both horizontal orientations of every exposed image—especially during the clean second half—will better align training with ten-view inference and exceed 9,142 correct validation predictions.
change: Concatenate each training batch with its horizontal mirror and compute the existing smoothed cross-entropy over both views in one forward pass.
mechanism: Antithetic flip-paired supervision
evidence_used: The current design reached 9,142 correct with flip-averaged inference, but flip augmentation stops halfway through training; maintaining explicit flip supervision throughout directly reinforces the successful inference symmetry.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 80.0304807908833, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.24057395553588867, "validation_score": 9208.403039252733}

RECENT RESULT
hypothesis: Pairing opposite one-pixel translations with both horizontal orientations during every update will align training with ten-view inference and exceed the prior best of 9,208 correct predictions.
change: Adopt the best batch-32 ten-view residual design, then train each example with its original view plus an alternating pair of opposite horizontal or vertical shifts, including both flips of all three views.
mechanism: Antithetic cardinal-shift and flip-orbit supervision
evidence_used: Explicit flip-paired supervision raised the same architecture from 9,142 to 9,208 correct; extending that successful symmetry-matching mechanism to the four translations already used at inference is the most direct next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on all ten inference view types during the third quarter, while retaining a final clean quarter, will exceed 9,208 correct predictions without the timeout caused by six-view-per-example training.
change: Adopt the proven batch-32 residual model, flip-paired supervision, and ten-view inference; distribute center and cardinal translations across examples in each third-quarter batch instead of duplicating every example into extra translated views.
mechanism: Compute-neutral stratified translation-orbit curriculum
evidence_used: Reference Design 3 reached 9,208 correct with flip-paired supervision in 80.0 seconds, whereas extending each example to three translations and both flips timed out; stratifying one translation per example preserves the successful two-view training cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.
change: Cycle batch-wide through center and four one-pixel cardinal translations during the augmented half, while removing redundant random flips because the loss already supervises both orientations.
mechanism: Compute-neutral inference-orbit translation curriculum
evidence_used: Flip-paired supervision improved accuracy from 9,142 to 9,208, while more expensive translation-paired variants timed out; matching the inference translations through the existing single-crop path tests the same alignment principle at lower cost.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 81.30000304197893, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.24333785018920898, "validation_score": 9214.402143311188}

RECENT RESULT
hypothesis: Cycling through the five ten-view spatial transformations during all 100,000 training exposures, while retaining flip-paired supervision, will exceed 9,214 correct predictions by eliminating the clean-half translation mismatch without materially increasing computation.
change: Adopt the proven batch-32 residual model with ten-view inference and flip-paired loss, then extend its center/cardinal one-pixel translation curriculum from the first half to the entire training run.
mechanism: Full-exposure inference-orbit supervision
evidence_used: Flip supervision throughout training improved the same model from 9,142 to 9,208 correct, and matching training crops to the five inference views raised this to 9,214; extending that successful alignment through the second half is the direct next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying a 0.995-decay parameter EMA to the proven batch-32, flip-paired ten-view residual design will exceed 9,214 correct predictions by reducing late-update noise without additional forward passes.
change: Adopt Reference Design 3 and replace its final raw parameters with an EMA accumulated across optimizer steps.
mechanism: Late-training exponential weight averaging
evidence_used: Reference Design 3 achieved the best result, 9,214 correct, while more computationally expensive augmentation variants timed out; EMA tests an orthogonal, nearly compute-free improvement suited to its noisy batch-32 optimization.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 82.89386120787822, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.23886637077331543, "validation_score": 9241.403594779707}

RECENT RESULT
hypothesis: A fused 0.995-decay EMA will retain the observed 9,241-correct benefit while batching each flip pair will recover enough verification time to finish and exceed 9,214 correct predictions.
change: Maintain an exact parameter EMA using fused foreach operations, install it after the final update, and evaluate each spatial view with its mirror in one forward pass.
mechanism: Fused parameter EMA with batched flip inference
evidence_used: The prior EMA attempt reached 9,241 correct versus the current 9,214 but timed out at 82.89 training seconds; fused updates and halving inference forward-call count target that computational failure without changing its successful learning rule.
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
