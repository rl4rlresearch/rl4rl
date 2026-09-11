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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 81.30000304197893, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.24333785018920898, "validation_score": 9214.402143311188}
prior_hypothesis: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247538, "training_seconds": 22.307899000123143, "validation_accuracy": 0.9105, "validation_correct": 9105, "validation_cross_entropy": 0.25879054794311523, "validation_score": 9105.39720666859}
prior_hypothesis: Expanding convolutional capacity to 247,538 parameters, doubling optimizer updates via batch size 128, and using one augmented pass followed by clean fine-tuning will exceed 8,928 correct validation predictions.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 43.185732250101864, "validation_accuracy": 0.9112, "validation_correct": 9112, "validation_cross_entropy": 0.2538625297546387, "validation_score": 9112.398767798011}
prior_hypothesis: Reallocating the near-250K parameter budget from the dense classifier into wider convolutional features, adding a residual connection, and doubling update frequency will exceed the reference’s 9,105 correct predictions.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Updating a 0.995-timescale EMA every eight steps during the second half will preserve most of the observed 9,241-correct EMA benefit while reducing its overhead enough to finish verification and exceed 9,214 correct predictions.
change: Adopt the proven batch-32 flip-paired ten-view design, batch each inference flip pair into one forward call, and install a fused parameter EMA sampled every eight late-training steps.
mechanism: Sparse late-training EMA with batched flip inference
evidence_used: Exact per-step EMA reached 9,241 correct but timed out at 82.89 seconds; the non-EMA design finished with 9,214 correct at 81.30 seconds, while fused per-step EMA still timed out, motivating a large reduction in EMA update frequency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Linearly cooling the proven Reference Design 3 learning rate to zero over its final 10% of updates will exceed 9,214 correct predictions by suppressing late-update noise without EMA’s verification-time overhead.
change: Adopt the batch-32 residual model, flip-paired training, matched cardinal-view curriculum, and ten-view inference, then add a compute-negligible terminal learning-rate cooldown.
mechanism: Terminal learning-rate cooldown for EMA-like stabilization
evidence_used: Reference Design 3 finished with 9,214 correct in 81.30 seconds, while parameter EMA reached 9,241 correct but timed out at 82.89 seconds; this motivates retaining its stabilization effect through the schedule rather than weight averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing batch size from 32 to 40 will provide enough training and validation throughput for the proven 0.995-decay EMA to finish verification and exceed 9,214 correct predictions.
change: Use an evenly dividing batch size of 40 and maintain a fused per-step EMA of all learned parameters, installing the averaged parameters after the final update.
mechanism: Throughput-funded exact parameter EMA
evidence_used: Per-step EMA produced 9,241 correct predictions versus the current 9,214 but missed the time limit at 82.89 seconds; batch 40 reduces optimizer steps from 3,126 to 2,500 and eliminates partial batches while preserving substantially denser optimization than batch 64.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Single-view random flips during the augmented half and flip-paired supervision during the clean half will preserve most of the 9,214-correct design’s symmetry benefit while freeing enough computation for a 0.995-decay late EMA to finish and exceed 9,214 correct predictions.
change: Adopt Reference Design 3, batch each inference flip pair, use random flips instead of duplicated views during the first half, retain paired views during the second half, and install a fused EMA accumulated over the second half.
mechanism: Compute-reallocated late parameter averaging
evidence_used: Full flip pairing improved 9,142 to 9,208 correct but added about 9 seconds, while exact EMA reached 9,241 correct but exceeded the time limit by roughly 1.6 seconds; eliminating half of the duplicated training forwards creates substantially more runtime headroom than EMA requires.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling a 0.995-timescale EMA every 32 steps during the second half will retain enough of the observed 9,241-correct EMA benefit to exceed 9,214 correct while avoiding the verification timeout.
change: Adopt the best batch-32 residual, flip-paired, matched-view design; batch mirrored inference views; and install a terminal EMA updated only every 32 steps.
mechanism: Coarse-sampled terminal parameter EMA
evidence_used: Exact EMA improved Reference Design 3 from 9,214 to 9,241 correct but took 82.89 seconds, while an eight-step sparse EMA still timed out; reducing shadow updates another fourfold directly targets that overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a 25%-weighted loss on the same probability-averaged flip ensemble used at inference will exceed 9,214 correct predictions without materially increasing runtime.
change: Adopt the proven batch-32 residual ten-view design and augment its flip-paired cross-entropy with inference-matched ensemble cross-entropy.
mechanism: Probability-averaged flip-ensemble supervision
evidence_used: Flip-paired supervision improved the residual design from 9,142 to 9,208 correct, and matched spatial-view training raised it to 9,214; directly supervising the paired inference aggregation extends that successful alignment without another model forward pass.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging ten-view logits will exceed 9,214 correct predictions by favoring classes supported consistently across the matched spatial and flip views.
change: Replace probability-space test-time averaging with logit-space averaging, removing softmax aggregation without changing training or adding computation.
mechanism: Geometric ten-view consensus
evidence_used: Matched cardinal-view training and flip-paired supervision produced the current 9,214-correct best; this isolates the remaining ensemble aggregation choice, while the attempted probability-ensemble training added complexity and timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the final parameters with a snapshot from 400 updates earlier will capture much of the 9,241-correct EMA benefit while adding negligible overhead, allowing verification to finish and exceed 9,214 correct predictions.
change: Adopt the proven batch-32 residual design with matched cardinal-view training, flip-paired supervision, and ten-view inference, then perform one terminal parameter average with a late-training snapshot.
mechanism: Two-point terminal weight averaging
evidence_used: Reference Design 3 finished with 9,214 correct, while per-step 0.995 EMA reached 9,241 correct but timed out at 82.89 seconds; a single snapshot and final interpolation test the same late-weight stabilization with only two parameter-copy operations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing classifier dropout from Reference Design 3 will exceed 9,214 correct predictions by reducing late-update noise and train/evaluation mismatch without EMA overhead.
change: Adopt the batch-32 residual model, flip-paired supervision, matched cardinal-view curriculum, and ten-view inference from Reference Design 3, while replacing its classifier dropout with deterministic activations.
mechanism: Dropout-free deterministic residual training
evidence_used: Reference Design 3 achieved 9,214 correct, while EMA reached 9,241 but exceeded the time limit; this motivates a compute-free reduction in training noise.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing residual width from 112 to 108 channels while reallocating parameters to a 76-unit classifier will give the proven 0.995 EMA enough runtime headroom to finish and exceed 9,214 correct predictions.
change: Adopt Reference Design 3’s batch-32 matched-view, flip-paired training and ten-view inference; narrow its expensive convolutions, widen its classifier, batch mirrored inference, and install a fused per-step EMA for final evaluation.
mechanism: Capacity-reallocated narrow residual network funding exact parameter EMA
evidence_used: Reference Design 3 finished with 9,214 correct, while exact 0.995 EMA reached 9,241 correct but timed out at 82.89 seconds; modestly narrowing the dominant convolutions targets that runtime failure while retaining 249,090 learned parameters.
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
