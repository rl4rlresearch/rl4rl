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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 71.9894648338668, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18783146209716797, "validation_score": 9344.420935137648}
prior_hypothesis: A 0.9825-decay full-state EMA will exceed 9,344 correct predictions by interpolating between the near-best 0.985 horizon and the best 0.98 horizon.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.70533933397382, "validation_accuracy": 0.934, "validation_correct": 9340, "validation_cross_entropy": 0.18788887405395507, "validation_score": 9340.420914793396}
prior_hypothesis: A 0.975-decay full-state EMA will exceed 9,344 correct predictions by tracking final low-learning-rate convergence more closely than the successful 0.98 EMA.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.05553245800547, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18791449165344237, "validation_score": 9343.42090571629}
prior_hypothesis: A computationally batched 0.985-decay full-state EMA will finish within the time limit and exceed 9,334 correct predictions by tracking late low-learning-rate convergence more closely than the successful 0.99 EMA.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.50179904093966, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18781963577270508, "validation_score": 9344.420939328616}
prior_hypothesis: A 0.98-decay full-state EMA will exceed 9,343 correct predictions by tracking the final low-learning-rate solution more closely than the successful 0.985 EMA while retaining its averaging of BatchNorm statistics.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the best mean/max gate and annealing label smoothing to zero during the final 20% of optimization will exceed 9,322 correct predictions by reducing smoothing bias during low-learning-rate convergence.
change: Restore Reference Design 3’s equal mean/max gate descriptor and progressively reduce label smoothing from 0.02 to zero over the last fifth of training.
mechanism: Late hard-label sharpening on dual-statistic refinement gating
evidence_used: Reference Design 3 achieved the best result at 9,322 correct; later architectural, gate-capacity, and flip-ensemble-loss changes regressed, motivating a conservative optimization refinement of that proven model.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 54.95386716700159, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.19413365173339844, "validation_score": 9316.418713599835}

RECENT RESULT
hypothesis: Applying an eight-step approximation of 0.99-decay EMA to Reference Design 3 during the second half of training will exceed 9,322 correct predictions without repeating the per-step EMA experiment’s timeout.
change: Restore the proven equal mean/max refinement gate and calibrated temperature, then sparsely average parameters and floating-point normalization buffers and install the averaged state after the final optimizer step.
mechanism: Sparse late-training exponential weight consolidation
evidence_used: Reference Design 3 achieved the best verified correctness at 9,322, while later architectural and loss changes regressed. Per-step late EMA was promising but timed out, motivating an eightfold-lower-frequency approximation of that orthogonal optimization idea.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 65.65463312505744, "validation_accuracy": 0.9334, "validation_correct": 9334, "validation_cross_entropy": 0.1886333724975586, "validation_score": 9334.420651154149}

RECENT RESULT
hypothesis: A 0.985-decay sparse EMA on the proven mean/max-gated model will exceed 9,334 correct predictions by retaining the variance reduction of the successful 0.99 EMA while tracking the final low-learning-rate solution more closely.
change: Restore Reference Design 2’s equal mean/max descriptor and sparse late-training EMA, but shorten the EMA memory from roughly 100 to 67 optimizer steps.
mechanism: Short-horizon sparse exponential weight consolidation
evidence_used: Reference Design 2’s 0.99-decay sparse EMA achieved the best result at 9,334 correct, improving the same unaveraged architecture’s 9,322; this motivates localized tuning of the successful averaging horizon rather than another architectural change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.985-decay EMA sampled every 16 steps will exceed 9,334 correct predictions by tracking late convergence more closely while avoiding the timeout encountered by the eight-step 0.985 experiment.
change: Shorten the successful EMA horizon from roughly 100 to 67 optimizer steps and halve its snapshot frequency from every eight steps to every sixteen.
mechanism: Lower-overhead short-horizon sparse EMA
evidence_used: The current sparse 0.99 EMA improved the same unaveraged model from 9,322 to 9,334 correct; the proposed 0.985 horizon was unverified only because training timed out, motivating a lower-overhead implementation of that localized EMA test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging learned parameters while retaining final BatchNorm statistics will exceed 9,334 correct predictions by preserving the successful 0.99 EMA’s variance reduction without averaging stateful normalization buffers.
change: Add Reference Design 2’s sparse 0.99 EMA, but apply it only to model parameters using batched foreach operations and leave BatchNorm running statistics at their final trained values.
mechanism: Parameter-only sparse late-training EMA
evidence_used: Sparse late-training EMA improved the same architecture from 9,322 to 9,334 correct; isolating parameter averaging tests whether its averaged normalization buffers limit that gain while also reducing snapshot overhead implicated by subsequent EMA timeouts.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 77.51282549998723, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.18879320220947265, "validation_score": 9331.42059459885}

RECENT RESULT
hypothesis: Approximating the successful per-step 0.99 EMA by integrating a linear parameter trajectory between eight-step snapshots will exceed 9,334 correct predictions without incurring the per-step experiment’s timeout.
change: Restore the equal mean/maximum gate, then average parameters and floating-point BatchNorm buffers using exponentially weighted interpolation between consecutive sparse snapshots.
mechanism: Linearly interpolated sparse full-state EMA
evidence_used: Sparse full-state EMA improved the equal-statistic gated model from 9,322 to 9,334 correct, while parameter-only EMA reached 9,331 and per-step EMA timed out; this tests a more faithful approximation at the proven snapshot frequency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing both max-pool reductions and the flatten-to-56 bottleneck with pixel-unshuffle downsampling, deep local residual mixing, and multiscale regional logits—while retaining the successful sparse 0.99 EMA—will exceed 9,334 correct predictions.
change: Preserve every 2×2 sampling phase during downsampling, mix those phases with learned projections and seven dilated residual blocks, then predict directly from a 1×1/2×2/4×4 spatial pyramid.
mechanism: Lossless phase-preserving convolutional pyramid with regional evidence logits
evidence_used: Global pooling fell to 9,085 correct, showing spatial organization is load-bearing, while attention applied after the existing lossy pooling reached only 9,300. Reference Design 2’s sparse full-state EMA remains the best verified training protocol at 9,334 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A computationally batched 0.985-decay full-state EMA will finish within the time limit and exceed 9,334 correct predictions by tracking late low-learning-rate convergence more closely than the successful 0.99 EMA.
change: Replace dictionary-based sparse EMA with batched foreach updates over all floating parameters and BatchNorm buffers, while shortening the EMA horizon from 0.99 to 0.985.
mechanism: Foreach-accelerated short-horizon full-state EMA
evidence_used: Sparse full-state 0.99 EMA improved the same model from 9,322 to 9,334 correct, whereas the proposed 0.985 horizon produced no accuracy evidence because both prior implementations timed out; batching the state updates directly tests that localized hypothesis with less Python overhead.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.05553245800547, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18791449165344237, "validation_score": 9343.42090571629}

RECENT RESULT
hypothesis: A 0.98-decay full-state EMA will exceed 9,343 correct predictions by tracking the final low-learning-rate solution more closely than the successful 0.985 EMA while retaining its averaging of BatchNorm statistics.
change: Restore foreach-accelerated averaging of all floating-point parameters and normalization buffers, and shorten the EMA decay from 0.985 to 0.98.
mechanism: Shorter-horizon full-state exponential weight consolidation
evidence_used: Full-state 0.985 EMA achieved the best result at 9,343 correct, outperforming full-state 0.99 EMA at 9,334 and parameter-only 0.99 EMA at 9,331; this motivates one further localized shortening of the successful EMA horizon.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.50179904093966, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18781963577270508, "validation_score": 9344.420939328616}

RECENT RESULT
hypothesis: A 0.975-decay full-state EMA will exceed 9,344 correct predictions by tracking final low-learning-rate convergence more closely than the successful 0.98 EMA.
change: Restore the proven equal mean/max refinement descriptor and foreach-accelerated full-state EMA, shortening its decay from 0.98 to 0.975.
mechanism: Short-horizon full-state exponential weight consolidation
evidence_used: Shortening full-state EMA decay from 0.99 to 0.985 improved correctness from 9,334 to 9,343, and 0.98 further improved it to 9,344; this motivates one localized continuation of the successful trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.70533933397382, "validation_accuracy": 0.934, "validation_correct": 9340, "validation_cross_entropy": 0.18788887405395507, "validation_score": 9340.420914793396}

RECENT RESULT
hypothesis: A 0.9825-decay full-state EMA will exceed 9,344 correct predictions by interpolating between the near-best 0.985 horizon and the best 0.98 horizon.
change: Restore the proven pooled mean/max refinement gate and add foreach-accelerated averaging of all floating-point parameters and BatchNorm buffers from halfway through training, using decay 0.9825.
mechanism: Mid-horizon full-state exponential weight consolidation
evidence_used: Full-state EMA improved from 9,334 correct at decay 0.99 to 9,343 at 0.985 and peaked at 9,344 at 0.98 before regressing to 9,340 at 0.975, motivating a focused midpoint test in the bracket containing the observed optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 71.9894648338668, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18783146209716797, "validation_score": 9344.420935137648}

RECENT RESULT
hypothesis: A 0.98125-decay full-state EMA will exceed the current validation_score by selecting an averaging horizon between the two 9,344-correct endpoints, potentially improving correctness or tie-breaking cross-entropy.
change: Retain the proven architecture and training procedure while changing only the EMA decay from 0.98 to 0.98125.
mechanism: Fine-bracket full-state exponential weight consolidation
evidence_used: Decays 0.98 and 0.9825 both achieved the best observed 9,344 correct, while neighboring 0.975 and 0.985 achieved 9,340 and 9,343; testing their midpoint is the most focused refinement of the observed optimum plateau.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.96864345786162, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18781935539245606, "validation_score": 9343.420939427979}



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
