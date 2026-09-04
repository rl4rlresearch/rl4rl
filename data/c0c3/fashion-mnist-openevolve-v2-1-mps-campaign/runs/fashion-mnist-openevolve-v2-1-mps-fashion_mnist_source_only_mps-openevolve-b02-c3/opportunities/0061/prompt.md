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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 65.65463312505744, "validation_accuracy": 0.9334, "validation_correct": 9334, "validation_cross_entropy": 0.1886333724975586, "validation_score": 9334.420651154149}
prior_hypothesis: Applying an eight-step approximation of 0.99-decay EMA to Reference Design 3 during the second half of training will exceed 9,322 correct predictions without repeating the per-step EMA experiment’s timeout.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 49.296941000036895, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1918064540863037, "validation_score": 9319.419531206839}
prior_hypothesis: Image-adaptive gating of the final residual refinement, while preserving the verified backbone, spatial head, training protocol, and calibration, will exceed 9,315 correct predictions.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 75.43122933292761, "validation_accuracy": 0.9303, "validation_correct": 9303, "validation_cross_entropy": 0.19273306312561034, "validation_score": 9303.419205281934}
prior_hypothesis: Processing mean and maximum responses separately through the shared nonlinear gate will exceed Reference Design 3’s 9,322 correct predictions while retaining its regularized parameter count.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 77.51282549998723, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.18879320220947265, "validation_score": 9331.42059459885}
prior_hypothesis: Averaging learned parameters while retaining final BatchNorm statistics will exceed 9,334 correct predictions by preserving the successful 0.99 EMA’s variance reduction without averaging stateful normalization buffers.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the proven dual-statistic gate from seven to eight hidden channels while removing the dispensable ten-class output bias will exceed 9,322 correct predictions within the 250,000-parameter ceiling.
change: Use an eight-channel refinement-gate bottleneck and disable the final classifier bias, yielding exactly 250,000 learned parameters.
mechanism: Balanced-class bias reallocation to refinement-gate capacity
evidence_used: Image-conditioned gating improved the ungated backbone from 9,315 to 9,319 correct, and the equal mean/max descriptor further reached 9,322; increasing capacity without altering that proven descriptor directly tests whether the gate remains bottlenecked.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 250000, "training_seconds": 70.47354529099539, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.1899112991333008, "validation_score": 9310.420199388278}

RECENT RESULT
hypothesis: Replacing fixed depthwise spatial refinement and global channel gating with content-adaptive token interactions will exceed 9,322 correct predictions while preserving the load-bearing 7×7 spatial layout.
change: Replace `SpatialRefinement` with four-head self-attention using learned relative-position biases initialized at multiple locality scales, and reduce the dense bottleneck from 56 to 53 units to remain at 249,429 learned parameters.
mechanism: Multi-scale relative-position self-attention refinement
evidence_used: Global pooling fell to 9,085 correct, proving spatial layout is essential, while repeated global-gate variants peaked at 9,322 and usually regressed. This retains every spatial location but challenges the shared assumption that interactions between them should be fixed convolutions modulated only by a global descriptor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249429, "training_seconds": 49.79733208287507, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.18714674224853517, "validation_score": 9300.421177923678}

RECENT RESULT
hypothesis: Applying a 0.99-decay EMA during the second half of training to Reference Design 3’s proven dual-statistic gated model will exceed 9,322 correct predictions by reducing late cosine-schedule parameter noise.
change: Restore the 249,881-parameter mean/max refinement gate and calibrated temperature, then evaluate an EMA of parameters and floating-point normalization buffers accumulated over the second half of training.
mechanism: Late-training exponential weight consolidation
evidence_used: Reference Design 3 achieved the best verified correctness at 9,322, while several further architectural and gating changes regressed; this retains that design exactly and tests an orthogonal optimization-level improvement without adding learned parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring Reference Design 3 and reducing the batch size to 64 will exceed 9,322 correct predictions because the same 100,000-example exposure will provide roughly twice as many parameter updates while retaining adequate 128-view effective batches from paired training.
change: Restore the proven mean/max descriptor before its shared nonlinear gate and reduce the training batch size from 128 to 64 without changing the qualified learning-rate schedule or inference calibration.
mechanism: Higher-frequency stochastic optimization of the qualified dual-statistic gated model
evidence_used: Reference Design 3 achieved the best verified result at 9,322 correct, while subsequent gate-capacity, attention, and descriptor variants regressed; batch size remains an orthogonal, untested way to improve optimization under the fixed exposure budget.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending individual-view cross-entropy with cross-entropy on the averaged probabilities of each flip pair will exceed 9,322 correct predictions by aligning training with the probability-level flip ensemble used during validation.
change: Preserve the best 249,881-parameter architecture and training exposure, but optimize an equal mixture of the existing per-view loss and a label-smoothed loss on each original/flip probability ensemble.
mechanism: Flip-ensemble-aligned supervised training
evidence_used: The current mean/max-gated design is best at 9,322 correct, while subsequent architectural and gating variants regressed; its validation path averages flip probabilities, but its training loss supervises those views independently, motivating an orthogonal objective-alignment change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.00137554178946, "validation_accuracy": 0.9308, "validation_correct": 9308, "validation_cross_entropy": 0.19132550621032715, "validation_score": 9308.419700575027}

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
