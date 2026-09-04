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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.8552653749939, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.1980760238647461, "validation_score": 9280.417335786746}
prior_hypothesis: A batch size of 48 will finish within the verification limit and exceed 9,238 correct predictions by providing 33% more optimizer updates than batch size 64 without the roughly doubled runtime that caused batch size 32 to time out.

## Recent verification evidence

RECENT RESULT
hypothesis: A batch size of 48 will finish within the verification limit and exceed 9,238 correct predictions by providing 33% more optimizer updates than batch size 64 without the roughly doubled runtime that caused batch size 32 to time out.
change: Reduce the training batch size from 64 to 48 while preserving the proven architecture, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Intermediate-batch higher-update optimization
evidence_used: Reducing batch size from 128 to 64 doubled updates and improved validation_correct from 9,232 to 9,238 while lowering cross-entropy; batch size 32 timed out, motivating an intermediate setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.8552653749939, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.1980760238647461, "validation_score": 9280.417335786746}

RECENT RESULT
hypothesis: Reducing batch size from 48 to 40 will exceed 9,280 correct predictions by increasing optimizer steps from 2,084 to approximately 2,500 while remaining faster than the batch-32 configuration that timed out.
change: Change only the training batch size from 48 to 40, preserving the proven architecture, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Intermediate-batch higher-update optimization
evidence_used: Reducing batch size from 64 to 48 increased optimizer steps by 33% and improved validation_correct from 9,238 to 9,280; batch size 32 timed out, so 40 is the next conservative point between the best completed run and the runtime boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging learned parameters over the latter half of training will exceed 9,280 correct predictions by reducing variance from batch-48 optimization without requiring the slower batch-40 update count.
change: Maintain a low-overhead EMA every two optimizer steps during the second half of training, then install the averaged parameters after the final step.
mechanism: Tail exponential parameter averaging
evidence_used: Batch 48 improved validation_correct from 9,238 to 9,280, while batch 40 timed out; parameter averaging tests whether the benefits of smoother optimization can be captured within the proven runtime regime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Expanding the final convolutional stage from 96 to 106 channels will exceed 9,280 correct predictions by increasing learned feature capacity with minimal additional runtime.
change: Widen the final feature stage and matching classifier input, increasing parameters from 232,682 to 249,982 while preserving every proven training and evaluation setting.
mechanism: Near-ceiling final-stage channel expansion
evidence_used: The batch-48 design completed in 73.86 seconds and achieved 9,280 correct, while slower optimization and residual redesigns timed out; this controlled expansion uses nearly all remaining parameter headroom with only about 1–2% additional convolutional work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the batch-48 learning-rate schedule by 1.2× will exceed 9,280 correct predictions by approximating the cumulative optimization of batch size 40 without its additional steps or runtime.
change: Increase the initial, peak, and terminal learning rates by 20% while preserving batch size, warmup, cosine schedule, architecture, augmentation, and evaluation.
mechanism: Update-equivalent learning-rate scaling
evidence_used: Moving from batch size 64 to 48 increased optimizer steps by 33% and improved validation_correct from 9,238 to 9,280, while batch size 40 timed out; a 1.2× learning-rate scale matches the attempted 48/40 increase in update opportunity at unchanged computational cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 75.29956141696312, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.20057363929748534, "validation_score": 9274.416467581525}

RECENT RESULT
hypothesis: Reducing classifier dropout from 0.15 to 0.05 will exceed 9,280 correct predictions by improving feature utilization during the fixed two-pass exposure without the instability observed from a 1.2× learning-rate increase.
change: Lower only the classifier-head dropout probability, preserving architecture size, runtime, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Reduced short-horizon head regularization
evidence_used: More optimizer updates improved validation_correct from 9,238 to 9,280, indicating that the short training horizon benefits from more effective optimization, while increasing learning rates reduced it to 9,274; weaker stochastic masking is a low-cost alternative for accelerating head convergence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 70.8537731252145, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.20079868850708008, "validation_score": 9265.416389528724}

RECENT RESULT
hypothesis: Updating an EMA every 32 steps during the latter half of training will exceed 9,280 correct predictions while finishing within the time limit by smoothing batch-48 optimization with roughly 16× fewer averaging updates than the timed-out every-two-step implementation.
change: Maintain a 0.85-decay parameter EMA at 32-step intervals during the second half of training and install it after the final optimizer step.
mechanism: Sparse tail exponential parameter averaging
evidence_used: Batch size 48 achieved the best result at 9,280 correct, while denser tail averaging timed out; sparsifying the same averaging mechanism directly targets its observed runtime cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.42016266705468, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.1963139259338379, "validation_score": 9279.417950497074}

RECENT RESULT
hypothesis: Batch size 44 will exceed 9,280 correct predictions by providing approximately 9% more optimizer steps than batch size 48 while remaining within the runtime boundary that batch size 40 exceeded.
change: Reduce only the training batch size from 48 to 44, preserving the architecture, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Runtime-bracketed intermediate-batch optimization
evidence_used: Reducing batch size from 64 to 48 increased validation_correct from 9,238 to 9,280, while batch size 40 timed out; 44 is the nearest conservative midpoint between the best completed configuration and the observed runtime failure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing GELU with in-place ReLU will make batch-size-44 training finish within the runtime limit and exceed 9,280 correct predictions through approximately 9% more optimizer steps.
change: Use batch size 44 and replace all feature and classifier GELU activations with lower-cost in-place ReLU activations; preserve augmentation, optimizer, schedule, capacity, and validation ensemble.
mechanism: Faster activations fund higher-update optimization
evidence_used: Batch 48 reached 9,280 correct after outperforming batch 64 through additional updates, while batch 44 timed out with the architecture unchanged; reducing activation overhead directly targets that runtime failure while retaining the proposed higher-update regime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a direct global mean–max decision path to the proven spatial classifier will exceed 9,280 correct predictions by capturing translation-tolerant feature presence without the runtime cost of the timed-out residual pyramid.
change: Preserve the successful feature extractor and position-specific head, but augment every prediction with a learned, zero-initialized linear residual computed from global mean and maximum channel statistics.
mechanism: Zero-initialized invariant residual classification branch
evidence_used: Repeated optimizer and capacity changes failed or timed out, while the global mean–max design was never accuracy-tested because its seven-convolution extractor exceeded the time limit; this patch isolates that alternative prediction mechanism with only 1,920 parameters and negligible training computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 25% of the sparse tail EMA into the final weights will retain at least 9,280 correct predictions while lowering validation cross-entropy, improving validation_score.
change: Sample a parameter EMA every 32 steps during the second half of training, then interpolate it 25% into the final parameters instead of fully replacing them.
mechanism: Conservative tail-EMA weight interpolation
evidence_used: Full sparse tail EMA finished within the time limit and reduced cross-entropy from 0.19808 to 0.19631 but lost one correct prediction; a conservative interpolation should capture some smoothing benefit while staying closer to the higher-accuracy final weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.15 to 0.20 will exceed 9,280 correct predictions by improving regularization without changing runtime or parameter count.
change: Raise only the classifier-head dropout probability, preserving the proven architecture, optimizer, schedule, augmentation, batch size, and validation ensemble.
mechanism: Moderately stronger classifier-head dropout
evidence_used: Reducing dropout from 0.15 to 0.05 lowered validation_correct from 9,280 to 9,265, indicating that head regularization is beneficial; a conservative increase tests the supported direction.
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
