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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 66.00526645802893, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.2163509880065918, "validation_score": 9239.411065559965}
prior_hypothesis: A cosine transition from 0.60 to 0.90 will exceed 9,237 correct predictions by strengthening early per-view learning and late ensemble alignment without exceeding the empirically best curriculum endpoints or changing the successful mean weight of 0.75.

## Recent verification evidence

RECENT RESULT
hypothesis: A cosine transition from 0.60 to 0.90 will exceed 9,237 correct predictions by strengthening early per-view learning and late ensemble alignment without exceeding the empirically best curriculum endpoints or changing the successful mean weight of 0.75.
change: Replace the linear ensemble-loss curriculum with a smooth cosine curriculum having the same endpoints and mean.
mechanism: Cosine-shaped ensemble-supervision curriculum
evidence_used: The 0.60–0.90 linear curriculum improved the static 0.75 result from 9,236 to 9,237 correct, while both narrower and wider ranges regressed; this isolates temporal concentration while preserving the supported amplitude.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 66.00526645802893, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.2163509880065918, "validation_score": 9239.411065559965}

RECENT RESULT
hypothesis: A quintic smoothstep transition from 0.60 to 0.90 will exceed 9,239 correct predictions by preserving the successful endpoints and mean while spending more training time in the early per-view and late ensemble-specialized regimes that cosine scheduling improved.
change: Replace the cosine ensemble-loss curriculum with a symmetric quintic smoothstep curriculum having identical endpoints, midpoint, and mean weight.
mechanism: Endpoint-dwelling smoothstep supervision curriculum
evidence_used: The 0.60–0.90 cosine curriculum improved correctness from 9,237 for the linear curriculum to 9,239, while changing the endpoint range regressed; increasing endpoint dwell without altering that supported range isolates the apparent benefit of temporal concentration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 53.06351050012745, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21632826919555664, "validation_score": 9236.411073237927}

RECENT RESULT
hypothesis: Adding one lightweight spatial bottleneck block will exceed 9,239 correct predictions by improving feature extraction after loss-curriculum refinements have saturated, without repeating the computational cost of the timed-out backbone redesign.
change: Add a 48→24→24→48 residual refinement block at 7×7 resolution, increasing learned parameters from 230,442 to 238,122 while preserving the proven classifier, six-view objective, cosine curriculum, and optimizer.
mechanism: Compute-efficient residual bottleneck refinement
evidence_used: The cosine curriculum is best at 9,239 correct, while linear, narrower, wider, and quintic variants all regressed; the prior deeper architectural replacement timed out, motivating a narrowly scoped, compute-efficient representational improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 238122, "training_seconds": 56.06276324996725, "validation_accuracy": 0.9188, "validation_correct": 9188, "validation_cross_entropy": 0.2210491325378418, "validation_score": 9188.409483932035}

RECENT RESULT
hypothesis: Reducing the batch size from 128 to 64 while halving the learning-rate schedule will exceed 9,239 correct predictions by doubling optimizer updates and increasing gradient diversity without changing the example-normalized optimization magnitude.
change: Train with batch size 64 and scale both the initial and scheduled learning rates by 0.5; preserve the architecture, six-view objective, cosine supervision curriculum, and AdamW configuration.
mechanism: Smaller-batch trajectory refinement
evidence_used: The cosine curriculum reached 9,239 correct, while further curriculum shaping and a residual refinement block regressed to 9,236 and 9,188; changing exposure granularity tests an orthogonal optimization mechanism while retaining the strongest established design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the proven flattened classifier from 80 to 88 hidden units will exceed 9,239 correct predictions by using nearly all remaining parameter capacity without adding costly feature-map operations or disrupting the successful backbone and cosine supervision curriculum.
change: Increase the classifier hidden width from 80 to 88, raising learned parameters from 230,442 to 249,362 while leaving training compute, augmentation, loss, and optimization essentially unchanged.
mechanism: Capacity-completing classifier widening
evidence_used: The added spatial residual block regressed from 9,239 to 9,188 correct, and a deeper backbone timed out; allocating the remaining capacity to the existing successful head tests representational capacity without repeating either failure mode.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249362, "training_seconds": 49.09446212509647, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.21892979316711425, "validation_score": 9210.410195897091}

RECENT RESULT
hypothesis: Replacing absolute 7×7 flattening with multi-scale average-and-max pooled features will exceed 9,239 correct predictions by preserving coarse garment layout and localized feature presence while reducing sensitivity to exact spatial coordinates.
change: Reallocate the dense head into a 1×1/2×2/4×4 dual-statistic spatial pyramid and a 100-unit classifier, yielding 244,142 learned parameters with similar dense-head compute.
mechanism: Dual-statistic spatial-pyramid representation
evidence_used: The coordinate-specific head remains the shared load-bearing assumption; widening it regressed to 9,210, and adding spatial convolution regressed to 9,188. The timed-out semantic-token design confounded adaptive pooling with a deeper backbone, whereas this patch cleanly tests position-tolerant image representation without additional learned feature-map operations.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Appending global average and maximum feature summaries to the proven flattened representation will exceed 9,239 correct predictions by adding translation-tolerant shape and feature-presence cues without discarding the load-bearing spatial layout.
change: Concatenate 48-channel global average and max descriptors with the flattened 7×7 feature map before the existing 80-unit classifier, increasing parameters from 230,442 to 238,122 with negligible additional convolutional compute.
mechanism: Hybrid coordinate-and-global-statistic classifier
evidence_used: Widening the unchanged flattened head regressed to 9,210 correct and adding spatial convolution regressed to 9,188, so the next test preserves the best backbone and head width while introducing a distinct, inexpensive position-tolerant signal.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 238122, "training_seconds": 48.40983891696669, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21683223304748536, "validation_score": 9236.410902987627}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.04 to 0.00 while preserving its mean at 0.02 will exceed 9,239 correct predictions by emphasizing regularized feature learning early and sharper decision boundaries late.
change: Apply the existing cosine training progress to label smoothing in all individual-view and ensemble cross-entropy terms, leaving architecture, compute, optimizer, and ensemble supervision unchanged.
mechanism: Cosine-decayed label smoothing
evidence_used: The cosine ensemble-loss curriculum improved correctness from 9,237 to 9,239 while architecture and head changes regressed, motivating another compute-neutral temporal allocation of an established loss component.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 56.853461917024106, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2164833023071289, "validation_score": 9237.41102084924}

RECENT RESULT
hypothesis: Sampling crop offsets independently for every image will exceed 9,239 correct predictions by reducing batch-correlated augmentation noise and exposing each optimizer step to broader translation coverage while preserving the proven architecture, six-view objective, and cosine curriculum.
change: Replace the three batch-shared crop offsets with independently sampled per-example offsets, using an unfolded padded-image view for vectorized crop selection.
mechanism: Per-example Monte Carlo crop sampling
evidence_used: The cosine-curriculum design remains best at 9,239 correct, while subsequent loss-schedule, capacity, and feature-head changes regressed; this motivates a compute-neutral improvement to the remaining batch-correlated augmentation mechanism without disturbing the successful model or objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.99-decay parameter EMA will exceed 9,239 correct predictions by suppressing late-stage optimizer noise while preserving the proven model, objective, augmentation, and training trajectory.
change: Track an EMA of trainable parameters after every optimizer step and install the averaged weights after the final step for validation.
mechanism: Exponential moving-average endpoint weights
evidence_used: The cosine-curriculum design achieved the best result at 9,239 correct, while subsequent architecture, head, loss-schedule, and augmentation changes regressed or timed out; endpoint weight averaging tests a compute-light optimization mechanism without disturbing that successful design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An interval-8 approximation of the previously timed-out 0.99-decay EMA will exceed 9,239 correct predictions by retaining its late-trajectory noise suppression with roughly one-eighth the averaging overhead.
change: Maintain parameter EMA checkpoints every eight optimizer steps using decay adjusted for elapsed steps, then install the averaged parameters for validation.
mechanism: Checkpointed parameter EMA
evidence_used: Per-step 0.99 EMA timed out before producing accuracy evidence, while subsequent architecture, head, loss, and augmentation changes failed to improve the 9,239-correct cosine-curriculum design; reducing EMA maintenance frequency isolates the still-untested averaging mechanism without changing that design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A foreach-fused 0.99-decay EMA sampled every 32 steps will exceed 9,239 correct predictions by suppressing endpoint noise without the runtime overhead that caused per-step EMA to time out.
change: Maintain a fused parameter EMA at 32-step intervals, account for elapsed steps in its decay, and install the averaged parameters after the final optimizer step.
mechanism: Fused coarse-checkpoint parameter EMA
evidence_used: The cosine-curriculum model remains best at 9,239 correct, while architecture, head, loss, and augmentation changes failed to improve it; per-step EMA timed out, so substantially reducing update frequency and kernel dispatch isolates its still-unresolved optimization benefit.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 73.86709962482564, "validation_accuracy": 0.9223, "validation_correct": 9223, "validation_cross_entropy": 0.21963992080688477, "validation_score": 9223.409957063122}



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
