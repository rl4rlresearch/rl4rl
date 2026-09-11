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
hypothesis: An ensemble-loss weight of 0.70 will exceed 9,236 correct predictions by preserving stronger individual-view regularization than 0.75 while retaining most of the gain from ensemble-aligned supervision.
change: Reweight the six-view objective from 25% individual and 75% ensemble loss to 30% individual and 70% ensemble loss.
mechanism: Bracketed ensemble-loss weight refinement
evidence_used: Increasing the ensemble weight from 0.50 to 0.75 raised correctness from 9,218 to 9,236, but further increases to 0.80 and 0.875 reduced it to 9,230 and 9,232; testing 0.70 probes the untested side of the apparent optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 59.82430004095659, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.2174182384490967, "validation_score": 9232.410705199092}

RECENT RESULT
hypothesis: Replacing the coordinate-specific flattened head with a deeper residual backbone and content-adaptive global-plus-part tokens will exceed 9,236 correct predictions by learning discriminative regions while making predictions intrinsically less dependent on absolute position.
change: Reallocate the flattened head’s 189,210 parameters into grouped and bottleneck residual feature extraction, then classify one global token and three learned attention-pooled semantic part tokens; preserve the best six-view loss and optimizer to isolate the representational change. The resulting model has 234,357 learned parameters.
mechanism: Learned semantic part-token pooling
evidence_used: The 0.75 ensemble-loss design reached 9,236 correct, while nearby weights of 0.70, 0.80, and 0.875 all regressed, indicating that loss-weight refinement has saturated. The best model still spends most of its capacity on an absolute-position flattened head, so adaptive part localization tests a distinct mechanism rather than another ensemble calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the final half of the AdamW trajectory with decay 0.99 will exceed 9,236 correct predictions by reducing parameter noise while preserving the proven architecture and six-view objective.
change: Replace AdamW with an interface-compatible optimizer that maintains an EMA after the midpoint of training and installs the averaged parameters on the final step.
mechanism: Late-training exponential weight averaging
evidence_used: The 0.75 ensemble-loss design achieved 9,236 correct, while nearby loss weights regressed and the larger architectural replacement timed out; inexpensive weight averaging tests an orthogonal generalization improvement without increasing learned parameters or training forwards.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 49.8437739580404, "validation_accuracy": 0.9216, "validation_correct": 9216, "validation_cross_entropy": 0.2209544494628906, "validation_score": 9216.409515686863}

RECENT RESULT
hypothesis: Linearly increasing ensemble supervision from 0.60 to 0.90 while preserving an average weight of 0.75 will exceed 9,236 correct predictions by prioritizing stable per-view feature learning early and deployed-ensemble alignment late.
change: Replace the static 75% ensemble-loss weight with a training-progress curriculum from 60% to 90%, leaving the architecture, views, optimizer, and average loss weighting unchanged.
mechanism: Mean-preserving ensemble-loss curriculum
evidence_used: Static 0.75 achieved the best result at 9,236 correct, while both lower 0.70 and higher 0.80/0.875 weights regressed; preserving the successful mean while changing only its temporal allocation directly tests whether those competing supervision needs are stage-dependent.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 55.51774841709994, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21660788536071776, "validation_score": 9237.410978759892}

RECENT RESULT
hypothesis: Increasing the curriculum range from 0.60–0.90 to 0.55–0.95 will exceed 9,237 correct predictions by strengthening early per-view feature learning and late deployed-ensemble alignment while preserving the successful average ensemble weight of 0.75.
change: Steepen the linear ensemble-supervision curriculum without changing its mean, architecture, views, optimizer, or evaluation.
mechanism: Stronger mean-preserving ensemble-loss curriculum
evidence_used: Static 0.75 produced 9,236 correct, while the mean-preserving 0.60–0.90 curriculum improved this to 9,237; widening that curriculum directly tests whether greater stage specialization extends the observed gain.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 59.04651687480509, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.21623482475280761, "validation_score": 9235.411104821063}

RECENT RESULT
hypothesis: A 0.65–0.85 curriculum will exceed 9,237 correct predictions by retaining beneficial stage-dependent supervision while avoiding the excessive specialization observed with the wider 0.55–0.95 range.
change: Narrow the linear ensemble-loss curriculum from 0.60–0.90 to 0.65–0.85 while preserving its mean weight of 0.75.
mechanism: Narrower mean-preserving ensemble-loss curriculum
evidence_used: Static 0.75 reached 9,236 correct, the 0.60–0.90 curriculum improved to 9,237, and widening it to 0.55–0.95 regressed to 9,235; this motivates testing an intermediate curriculum amplitude.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 49.21014712494798, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21673590965270997, "validation_score": 9236.41093551693}

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
