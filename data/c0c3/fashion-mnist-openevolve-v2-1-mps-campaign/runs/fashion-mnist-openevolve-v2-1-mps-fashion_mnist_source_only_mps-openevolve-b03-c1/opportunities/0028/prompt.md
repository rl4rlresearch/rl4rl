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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 49.584366667084396, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24664095916748047, "validation_score": 9133.401077789336}
prior_hypothesis: A 28.61328125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466473.

## Recent verification evidence

RECENT RESULT
hypothesis: Blending 21.875% of the bias-corrected EMA into the endpoint will exceed 9,130 correct predictions by refining the narrow optimum bracketed by the 18.75% and 25% results.
change: Reduce the final EMA interpolation coefficient from 0.25 to 0.21875 while leaving training and evaluation unchanged.
mechanism: Fine-grained endpoint–EMA interpolation
evidence_used: The 25% blend achieved 9,130 correct, while 18.75% achieved 9,129 and 50% fell to 9,123, indicating a narrow optimum near 25% and motivating a midpoint refinement on its better-supported lower side.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 71.84408129099756, "validation_accuracy": 0.913, "validation_correct": 9130, "validation_cross_entropy": 0.24713736572265624, "validation_score": 9130.40091814562}

RECENT RESULT
hypothesis: Blending 28.125% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,130 correct predictions by moving slightly beyond the best observed 25% blend while remaining well below the degraded 50% blend.
change: Increase the final EMA interpolation coefficient from 0.25 to 0.28125 without changing training, architecture, or evaluation.
mechanism: Upper-side endpoint–EMA interpolation refinement
evidence_used: Accuracy increased from 9,129 at 18.75% blending to 9,130 at both 21.875% and 25%, with 25% also giving better cross-entropy; 50% fell to 9,123, so 28.125% is the nearest untested upper-side refinement around the apparent optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.61174912494607, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.2466726921081543, "validation_score": 9133.40106758026}

RECENT RESULT
hypothesis: Blending 31.25% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,133 correct predictions by continuing the improvement from 25% to 28.125% while remaining substantially below the degraded 50% blend.
change: Increase the final EMA interpolation coefficient from 0.28125 to 0.3125 without changing architecture, training, or evaluation.
mechanism: Upper-side endpoint–EMA interpolation refinement
evidence_used: Accuracy rose from 9,130 at 25% EMA blending to 9,133 at 28.125%, while 50% blending fell to 9,123; the nearest untested upper-side coefficient is therefore the most informative refinement of the apparent interior optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 72.12914808397181, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24648008041381836, "validation_score": 9132.401129555023}

RECENT RESULT
hypothesis: Blending 29.6875% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,133 correct predictions by refining the interval between the best 28.125% blend and the slightly degraded 31.25% blend.
change: Increase the final EMA interpolation coefficient from 0.28125 to 0.296875 without changing architecture, training, or evaluation.
mechanism: Local endpoint–EMA interpolation refinement
evidence_used: The 28.125% blend achieved 9,133 correct, while 31.25% achieved 9,132 and 25% achieved 9,130; the midpoint of the narrow best-supported interval is the most informative next coefficient.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 55.91102033411153, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.2465733055114746, "validation_score": 9132.401099556511}

RECENT RESULT
hypothesis: Replacing the fixed 3×3 flattening head with learned spatial attention plus peak pooling, while reallocating parameters to wider residual features, will exceed 9,133 correct predictions by making class evidence less dependent on absolute crop position.
change: Widen the convolutional hierarchy, preserve a 4×4 final feature map, add a depthwise residual refinement block, and classify from attended and maximum spatial summaries while remaining under the 250,000-parameter ceiling.
mechanism: Content-adaptive spatial evidence pooling with residual feature extraction
evidence_used: Differential input features raised accuracy from 9,091 to 9,122, showing that better learned representations remain valuable, while translation augmentation and multi-view inference produced repeated gains; this challenges the old assumption that a large fixed-position flattening head is the best use of parameters and motivates content-dependent spatial aggregation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 28.90625% of the bias-corrected EMA into the endpoint will retain 9,133 correct predictions while lowering cross-entropy below 0.2466727, exceeding the current validation score.
change: Increase the final EMA interpolation coefficient from 0.28125 to 0.2890625 without changing training, architecture, or evaluation.
mechanism: Upper-boundary EMA interpolation refinement
evidence_used: The 28.125% blend achieved 9,133 correct, while 29.6875% lost only one correct prediction but improved cross-entropy; their midpoint tests whether more smoothing can improve the tie-breaker before accuracy declines.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.710707999998704, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.2466222152709961, "validation_score": 9132.401083819841}

RECENT RESULT
hypothesis: Blending 27.34375% of the bias-corrected EMA into the endpoint will exceed 9,133 correct predictions by testing symmetrically below the best 28.125% blend after the equally distant 28.90625% blend lost one correct prediction.
change: Reduce the final EMA interpolation coefficient from 0.28125 to 0.2734375 without changing architecture, training, or evaluation.
mechanism: Lower-boundary EMA interpolation refinement
evidence_used: The 28.125% blend achieved the best result with 9,133 correct, while 28.90625% achieved 9,132; testing the symmetric lower-side coefficient is the most direct refinement of the apparent local optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 44.64380883402191, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24672472305297852, "validation_score": 9132.401050842062}

RECENT RESULT
hypothesis: A 28.515625% EMA blend will retain 9,133 correct predictions while lowering cross-entropy below 0.2466727, producing a strict validation-score improvement.
change: Set the endpoint-to-EMA interpolation coefficient to the midpoint between the best 28.125% blend and the accuracy-degrading 28.90625% blend.
mechanism: Upper-plateau EMA boundary refinement
evidence_used: The 28.125% blend achieved 9,133 correct, while 28.90625% achieved 9,132 with lower cross-entropy; bisecting this interval directly tests how much additional smoothing the best accuracy plateau permits.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 69.16361825005151, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24664725646972657, "validation_score": 9133.401075763337}

RECENT RESULT
hypothesis: A 28.7109375% EMA blend will retain 9,133 correct predictions while reducing cross-entropy below 0.2466473.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.28515625 to 0.287109375.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The 28.515625% blend retained 9,133 correct with lower cross-entropy than 28.125%, while 28.90625% lost one correct but lowered cross-entropy further; their midpoint tests the accuracy plateau’s upper edge.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 75.72709625004791, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24663469009399414, "validation_score": 9132.401079806275}

RECENT RESULT
hypothesis: A 28.61328125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466473.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.28515625 to 0.2861328125.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The 28.515625% blend retained 9,133 correct, while 28.7109375% lost one correct but further reduced cross-entropy; bisecting this interval is the most informative test of the accuracy plateau’s upper boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 49.584366667084396, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24664095916748047, "validation_score": 9133.401077789336}

RECENT RESULT
hypothesis: A 28.662109375% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664096.
change: Increase the endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28662109375.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The 28.61328125% blend retained 9,133 correct, while 28.7109375% lost one correct but achieved lower cross-entropy; their midpoint is the most informative refinement of the accuracy plateau’s upper boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 79.06733824987896, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24663781204223634, "validation_score": 9132.401078801855}

RECENT RESULT
hypothesis: A 28.6376953125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664096.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.286376953125.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The 28.61328125% blend retained 9,133 correct, while 28.662109375% lost one correct but lowered cross-entropy; their midpoint is the most informative refinement of the accuracy plateau’s upper boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 63.53454008302651, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.2466393798828125, "validation_score": 9132.401078297436}



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
