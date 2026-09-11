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

RECENT RESULT
hypothesis: A 28.62548828125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664096.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.2862548828125.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The 28.61328125% blend retained 9,133 correct, while 28.6376953125% lost one correct but lowered cross-entropy; their midpoint directly refines the accuracy plateau’s upper boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 28.619384765625% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28619384765625.
mechanism: Conservative upper-edge EMA plateau refinement
evidence_used: The current 28.61328125% blend retains 9,133 correct, while 28.6376953125% loses one; the intervening 28.62548828125% test timed out without accuracy evidence, so bisecting its lower half cautiously advances toward the lower-cross-entropy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 52.18973754090257, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.2466405647277832, "validation_score": 9132.401077916238}

RECENT RESULT
hypothesis: Replacing the fixed 3×3 flattening head with mean-and-maximum spatial summaries, while spending the recovered capacity on a late residual bottleneck, will exceed 9,133 correct predictions by learning stronger whole-object features that are less sensitive to crop position.
change: Preserve the successful differential inputs and early hierarchy, widen only the final 3×3 representation to 128 channels, add a low-cost residual refinement block, and classify normalized global mean/max features; the resulting model has 245,258 learned parameters.
mechanism: Late residual semantic refinement with dual-statistic invariant readout
evidence_used: Translation augmentation and multi-view inference produced repeated gains, challenging the assumption that absolute final-grid coordinates are essential. The earlier attention/peak-pooling design timed out after also widening the hierarchy and preserving a 4×4 map; this patch isolates spatially invariant prediction with a smaller 3×3 bottleneck and no attention or depthwise operations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 28.6163330078125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.286163330078125.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The current 0.2861328125 blend retains 9,133 correct, while 0.28619384765625 loses one correct but lowers cross-entropy; their midpoint is the most informative refinement of the accuracy plateau boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 48.392976708943024, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24664076232910156, "validation_score": 9132.401077852664}

RECENT RESULT
hypothesis: A 28.61480712890625% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.2861480712890625.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The current 0.2861328125 blend retains 9,133 correct, while 0.286163330078125 loses one correct with lower cross-entropy; bisecting this interval is the most informative remaining test of the accuracy plateau’s upper boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 80.82721616700292, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.2466408561706543, "validation_score": 9132.401077822475}

RECENT RESULT
hypothesis: A 28.614044189453125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28614044189453125.
mechanism: Upper-edge EMA plateau bisection
evidence_used: The current 0.2861328125 blend retains 9,133 correct, while 0.2861480712890625 loses one correct with slightly lower cross-entropy; bisecting this interval most directly refines the accuracy plateau’s upper boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 59.24028337490745, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24664090957641602, "validation_score": 9132.401077805293}



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
