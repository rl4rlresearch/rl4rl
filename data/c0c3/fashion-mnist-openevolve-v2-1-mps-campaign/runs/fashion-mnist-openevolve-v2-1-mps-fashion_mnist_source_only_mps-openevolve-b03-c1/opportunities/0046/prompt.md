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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 57.836604417068884, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24431087265014648, "validation_score": 9133.401828844375}
prior_hypothesis: Scaling ensemble logits by 1.03 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.245031295.

## Recent verification evidence

RECENT RESULT
hypothesis: A 28.6136627197265625% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.286136627197265625.
mechanism: Final upper-edge EMA plateau bisection
evidence_used: The current 0.2861328125 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.28614044189453125, loses one correct with lower cross-entropy; their midpoint is the most informative remaining boundary refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 66.88414016598836, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24664092712402344, "validation_score": 9132.401077799646}

RECENT RESULT
hypothesis: A 28.61347198486328125% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.2466409592.
change: Increase the endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.2861347198486328125.
mechanism: Final upper-edge EMA plateau bisection
evidence_used: The current 0.2861328125 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.286136627197265625, loses one correct with lower cross-entropy; their midpoint is the most informative remaining boundary refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.810739333042875, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24664094619750976, "validation_score": 9132.40107779351}

RECENT RESULT
hypothesis: A 28.613376617431640625% EMA blend will retain 9,133 correct predictions while lowering cross-entropy below 0.2466409592.
change: Increase the endpoint-to-EMA interpolation coefficient from 0.2861328125 to 0.28613376617431640625.
mechanism: Final upper-edge EMA plateau bisection
evidence_used: The current 0.2861328125 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint most precisely probes the remaining accuracy-plateau boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 40.587477958062664, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.246640958404541, "validation_score": 9133.401077789582}

RECENT RESULT
hypothesis: A 28.6134243011474609375% EMA blend will retain 9,133 correct predictions while lowering cross-entropy below 0.2466409584.
change: Increase the endpoint-to-EMA interpolation coefficient from 0.28613376617431640625 to 0.286134243011474609375.
mechanism: Final upper-edge EMA plateau bisection
evidence_used: The current 0.28613376617431640625 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint most precisely probes the remaining accuracy-plateau boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 73.63452066713944, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24664094924926758, "validation_score": 9133.401077792529}

RECENT RESULT
hypothesis: A 28.61344814300537109375% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.24664094925.
change: Increase the endpoint-to-EMA interpolation coefficient from 0.286134243011474609375 to 0.2861344814300537109375.
mechanism: Final upper-edge EMA plateau bisection
evidence_used: The current 0.286134243011474609375 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy plateau boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 48.4615865831729, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24664094009399415, "validation_score": 9133.401077795474}

RECENT RESULT
hypothesis: A 28.613460063934326171875% EMA blend will retain 9,133 correct predictions while lowering validation cross-entropy below 0.246640940094.
change: Increase the endpoint-to-EMA interpolation coefficient from 0.2861344814300537109375 to 0.28613460063934326171875.
mechanism: Final upper-edge EMA plateau bisection
evidence_used: The current 0.2861344814300537109375 blend retains 9,133 correct, while the nearest tested upper coefficient, 0.2861347198486328125, loses one correct with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy plateau boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 75.37275275005959, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.2466409439086914, "validation_score": 9133.401077794246}

RECENT RESULT
hypothesis: Feeding coarse stage-2 and stage-3 feature maps directly into the positional classifier will exceed 9,133 correct predictions by preserving mid-level shape and texture evidence that the current deepest-only representation discards.
change: Split the convolutional hierarchy into stages, fuse normalized 12-channel projections from its intermediate scales with the final 3×3 map, and resize the dense head; the model remains lightweight at 249,443 learned parameters.
mechanism: Normalized multi-scale hypercolumn fusion
evidence_used: Translation augmentation and multi-view inference produced repeated gains, indicating that representations stable across nearby spatial scales are useful. Unlike the timed-out widened residual/global-pooling design, this tests multi-scale prediction with negligible added convolutional work while retaining the successful positional final-grid readout.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the EMA blend by one float32 ULP to 28.61345112323760986328125% will retain 9,133 correct predictions while lowering validation cross-entropy below 0.246640940094.
change: Increase the final endpoint-to-EMA interpolation coefficient from 0.2861344814300537109375 to its next float32 value, 0.2861345112323760986328125.
mechanism: One-ULP EMA interpolation refinement
evidence_used: Increasing the blend from 0.286134243011474609375 to the current coefficient lowered cross-entropy while retaining 9,133 correct, and the tested coefficient four ULPs above current also retained 9,133; a one-ULP increase is the most precise probe of the local cross-entropy optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.177333999890834, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.24664094543457032, "validation_score": 9132.401077793755}

RECENT RESULT
hypothesis: Multiplying ensemble logits by 1.02 will preserve exactly 9,133 predictions while lowering validation cross-entropy below 0.246640940094.
change: Apply a positive calibration scale after multi-view logit averaging, without changing training or class rankings.
mechanism: Evaluation-only logit sharpening
evidence_used: One-float32-ULP EMA movement lost a correct prediction, showing further EMA tuning is accuracy-sensitive; evaluation-only positive scaling preserves argmax while testing whether label smoothing and ten-view averaging leave logits underconfident.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 69.14879058394581, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24503129501342774, "validation_score": 9133.401596330954}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.04 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.245031295.
change: Increase the positive post-ensemble calibration scale from 1.02 to 1.04.
mechanism: Evaluation-only logit sharpening continuation
evidence_used: Raising the scale from 1.00 to 1.02 preserved exactly 9,133 correct predictions and reduced cross-entropy from 0.246640940 to 0.245031295, indicating the ensemble remains underconfident at 1.02.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.03 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.245031295.
change: Increase the positive post-ensemble calibration scale from 1.02 to 1.03.
mechanism: Evaluation-only logit sharpening midpoint
evidence_used: Scaling from 1.00 to 1.02 preserved 9,133 correct predictions and reduced cross-entropy from 0.246640940 to 0.245031295, indicating residual underconfidence; the 1.04 verification timed out without contradicting that trend, so 1.03 is the most informative intermediate probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 57.836604417068884, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24431087265014648, "validation_score": 9133.401828844375}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.04 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.244310873.
change: Increase the positive post-ensemble calibration scale from 1.03 to 1.04.
mechanism: Evaluation-only logit sharpening continuation
evidence_used: Scaling from 1.02 to 1.03 preserved 9,133 correct predictions and reduced cross-entropy from 0.245031295 to 0.244310873; the prior 1.04 attempt timed out and therefore provides no contradictory performance evidence.
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
