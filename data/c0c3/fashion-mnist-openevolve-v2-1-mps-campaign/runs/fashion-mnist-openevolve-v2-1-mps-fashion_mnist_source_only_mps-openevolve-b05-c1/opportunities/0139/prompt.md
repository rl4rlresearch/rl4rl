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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 70.40735145891085, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600157318115235, "validation_score": 9360.421584600987}
prior_hypothesis: A center weight of 1.81732177734375 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.

## Recent verification evidence

RECENT RESULT
hypothesis: A center weight of 1.8173828125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860016449.
change: Halve the step from the verified 1.818359375 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Reductions from 1.875 through 1.84375, 1.828125, 1.8203125, and 1.818359375 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 verification timed out and supplied no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 77.54128845804371, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600157852172852, "validation_score": 9360.42158459909}

RECENT RESULT
hypothesis: A center weight of 1.81689453125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.
change: Halve the remaining interval from the verified 1.8173828125 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.
mechanism: Fine-grained center-prior boundary search
evidence_used: Every verified reduction from 1.875 through 1.8173828125 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 run timed out and therefore provides no contrary accuracy evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 63.04985083290376, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18600154457092286, "validation_score": 9359.421584611157}

RECENT RESULT
hypothesis: A center weight of 1.817138671875 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.
change: Bisect the interval between the best verified 1.8173828125 weight and the 1.81689453125 weight that lost one prediction.
mechanism: Classification-boundary bisection
evidence_used: Reducing the weight to 1.81689453125 improved cross-entropy but reduced correct predictions to 9,359, establishing a nearby accuracy boundary; midpoint testing is the most informative refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A zero-initialized covariance head will exceed 9,360 correct predictions by exposing second-order channel co-activation evidence unavailable to the existing per-channel statistics and 30-unit spatial bottleneck.
change: Project gated feature maps to 15 channels, compute their 120 unique spatial covariance terms, normalize them, and add learned residual logits. This raises the model to 249,956 parameters.
mechanism: Learned projected channel-covariance residual head
evidence_used: Repeated center-weight refinements plateaued at 9,360 correct with only microscopic cross-entropy gains, indicating inference aggregation is saturated. The old assumption is that marginal channel statistics are sufficient; this patch instead models joint channel interactions with little additional computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.8172607421875 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.
change: Move one quarter of the known interval from the best verified weight toward the weight that lost one prediction, preserving all evaluation views and normalization.
mechanism: Conservative classification-boundary refinement
evidence_used: Weight 1.8173828125 achieved 9,360 correct, while 1.81689453125 improved cross-entropy but lost one prediction; the smaller step limits boundary risk after the direct midpoint verification timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.81732177734375 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.
change: Move one eighth of the known interval from the best verified center weight toward the weight that lost one prediction, preserving all views and normalization.
mechanism: Conservative classification-boundary refinement
evidence_used: Weight 1.8173828125 achieved 9,360 correct, while 1.81689453125 lowered cross-entropy but lost one prediction; prior larger-step refinements timed out, so this smaller reduction tests the established monotonic cross-entropy trend with less boundary risk.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 70.40735145891085, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600157318115235, "validation_score": 9360.421584600987}

RECENT RESULT
hypothesis: A center weight of 1.817291259765625 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015732.
change: Halve the previous successful downward step in center-view weight, preserving all evaluation views and normalization.
mechanism: Conservative classification-boundary refinement
evidence_used: Reducing the weight from 1.8173828125 to 1.81732177734375 retained 9,360 correct and lowered cross-entropy, while 1.81689453125 lost one prediction; a half-sized continuation exploits the monotonic cross-entropy trend with less boundary risk.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.8173065185546875 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015732.
change: Halve the most recent attempted downward step in center-view weight while preserving all evaluation views and normalization.
mechanism: Conservative classification-boundary refinement
evidence_used: Weight 1.81732177734375 retained 9,360 correct, while 1.81689453125 lost one prediction despite lower cross-entropy; the intervening 1.817291259765625 verification timed out without contrary accuracy evidence, motivating a smaller boundary-safe step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.81731414794921875 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015732.
change: Halve the latest attempted downward adjustment to the center-view ensemble weight, preserving all views and normalization.
mechanism: Conservative center-view boundary refinement
evidence_used: Weight 1.81732177734375 retained 9,360 correct, whereas 1.81689453125 lost one prediction despite lower cross-entropy; the intervening 1.8173065185546875 run timed out, so a smaller step tests the established cross-entropy trend with less boundary risk.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 69.32460195780732, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600157432556153, "validation_score": 9360.42158460058}

RECENT RESULT
hypothesis: A center weight of 1.8173396587371826 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.18600157318115235.
change: Replace the center-view ensemble weight with the nearest float32 value to the minimum predicted by quadratic interpolation of the three closest completed measurements.
mechanism: Three-point parabolic center-weight calibration
evidence_used: The current 1.81732177734375 weight outperformed both 1.81731414794921875 and 1.8173828125 at the same 9,360 correct; their asymmetric cross-entropy increases predict a local minimum slightly above the current weight.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center weight of 1.8173396587371826 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.18600157318115235.
change: Set the center-view ensemble weight to the float32-representable minimum estimated from the three nearest completed calibration measurements.
mechanism: Three-point parabolic center-weight calibration
evidence_used: The current 1.81732177734375 weight produced lower cross-entropy than both 1.81731414794921875 and 1.8173828125 while all retained 9,360 correct; quadratic interpolation predicts a minimum at 1.8173396587371826, and its prior verification timed out without contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the final positive logit scale from 1.104 to 1.1045 will retain all 9,360 correct predictions while reducing validation cross-entropy below 0.18600157318115235.
change: Raise only the evaluation-time logit multiplier, leaving training, TTA views, center weighting, normalization, and class rankings unchanged.
mechanism: Accuracy-invariant logit temperature refinement
evidence_used: Center-weight refinement has plateaued at 9,360 correct with microscopic cross-entropy differences; a positive global logit scale preserves argmax predictions exactly while directly probing the remaining calibration objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 77.16674595791847, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18600544357299806, "validation_score": 9360.421583225194}



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
