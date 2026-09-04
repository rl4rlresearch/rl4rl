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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 60.546387874986976, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18585695190429688, "validation_score": 9360.42163601537}
prior_hypothesis: An evaluation-time logit scale of 1.0671112 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.1858569564819336.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Decreasing the evaluation-time logit scale from 1.104 to 1.1035 will preserve all 9,360 argmax predictions while reducing validation cross-entropy below 0.18600157318115235.
change: Lower only the positive evaluation-time logit multiplier, leaving training, TTA views, center weighting, normalization, and class rankings unchanged.
mechanism: Accuracy-invariant logit temperature refinement
evidence_used: Increasing the scale to 1.1045 preserved 9,360 correct but worsened cross-entropy to 0.18600544357299806, indicating that the locally favorable calibration direction is downward.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 71.92042370908894, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18599775314331055, "validation_score": 9360.42158595889}

RECENT RESULT
hypothesis: A lightweight class-specific attention head will exceed 9,360 correct predictions by learning where each class’s discriminative features occur, while remaining substantially cheaper than the timed-out covariance head.
change: Add zero-output-initialized per-class evidence maps and learned per-class spatial attention, then aggregate localized evidence into residual logits.
mechanism: Class-conditioned spatial attention residual
evidence_used: Center-weight and temperature calibration plateaued at 9,360 correct, showing that inference calibration is saturated. The covariance experiment targeted richer interactions but timed out; this instead challenges the shared assumption that fixed flattening and class-agnostic global statistics adequately preserve spatial evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 248846, "training_seconds": 71.59655520808883, "validation_accuracy": 0.9337, "validation_correct": 9337, "validation_cross_entropy": 0.18638620071411133, "validation_score": 9337.42144792286}

RECENT RESULT
hypothesis: A positive evaluation-time logit scale of 1.0658182 will preserve all 9,360 argmax predictions while reducing validation cross-entropy below 0.18599775314331055.
change: Replace the current evaluation multiplier with the quadratic-interpolation minimum estimated from the completed 1.1035, 1.104, and 1.1045 measurements.
mechanism: Newton temperature calibration from three fixed-logit measurements
evidence_used: Lowering the scale from 1.104 to 1.1035 improved cross-entropy, while raising it to 1.1045 worsened it; because positive global scaling cannot change class rankings, these three measurements support a larger accuracy-invariant step toward their estimated convex minimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 76.95708775008097, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18585714950561524, "validation_score": 9360.421635945113}

RECENT RESULT
hypothesis: An evaluation-time logit scale of 1.0664346 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.18585714950561524.
change: Replace the positive evaluation multiplier with the quadratic minimum estimated using the completed 1.0658182, 1.1035, and 1.104 measurements.
mechanism: Convex temperature recalibration
evidence_used: Scale 1.0658182 achieved 9,360 correct and 0.18585714950561524 cross-entropy; combining this new measurement with the prior 1.1035 and 1.104 results places the updated convex interpolation minimum near 1.0664346.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 60.62063245801255, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18585700874328615, "validation_score": 9360.42163599516}

RECENT RESULT
hypothesis: An evaluation-time logit scale of 1.0671952 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.18585700874328615.
change: Replace only the positive evaluation-time logit multiplier with the quadratic minimum estimated from three completed measurements.
mechanism: Three-point convex temperature refinement
evidence_used: Scale 1.0664346 improved cross-entropy to 0.18585700874328615; the higher losses at 1.0658182 and 1.1035 place the three-point interpolated minimum near 1.0671952, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 59.41596991708502, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1858569564819336, "validation_score": 9360.421636013743}

RECENT RESULT
hypothesis: An evaluation-time logit scale of 1.0671112 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.1858569564819336.
change: Replace only the positive evaluation-time logit multiplier with the minimum interpolated from the three nearest completed scale measurements.
mechanism: Local quadratic temperature calibration
evidence_used: Scales 1.0658182, 1.0664346, and 1.0671952 produced cross-entropies 0.18585714950561524, 0.18585700874328615, and 0.1858569564819336; quadratic interpolation places the local minimum near 1.0671112, and positive scaling preserves every argmax.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An evaluation-time logit scale of 1.0671112 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.1858569564819336.
change: Replace only the positive evaluation-time logit multiplier, leaving training and class rankings unchanged.
mechanism: Local quadratic temperature calibration
evidence_used: Quadratic interpolation of the three nearest completed scale measurements predicts a minimum near 1.0671112; its previous verification timed out and supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 60.546387874986976, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18585695190429688, "validation_score": 9360.42163601537}

RECENT RESULT
hypothesis: An evaluation-time logit scale of 1.067003607749939 will preserve all 9,360 argmax predictions while reducing validation cross-entropy below 0.18585695190429688.
change: Replace only the positive evaluation-time logit multiplier with the nearest float32 value to the local quadratic minimum.
mechanism: Local three-point temperature interpolation
evidence_used: Scales 1.0664346, 1.0671112, and 1.0671952 yielded cross-entropies 0.18585700874328615, 0.18585695190429688, and 0.1858569564819336; quadratic interpolation predicts a minimum near 1.0670036, while positive scaling preserves class rankings.
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
