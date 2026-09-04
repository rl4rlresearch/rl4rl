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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 53.118747250176966, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21258392791748046, "validation_score": 9253.412342592119}
prior_hypothesis: Scaling ensemble scores by 1.3715955 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839283.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3698 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125948353.
change: Increase only the positive inference-time ensemble calibration multiplier from 1.3610 to 1.3698.
mechanism: Quadratic-fit post-ensemble temperature calibration
evidence_used: Verified cross-entropy improvements through 1.3610 establish that the improving direction remains upward, while the earlier finite-difference quadratic fit places the estimated minimum near 1.3698; previous attempts at that scale timed out without contradictory score evidence, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 81.76893245894462, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2125842613220215, "validation_score": 9253.412342478745}

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3715 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125842613.
change: Increase only the positive inference-time ensemble calibration multiplier from 1.3698 to 1.3715.
mechanism: Updated secant-curvature temperature calibration
evidence_used: Cross-entropy decreased from 0.2125948353 at 1.3610 to 0.2125842613 at 1.3698. Combining that secant slope with the earlier local slopes indicates a convex minimum near 1.3715; positive scaling preserves every argmax prediction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3715 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125842613.
change: Increase only the positive inference-time ensemble calibration multiplier from 1.3698 to 1.3715.
mechanism: Secant-curvature post-ensemble temperature calibration
evidence_used: Cross-entropy improved from 0.2125948353 at 1.3610 to 0.2125842613 at 1.3698, and the estimated convex minimum is near 1.3715. Positive scaling preserves argmax predictions; the prior 1.3715 timeout supplied no contradictory score evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3715 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125842613.
change: Increase only the positive inference-time ensemble calibration multiplier from 1.3698 to 1.3715.
mechanism: Secant-curvature post-ensemble temperature calibration
evidence_used: Cross-entropy improved from 0.2125948353 at 1.3610 to 0.2125842613 at 1.3698, and the observed curvature estimates the minimum near 1.3715. Positive scaling preserves argmax predictions; the two 1.3715 timeouts provided no contradictory score evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3706 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125842613.
change: Increase only the positive inference-time ensemble calibration multiplier from 1.3698 to 1.3706.
mechanism: Conservative post-ensemble temperature calibration
evidence_used: Cross-entropy improved through 1.3698, and the estimated convex minimum lies near 1.3715. Testing the midpoint toward that estimate should retain most of the predicted gain; positive scaling preserves argmax predictions, while the 1.3715 attempts timed out without contradictory score evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 78.33699099998921, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21258403244018553, "validation_score": 9253.412342556576}

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3717 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125840324.
change: Increase only the positive inference-time ensemble calibration multiplier from 1.3706 to 1.3717.
mechanism: Three-point quadratic post-ensemble temperature calibration
evidence_used: Cross-entropy fell from 0.2125948353 at 1.3610 to 0.2125842613 at 1.3698 and 0.2125840324 at 1.3706; a quadratic fit to these verified points places the minimum near 1.3717, while positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 64.72033174987882, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21258392944335938, "validation_score": 9253.4123425916}

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3716 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839294.
change: Decrease only the positive inference-time ensemble calibration multiplier from 1.3717 to 1.3716.
mechanism: Refined finite-difference temperature calibration
evidence_used: Cross-entropy improved through scales 1.3698, 1.3706, and 1.3717; the slopes across the last two intervals imply a convex minimum near 1.3716. Positive scaling preserves argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3716 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839294.
change: Decrease only the positive inference-time ensemble calibration multiplier from 1.3717 to 1.3716.
mechanism: Refined quadratic post-ensemble temperature calibration
evidence_used: The verified cross-entropies at scales 1.3698, 1.3706, and 1.3717 imply a convex minimum near 1.3716. Positive scaling preserves argmax predictions, and the previous 1.3716 attempt timed out without contradictory score evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 56.76525091705844, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2125839282989502, "validation_score": 9253.41234259199}

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3715955 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839283.
change: Refine only the positive inference-time ensemble calibration multiplier from 1.3716 to 1.3715955.
mechanism: Quadratic-vertex post-ensemble temperature calibration
evidence_used: The verified cross-entropies at 1.3706, 1.3716, and 1.3717 form a locally convex curve whose quadratic vertex is approximately 1.3715955; positive scaling cannot change argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble scores by 1.3715955 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839283.
change: Refine only the positive inference-time ensemble calibration multiplier from 1.3716 to 1.3715955.
mechanism: Quadratic-vertex post-ensemble temperature calibration
evidence_used: Verified cross-entropies at scales 1.3706, 1.3716, and 1.3717 place the local convex minimum near 1.3715955; the prior attempt at this value timed out without contradictory score evidence, and positive scaling preserves argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 53.118747250176966, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21258392791748046, "validation_score": 9253.412342592119}

RECENT RESULT
hypothesis: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by reducing distortion from overrepresented shifted views.
change: Match inference ensemble weights to the training augmentation’s exact center-versus-shift sampling probabilities while retaining flip averaging, ensemble blending, and calibration.
mechanism: Augmentation-prior-weighted test-time ensemble
evidence_used: Calibration has converged near 1.3715955 with 9,253 predictions unchanged; training samples the center offset 5/13 of the time and each cardinal shift 2/13, whereas inference currently weights all five offsets equally.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by matching the training augmentation distribution and reducing distortion from overrepresented shifted views.
change: Replace uniform averaging across the ten original/flipped inference views with probability and geometric ensembles weighted by the exact training offset probabilities, while retaining flip averaging, ensemble blending, and calibration.
mechanism: Augmentation-prior-weighted test-time ensemble
evidence_used: Training selects the center offset with probability 5/13 and each cardinal shift with probability 2/13, but verified inference weights all offsets equally; the previous test of this targeted change timed out and supplied no contradictory score evidence.
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
