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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.05628245905973, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227477645874024, "validation_score": 9254.412447746756}
prior_hypothesis: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by matching the training augmentation distribution.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by matching the training augmentation distribution and reducing distortion from overrepresented shifts.
change: Replace uniform ten-view probability and geometric averaging with center/shift weights matching the exact training offset probabilities, while retaining flip averaging, ensemble blending, and calibration.
mechanism: Augmentation-prior-weighted test-time ensemble
evidence_used: The verified 9,253-correct design weights all offsets equally although training samples center 5/13 and each cardinal shift 2/13; two prior tests timed out without producing contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by matching the training augmentation distribution.
change: Replace uniform ten-view probability and geometric averaging with center/shift weights matching the exact training offset probabilities, while retaining flip averaging, ensemble blending, and calibration.
mechanism: Augmentation-prior-weighted test-time ensemble
evidence_used: The verified 9,253-correct design weights all offsets equally although training samples center 5/13 and each cardinal shift 2/13; prior tests timed out without contradictory validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.05628245905973, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21227477645874024, "validation_score": 9254.412447746756}

RECENT RESULT
hypothesis: Increasing the center-to-shift view weight ratio from 2.5:1 to 3:1 will exceed 9,254 correct predictions by further reducing translation-induced distortion while retaining useful shifted-view averaging.
change: Increase each center-view ensemble weight from 5/26 to 6/28 and renormalize all ten inference weights; retain the existing probability/geometric blend and calibration.
mechanism: Conservative center-weight extrapolation in test-time augmentation
evidence_used: Moving from uniform offset weights to the training-prior 2.5:1 center weighting improved validation_correct from 9,253 to 9,254, providing directional evidence that moderately greater center emphasis is beneficial.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the center-to-shift view-weight ratio from 2.5:1 to 3:1 will exceed 9,254 correct predictions by further reducing translation-induced distortion while preserving useful shifted-view averaging.
change: Increase both center-view weights from 5 to 6 and renormalize the ten-view ensemble from 26 to 28, retaining the probability/geometric blend and calibration.
mechanism: Conservative center-weight extrapolation in test-time augmentation
evidence_used: Moving from uniform weighting to a 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; the sole 3:1 verification timed out and provided no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the center-to-shift view-weight ratio from 2.5:1 to 2.75:1 will exceed 9,254 correct predictions while preserving more useful shifted-view averaging than the repeatedly timed-out 3:1 proposal.
change: Increase both center-view weights from 5.0 to 5.5 and renormalize the ten-view ensemble from 26.0 to 27.0, retaining the probability/geometric blend and calibration.
mechanism: Intermediate center-weight extrapolation in test-time augmentation
evidence_used: Moving from uniform weights to a 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; 2.75:1 is the most informative untested interpolation toward the twice-unresolved 3:1 design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the center-to-shift weight ratio from 2.5:1 to 2.625:1 will exceed 9,254 correct predictions while retaining more shifted-view information than the unresolved 2.75:1 and 3:1 designs.
change: Increase both center-view weights from 5.0 to 5.25 and renormalize the ten-view ensemble from 26.0 to 26.5.
mechanism: Local center-weight interpolation in test-time augmentation
evidence_used: Increasing center emphasis from uniform weighting to 2.5:1 improved validation_correct from 9,253 to 9,254; the higher-ratio attempts timed out without contradictory score evidence, motivating a smaller interpolation beyond the verified optimum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the center-to-shift view-weight ratio from 2.5:1 to 2.5625:1 will exceed 9,254 correct predictions while preserving more shifted-view information than the unresolved 2.625:1 design.
change: Increase both center-view weights from 5.0 to 5.125 and renormalize the ten-view ensemble from 26.0 to 26.25.
mechanism: Fine-grained center-weight extrapolation in test-time augmentation
evidence_used: Increasing center emphasis from uniform weighting to 2.5:1 improved validation_correct from 9,253 to 9,254; higher-ratio attempts timed out without contradictory validation evidence, motivating the smallest tested extrapolation beyond the verified design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the center-to-shift ratio from 2.5:1 to 2.53125:1 will exceed 9,254 correct predictions while retaining more useful shifted-view information than the unresolved higher-ratio designs.
change: Increase both center-view weights from 5.0 to 5.0625 and renormalize the ten-view ensemble from 26.0 to 26.125.
mechanism: Fine-grained center-weight extrapolation in test-time augmentation
evidence_used: Moving from uniform weighting to 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; all larger extrapolations timed out without contradictory validation evidence, motivating a smaller untested step in the supported direction.
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
