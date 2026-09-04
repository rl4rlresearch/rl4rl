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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.79398037493229, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.19407607765197754, "validation_score": 9324.418733788707}
prior_hypothesis: Retaining the validated 37.5% cross-offset rate while center-anchoring 71.37% of those pairs will exceed 9,325 correct predictions by matching training-view exposure to the inference ensemble’s measured 1.5578× center reliability.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 52.62951845792122, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19298710632324217, "validation_score": 9325.419116013367}
prior_hypothesis: Downweighting flip-inconsistent offsets by 12.5% while using the best verified calibration scale will exceed 9,325 correct predictions by reducing unreliable transformed evidence.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 51.44045537500642, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19298956680297852, "validation_score": 9325.419115148961}
prior_hypothesis: Setting the calibration scale to 1.24754 will preserve all 9,325 argmax predictions while reducing validation cross-entropy below 0.1929895683.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 55.377164707984775, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1929895683288574, "validation_score": 9325.419115148425}
prior_hypothesis: Setting the calibration scale to 1.24748 will preserve all 9,325 argmax predictions while reducing validation cross-entropy below 0.1929895706.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the validated uniform 37.5% paired-offset training while increasing the inference penalty for flip-inconsistent offsets from 10% to 12.5% will exceed 9,325 correct predictions by suppressing unreliable transformed evidence without altering training compute.
change: Restore Reference Design 2’s paired-view loss and agreement-curved calibration, then strengthen only the flip-inconsistency downweighting.
mechanism: Slightly stronger flip-disagreement gating
evidence_used: Reference Design 2 achieved the best verified result of 9,325 correct using uniform 37.5% cross-offset pairs, 5% consistency, and a 10% flip-disagreement penalty; this patch preserves that control and isolates a conservative inference-only refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 12.5% reliability-weighted arithmetic probability pooling into the validated logit ensemble will exceed 9,325 correct predictions by damping isolated overconfident transformed views while preserving the stronger geometric-pooling signal.
change: Keep training unchanged and replace evaluation logits with the log of an 87.5% logit-pooled and 12.5% probability-pooled mixture.
mechanism: Conservative hybrid geometric–arithmetic view pooling
evidence_used: The weighted logit ensemble produced the best verified 9,325 correct; the prior 25% probability-blend attempt timed out rather than yielding negative accuracy evidence, motivating a more conservative blend with negligible added computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.07845108397305, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1930408390045166, "validation_score": 9325.41909713704}

RECENT RESULT
hypothesis: Restoring uniform 37.5% cross-offset training and increasing the validated 12.5% probability blend only when flip views disagree will exceed 9,325 correct predictions by damping unreliable extreme logits without perturbing fully consistent examples.
change: Remove detrimental center-anchored pair sampling, restore uniform cross-offset partners, and adapt the arithmetic-probability blend from 12.5% to at most 25% according to reliability-weighted flip disagreement.
mechanism: Disagreement-adaptive geometric–arithmetic view pooling
evidence_used: Uniform 37.5% sampling with a 12.5% global probability blend achieved the best verified result of 9,325 correct and 0.1930408 cross-entropy, whereas center anchoring fell to 9,324; the 25% blend timed out rather than providing negative accuracy evidence, motivating a targeted interpolation only on uncertain examples.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the validated probability-pooling blend from 12.5% to 18.75% will exceed 9,325 correct predictions or, if correctness ties, reduce cross-entropy below 0.193040839 by further damping unreliable extreme view logits.
change: Restore the best uniform 37.5% cross-offset training with 5% consistency, then evaluate with an 81.25% logit-pooled and 18.75% arithmetic-probability ensemble.
mechanism: Intermediate geometric–arithmetic view pooling
evidence_used: Uniform 37.5% cross-offset training with a 12.5% probability blend achieved the best verified score, 9,325 correct at 0.193040839 cross-entropy; the 25% attempt timed out without negative accuracy evidence, motivating an intermediate fixed dose.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the validated 12.5% hybrid probability ensemble and increasing its inherited calibration scale by 1% will retain 9,325 correct predictions while reducing validation cross-entropy below 0.193040839.
change: Restore Reference Design 2’s uniform 37.5% cross-offset paired training and hybrid inference, then resharpen the softened hybrid probabilities with a 1.24003 calibration multiplier.
mechanism: Hybrid-pool temperature retuning
evidence_used: Reference Design 2 achieved the best verified result—9,325 correct with 0.193040839 cross-entropy—after adding arithmetic-probability pooling but retaining the earlier logit-only calibration; a small temperature adjustment isolates calibration without changing predictions or training compute.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the hybrid ensemble’s calibration scale by 1% will preserve its 9,325 predictions while reducing validation cross-entropy below 0.193040839.
change: Increase only the positive post-pooling logit multiplier from 1.22775 to 1.24003.
mechanism: Hybrid-pool temperature resharpening
evidence_used: The 12.5% probability blend produced the best verified result but softens pooled probabilities; the prior identical calibration test timed out, providing no negative accuracy evidence, and positive scaling cannot change argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 74.14410929195583, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1929967903137207, "validation_score": 9325.41911261125}

RECENT RESULT
hypothesis: On the validated uniform-pair baseline, explicitly supplying horizontal and vertical edge magnitudes will exceed 9,325 correct predictions by learning shape-sensitive features more efficiently within 782 optimizer steps.
change: Restore the best uniform 37.5% pairing and hybrid inference control, then replace the raw-intensity-only stem input with intensity plus fixed oriented gradient magnitudes; narrow the MLP to remain below 250,000 parameters.
mechanism: Oriented gradient-lifted convolutional stem
evidence_used: Uniform pairing with hybrid pooling reached 9,325 correct, while later loss and inference refinements did not improve correctness and center anchoring fell to 9,324. This challenges the shared assumption that the shallow stem can discover all useful nonlinear edge representations from raw intensity during the fixed exposure, without repeating the computationally heavier spatial-head designs that timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249928, "training_seconds": 64.09771041595377, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20191325302124025, "validation_score": 9286.416003400198}

RECENT RESULT
hypothesis: Raising the validated hybrid ensemble’s calibration scale by another 0.5% will preserve its 9,325 argmax predictions while reducing validation cross-entropy below 0.1929967903.
change: Restore uniform 37.5% cross-offset paired training with uniform 5% consistency and the validated 12.5% arithmetic-probability blend, then increase calibration from 1.24003 to 1.24623.
mechanism: Conservative hybrid-pool temperature resharpening
evidence_used: Reference Design 2 achieved 9,325 correct at 0.1929967903 cross-entropy; its prior 1% calibration increase improved cross-entropy from 0.193040839 without changing predictions, motivating a smaller continuation of that line search.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.93736645812169, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19298977279663085, "validation_score": 9325.419115076593}

RECENT RESULT
hypothesis: Restoring the validated hybrid ensemble and setting its calibration scale to 1.24739 will preserve 9,325 correct predictions while reducing cross-entropy below 0.1929897728.
change: Restore uniform 37.5% paired-offset consistency training and 12.5% probability-blended inference, then move the calibration scale to the estimated optimum from the three verified calibration points.
mechanism: Quadratic-interpolated hybrid calibration
evidence_used: Scales 1.22775, 1.24003, and 1.24623 produced progressively lower cross-entropies while retaining exactly 9,325 correct predictions; quadratic interpolation places the local minimum near 1.24739.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 55.21924633393064, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19298957061767577, "validation_score": 9325.41911514762}

RECENT RESULT
hypothesis: Setting the calibration scale to 1.24748 will preserve all 9,325 argmax predictions while reducing validation cross-entropy below 0.1929895706.
change: Increase only the positive post-pooling calibration multiplier from 1.24739 to the local quadratic optimum of 1.24748.
mechanism: Local quadratic temperature refinement
evidence_used: Scales 1.24003, 1.24623, and 1.24739 successively reduced cross-entropy without changing correctness; the two nearest measured intervals estimate the minimum near 1.24748.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 55.377164707984775, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1929895683288574, "validation_score": 9325.419115148425}

RECENT RESULT
hypothesis: Setting the calibration scale to 1.24754 will preserve all 9,325 argmax predictions while reducing validation cross-entropy below 0.1929895683.
change: Increase only the positive post-pooling calibration multiplier from 1.24623 to 1.24754.
mechanism: Updated local quadratic temperature refinement
evidence_used: Calibration scales 1.24623, 1.24739, and 1.24748 successively reduced cross-entropy without changing correctness; incorporating the latest measurement places the fitted local minimum near 1.24754.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 51.44045537500642, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19298956680297852, "validation_score": 9325.419115148961}

RECENT RESULT
hypothesis: Downweighting flip-inconsistent offsets by 12.5% while using the best verified calibration scale will exceed 9,325 correct predictions by reducing unreliable transformed evidence.
change: Increase only the inference penalty for flip-inconsistent offsets from 10% to 12.5% and restore the verified 1.24754 calibration multiplier.
mechanism: Stronger flip-disagreement gating
evidence_used: The uniform-pair hybrid ensemble achieved 9,325 correct at the best verified cross-entropy of 0.1929895668; the prior 12.5% gating attempt timed out, so it supplied no negative accuracy evidence and remains an unresolved compute-neutral inference refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 52.62951845792122, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19298710632324217, "validation_score": 9325.419116013367}



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
