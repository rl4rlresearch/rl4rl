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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.7220510840416, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.70705533307046, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655155601501465, "validation_score": 9251.414404173205}
prior_hypothesis: Setting the inference scale to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 79.0855537080206, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655190887451172, "validation_score": 9251.414404052011}
prior_hypothesis: Increasing the positive inference scale from 1.044 to 1.048 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.20655614318847657.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.99391554109752, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20657955322265625, "validation_score": 9251.414394557461}
prior_hypothesis: Blending the stronger 0.99 EMA prediction with 25% of the final live-model prediction will exceed 9,247 correct predictions, or tie while lowering validation cross-entropy.

## Recent verification evidence

RECENT RESULT
hypothesis: A 76.25/23.75 EMA/live probability blend will exceed 9,251 correct predictions, or retain 9,251 while reducing cross-entropy below 0.20655155601501465.
change: Restore the verified EMA evaluation machinery and calibrated flip ensemble, then reduce the live-model contribution from 25% to 23.75% while retaining the 1.0495 scale.
mechanism: Lower-side EMA–endpoint blend interpolation
evidence_used: The 75/25 blend achieved 9,251 correct, whereas increasing live weight to 27.5% reduced that to 9,249; a small lower-side interpolation tests whether the verified optimum lies just below 25%.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 76.25/23.75 EMA/live probability blend will exceed 9,251 correct predictions, or retain 9,251 while reducing cross-entropy below 0.20655155601501465.
change: Reduce the live-model ensemble weight from 25% to 23.75% while preserving the verified architecture, training procedure, flip ensemble, and 1.0495 calibration.
mechanism: Lower-side EMA–endpoint blend interpolation
evidence_used: The 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249; the prior 23.75% attempt timed out and therefore provides no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the positive inference scale from 1.048 to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.
change: Raise only the final EMA–live ensemble logit scale, leaving training, rankings, parameter count, and runtime characteristics unchanged.
mechanism: Verified confidence-scale calibration
evidence_used: Reference Design 1 verified scale 1.0495 at 9,251 correct with cross-entropy 0.20655155601501465, strictly improving the current scale 1.048 result of 0.20655190887451172 without changing correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the positive inference scale to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy from 0.20657955322265625 to approximately 0.20655155601501465.
change: Change only the final EMA–live ensemble logit scale, leaving training, class rankings, parameters, and runtime characteristics unchanged.
mechanism: Verified confidence-scale calibration
evidence_used: Reference Design 1 verified this exact implementation at 9,251 correct with cross-entropy 0.20655155601501465, strictly outperforming the current 1.03592-scaled design; the later unverifiable attempt provides no contradictory measurement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact Reference Design 1 implementation will increase validation_correct from 9,243 to 9,251 and reduce validation cross-entropy from 0.20883300514221192 to approximately 0.20655155601501465.
change: Add 0.99 EMA tracking and validation-time 75/25 EMA/live probability blending, retaining flip ensembling and applying the verified 1.0495 logit scale.
mechanism: EMA–endpoint probability ensemble with flip test-time augmentation
evidence_used: Reference Design 1 verified this exact architecture and procedure at 9,251 correct and 0.20655155601501465 cross-entropy, the strongest completed result provided.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 76/24 EMA/live probability blend will exceed 9,251 correct predictions, or retain 9,251 while reducing cross-entropy below 0.20655155601501465.
change: Reduce the live-model contribution from 25% to 24% while preserving the verified architecture, training procedure, flip ensemble, and 1.0495 calibration.
mechanism: Lower-side EMA–endpoint blend refinement
evidence_used: The verified 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy; this motivates a conservative test on the lower-live-weight side. Prior 23.75% attempts timed out and provide no contradictory validation result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the parameter-heavy flattened 7×7 bottleneck with learned row/column mean-and-maximum projections plus deeper high-level spatial processing will exceed 9,251 correct predictions, or tie while reducing cross-entropy below 0.20655155601501465.
change: Add one efficient 96-channel residual block, classify axis-wise spatial statistics through a wider nonlinear head, and retain the best verified 1.0495 inference calibration.
mechanism: Orthogonal silhouette-statistic pooling
evidence_used: All completed 9,251-correct designs share the same 4,704-value flattening head, while prediction-side calibration and EMA blending have plateaued. This challenges the load-bearing assumption that most parameters should model every fixed spatial cell; axis projections instead emphasize clothing silhouettes and local feature presence while reallocating capacity to a 64-unit decision layer.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the inference scale to 1.0495 will preserve 9,251 correct predictions while reducing validation cross-entropy from 0.20657955322265625 to approximately 0.20655155601501465.
change: Change only the final EMA–live ensemble logit scale, leaving training, class rankings, parameters, and runtime characteristics unchanged.
mechanism: Verified confidence-scale calibration
evidence_used: Reference Design 1 verified this exact implementation at 9,251 correct with cross-entropy 0.20655155601501465, strictly improving the current design’s cross-entropy at the same correct count.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring Reference Design 1 will increase validation_correct from 9,243 to 9,251 and reduce validation cross-entropy from 0.20883300514221192 to approximately 0.20655155601501465.
change: Add 0.99 EMA tracking, blend EMA and live-model flip-ensembled probabilities at 75/25, and apply the verified 1.0495 inference scale.
mechanism: EMA–endpoint probability ensemble with flip test-time augmentation
evidence_used: Reference Design 1 verified this exact architecture and procedure at 9,251 correct and 0.20655155601501465 cross-entropy, the strongest completed result provided.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 75.5/24.5 EMA/live probability blend will exceed 9,251 correct predictions, or preserve 9,251 while reducing validation cross-entropy below 0.20655155601501465.
change: Reduce the live-model validation contribution from 25% to 24.5% while retaining the verified architecture, training procedure, flip ensemble, EMA decay, and 1.0495 calibration.
mechanism: Lower-side EMA–endpoint blend refinement
evidence_used: The verified 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy. The unmeasured lower side remains the most direct local refinement; 24.5% tests it conservatively.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the inference scale from 1.048 to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.
change: Change only the final EMA–live ensemble logit scale, leaving training, class rankings, parameters, and runtime characteristics unchanged.
mechanism: Verified confidence-scale calibration
evidence_used: Reference Design 1 verified this exact implementation at 9,251 correct with cross-entropy 0.20655155601501465, strictly improving the current design at the same correct count.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the inference scale to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy from 0.20657955322265625 to approximately 0.20655155601501465.
change: Change only the final ensemble logit scale, leaving training, class rankings, parameters, and runtime characteristics unchanged.
mechanism: Verified confidence-scale calibration
evidence_used: Reference Design 1 verified this exact implementation at 9,251 correct and 0.20655155601501465 cross-entropy, strictly outperforming the current design at the same correct count; failed verification attempts provide no contradictory measurement.
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
