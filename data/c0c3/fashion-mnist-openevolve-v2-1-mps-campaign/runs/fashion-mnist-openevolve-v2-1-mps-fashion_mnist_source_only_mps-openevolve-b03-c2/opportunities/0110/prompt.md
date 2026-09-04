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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.661927457898855, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445223388671875, "validation_score": 9256.411708246771}
prior_hypothesis: Scaling the ensemble logits by 1.356 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.214452343.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 36.167439040960744, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21452123413085938, "validation_score": 9256.411684856508}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.33 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2147829.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 44.383009457960725, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2147829204559326, "validation_score": 9256.411596172107}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.30 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2152603.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.09292483399622, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144522300720215, "validation_score": 9256.411708248064}
prior_hypothesis: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2145212341 to approximately 0.2144522301.
change: Replace only the evaluation-time logit multiplier, leaving training and ensemble weighting unchanged.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 directly verified the same implementation at 9,256 correct with cross-entropy 0.2144522301, strictly improving the current design’s tie-break metric.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions and reduce validation cross-entropy from 0.2147829205 to approximately 0.2144522301.
change: Replace only the evaluation-time logit multiplier, leaving training and ensemble weighting unchanged.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 directly verified this exact implementation at 9,256 correct with cross-entropy 0.2144522300720215, a strict improvement over the current design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding an identity skip around the final 96-channel convolution will improve optimization under the fixed two-pass exposure and raise validation_correct above 9,256 without increasing the parameter count.
change: Reorganize the existing final convolution and pooling stage into a parameter-neutral residual block while preserving all training, augmentation, optimizer, and ensemble settings.
mechanism: Residual high-level feature refinement
evidence_used: Evaluation weighting, diagonal views, orientation bias, translation-duration changes, and reduced label smoothing all lowered validation_correct, indicating those dimensions are locally exhausted; the existing same-shape 96-to-96 convolution permits an untested residual path without spending parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.21445223388671875.
change: Change only the evaluation-time logit multiplier from 1.356 to 1.35605.
mechanism: Verified fine-grained ensemble temperature calibration
evidence_used: Reference Design 3 verified this exact implementation at 9,256 correct with cross-entropy 0.2144522300720215, a strict validation_score improvement; prior attempts were not verified and therefore do not contradict that result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2145212341 to approximately 0.2144522301.
change: Change only the evaluation-time ensemble logit multiplier from 1.33 to 1.35605.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 verified this exact implementation at 9,256 correct with cross-entropy 0.2144522300720215, strictly improving the current design’s tie-break metric.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2147829205 to approximately 0.2144522301.
change: Change only the evaluation-time logit multiplier from 1.30 to 1.35605.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 verified this exact otherwise-identical implementation at 9,256 correct and 0.2144522301 cross-entropy, strictly improving the current tie-break metric; unsuccessful verification attempts provide no contradictory measured result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling logits by 1.35524 will preserve exactly 9,256 predictions while reducing validation cross-entropy below 0.2144522300720215.
change: Replace only the evaluation-time ensemble multiplier with the estimated cross-entropy-minimizing value.
mechanism: Quadratic ensemble-temperature interpolation
evidence_used: The verified cross-entropies at multipliers 1.30, 1.33, and 1.356 imply a local quadratic minimum near 1.35524; any positive multiplier preserves argmax predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 59.919017457868904, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445229873657226, "validation_score": 9256.411708224787}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy below 0.21445223388671875.
change: Change only the evaluation-time ensemble logit multiplier from 1.356 to 1.35605.
mechanism: Verified fine-grained ensemble temperature calibration
evidence_used: Reference Design 3 directly verified this otherwise-identical implementation at 9,256 correct and 0.2144522300720215 cross-entropy, a strict validation_score improvement over the current design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2145212341 to approximately 0.2144522301.
change: Change only the evaluation-time ensemble logit multiplier from 1.33 to 1.35605.
mechanism: Verified ensemble temperature calibration
evidence_used: Reference Design 2 directly verified this otherwise-identical implementation at 9,256 correct and 0.2144522300720215 cross-entropy, strictly improving the current validation score; failed verification attempts reported no contradictory measurement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a 25% probability-ensemble loss will raise validation_correct above 9,256 by aligning training with evaluation-time flip averaging; otherwise, the verified 1.35605 scale should improve the cross-entropy tie-break.
change: Blend the existing per-view loss with label-smoothed NLL on the averaged original/flip probabilities, and apply the best verified evaluation multiplier.
mechanism: TTA-aligned flip-ensemble training with verified calibration
evidence_used: Evaluation-view modifications reduced validation_correct, favoring preservation of the established ensemble; Reference Design 2 verified 1.35605 at 9,256 correct with the lowest reported cross-entropy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 81.45665208389983, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.21451671142578124, "validation_score": 9248.41168638957}

RECENT RESULT
hypothesis: Raising label smoothing from 0.02 to 0.025 will improve generalization enough to exceed 9,256 correct predictions while avoiding the larger unverified jump to 0.03.
change: Increase only the training loss label-smoothing coefficient, preserving the best verified architecture, optimizer, augmentation, ensemble, and calibration.
mechanism: Conservative increase in label-smoothing regularization
evidence_used: Reducing label smoothing to 0.01 lowered validation_correct from 9,256 to 9,239, directly indicating that weaker smoothing is harmful; 0.025 tests the favorable direction conservatively.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 78.18015862489119, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21402598724365235, "validation_score": 9253.411852798254}



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
