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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 67.20907104178332, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883301086425782, "validation_score": 9243.413622059876}
prior_hypothesis: Scaling inference logits by 1.0360 will preserve exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20883404388427734.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}
prior_hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.7220510840416, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.751318583032116, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.2066554630279541, "validation_score": 9247.414368488206}
prior_hypothesis: Evaluating a 0.99-decay EMA of parameters and BatchNorm state will increase validation_correct above 9,243 while preserving the existing training budget and 245,040 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the flip-ensemble logits by 1.03592 will preserve all 9,243 class predictions while reducing validation cross-entropy from 0.20883301086425782 to approximately 0.20883300514221192.
change: Change only the positive inference-time logit scale from 1.0360 to 1.03592.
mechanism: Verified inference-temperature calibration
evidence_used: Reference Designs 2 and 3 achieved the highest verified score, 9243.413622061835, using this exact scale with otherwise identical code; prior timeouts provide no contradictory performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring power 0.5831695556640625 and logit scale 1.03592 will preserve 9,243 correct predictions while reducing validation cross-entropy from 0.20904548225402833 to approximately 0.20883300514221192.
change: Apply the highest-scoring verified inference configuration without changing training, parameters, or optimizer steps.
mechanism: Verified power-mean flip ensembling with inference-temperature calibration
evidence_used: Reference Designs 2 and 3 both achieved the best verified score, 9243.413622061835, with this exact configuration; more computationally expensive augmentation and architecture changes timed out, while label smoothing reduced accuracy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Evaluating a 0.99-decay EMA of parameters and BatchNorm state will increase validation_correct above 9,243 while preserving the existing training budget and 245,040 learned parameters.
change: Maintain non-persistent EMA state after each optimizer step and temporarily swap it into the model during evaluation, retaining the verified flip ensemble and calibration.
mechanism: Exponential moving-average weight evaluation
evidence_used: Inference calibration has plateaued at exactly 9,243 correct, label smoothing reduced accuracy to 9,225, and heavier augmentation or architecture changes timed out; EMA can change class rankings with no extra training forwards or learned parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.751318583032116, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.2066554630279541, "validation_score": 9247.414368488206}

RECENT RESULT
hypothesis: Evaluating a 0.99-decay EMA will reproduce the verified improvement from 9,243 to at least 9,247 correct predictions.
change: Maintain non-persistent EMA state after every optimizer step and automatically swap it into the model for validation.
mechanism: Exponential moving-average parameter and BatchNorm evaluation
evidence_used: Reference Design 3 used this exact EMA implementation and achieved the best result: 9,247 correct with 0.206655 cross-entropy, versus the current 9,243 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 0.99-decay EMA will reproduce the qualified Reference Design 3 result of at least 9,247 correct predictions, improving over the current 9,243.
change: Restore Reference Design 3’s non-persistent EMA state, validation-time state swap, optimizer model link, update hook, and calibrated 1.03592 inference scale.
mechanism: Exponential moving-average parameter and BatchNorm evaluation
evidence_used: Reference Design 3 achieved the best verified result—9,247 correct and 0.206655 cross-entropy—while the current non-EMA design achieved 9,243; the later unverifiable attempt supplied no contradictory performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing the narrow full-grid head with horizontal mean/max pooling, a 128-wide classifier, and one additional residual refinement—evaluated with the verified EMA—will exceed 9,247 correct predictions without exceeding the parameter or runtime limits.
change: Treat horizontal position as a nuisance while preserving vertical structure, increasing classifier width from 38 to 128; restore the verified EMA and inference calibration. The model has 249,954 learned parameters.
mechanism: Learned vertical-strip statistics with a wide nonlinear classifier
evidence_used: Calibration repeatedly plateaued at 9,243 correct, while EMA reached 9,247. Existing flip augmentation supports horizontal invariance; this single-pass linear-cost representation tests it without the runtime overhead that caused translation ensembles and covariance modeling to time out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing EMA decay from 0.99 to 0.995 will reduce update noise over an effective horizon of roughly 200 steps and exceed 9,247 correct predictions, or tie while lowering validation cross-entropy.
change: Change only the EMA decay; architecture, training budget, optimizer, augmentation, and inference calibration remain unchanged.
mechanism: Longer-horizon exponential moving average
evidence_used: The 0.99 EMA improved validation_correct from 9,243 to 9,247 with no runtime penalty, while heavier augmentation and architectural experiments timed out; extending the successful averaging mechanism is the lowest-cost ranking-changing test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Evaluating the verified 0.99-decay EMA will increase validation_correct from 9,243 to at least 9,247.
change: Add non-persistent EMA state, update it after every optimizer step, and automatically swap it into the model for validation.
mechanism: Exponential moving-average parameter and BatchNorm evaluation
evidence_used: Reference Design 3 achieved the best verified result—9,247 correct and 0.206655 cross-entropy—using this exact EMA implementation, while the current non-EMA design reached 9,243.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Evaluating a 0.99-decay EMA will reproduce the qualified Reference Design 3 result of at least 9,247 correct predictions and improve over the current 9,243.
change: Add Reference Design 3’s non-persistent EMA state, validation-time state swap, optimizer update hook, and calibrated 1.03592 inference scale.
mechanism: Exponential moving-average parameter evaluation
evidence_used: Reference Design 3 is the strongest verified implementation, achieving 9,247 correct and 0.206655 cross-entropy; subsequent unverifiable attempts provide no contradictory performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Evaluating the 0.99-decay EMA will increase validation_correct from 9,243 to at least 9,247 while reducing cross-entropy.
change: Add non-persistent EMA state and validation-time swapping, update it after each optimizer step, and restore the verified power and logit scale.
mechanism: 0.99-decay EMA evaluation with calibrated flip power-mean inference
evidence_used: Reference Design 3 used this exact configuration and achieved the best verified result: 9,247 correct and 0.206655 cross-entropy; later unverifiable attempts provide no contradictory performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Removing 0.1 dropout from the 38-unit classifier head will increase validation_correct above 9,247 by improving convergence within the fixed two-pass exposure budget.
change: Disable the classifier-head dropout while preserving the verified architecture, EMA, augmentation, optimizer schedule, and calibrated flip ensemble.
mechanism: Dropout-free narrow classifier head
evidence_used: Adding 0.05 label smoothing reduced validation_correct from 9,243 to 9,225 and worsened cross-entropy, indicating that additional regularization impedes this short training regime; the verified EMA already supplies parameter averaging and regularization.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.72152316686697, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.20715935821533202, "validation_score": 9237.414195521575}

RECENT RESULT
hypothesis: Evaluating the qualified 0.99 EMA will reproduce Reference Design 3’s improvement from 9,243 to at least 9,247 correct predictions.
change: Add non-persistent EMA state, swap it in for validation, and update it after every optimizer step while preserving the calibrated flip ensemble.
mechanism: 0.99-decay exponential moving-average evaluation
evidence_used: Reference Design 3 achieved the best verified result—9,247 correct and 0.206655 cross-entropy—using this exact EMA implementation; later unverifiable attempts provide no contradictory performance evidence.
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
