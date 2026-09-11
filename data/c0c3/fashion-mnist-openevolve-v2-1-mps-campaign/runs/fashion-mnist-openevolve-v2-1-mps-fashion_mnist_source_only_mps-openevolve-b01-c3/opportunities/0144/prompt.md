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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.97645550011657, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059280395508, "validation_score": 9287.41435115912}
prior_hypothesis: A 1.557939852073332× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059269.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.46929245791398, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935852050782, "validation_score": 9287.414356848736}
prior_hypothesis: Restoring the verified 7.788524% disagreement adjustment will preserve all 9,287 pooled predictions and reduce validation cross-entropy from 0.20668935928344725 to approximately 0.20668935852050782.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.03452316601761, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2067018039703369, "validation_score": 9289.414352575222}
prior_hypothesis: A 15% penalty on flip-inconsistent offset pairs will exceed 9,287 correct predictions by suppressing locally unreliable transformed evidence while leaving flip-stable pooling unchanged.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 73.09784612501971, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935852050782, "validation_score": 9287.414356848736}
prior_hypothesis: A 7.788524% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935928344725.

## Recent verification evidence

RECENT RESULT
hypothesis: A 7.708% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.
change: Restore the center-view weight used by the best reference and apply a weaker, quadratically estimated agreement-conditioned logit scale.
mechanism: Quadratically interpolated agreement-conditioned calibration
evidence_used: The 0%, 15%, and 22.5% adjustments produced cross-entropies of 0.20670592765808105, 0.20670405883789061, and 0.20675360946655275; quadratic interpolation estimates the minimum near 7.708%, while positive scaling preserves the pooled argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 61.30429754196666, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668936004638672, "validation_score": 9287.414356848212}

RECENT RESULT
hypothesis: A 7.727% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.
change: Move the agreement-conditioning strength from 7.708% to the quadratic minimum estimated from the verified 0%, 7.708%, and 15% results.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: Those calibration strengths produced cross-entropies of 0.20670592765808105, 0.20668936004638672, and 0.20670405883789061; local quadratic interpolation places the minimum near 7.727%, and positive scaling leaves every pooled argmax unchanged.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 7.727% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.
change: Replace the current 15% agreement-conditioning strength with the quadratically estimated 7.727% optimum, leaving training and pooled logits unchanged.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: The verified 0%, 7.708%, and 15% adjustments yielded cross-entropies of 0.20670592765808105, 0.20668936004638672, and 0.20670405883789061; their quadratic minimum is near 7.727%, and its only prior verification timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.16700104204938, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935928344725, "validation_score": 9287.414356848474}

RECENT RESULT
hypothesis: A 7.788524% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935928344725.
change: Restore the best verified pooling configuration and move its agreement-conditioning strength from 7.727% to the local quadratic minimum.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: The verified 7.708%, 7.727%, and 15% adjustments produced cross-entropies of 0.20668936004638672, 0.20668935928344725, and 0.20670405883789061; quadratic interpolation estimates a minimum near 7.788524%, while positive scaling cannot change pooled argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 73.09784612501971, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935852050782, "validation_score": 9287.414356848736}

RECENT RESULT
hypothesis: A 7.7757513% disagreement adjustment will preserve all 9,287 pooled predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935852050782.
change: Restore the best verified center-view pooling weight and apply the quadratic-minimum agreement-conditioned logit calibration.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: The verified 7.708%, 7.727%, and 7.788524% adjustments produced cross-entropies of 0.20668936004638672, 0.20668935928344725, and 0.20668935852050782; quadratic interpolation places the next minimum near 7.7757513%, while positive scaling cannot change argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 7.7757513% disagreement adjustment will preserve all 9,287 pooled predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668935852050782.
change: Move the agreement-conditioning strength from 7.788524% to the unresolved quadratic-minimum estimate of 7.7757513%, leaving training and pooled logits unchanged.
mechanism: Local quadratic refinement of agreement-conditioned calibration
evidence_used: Verified strengths of 7.708%, 7.727%, and 7.788524% progressively reduced cross-entropy while retaining 9,287 correct; quadratic interpolation estimated 7.7757513%, and its only verification timed out without subject-level evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.63248324999586, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935852050782, "validation_score": 9287.414356848736}

RECENT RESULT
hypothesis: Restoring the verified 7.788524% disagreement adjustment will preserve all 9,287 pooled predictions and reduce validation cross-entropy from 0.20668935928344725 to approximately 0.20668935852050782.
change: Replace the current 7.727% agreement-conditioning strength with the best verified 7.788524% setting while leaving training and pooled predictions unchanged.
mechanism: Best-verified agreement-conditioned logit calibration
evidence_used: Reference Design 3 achieved the highest available validation_score with the same 9,287 correct predictions; the positive per-image calibration scale cannot change pooled argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.46929245791398, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935852050782, "validation_score": 9287.414356848736}

RECENT RESULT
hypothesis: Restoring the verified 7.788524% disagreement adjustment will preserve all 9,287 pooled predictions and reduce cross-entropy from 0.20668936004638672 to approximately 0.20668935852050782.
change: Replace the current 7.708% agreement-conditioning strength with the best verified 7.788524% setting.
mechanism: Best-verified agreement-conditioned logit calibration
evidence_used: Reference Designs 1 and 3 independently achieved the highest available validation_score using this calibration; its positive per-image scaling cannot change pooled argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 70.89092820812948, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20668935852050782, "validation_score": 9287.414356848736}

RECENT RESULT
hypothesis: Identity-initialized channel recalibration will exceed 9,287 correct predictions by letting global image context modulate semantic feature channels without the harmful spatial selection seen in prior pooling experiments.
change: Add a bottleneck channel-recalibration block before global pooling, reduce the classifier width to keep 249,997 parameters, and retain the best verified view-agreement calibration.
mechanism: Residual global-context channel recalibration
evidence_used: Spatial attention fell to 9,266 correct and static spatial pooling to 9,252, suggesting location selection is harmful; channel recalibration instead preserves all spatial evidence while introducing learned context-dependent feature interactions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249997, "training_seconds": 60.52951525012031, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.20387982788085937, "validation_score": 9274.415323845802}

RECENT RESULT
hypothesis: Averaging bounded per-view probabilities instead of logits will exceed 9,287 correct predictions by reducing the influence of confidently wrong transformed views on disagreement cases.
change: Keep training and all learned parameters unchanged; replace logit-space test-time pooling with center-weighted probability pooling, then convert the ensemble back to logits and retain the best verified agreement calibration.
mechanism: Confidence-bounded probability-space test-time ensembling
evidence_used: Agreement-conditioned calibration improved cross-entropy while preserving 9,287 correct, identifying view disagreement as useful reliability information; spatial and channel-pooling changes reduced accuracy, motivating a targeted inference-only aggregation change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 61.25002041598782, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20672935791015626, "validation_score": 9286.414343114073}

RECENT RESULT
hypothesis: Downweighting only translation offsets whose original and flipped views disagree will exceed 9,287 correct predictions by suppressing locally unreliable evidence while reproducing the verified pooling exactly for flip-stable offsets.
change: Replace fixed offset pooling with a mild 15% penalty for flip-inconsistent view pairs, while retaining the verified center-view preference and agreement-conditioned calibration.
mechanism: Flip-consistency-weighted logit ensembling
evidence_used: Agreement-conditioned calibration showed that view consensus contains useful reliability information, while wholesale probability pooling lost one correct prediction; using consensus only to adjust unreliable offset pairs is a narrower aggregation change.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 15% penalty on flip-inconsistent offset pairs will exceed 9,287 correct predictions by suppressing locally unreliable transformed evidence while leaving flip-stable pooling unchanged.
change: Replace fixed offset averaging with per-image offset weights derived from agreement between each offset’s original and flipped predictions, retaining the verified center preference and calibration.
mechanism: Flip-consistency-weighted logit ensembling
evidence_used: Agreement-conditioned calibration improved cross-entropy without changing predictions, showing view consensus carries reliability information; probability pooling lost one correct prediction, motivating this narrower unresolved aggregation change.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.03452316601761, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2067018039703369, "validation_score": 9289.414352575222}



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
