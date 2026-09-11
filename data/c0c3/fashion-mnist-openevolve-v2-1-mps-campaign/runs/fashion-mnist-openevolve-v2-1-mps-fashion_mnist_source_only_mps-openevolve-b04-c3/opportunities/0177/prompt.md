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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 67.45724108302966, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655155067443848, "validation_score": 9251.414404175039}
prior_hypothesis: An inference scale of 1.04953 will preserve all 9,251 class predictions while slightly reducing validation cross-entropy below 0.20655155601501465.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.70705533307046, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655155601501465, "validation_score": 9251.414404173205}
prior_hypothesis: Setting the inference scale to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.7220510840416, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 66.43872454203665, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655155258178712, "validation_score": 9251.414404174384}
prior_hypothesis: An inference scale of 1.0496144 will preserve all 9,251 class predictions while reducing validation cross-entropy below 0.20655155067443848.

## Recent verification evidence

RECENT RESULT
hypothesis: A 76/24 EMA/live probability blend will exceed 9,251 correct predictions, or preserve 9,251 while reducing validation cross-entropy below 0.20655155067443848.
change: Reduce the live-model validation contribution from 25% to 24% while retaining the verified training procedure, flip ensemble, EMA decay, and 1.04953 calibration.
mechanism: Lower-side EMA–endpoint blend refinement
evidence_used: The verified 75/25 blend achieved 9,251 correct, while raising live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy. The only lower-side test, 24.5%, timed out without producing contradictory performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.25/24.75 EMA/live probability blend will exceed 9,251 correct predictions, or preserve 9,251 while reducing validation cross-entropy below 0.20655155067443848.
change: Reduce the live-model validation contribution from 25% to 24.75%, increase the EMA contribution correspondingly, and use the best verified 1.04953 calibration.
mechanism: Conservative lower-side EMA–endpoint blend refinement
evidence_used: The verified 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy; unsuccessful 24–24.5% runs produced no contradictory validation measurement, motivating a conservative lower-side interpolation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling validation logits by 1.04953 will preserve all 9,251 predictions while reducing validation cross-entropy from 0.2065795532 to approximately 0.2065515507.
change: Change only the final validation-time logit scale; training and class rankings remain unchanged.
mechanism: Verified confidence-scale calibration
evidence_used: Reference Design 2 verified this otherwise identical implementation at 9,251 correct and 0.20655155067443848 cross-entropy, the best completed validation score.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact Reference Design 2 procedure will increase validation_correct from 9,243 to 9,251 and reduce validation_cross_entropy to approximately 0.206551551.
change: Track a 0.99 EMA during training, blend 75% EMA and 25% live flip-ensembled probabilities, and apply the verified 1.04953 calibration.
mechanism: EMA–endpoint probability ensemble with flip test-time augmentation
evidence_used: Reference Design 2 completed in 67.46 seconds and achieved the strongest measured result: 9,251 correct with 0.20655155067443848 cross-entropy; unsuccessful verification attempts supplied no contradictory performance measurement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the terminal learning-rate floor from 12.5% to 5% will exceed 9,251 correct predictions by limiting harmful late-stage parameter drift while preserving early optimization.
change: Keep the verified architecture, EMA ensemble, augmentation, and calibration unchanged; lower only the cosine schedule’s minimum learning rate from 3.125e-4 to 1.25e-4.
mechanism: Lower-tail cosine annealing
evidence_used: The verified 75/25 EMA–live blend reached 9,251 correct, while increasing the live-model contribution to 27.5% reduced correctness to 9,249, suggesting that the final live weights are less reliable than their recent EMA average and motivating gentler endpoint updates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dynamically emphasizing image-dependent channels and 7×7 regions will exceed 9,251 correct predictions, or tie while lowering cross-entropy below 0.20655155067443848.
change: Add a near-identity channel-and-spatial attention gate after the final residual block, using the remaining parameter budget and negligible extra convolutional work; retain the best verified calibration.
mechanism: Content-adaptive channel–spatial gating
evidence_used: The current static feature routing reached 9,251 correct, while an added high-resolution residual block timed out. Translation augmentation also reduced correctness, motivating a lightweight, coordinate-preserving gate rather than more invariance or expensive spatial processing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An inference scale of 1.05016 will preserve all 9,251 class predictions while reducing validation cross-entropy below 0.20655155067443848.
change: Change only the positive validation-time logit scale, leaving training, parameters, runtime, and predicted class rankings unchanged.
mechanism: Empirical quadratic logit-temperature calibration
evidence_used: The otherwise identical scales 1.03592, 1.0495, and 1.04953 produced cross-entropies 0.2065795532, 0.2065515560, and 0.2065515507 with 9,251 correct; quadratic interpolation places the local cross-entropy minimum near 1.05016.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.21973083307967, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655160598754882, "validation_score": 9251.414404156041}

RECENT RESULT
hypothesis: Restoring the exact Reference Design 2 procedure will increase validation_correct from 9,243 to 9,251 and reduce validation_cross_entropy to approximately 0.206551551.
change: Track a 0.99 EMA during training, blend 75% EMA and 25% live flip-ensembled probabilities, and apply the verified 1.04953 calibration.
mechanism: EMA–endpoint probability ensemble with flip test-time augmentation
evidence_used: Reference Design 2 achieved the highest completed validation_score, with 9,251 correct and 0.20655155067443848 cross-entropy; unsuccessful restoration attempts supplied no contradictory performance result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling validation logits by 1.049736 will preserve all 9,251 predictions while reducing validation cross-entropy below 0.20655155067443848.
change: Change only the positive validation-time logit scale, leaving training, parameters, runtime, and class rankings unchanged.
mechanism: Local quadratic confidence calibration
evidence_used: The identical implementation measured cross-entropies of 0.20655155601501465, 0.20655155067443848, and 0.20655160598754882 at scales 1.0495, 1.04953, and 1.05016; local quadratic interpolation places the minimum near 1.049736.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 78.926611749921, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655155754089355, "validation_score": 9251.414404172681}

RECENT RESULT
hypothesis: Reducing the live-model contribution from 25% to 22.5% will exceed 9,251 correct predictions, or preserve 9,251 while lowering cross-entropy, because increasing it to 27.5% reduced correctness to 9,249 and worsened cross-entropy.
change: Change the validation probability ensemble from 75/25 to 77.5/22.5 EMA/live while leaving training, runtime, and calibration unchanged.
mechanism: Lower-live EMA endpoint blending
evidence_used: The verified 75/25 blend achieved 9,251 correct, whereas 72.5/27.5 achieved only 9,249; this symmetric lower-side test directly evaluates whether the EMA prediction is more reliable than the noisy final endpoint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.69792249985039, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.20653982543945312, "validation_score": 9249.414408202247}

RECENT RESULT
hypothesis: An inference scale of 1.0496144 will preserve all 9,251 class predictions while reducing validation cross-entropy below 0.20655155067443848.
change: Change only the positive validation-time logit scale; training, runtime, parameters, and class rankings remain unchanged.
mechanism: Three-point quadratic logit calibration
evidence_used: Scales 1.0495, 1.04953, and 1.049736 yielded cross-entropies 0.20655155601501465, 0.20655155067443848, and 0.20655155754089355 respectively; quadratic interpolation places the bracketed minimum near 1.0496144.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 66.43872454203665, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655155258178712, "validation_score": 9251.414404174384}

RECENT RESULT
hypothesis: Restoring the exact Reference Design 2 procedure will increase validation_correct from 9,243 to 9,251 and reduce validation_cross_entropy to approximately 0.206551551.
change: Track a 0.99 parameter-and-buffer EMA, blend 75% EMA and 25% live flip-ensembled probabilities during validation, and apply the verified 1.04953 calibration.
mechanism: EMA–endpoint probability ensemble with flip test-time augmentation
evidence_used: Reference Design 2 achieved the strongest measured result—9,251 correct with 0.20655155067443848 cross-entropy—while unsuccessful restoration attempts supplied no contradictory validation measurement.
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
