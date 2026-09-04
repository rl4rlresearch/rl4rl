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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}
prior_hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.72801075014286, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22454801788330078, "validation_score": 9208.408313918848}
prior_hypothesis: Halving the two unshifted-view weights will exceed 9,206 correct predictions by moving aggregation opposite the clean-view doubling that reduced correctness to 9,202.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.40000991616398, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22445868225097657, "validation_score": 9206.408343709141}
prior_hypothesis: The verified per-view sharpening and 0.0325 BatchNorm EMA will recover 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.10× will reduce cross-entropy below 0.227083.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring Reference Design 2 exactly will increase validation correctness from 9,168 toward 9,206 and reduce cross-entropy toward 0.224459.
change: Add residual stage connections, phase-specific EMA for parameters and BatchNorm buffers, and verified 1.05× per-view plus 1.10× post-ensemble sharpening.
mechanism: Verified residual split-EMA ensemble with calibrated probability TTA
evidence_used: Reference Design 2 completed within 72.4 seconds and achieved the best supplied score—9,206 correct with 0.224459 cross-entropy—while recent unsuccessful restorations supplied no contrary performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Doubling the influence of the two unshifted views will exceed 9,206 correct predictions by preventing the eight padded shifts from dominating close ensemble decisions.
change: Weight the original and horizontally flipped unshifted predictions twice, retain all shifted views and existing sharpening, and normalize the resulting twelve units of ensemble weight.
mechanism: Clean-view-anchored probability TTA
evidence_used: Unsharpened Reference Design 1 and the sharpened current design both produced exactly 9,206 correct predictions, so confidence scaling changed cross-entropy but not decisions; view weighting is an untested, decision-changing aggregation axis with no extra model evaluations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 59.503497540950775, "validation_accuracy": 0.9202, "validation_correct": 9202, "validation_cross_entropy": 0.22441465225219726, "validation_score": 9202.408358393197}

RECENT RESULT
hypothesis: Increasing only the post-ensemble sharpening from 1.05× to 1.10× will preserve all 9,206 correct predictions while reducing validation cross-entropy toward 0.224459.
change: Change the final TTA log-probability multiplier to 1.10 without altering training or ensemble argmaxes.
mechanism: Verified decision-preserving ensemble calibration
evidence_used: Reference Design 3 differs from the current design only in this multiplier and verified 9,206 correct with 0.224459 cross-entropy, improving the current tied-correctness cross-entropy of 0.227083.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying the verified 1.05× per-view and 1.10× post-ensemble sharpening will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.235889 toward 0.224459.
change: Sharpen every TTA view before probability averaging, then sharpen the ensemble log-probabilities without changing the model or training procedure.
mechanism: Calibrated confidence-weighted probability TTA
evidence_used: Reference Design 2 verified this exact change at 9,206 correct and 0.224459 cross-entropy, strictly improving the current tied-correctness score; unsuccessful later verification attempts supplied no contrary performance result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing hard max pooling with learned, channel-specific max/mean fusion will exceed 9,206 correct predictions by retaining both salient peaks and spatially distributed evidence while reducing shift aliasing.
change: Use residual stages with lightweight learned mixed pooling, reduce the hidden classifier width to 47 to remain under the parameter ceiling, and retain the qualified split-EMA and calibrated TTA configuration.
mechanism: Learned per-channel mixed-statistic residual downsampling
evidence_used: Reference Design 2 reached 9,206 correct, while changing only TTA view weighting fell to 9,202; this suggests further progress requires improving the learned representation rather than reweighting unchanged view predictions. Mixed pooling directly challenges the shared assumption that every scale should discard all but its maximum activation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing label smoothing only during the final 13/32 training phase will exceed 9,206 correct predictions by refining class boundaries under the validation-aligned ±1-shift curriculum without adding runtime or parameters.
change: Retain 0.02 label smoothing during broad augmentation, then switch to ordinary cross-entropy when the existing terminal augmentation and fast EMA phase begins.
mechanism: Terminal hard-label boundary refinement
evidence_used: Per-view and post-ensemble calibration preserved exactly 9,206 correct, while clean-view weighting reduced correctness to 9,202; this motivates changing learned decision boundaries rather than further manipulating unchanged TTA predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gradually reducing label smoothing from 0.02 to zero during the validation-aligned terminal phase will exceed 9,206 correct predictions by refining learned class boundaries without an abrupt loss transition; verified 1.10× outer calibration will improve tied-count cross-entropy without changing argmaxes.
change: Linearly anneal label smoothing only during the final 13/32 of training and replace the final 1.05× ensemble-logit multiplier with the verified 1.10× value.
mechanism: Terminal label-smoothing annealing
evidence_used: Inference-only calibration preserved exactly 9,206 correct while clean-view weighting fell to 9,202, indicating that further correctness gains likely require changed learned boundaries. The terminal hard-label experiment supplied no performance result, so gradual annealing tests its mechanism more conservatively; Reference Design 3 verifies 1.10× outer calibration at 9,206 correct and 0.224459 cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Halving the two unshifted-view weights will exceed 9,206 correct predictions by moving aggregation opposite the clean-view doubling that reduced correctness to 9,202.
change: Apply verified 1.05× per-view and 1.10× final sharpening while weighting each unshifted view 0.5 and each shifted view 1.0.
mechanism: Shift-view-favored probability TTA
evidence_used: Equal weighting achieved 9,206 correct, whereas doubling unshifted views fell to 9,202; this motivates testing the opposite weighting direction without extra evaluations or parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.72801075014286, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22454801788330078, "validation_score": 9208.408313918848}

RECENT RESULT
hypothesis: Reducing each unshifted-view weight from 0.5 to 0.25 will exceed 9,208 correct predictions, continuing the observed improvement as unshifted influence decreased from 2.0 to 1.0 to 0.5.
change: Restore the verified residual architecture, phase-specific parameter-and-BatchNorm EMA, and calibrated TTA, then give the two unshifted views 0.25 weight each while retaining unit weight for all eight shifted views.
mechanism: Further shift-view-favored calibrated residual ensemble
evidence_used: Unshifted weights of 2.0, 1.0, and 0.5 produced 9,202, 9,206, and 9,208 correct respectively; another conservative halving directly tests whether that monotonic decision improvement continues without adding parameters or model evaluations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing each unshifted-view weight from 0.5 to 0.25 will exceed 9,208 correct predictions by continuing the observed improvement as that weight decreased from 2.0 to 1.0 to 0.5.
change: Halve both unshifted-view weights to 0.25 and renormalize the ten-view probability ensemble by its total weight of 8.5.
mechanism: Further shift-view-favored probability TTA
evidence_used: Unshifted weights of 2.0, 1.0, and 0.5 yielded 9,202, 9,206, and 9,208 correct respectively; the previous 0.25 attempt timed out and provided no contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Halving both unshifted-view weights will reproduce the qualified 9,208-correct result, improving over the current 9,206 correct predictions.
change: Weight the original and flipped unshifted predictions by 0.5 while retaining unit weights for all eight shifted views, then normalize by total weight 9.0.
mechanism: Shift-view-favored probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy, outperforming the current equal-weight aggregation at 9,206 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Halving both unshifted-view weights and applying 1.10× post-ensemble sharpening will reproduce the qualified 9,208-correct result, exceeding the current 9,206 correct predictions.
change: Weight the original and flipped unshifted predictions by 0.5, normalize the ten-view ensemble by its total weight of 9.0, and increase argmax-invariant final sharpening from 1.05× to 1.10×.
mechanism: Shift-view-favored calibrated probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy, strictly outperforming the current equal-weight design’s 9,206 correct and 0.227083 cross-entropy.
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
