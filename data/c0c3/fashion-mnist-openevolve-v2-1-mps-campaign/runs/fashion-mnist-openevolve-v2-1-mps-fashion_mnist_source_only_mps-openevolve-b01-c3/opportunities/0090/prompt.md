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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.13886966602877, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670779724121094, "validation_score": 9287.414350517287}
prior_hypothesis: Giving the unshifted offset 1.5× weight while retaining all ten views will exceed 9,287 correct predictions by favoring the validation-aligned center crop without discarding the complementary shifted evidence.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.3550312500447, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20673983764648438, "validation_score": 9287.414339515777}
prior_hypothesis: Scaling the linear-recency ten-view logits by 1.225 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2071991.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.4552982088644, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20854735527038573, "validation_score": 9287.41371982473}
prior_hypothesis: Scaling the linear-recency ten-view logits by 1.125 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2096186.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 64.05267137498595, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670711212158202, "validation_score": 9287.414350752537}
prior_hypothesis: A 1.546875× center-offset weight will preserve 9,287 correct predictions while reducing validation cross-entropy below 0.2067073055.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling pooled validation logits by 1.15 will preserve all 9,287 argmax predictions while reducing cross-entropy below 0.2096186.
change: Restore the verified linear-recency baseline’s evaluation sharpening and increase its logit scale from 1.10 to the untested midpoint 1.15.
mechanism: Decision-invariant intermediate temperature sharpening
evidence_used: A 1.10 scale preserved 9,287 correct predictions and reduced cross-entropy from 0.2173399 to 0.2096186; the 1.20 trial timed out, so 1.15 conservatively probes further sharpening without changing decisions, parameters, or meaningful computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the linear-recency ten-view logits by 1.125 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2096186.
change: Restore the verified linear-recency tail average and apply a conservative evaluation-only scale between the successful 1.10 setting and the unverified 1.15 setting.
mechanism: Decision-invariant fine-grained temperature sharpening
evidence_used: Linear recency with 1.10 scaling achieved 9,287 correct and reduced cross-entropy from 0.2173399 to 0.2096186; positive scaling cannot change predicted classes, and the 1.15 attempt timed out without subject-level evidence against further sharpening.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.4552982088644, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20854735527038573, "validation_score": 9287.41371982473}

RECENT RESULT
hypothesis: Replacing hard global maxima with learned per-channel log-mean-exp pooling will exceed 9,287 correct predictions by preserving distributed detector evidence while suppressing isolated activation outliers.
change: Challenge the assumption that each channel is best summarized by a single maximum; learn each channel’s pooling sharpness, slightly narrow the classifier to remain below 250,000 parameters, and retain the strongest linear-recency averaging with 1.125 logit scaling.
mechanism: Channel-adaptive smooth-extremum feature pooling
evidence_used: Uniform ten-view pooling reached 9,287 correct, whereas trimming view evidence fell to 9,276; this favors retaining all views while addressing outlier sensitivity inside the representation, before view aggregation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the pooled logits by 1.175 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2085474.
change: Increase only the final ten-view evaluation logit scale from 1.125 to 1.175.
mechanism: Decision-invariant temperature calibration
evidence_used: Increasing the scale from 1.10 to 1.125 preserved 9,287 correct predictions and reduced cross-entropy from 0.2096186 to 0.2085474; positive scaling cannot change argmax predictions, and the timed-out 1.15 and 1.20 trials provide no subject-level evidence against further sharpening.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 74.46106370794587, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20719905242919923, "validation_score": 9287.414181902308}

RECENT RESULT
hypothesis: Scaling the linear-recency ten-view logits by 1.225 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2071991.
change: Increase only the final evaluation logit scale from 1.10 to 1.225, leaving training, parameters, and view pooling unchanged.
mechanism: Decision-invariant near-optimal temperature sharpening
evidence_used: Successive scales of 1.10, 1.125, and 1.175 preserved 9,287 correct while reducing cross-entropy to 0.2096186, 0.2085474, and 0.2071991; the diminishing gains suggest testing 1.225 near the implied calibration optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.3550312500447, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20673983764648438, "validation_score": 9287.414339515777}

RECENT RESULT
hypothesis: Giving the unshifted offset 1.5× weight while retaining all ten views will exceed 9,287 correct predictions by favoring the validation-aligned center crop without discarding the complementary shifted evidence.
change: Replace uniform offset pooling with a modest center-weighted average and apply the best-verified 1.225 evaluation-logit scale.
mechanism: Center-biased full-view logit pooling
evidence_used: Uniform ten-view pooling reached 9,287 correct, while trimming view evidence regressed to 9,276; therefore all views should remain, but the unshifted validation-aligned offset can receive a mild prior. A 1.225 scale produced the best verified tied-count cross-entropy of 0.2067398.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.13886966602877, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670779724121094, "validation_score": 9287.414350517287}

RECENT RESULT
hypothesis: Increasing the unshifted offset weight from 1.5× to 2× while retaining every shifted and flipped view will exceed 9,287 correct predictions by more strongly favoring validation-aligned evidence.
change: Restore linear-recency tail averaging, pool the center offset at 2× weight, retain the best-verified 1.225 logit scale, and leave training and parameter count unchanged.
mechanism: Stronger center-biased full-view logit pooling
evidence_used: Center weighting at 1.5× preserved 9,287 correct and improved cross-entropy from 0.2067398 to 0.2067078; a stronger weight tests whether the supported direction was too modest to change accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the center-offset weight from 1.5× to 1.75× will exceed 9,287 correct predictions by strengthening validation-aligned evidence while retaining all shifted and flipped views.
change: Increase only the unshifted offset’s pooling weight and adjust the normalization accordingly; preserve training and the 1.225 logit scale.
mechanism: Intermediate center-biased full-view pooling
evidence_used: Moving from uniform pooling to 1.5× center weighting preserved 9,287 correct while lowering cross-entropy from 0.2067398 to 0.2067078; the 2× attempt timed out, making 1.75× the informative untested midpoint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 57.55518033308908, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20670757904052733, "validation_score": 9286.414350592211}

RECENT RESULT
hypothesis: A 1.625× center-offset weight will preserve 9,287 correct predictions while reducing cross-entropy below 0.2067078.
change: Retain all ten evaluation views and the 1.225 logit scale, but weight the unshifted offset 1.625×.
mechanism: Boundary-seeking center-biased full-view pooling
evidence_used: Center weighting at 1.5× retained 9,287 correct and improved cross-entropy, whereas 1.75× lost one correct prediction; 1.625× is the informative midpoint between those outcomes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 73.89989495906048, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20670665245056152, "validation_score": 9286.414350910376}

RECENT RESULT
hypothesis: A 1.5625× center-offset weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.206707797.
change: Retain all ten views, weight the unshifted offset 1.5625×, normalize the pooled logits, and use the best-verified 1.225 evaluation scale.
mechanism: Boundary-refined center-biased full-view pooling
evidence_used: A 1.5× center weight achieved 9,287 correct at 0.206707797 cross-entropy, while 1.625× reduced cross-entropy slightly but lost one prediction; 1.5625× is the informative midpoint at that accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 64.4250784998294, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2067069496154785, "validation_score": 9286.414350808338}

RECENT RESULT
hypothesis: A 1.53125× center-offset weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.206707797.
change: Restore linear-recency tail averaging, retain all ten evaluation views, weight the unshifted offset 1.53125×, and apply 1.225 logit scaling.
mechanism: Boundary-refined center-biased full-view pooling
evidence_used: A 1.5× center weight retained 9,287 correct, while 1.5625× lost one prediction but slightly lowered cross-entropy; their midpoint is the most informative refinement of that accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 61.73870625020936, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067073055267334, "validation_score": 9287.414350686127}

RECENT RESULT
hypothesis: A 1.546875× center-offset weight will preserve 9,287 correct predictions while reducing validation cross-entropy below 0.2067073055.
change: Increase the unshifted offset weight from 1.53125× to the midpoint 1.546875× and update the pooling normalization.
mechanism: Accuracy-boundary bisection for center-biased view pooling
evidence_used: A 1.53125× weight retained 9,287 correct, while 1.5625× lost one prediction but achieved slightly lower cross-entropy; their midpoint is the most informative refinement of the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 64.05267137498595, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670711212158202, "validation_score": 9287.414350752537}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
