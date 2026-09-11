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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.64136466709897, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2229105945587158, "validation_score": 9210.408860633168}
prior_hypothesis: Restoring 1.10× per-view sharpening will recover exactly 9,210 correct predictions, while a 1.1875× argmax-invariant final multiplier will reduce cross-entropy below 0.2228996.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.10024604201317, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22289843368530274, "validation_score": 9210.408864699002}
prior_hypothesis: A 1.178× final multiplier will preserve all 9,210 predictions while reducing validation cross-entropy below 0.22289956.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 64.93181979213841, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.22246544494628906, "validation_score": 9214.40900951603}
prior_hypothesis: Reducing each centered-view weight from 0.375 to 0.25 will exceed 9,215 correct predictions by favoring the translation-robust symmetric shifted views.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 53.110233209095895, "validation_accuracy": 0.9215, "validation_correct": 9215, "validation_cross_entropy": 0.22240231246948242, "validation_score": 9215.40903063983}
prior_hypothesis: Splitting the qualified horizontal-view weight equally between left- and right-shifted views will exceed 9,210 correct predictions by removing directional inference bias while preserving the best ensemble’s total horizontal, centered, and vertical weight.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the exact Reference Design 1 ensemble and one-third centered terminal sampler will increase validation_correct from 9,206 to 9,210 while achieving cross-entropy near 0.22289843.
change: Replace equal-weight ten-view inference with the verified weighted eight-view ensemble using 1.10× per-view and 1.178× final sharpening, and increase terminal centered-crop probability from one-fifth to one-third.
mechanism: Qualified confidence-weighted eight-view TTA and centered terminal sampling
evidence_used: Reference Design 1 used these exact settings and achieved the best verified result: 9,210 correct and 0.2228984337 cross-entropy, versus the current design’s 9,206 correct and 0.2270833004 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Lowering each centered-view weight from 0.375 to 0.25 will exceed 9,210 correct predictions by further favoring translation-robust shifted views while preserving the qualified training trajectory and per-view sharpening.
change: Reduce only the original and horizontally flipped centered-view weights, updating the ensemble normalization from 6.75 to 6.50.
mechanism: Reduced centered-view influence in probability-space TTA
evidence_used: The weighted eight-view ensemble achieved 9,210 correct versus 9,206 for the equal-weight ten-view design, while final-scalar calibration cannot alter predictions; this isolates the still-untested centered-view weighting as the next accuracy-sensitive variable.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 81.45047604199499, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22299810676574708, "validation_score": 9210.408831376953}

RECENT RESULT
hypothesis: Input-conditioned recalibration of the final 96 feature channels will exceed 9,210 correct predictions by using global image context to emphasize class-relevant texture and shape channels, while identity initialization and the qualified training/TTA pipeline preserve the strong baseline trajectory.
change: Add a parameter-efficient squeeze/recalibration branch to the final residual stage, initialized as an exact identity, and restore the best-verified 1.178× output calibration; the model remains below the parameter ceiling at 249,002 parameters.
mechanism: Identity-initialized global channel recalibration
evidence_used: The qualified architecture plateaued at 9,210 correct despite extensive TTA tuning, while direct multi-scale feature fusion fell to 9,193; this suggests that merely exposing earlier features is unhelpful, motivating a distinct test of the shared assumption that final-stage channels should have fixed importance for every image.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249002, "training_seconds": 65.44978337502107, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22097557678222657, "validation_score": 9208.409508600751}

RECENT RESULT
hypothesis: Widening the classifier bottleneck from 48 to 52 units will exceed 9,210 correct predictions by increasing learned class-separation capacity while remaining under the 250,000-parameter ceiling.
change: Expand both classifier linear layers consistently to 52 hidden units, producing 249,318 parameters, and restore the verified 1.178× output calibration.
mechanism: Budget-filling classifier bottleneck expansion
evidence_used: Reference Design 1 reached 9,210 correct with the unchanged 48-unit head, while further TTA tuning preserved or reduced accuracy and specialized multi-scale and channel-recalibration additions regressed; this motivates a minimal capacity increase that preserves the qualified feature extractor and training pipeline.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249318, "training_seconds": 75.97553304187022, "validation_accuracy": 0.9204, "validation_correct": 9204, "validation_cross_entropy": 0.2251865119934082, "validation_score": 9204.408101129995}

RECENT RESULT
hypothesis: Splitting the qualified horizontal-view weight equally between left- and right-shifted views will exceed 9,210 correct predictions by removing directional inference bias while preserving the best ensemble’s total horizontal, centered, and vertical weight.
change: Restore the qualified training sampler and calibration, then use ten inference passes with half-weighted left/right shift pairs, full-weighted vertical pairs, and unchanged total normalization.
mechanism: Horizontally symmetric probability-space TTA
evidence_used: Reference Design 1 achieved 9,210 correct with an asymmetric eight-pass ensemble, while the equal-weight ten-pass current design achieved 9,206; preserving the qualified view-group weights isolates horizontal symmetry without the current design’s confounded reweighting.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 53.110233209095895, "validation_accuracy": 0.9215, "validation_correct": 9215, "validation_cross_entropy": 0.22240231246948242, "validation_score": 9215.40903063983}

RECENT RESULT
hypothesis: Increasing each centered-view weight from 0.375 to 0.5 will retain at least 9,215 correct predictions and reduce validation cross-entropy below 0.222402 by better balancing unshifted evidence against translation-robust views.
change: Increase both centered-view ensemble weights to 0.5 and update the normalization from 6.75 to 7.0, leaving training and all shifted-view weights unchanged.
mechanism: Centered-view weight bracketing in symmetric probability-space TTA
evidence_used: Reducing centered weights to 0.25 preserved accuracy but worsened cross-entropy from 0.22289843 to 0.22299811; testing the equally sized upward perturbation is the informative opposite side of that bracket on the new 9,215-correct symmetric ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 63.98163733398542, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.22235396995544435, "validation_score": 9214.409046816461}

RECENT RESULT
hypothesis: Splitting the horizontal-view weight equally between left- and right-shifted views will reproduce at least 9,215 correct predictions, exceeding the current design’s 9,210.
change: Replace the asymmetric eight-pass ensemble with the verified ten-pass symmetric ensemble while preserving total view-group weight and calibration.
mechanism: Horizontally symmetric probability-space test-time augmentation
evidence_used: Reference Design 3 made this isolated change and achieved the best available result: 9,215 correct with 0.2224023125 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The verified symmetric ten-pass ensemble will increase validation_correct from 9,210 to at least 9,215 and achieve cross-entropy near 0.222402.
change: Split the existing horizontal-shift weight equally between left and right views while preserving total ensemble weight, and restore the verified 1.178× calibration.
mechanism: Horizontally symmetric probability-space test-time augmentation
evidence_used: Reference Design 3 made this inference-only change and achieved the best available result: 9,215 correct predictions and 0.2224023125 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing each centered-view weight from 0.375 to 0.25 will exceed 9,215 correct predictions by favoring the translation-robust symmetric shifted views.
change: Install the verified symmetric ten-pass ensemble, reduce both centered-view weights to 0.25, normalize by 6.5, and restore 1.178× calibration.
mechanism: Reduced centered-view influence in symmetric probability-space TTA
evidence_used: Symmetric horizontal weighting improved validation_correct from 9,210 to 9,215; increasing centered weights to 0.5 then lost one correct prediction, motivating the opposite centered-weight perturbation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 64.93181979213841, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.22246544494628906, "validation_score": 9214.40900951603}

RECENT RESULT
hypothesis: Increasing each centered-view weight from 0.375 to 0.4 will preserve all 9,215 correct predictions while reducing validation cross-entropy below 0.222402.
change: Raise both centered-view weights to 0.4 and adjust ensemble normalization from 6.75 to 6.8, leaving training, shifted-view weights, and calibration unchanged.
mechanism: Fine-grained centered-view weight interpolation
evidence_used: Centered weights of 0.25 and 0.5 each lost one correct prediction, while 0.5 reduced cross-entropy to 0.222354; a smaller upward step tests whether that calibration benefit begins before the accuracy boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Centered-view weights of 0.3875 will preserve all 9,215 predictions achieved at 0.375 while slightly reducing cross-entropy toward the improvement observed at 0.5.
change: Restore centered-view influence past the regressed 0.25 setting, using a smaller upward step than the unverified 0.4 experiment, and normalize the ensemble by its new total weight of 6.775.
mechanism: Boundary-seeking centered-view interpolation
evidence_used: Weight 0.375 produced the best result of 9,215 correct at 0.22240231 cross-entropy; 0.5 lost one prediction but improved cross-entropy to 0.22235397, while 0.25 also lost one and worsened cross-entropy. This motivates a conservative interpolation just above 0.375.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Splitting the horizontal-shift contribution equally between left- and right-shifted views will increase validation_correct from 9,210 to at least 9,215 while retaining cross-entropy near 0.222402.
change: Replace the asymmetric six-shift-pass ensemble with the verified symmetric eight-shift-pass ensemble, preserving centered, vertical, total horizontal weights, normalization, and calibration.
mechanism: Horizontally symmetric probability-space test-time augmentation
evidence_used: Reference Design 3 made this isolated change and achieved the best available result: 9,215 correct predictions and 0.2224023125 cross-entropy.
result: the implementation could not be verified

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
