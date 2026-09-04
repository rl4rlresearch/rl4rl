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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.96161983301863, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670600929260255, "validation_score": 9287.41435113122}
prior_hypothesis: A 1.22775 evaluation-logit scale will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2067060364.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.3550312500447, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20673983764648438, "validation_score": 9287.414339515777}
prior_hypothesis: Scaling the linear-recency ten-view logits by 1.225 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2071991.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 76.57101987511851, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059928894043, "validation_score": 9287.414351136853}
prior_hypothesis: A 1.548828125× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067060093.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.66874208394438, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670596809387207, "validation_score": 9287.414351145366}
prior_hypothesis: A 1.552032470703125× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059761.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1.55078125× center-offset weight with the calibrated 1.22775 logit scale will retain 9,287 correct predictions while reducing validation cross-entropy below 0.2067060093.
change: Restore the best calibrated scale and move the center weight one quarter of the way from the verified-safe 1.546875× setting toward the accuracy-losing 1.5625× setting.
mechanism: Conservative accuracy-boundary refinement for center-biased view pooling
evidence_used: A 1.546875× center weight retained 9,287 correct, while 1.5625× lost one prediction despite lower cross-entropy; 1.55078125× is a conservative unresolved boundary probe, and 1.22775 is the best verified calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling the centered crop with the same 1.546875× prior used by the best verified evaluation pool will exceed 9,287 correct predictions by emphasizing validation-aligned evidence during learning while preserving shifted-view robustness.
change: Restore the best verified 1.546875× center-biased pooling and 1.22775 calibration, then match that center bias in training augmentation.
mechanism: Train–evaluation view-prior alignment
evidence_used: Center-biased evaluation retained 9,287 correct and reduced cross-entropy to 0.2067060093, while stronger evaluation-only bias lost a prediction; aligning the training distribution to the proven safe bias is the most direct untested extension.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling training crops with the evaluation pool’s exact 1.546875× center prior will exceed 9,287 correct predictions by emphasizing validation-aligned examples while retaining substantial shifted-view augmentation.
change: Replace uniform crop-position sampling with a low-overhead 99:64:64:64:64 integer-bucket sampler; preserve the best verified evaluation pooling and calibration.
mechanism: Integer-bucketed train–evaluation view-prior alignment
evidence_used: The 1.546875× center-biased evaluation pool retained 9,287 correct and improved cross-entropy; the prior training-alignment attempt timed out, so its accuracy effect remains untested, and integer bucketing avoids multinomial-sampling overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the center-biased pooled logits by 1.22775 will preserve all 9,287 correct predictions while reducing validation cross-entropy below 0.2067060364.
change: Increase only the evaluation-logit scale from 1.227325 to the best-verified 1.22775.
mechanism: Verified local temperature-calibration refinement
evidence_used: Reference Design 2 used identical 1.546875× center-biased pooling with a 1.22775 scale and achieved 9,287 correct at 0.2067060093 cross-entropy, strictly improving the current design.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.28837712504901, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670600929260255, "validation_score": 9287.41435113122}

RECENT RESULT
hypothesis: Using the verified 1.546875× center weight and 1.22775 logit scale will retain 9,287 correct predictions while reducing cross-entropy below the current 0.2067077972.
change: Increase the centered-view pooling weight from 1.5 to 1.546875 with matching normalization, and increase the final evaluation-logit scale from 1.225 to 1.22775.
mechanism: Verified center-biased ten-view pooling with temperature calibration
evidence_used: Reference Designs 2 and 3 used this exact evaluation rule and achieved 9,287 correct at 0.2067060093 cross-entropy, strictly improving the current design with the same model and training procedure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 77.38550650002435, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670600929260255, "validation_score": 9287.41435113122}

RECENT RESULT
hypothesis: Learning a per-channel blend of max and average pooling will exceed 9,287 correct predictions by reducing one-pixel phase sensitivity while preserving peak-sensitive channels.
change: Replace both fixed max-pooling stages with channel-wise learned mixed pooling, reduce the classifier width by one unit to remain below 250,000 parameters, and retain the best-verified center-biased evaluation pooling and calibration.
mechanism: Channel-adaptive mixed downsampling
evidence_used: Shift-view and center-weight refinements plateaued at 9,287 correct, showing that post-hoc aggregation is saturated. All available architectures assume winner-take-all max downsampling for every channel; learning which channels require smoother averaging directly targets the translation sensitivity that motivated the ten-view evaluation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Preserving separate upper- and lower-image feature averages while retaining global peak features will exceed 9,287 correct predictions by capturing class-relevant vertical layout discarded by global mean pooling.
change: Replace global mean features with 2×1 vertical pooled features and resize the classifier from 192→61 to 288→40, yielding approximately 249,730 learned parameters while retaining the verified evaluation ensemble.
mechanism: Parameter-neutral vertical spatial pooling head
evidence_used: Evaluation-pooling refinements plateaued at 9,287 correct, while the coarse-spatial-head attempt could not be verified; this tests the unresolved spatial-layout hypothesis without its added 1×1 projection and preserves more hidden width than a full 2×2 representation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 1.548828125× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067060093.
change: Move the center weight one eighth of the way from the verified-safe 1.546875 setting toward the accuracy-losing 1.5625 setting, retaining the best-verified logit scale.
mechanism: Conservative center-view boundary refinement
evidence_used: The 1.546875 weight achieved 9,287 correct, while 1.5625 reduced cross-entropy but lost one prediction; smaller intermediate probes were unverified only because of time limits, so this low-cost conservative probe targets a score improvement without changing training.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 76.57101987511851, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059928894043, "validation_score": 9287.414351136853}

RECENT RESULT
hypothesis: A 1.550537109375× center-view weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.2067059929.
change: Increase the center-view pooling weight one eighth of the remaining distance from the best verified-safe setting toward the accuracy-losing setting, retaining the calibrated 1.22775 logit scale.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: A 1.548828125× center weight retained 9,287 correct at 0.2067059929 cross-entropy, while 1.5625× lost one prediction; the nearby 1.55078125 probe timed out and provided no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 68.773385041859, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670597610473632, "validation_score": 9287.414351142615}

RECENT RESULT
hypothesis: A 1.552032470703125× center-view weight with 1.22775 logit scaling will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059761.
change: Replace uniform ten-view averaging with center-biased pooling one eighth of the remaining distance toward the accuracy-losing 1.5625× boundary, using the best-verified calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: A 1.550537109375× center weight retained 9,287 correct and improved cross-entropy to 0.2067059761, while 1.5625× lost one prediction; the same conservative one-eighth refinement previously improved the score.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.552032470703125× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059761.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The current 1.550537109375× weight retained 9,287 correct and improved cross-entropy, while 1.5625× lost one prediction; the intermediate 1.552032470703125× probe produced no accuracy evidence because verification timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.66874208394438, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670596809387207, "validation_score": 9287.414351145366}

RECENT RESULT
hypothesis: A 1.5533409118652344× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059681.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The verified 1.552032470703125× weight retained 9,287 correct and improved cross-entropy to 0.2067059681, while 1.5625× lost one prediction; advancing one eighth of the remaining interval continues the previously successful conservative boundary search.
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
