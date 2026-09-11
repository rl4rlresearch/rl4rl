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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 76.57101987511851, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059928894043, "validation_score": 9287.414351136853}
prior_hypothesis: A 1.548828125× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067060093.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 70.55100329197012, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670592765808105, "validation_score": 9287.41435115925}
prior_hypothesis: A 1.5578022218542173× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059322.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 55.823656832799315, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670595321655275, "validation_score": 9287.414351150475}
prior_hypothesis: A 1.5544857978820801× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059601.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 73.6388604589738, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670593223571776, "validation_score": 9287.414351157679}
prior_hypothesis: A 1.557131110690534× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059384.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1.5544857978820801× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059601.
change: Replace uniform ten-view averaging with center-biased pooling one eighth of the remaining distance toward the accuracy-losing 1.5625× boundary, retaining the verified 1.22775 calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The verified 1.5533409118652344× weight retained 9,287 correct and achieved the best available cross-entropy, while 1.5625× lost one prediction; prior one-eighth boundary refinements repeatedly improved cross-entropy without changing correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.5544857978820801× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059601.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The verified 1.5533409118652344× weight retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; the prior 1.5544857978820801× verification timed out and supplied no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 55.823656832799315, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670595321655275, "validation_score": 9287.414351150475}

RECENT RESULT
hypothesis: A five-parameter channel-attention gate will exceed 9,287 correct predictions by dynamically emphasizing class-relevant late features for each image, while preserving the baseline computation at initialization and staying below 250,000 parameters.
change: Challenge the assumption that late feature channels should have fixed, sample-independent relevance by adding local cross-channel attention before global mean/max pooling; the zero-initialized residual scaling begins as an exact identity and adds only five learned parameters.
mechanism: Identity-initialized input-conditioned channel recalibration
evidence_used: Repeated center-view weighting refinements plateaued at exactly 9,287 correct and produced only microscopic cross-entropy gains, indicating that evaluation aggregation is saturated. The learned mixed-downsampling attempt timed out, so this tests a genuinely different adaptive representation mechanism only on the compact 7×7 late feature map with negligible overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.55548757314682× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059532.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 2 verified that 1.5544857978820801× retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; advancing one eighth of the remaining interval continues the repeatedly successful conservative boundary search.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.55548757314682× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059532.
change: Replace uniform ten-view averaging with center-biased pooling one eighth closer to the known accuracy-loss boundary, and use the best-verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 2 retained 9,287 correct with the best available cross-entropy at a 1.5544857978820801× center weight, while 1.5625× lost one prediction; the 1.55548757314682× probe timed out and remains the nearest unresolved conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 72.95156274992041, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670594635009765, "validation_score": 9287.414351152833}

RECENT RESULT
hypothesis: A 1.5563641265034675× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059464.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The verified 1.55548757314682× weight retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; another one-eighth advance toward that boundary follows the repeatedly successful refinement sequence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.5563641265034675× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059464.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The verified 1.55548757314682× weight retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; the intermediate 1.5563641265034675× probe timed out and therefore remains unresolved without contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.5563641265034675× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059464.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The verified 1.55548757314682× weight retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; the two 1.5563641265034675× attempts timed out and supplied no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.11326883290894, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2067059383392334, "validation_score": 9287.414351155583}

RECENT RESULT
hypothesis: A 1.557131110690534× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059384.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 3 retained 9,287 correct and achieved the best cross-entropy at 1.5563641265034675×, while 1.5625× lost one prediction; advancing one eighth of the remaining interval follows the repeatedly successful conservative refinement sequence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.557131110690534× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059384.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: The current 1.5563641265034675× weight retained 9,287 correct with the best cross-entropy, while 1.5625× lost one prediction; the 1.557131110690534× verification timed out and supplied no contrary accuracy evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 73.6388604589738, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670593223571776, "validation_score": 9287.414351157679}

RECENT RESULT
hypothesis: A 1.5578022218542173× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059322.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 3 retained 9,287 correct and achieved the best verified cross-entropy at 1.557131110690534×, while 1.5625× lost one prediction; advancing one eighth of the remaining interval continues the refinement sequence with limited accuracy risk.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 70.55100329197012, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20670592765808105, "validation_score": 9287.41435115925}

RECENT RESULT
hypothesis: A 1.5583894441224401× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059277.
change: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.
mechanism: Conservative center-view accuracy-boundary refinement
evidence_used: Reference Design 1 achieved the best verified cross-entropy at 1.5578022218542173× while retaining 9,287 correct; advancing one eighth of the remaining interval toward the accuracy-losing 1.5625× boundary continues the consistently successful refinement sequence with limited accuracy risk.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.13762945798226, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20670592346191408, "validation_score": 9286.414351160693}

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
