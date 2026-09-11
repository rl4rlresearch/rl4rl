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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 62.796055833110586, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.216883642578125, "validation_score": 9258.41088562826}
prior_hypothesis: Scaling evaluation logits by 1.24 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2180025.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.24 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2180025.
change: Increase only the evaluation-time logit multiplier from 1.22 to 1.24.
mechanism: Incremental inference-logit calibration
evidence_used: Increasing the scale from 1.20 to 1.22 preserved 9,258 correct predictions and reduced cross-entropy from 0.2193095 to 0.2180025; the previous 1.24 verification timed out and therefore provided no contradictory calibration evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 62.796055833110586, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.216883642578125, "validation_score": 9258.41088562826}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.26.
mechanism: Incremental inference-logit calibration
evidence_used: Increasing the multiplier from 1.20 to 1.22 and then 1.24 preserved all 9,258 correct predictions while cross-entropy fell from 0.2193095 to 0.2180025 and then 0.2168836, supporting another small upward calibration step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the evaluation-time logit multiplier from 1.24 to 1.26.
mechanism: Incremental inference-logit calibration
evidence_used: Increases from 1.20 to 1.22 and 1.24 preserved all 9,258 correct predictions while cross-entropy fell from 0.2193095 to 0.2180025 and 0.2168836; the prior 1.26 verification timed out and supplied no contradictory calibration result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.26.
mechanism: Incremental inference-logit calibration
evidence_used: Increasing the scale from 1.20 to 1.22 and 1.24 preserved all 9,258 predictions while cross-entropy successively fell from 0.2193095 to 0.2180025 and 0.2168836; two 1.26 attempts timed out without producing contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.25 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to the untested midpoint 1.25.
mechanism: Midpoint inference-logit calibration
evidence_used: Increasing the scale from 1.20 through 1.22 to 1.24 preserved all 9,258 correct predictions while cross-entropy decreased at each step; repeated 1.26 runs timed out without contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.26.
mechanism: Incremental inference-logit calibration
evidence_used: Increasing the scale from 1.20 to 1.22 and 1.24 preserved all 9,258 correct predictions while cross-entropy successively fell from 0.2193095 to 0.2180025 and 0.2168836; prior 1.26 attempts timed out without producing contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the evaluation-time logit multiplier from 1.24 to 1.26.
mechanism: Incremental inference-logit calibration
evidence_used: Each completed increase from 1.20 to 1.22 and 1.24 preserved 9,258 correct predictions while reducing cross-entropy from 0.2193095 to 0.2180025 and then 0.2168836; the 1.26 attempts timed out without contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.26 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the evaluation-time logit multiplier from 1.24 to 1.26.
mechanism: Incremental inference-logit calibration
evidence_used: Completed increases from 1.20 to 1.22 and 1.24 preserved all 9,258 correct predictions while cross-entropy successively decreased from 0.2193095 to 0.2180025 and 0.2168836; prior 1.26 attempts timed out without producing contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits to 1.32 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.32.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Cross-entropy decreased from 0.2193095 at 1.20× to 0.2180025 at 1.22× and 0.2168836 at 1.24× without changing correctness; a local quadratic extrapolation from these completed results places the minimum near 1.35×.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging original and horizontally flipped evaluation logits will increase validation_correct above 9,258 while retaining the calibrated 1.24 logit scale.
change: During evaluation, classify each image and its horizontal reflection in one concatenated batch, average their logits, and apply the existing calibration; training remains unchanged.
mechanism: Horizontal-reflection test-time ensembling
evidence_used: Every completed logit-scaling change preserved exactly 9,258 predictions and improved only the tie-breaker, so the next informative change should target the primary correct-count objective without altering the verified training procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.35 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the positive evaluation-time logit multiplier from 1.24 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Cross-entropy fell at every completed increase—0.2193095 at 1.20×, 0.2180025 at 1.22×, and 0.2168836 at 1.24×—without changing correctness; quadratic extrapolation from those results places the estimated minimum near 1.35×, while the timed-out 1.32× attempt supplied no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.35 evaluation multiplier will preserve exactly 9,258 argmax predictions while reducing validation cross-entropy below 0.2168836.
change: Increase only the evaluation-time logit multiplier from 1.24 to 1.35.
mechanism: Quadratic-guided evaluation-logit calibration
evidence_used: Completed scales of 1.20, 1.22, and 1.24 preserved all predictions while cross-entropy fell from 0.2193095 to 0.2180025 to 0.2168836; quadratic interpolation places the estimated minimum near 1.35, and timed-out attempts supplied no contradictory validation result.
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
