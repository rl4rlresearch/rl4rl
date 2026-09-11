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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 77.07668395782821, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249082069396973, "validation_score": 9290.415803589845}
prior_hypothesis: A 50.4% live / 49.6% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.2024912166595459.

## Recent verification evidence

RECENT RESULT
hypothesis: The next representable float32 temperature below the current value will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20249243125915528.
change: Replace the final arithmetic-ensemble temperature with its exact lower-adjacent float32 value, 0.7171436548233032.
mechanism: Lower-adjacent float32 temperature retry
evidence_used: The preceding lower-adjacent move improved cross-entropy while preserving all predictions; the only verification of this next neighbor exceeded the time limit, so it remains without an accepted objective result.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.38036145805381, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249243545532228, "validation_score": 9290.415803031485}

RECENT RESULT
hypothesis: Weighting the smoothed EMA predictions at 51% and the endpoint model at 49% will either exceed 9,290 correct predictions or preserve them while reducing cross-entropy below 0.20249243125915528.
change: Replace equal live/EMA probability averaging with a normalized 49:51 mixture while retaining the best verified temperature.
mechanism: Slight EMA-favoring probability ensemble
evidence_used: Both adjacent float32 temperatures were worse than the current value, exhausting local scalar calibration; ensemble composition is the next orthogonal refinement, and the EMA represents a smoothed model estimate rather than a single training endpoint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A normalized 49:51 live-to-EMA probability mixture will either exceed 9,290 correct predictions or preserve them while reducing validation cross-entropy below 0.20249243125915528.
change: Compute live and EMA test-time augmentation ensembles separately, then combine their probabilities with 49% live and 51% EMA weight while retaining the best verified temperature.
mechanism: Slight EMA-favoring probability ensemble
evidence_used: Local temperature calibration is exhausted: both adjacent float32 temperatures were worse than the current 0.717143714427948 value. The only 49:51 EMA-favoring trial timed out, so its accuracy and cross-entropy effect remain unresolved and merit a direct retry.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 73.14116308395751, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249674949645996, "validation_score": 9290.415801539762}

RECENT RESULT
hypothesis: Weighting live predictions at 51% and EMA predictions at 49% will exceed 9,290 correct predictions or preserve them while reducing validation cross-entropy below 0.20249243125915528.
change: Compute live and EMA TTA ensembles separately, then combine their probabilities with 51% live and 49% EMA weight while retaining the best verified temperature.
mechanism: Slight live-model-favoring probability ensemble
evidence_used: The symmetric 49% live/51% EMA mixture preserved 9,290 predictions but worsened cross-entropy to 0.20249674949645996, indicating that moving ensemble weight toward EMA is locally harmful and motivating the opposite-side probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 72.09991770796478, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20248852615356444, "validation_score": 9289.415804383265}

RECENT RESULT
hypothesis: A 50.1% live / 49.9% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249243125915528.
change: Compute live and EMA TTA ensembles separately and shift only 0.1 percentage point of weight from EMA to the live model, retaining the best verified temperature.
mechanism: Conservative live-model ensemble-weight interpolation
evidence_used: The 51% live mixture reduced cross-entropy to 0.20248852615356444 but lost one correct prediction, while the 49% live mixture retained 9,290 correct but increased cross-entropy; a much smaller live-favoring interpolation tests for cross-entropy improvement before crossing the observed accuracy boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.1% live / 49.9% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249243125915528.
change: Compute live and EMA TTA probability ensembles separately, combine them with 50.1% live weight and 49.9% EMA weight, and retain the best verified temperature.
mechanism: Conservative live-model ensemble-weight interpolation
evidence_used: The 51% live mixture lowered cross-entropy to 0.20248852615356444 but lost one correct prediction, whereas 49% live retained 9,290 correct but worsened cross-entropy; the prior 50.1% probe timed out, leaving this conservative interpolation unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 72.92021999997087, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249202919006348, "validation_score": 9290.415803171965}

RECENT RESULT
hypothesis: A 50.2% live / 49.8% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249202919006348.
change: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.
mechanism: Conservative live-weight continuation
evidence_used: Moving from 50.0% to 50.1% live preserved 9,290 correct predictions and reduced cross-entropy, while 51.0% live reduced cross-entropy further but crossed an accuracy boundary and lost one correct prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 77.95743499998935, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249161682128905, "validation_score": 9290.415803314556}

RECENT RESULT
hypothesis: A 50.3% live / 49.7% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249161682128905.
change: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.
mechanism: Conservative live-weight continuation
evidence_used: Increasing live weight from 50.0% to 50.1% and then 50.2% preserved 9,290 correct predictions while successively reducing cross-entropy; 51.0% reduced it further but lost one correct prediction, motivating another conservative step toward that boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.3% live / 49.7% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249161682128905.
change: Shift 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.
mechanism: Conservative live-weight continuation retry
evidence_used: Increasing live weight from 50.0% through 50.2% successively reduced cross-entropy while preserving 9,290 correct predictions; the 50.3% trial timed out, so its objective result remains unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 71.98972825007513, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024912166595459, "validation_score": 9290.415803452926}

RECENT RESULT
hypothesis: A 50.4% live / 49.6% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.2024912166595459.
change: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.
mechanism: Conservative live-weight continuation
evidence_used: Increasing live weight from 50.0% through 50.3% successively reduced cross-entropy while preserving 9,290 correct predictions; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next conservative step toward that boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.4% live / 49.6% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.2024912166595459.
change: Shift 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.
mechanism: Conservative live-weight continuation retry
evidence_used: Live weights from 50.0% through 50.3% successively reduced cross-entropy while preserving 9,290 correct predictions; the only 50.4% verification timed out, leaving this next interpolation unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 77.07668395782821, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249082069396973, "validation_score": 9290.415803589845}

RECENT RESULT
hypothesis: A 50.5% live / 49.5% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249082069396973.
change: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.
mechanism: Conservative live-weight continuation
evidence_used: Increasing live weight from 50.0% through 50.4% successively reduced cross-entropy while preserving 9,290 correct predictions; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next conservative step toward that boundary.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 76.4534042919986, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.202490425491333, "validation_score": 9290.4158037265}



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
