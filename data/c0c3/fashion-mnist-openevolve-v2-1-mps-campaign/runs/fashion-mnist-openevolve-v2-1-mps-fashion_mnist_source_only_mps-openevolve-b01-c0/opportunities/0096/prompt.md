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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 60.72968504205346, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248970108032227, "validation_score": 9290.415803976992}
prior_hypothesis: A 50.6875% live / 49.3125% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248974838256836.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 50.5% live / 49.5% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249082069396973.
change: Shift 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.
mechanism: Conservative live-weight continuation retry
evidence_used: The timed-out 50.5% trial reported 9,290 correct predictions and cross-entropy of 0.202490425491333, improving on the accepted 50.4% result; retrying can establish that gain as a valid design.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 66.1412651669234, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.202490425491333, "validation_score": 9290.4158037265}

RECENT RESULT
hypothesis: A 50.6% live / 49.4% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.202490425491333.
change: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.
mechanism: Conservative live-weight continuation
evidence_used: Every increase from 50.0% through 50.5% live weight preserved 9,290 correct predictions and successively reduced cross-entropy; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next incremental boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 73.67094070813619, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20249003524780274, "validation_score": 9290.41580386144}

RECENT RESULT
hypothesis: A 50.7% live / 49.3% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249003524780274.
change: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.
mechanism: Conservative live-weight continuation
evidence_used: Every increase from 50.0% through 50.6% live weight preserved 9,290 correct predictions and successively reduced cross-entropy; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next incremental boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 54.9877434161026, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20248965530395507, "validation_score": 9289.41580399282}

RECENT RESULT
hypothesis: A 50.65% live / 49.35% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249003524780274.
change: Move the ensemble weight halfway between the successful 50.6% live mixture and the accuracy-losing 50.7% mixture.
mechanism: Binary search of the live/EMA decision boundary
evidence_used: The 50.6% live mixture retained 9,290 correct with cross-entropy 0.20249003524780274, while 50.7% reduced cross-entropy to 0.20248965530395507 but lost one correct prediction; their midpoint is the most informative boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 51.13004658278078, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024898406982422, "validation_score": 9290.415803928714}

RECENT RESULT
hypothesis: A 50.675% live / 49.325% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024898406982422.
change: Move the ensemble weight halfway between the successful 50.65% live mixture and the accuracy-losing 50.7% mixture.
mechanism: Binary search of the live/EMA accuracy boundary
evidence_used: The 50.65% live mixture retained 9,290 correct at 0.2024898406982422 cross-entropy, while 50.7% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative next boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.675% live / 49.325% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024898406982422.
change: Move the ensemble weight halfway between the successful 50.65% live mixture and the accuracy-losing 50.7% mixture.
mechanism: Binary search retry at the live/EMA accuracy boundary
evidence_used: The 50.65% mixture retained 9,290 correct, while 50.7% lost one correct despite lower cross-entropy; the only 50.675% verification timed out, leaving the midpoint unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 73.00771166710183, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248974838256836, "validation_score": 9290.415803960635}

RECENT RESULT
hypothesis: A 50.6875% live / 49.3125% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248974838256836.
change: Move the ensemble weight halfway between the successful 50.675% live mixture and the accuracy-losing 50.7% mixture, retaining the verified temperature.
mechanism: Binary search of the live/EMA accuracy boundary
evidence_used: The 50.675% live mixture retained 9,290 correct at 0.20248974838256836 cross-entropy, while 50.7% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative unresolved boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 60.72968504205346, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248970108032227, "validation_score": 9290.415803976992}

RECENT RESULT
hypothesis: A 50.69375% live / 49.30625% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248970108032227.
change: Move the ensemble weight halfway between the successful 50.6875% live mixture and the accuracy-losing 50.7% mixture while retaining the verified temperature.
mechanism: Binary search of the live/EMA accuracy boundary
evidence_used: The 50.6875% live mixture retained 9,290 correct at 0.20248970108032227 cross-entropy, while 50.7% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative unresolved boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 68.81442895811051, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2024896774291992, "validation_score": 9289.41580398517}



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
