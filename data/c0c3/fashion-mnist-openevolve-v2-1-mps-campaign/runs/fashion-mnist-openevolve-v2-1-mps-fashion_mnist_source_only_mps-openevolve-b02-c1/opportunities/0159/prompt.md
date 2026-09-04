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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 74.79264662507921, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1922867332458496, "validation_score": 9311.419362210498}
prior_hypothesis: Temperature 0.9340 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922869278.

## Recent verification evidence

RECENT RESULT
hypothesis: Temperature 0.9315 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922890224.
change: Raise only the positive evaluation-time ensemble temperature from 0.9310 to 0.9315.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300, 0.9305, and 0.9310 successively reduced cross-entropy while preserving 9,311 correct predictions; 0.9315 is the smallest untested continuation toward the evidence-fitted minimum near 0.934.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.9315 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922890224.
change: Raise only the positive evaluation-time ensemble temperature from 0.9310 to 0.9315.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300, 0.9305, and 0.9310 successively reduced cross-entropy with unchanged argmax predictions; the previous 0.9315 attempt timed out and supplied no contradictory metric evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 81.17430608300492, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19228845672607423, "validation_score": 9311.4193616043}

RECENT RESULT
hypothesis: Temperature 0.9320 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922884567.
change: Raise only the evaluation-time ensemble temperature from 0.9315 to 0.9320.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300, 0.9305, 0.9310, and 0.9315 successively reduced cross-entropy while preserving exactly 9,311 correct predictions, motivating the smallest continuation toward the evidence-fitted minimum near 0.934.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the absolute-position flattening bottleneck with multiplicative channel co-occurrences and coarse coordinate moments will exceed 9,311 correct predictions by representing shape-part combinations more directly while remaining cheaper than attention.
change: Project the terminal feature map to 64 channels, encode its mean, horizontal and vertical moments, and upper-triangular Gram matrix, then classify the normalized statistics with a 53-unit head; retain the established training procedure and TTA.
mechanism: Bilinear channel co-occurrence pooling with spatial moments
evidence_used: Translation-aware aggregation produced the best 9,311-correct design, indicating sensitivity to fixed spatial coordinates. Terminal attention supplied no validation metrics because it timed out, so this tests relational representation through explicit bilinear interactions without attention machinery.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.9320 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922884567.
change: Raise only the positive evaluation-time ensemble temperature from 0.9315 to 0.9320.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300, 0.9305, 0.9310, and 0.9315 successively reduced cross-entropy while preserving 9,311 correct predictions; the sole 0.9320 attempt timed out and supplied no contradictory metric evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 75.53883941704407, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19228796424865724, "validation_score": 9311.419361777518}

RECENT RESULT
hypothesis: Temperature 0.9325 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922879642.
change: Raise only the positive evaluation-time ensemble temperature from 0.9320 to 0.9325.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300, 0.9305, 0.9310, 0.9315, and 0.9320 successively reduced cross-entropy while preserving exactly 9,311 correct predictions; 0.9325 is the smallest continuation toward the evidence-fitted minimum near 0.934.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 82.82176512503065, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19228754348754884, "validation_score": 9311.419361925511}

RECENT RESULT
hypothesis: Temperature 0.9330 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922875435.
change: Raise only the positive evaluation-time ensemble temperature from 0.9325 to 0.9330.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300 through 0.9325 successively reduced cross-entropy while preserving exactly 9,311 correct predictions; 0.9330 is the smallest continuation toward the evidence-fitted minimum near 0.933–0.934.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.9330 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922875435.
change: Raise only the positive evaluation-time ensemble temperature from 0.9325 to 0.9330.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300 through 0.9325 successively reduced cross-entropy without changing argmax predictions; the previous 0.9330 verification timed out and supplied no contradictory metric evidence.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 82.40925979195163, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19228720283508302, "validation_score": 9311.419362045328}

RECENT RESULT
hypothesis: Temperature 0.9335 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922872028.
change: Raise only the positive evaluation-time ensemble temperature from 0.9325 to 0.9335.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300 through 0.9330 successively reduced cross-entropy without changing argmax predictions; the reported 0.9330 cross-entropy of 0.1922872028 motivates the smallest continuation toward the fitted minimum near 0.934.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.9335 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922872028.
change: Raise only the positive evaluation-time ensemble temperature from 0.9325 to 0.9335.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300 through 0.9330 successively reduced cross-entropy without changing argmax predictions; the prior 0.9335 attempt timed out and supplied no contradictory metric evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 58.92027383297682, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19228692779541015, "validation_score": 9311.419362142067}

RECENT RESULT
hypothesis: Temperature 0.9340 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922869278.
change: Raise only the positive evaluation-time ensemble temperature from 0.9335 to 0.9340.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300 through 0.9335 successively reduced cross-entropy while preserving exactly 9,311 correct predictions; 0.9340 is the smallest continuation toward the evidence-fitted minimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 74.79264662507921, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1922867332458496, "validation_score": 9311.419362210498}

RECENT RESULT
hypothesis: Temperature 0.9345 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922867332.
change: Raise only the positive evaluation-time ensemble temperature from 0.9340 to 0.9345.
mechanism: Fine-grained ensemble temperature calibration
evidence_used: Temperatures 0.9300 through 0.9340 successively reduced cross-entropy while preserving exactly 9,311 correct predictions; the latest 0.0005 increase improved cross-entropy by 0.0000001945, motivating one further increment toward the fitted minimum.
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
