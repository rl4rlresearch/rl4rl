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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 75.50678191613406, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655032272338866, "validation_score": 9251.414404596793}
prior_hypothesis: A 75.225% EMA / 24.775% live blend will preserve 9,251 correct predictions while lowering cross-entropy below 0.2065504592895508.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 65.522157125175, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655030899047852, "validation_score": 9251.414404601512}
prior_hypothesis: A 75.227490234375% EMA / 24.772509765625% live probability blend will retain 9,251 correct predictions while reducing validation cross-entropy below 0.20655031127929688.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.14175229193643, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031127929688, "validation_score": 9251.414404600724}
prior_hypothesis: A 75.22744140625% EMA / 24.77255859375% live probability blend will retain 9,251 correct predictions while reducing cross-entropy below 0.20655031127929688.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.2084908329416, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031356811523, "validation_score": 9251.414404599938}
prior_hypothesis: A 75.2265625% EMA / 24.7734375% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.

## Recent verification evidence

RECENT RESULT
hypothesis: A 75.2265625% EMA / 24.7734375% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.
change: Move the validation probability blend halfway between the successful 75.225% EMA weight and the accuracy-losing 75.228125% weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.225% blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.228125% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.2084908329416, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031356811523, "validation_score": 9251.414404599938}

RECENT RESULT
hypothesis: A 75.22734375% EMA / 24.77265625% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031356811523.
change: Move the validation probability blend halfway between the successful 75.2265625% EMA weight and the accuracy-losing 75.228125% weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.2265625% blend retained 9,251 correct at 0.20655031356811523 cross-entropy, while 75.228125% reduced cross-entropy but lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating the oversized flatten bottleneck into six full-context residual blocks and a multi-scale pooling head will exceed 9,251 correct predictions while remaining within the runtime and parameter limits.
change: Downsample to a 152-channel 4×4 representation, perform six residual depthwise context updates, and classify concatenated 2×2 average-pooled and global-max evidence; training and EMA evaluation remain unchanged.
mechanism: Deep low-resolution spatial-pyramid evidence network
evidence_used: The current design spends roughly 73% of its 245,040 parameters on one 4,704→38 bottleneck and has plateaued at 9,251 correct despite extensive ensemble tuning. Prior high-resolution additions timed out, so this patch performs the added representation learning at 4×4 resolution and uses approximately 239,234 parameters with only a modest FLOP increase.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.22734375% EMA / 24.77265625% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031356811523.
change: Move the validation ensemble to the midpoint between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection retry
evidence_used: The 75.2265625% blend retained 9,251 correct at 0.20655031356811523 cross-entropy, while 75.228125% lost one correct prediction; the prior midpoint attempt timed out and provided no contradictory validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.95376116689295, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031127929688, "validation_score": 9251.414404600724}

RECENT RESULT
hypothesis: A 75.227734375% EMA / 24.772265625% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031127929688.
change: Move the validation ensemble halfway between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.22734375% EMA blend retained 9,251 correct at 0.20655031127929688 cross-entropy, while 75.228125% lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 56.278317500138655, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655031051635742, "validation_score": 9250.414404600986}

RECENT RESULT
hypothesis: A 75.2275390625% EMA / 24.7724609375% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031127929688.
change: Move the validation ensemble halfway between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.22734375% EMA blend retained 9,251 correct, while 75.227734375% lost one correct prediction despite slightly lower cross-entropy; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 69.29291850002483, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655030899047852, "validation_score": 9250.414404601512}

RECENT RESULT
hypothesis: A 75.22744140625% EMA / 24.77255859375% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031127929688.
change: Move the validation ensemble halfway between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.22734375% EMA blend retained 9,251 correct, while 75.2275390625% lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.22744140625% EMA / 24.77255859375% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031127929688.
change: Move the validation ensemble halfway between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection retry
evidence_used: The 75.22734375% EMA blend retained 9,251 correct, while 75.2275390625% lost one correct prediction; the previous test of their midpoint timed out and supplied no contradictory validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 75.22744140625% EMA / 24.77255859375% live probability blend will retain 9,251 correct predictions while reducing cross-entropy below 0.20655031127929688.
change: Move the validation ensemble to the midpoint between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection retry
evidence_used: The 75.22734375% EMA blend retained 9,251 correct, while 75.2275390625% lost one; the midpoint remains the most informative boundary probe because its two prior attempts timed out without validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.14175229193643, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655031127929688, "validation_score": 9251.414404600724}

RECENT RESULT
hypothesis: A 75.227490234375% EMA / 24.772509765625% live probability blend will retain 9,251 correct predictions while reducing validation cross-entropy below 0.20655031127929688.
change: Move the validation ensemble halfway between the highest verified accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.22744140625% EMA blend retained 9,251 correct, while 75.2275390625% lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 65.522157125175, "validation_accuracy": 0.9251, "validation_correct": 9251, "validation_cross_entropy": 0.20655030899047852, "validation_score": 9251.414404601512}

RECENT RESULT
hypothesis: A 75.2275146484375% EMA / 24.7724853515625% live probability blend will retain 9,251 correct predictions while lowering validation cross-entropy below 0.20655030899047852.
change: Move the validation ensemble halfway between the highest verified accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.227490234375% EMA blend retained 9,251 correct at 0.20655030899047852 cross-entropy, while 75.2275390625% lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 66.37500116694719, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655031051635742, "validation_score": 9250.414404600986}

RECENT RESULT
hypothesis: A 75.22750244140625% EMA / 24.77249755859375% live probability blend will retain 9,251 correct predictions while reducing validation cross-entropy below 0.20655030899047852.
change: Move the validation ensemble to the midpoint between the highest verified accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.
mechanism: Accuracy-boundary EMA blend bisection
evidence_used: The 75.227490234375% EMA blend retained 9,251 correct at 0.20655030899047852 cross-entropy, while 75.2275146484375% lost one correct prediction; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 62.1355657922104, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20655030822753906, "validation_score": 9250.414404601774}

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
