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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 82.31951187504455, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385795364379882, "validation_score": 9359.418810293531}
prior_hypothesis: A 3.0194091796875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938580013.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the parameter EMA update to 3.75% while retaining the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by combining the stronger parameter smoothing associated with 0.193577 cross-entropy and the buffer horizon responsible for the current accuracy gain.
change: Change only the parameter EMA update rate from 4% to 3.75%; preserve the 3% floating-buffer update and all other behavior.
mechanism: Decoupled weight-and-buffer EMA refinement
evidence_used: The coherent 3.75% EMA produced the lowest observed cross-entropy, 0.193577, but lost one correct prediction; independently slowing buffer updates to 3% raised the best result to 9,359 correct, so decoupling these rates directly tests whether their benefits combine.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.93516691587865, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19377296028137206, "validation_score": 9358.418840111675}

RECENT RESULT
hypothesis: A 4.25% parameter EMA update with the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly improving weight responsiveness without disturbing the BatchNorm smoothing responsible for the current accuracy gain.
change: Increase only the parameter EMA update rate from 4% to 4.25%, preserving the 3% floating-buffer update and all other behavior.
mechanism: Upper-side decoupled parameter-EMA refinement
evidence_used: Reducing the parameter update to 3.75% with a 3% buffer update regressed from 9,359 to 9,358 correct; the symmetric upper-side refinement is the closest informative untested direction around the best 4% parameter rate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 4.125% parameter EMA update with the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly increasing weight responsiveness without disturbing BatchNorm smoothing.
change: Increase only the parameter EMA update rate from 4% to 4.125%.
mechanism: Conservative upper-side parameter-EMA refinement
evidence_used: Reducing the parameter update to 3.75% regressed to 9,358 correct, while 4% achieved 9,359; the 4.25% test timed out, making 4.125% the closest evaluable upper-side refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.03125% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by moving toward the better-calibrated 3.0625% rate without crossing its observed accuracy boundary.
change: Keep the parameter EMA update at 4% and increase only the floating-buffer EMA update from 3% to 3.03125%.
mechanism: Fine-grained BatchNorm-buffer EMA boundary refinement
evidence_used: The 3% buffer update achieved the best score with 9,359 correct, while 3.0625% lowered cross-entropy to 0.193843 but lost one correct prediction; their midpoint is the closest untested boundary refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.015625% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by moving slightly toward the better-calibrated 3.0625% rate without crossing its observed accuracy boundary.
change: Keep the parameter EMA update at 4% and increase only the floating-buffer EMA update from 3% to 3.015625%.
mechanism: Conservative BatchNorm-buffer EMA boundary refinement
evidence_used: The 3% update achieved the best score with 9,359 correct, while 3.0625% lowered cross-entropy to 0.193843 but lost one prediction; after the 3.03125% midpoint timed out, 3.015625% is the closest conservative refinement between the verified endpoints.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 79.72820237511769, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385929527282714, "validation_score": 9359.418809822882}

RECENT RESULT
hypothesis: A 3.0234375% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.1938593 by moving toward the better-calibrated higher-rate region without reaching the observed accuracy-loss boundary.
change: Keep the parameter EMA update at 4% and increase only the floating-buffer EMA update from 3.015625% to 3.0234375%.
mechanism: Conservative BatchNorm-buffer EMA boundary refinement
evidence_used: Raising the buffer update from 3% to 3.015625% preserved 9,359 correct and improved cross-entropy from approximately 0.193865 to 0.1938593, while 3.0625% lost one prediction; bisecting toward the uncompleted 3.03125% trial is the closest informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.23025304102339, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19385652961730956, "validation_score": 9358.418810793086}

RECENT RESULT
hypothesis: A 3.01953125% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.1938593.
change: Increase only the floating-buffer EMA update from 3.015625% to 3.01953125%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: The current 3.015625% rate achieved 9,359 correct, while 3.0234375% lowered cross-entropy but lost one prediction; their midpoint is the most informative untested boundary refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.017578125% buffer EMA update will preserve 9,359 correct predictions while lowering cross-entropy below 0.1938592953.
change: Increase only the floating-buffer EMA update from 3.015625% to 3.017578125%, preserving the 4% parameter EMA and all other behavior.
mechanism: Conservative BatchNorm-buffer EMA boundary bisection
evidence_used: The current 3.015625% rate achieved 9,359 correct, while 3.0234375% lowered cross-entropy but lost one prediction; after the intervening 3.01953125% trial timed out, this conservative midpoint is the closest untested refinement on the accuracy-preserving side.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.27313916687854, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938586181640625, "validation_score": 9359.418810060415}

RECENT RESULT
hypothesis: A 3.0185546875% buffer EMA update will preserve 9,359 correct predictions while lowering cross-entropy below 0.1938586182.
change: Increase only the floating-buffer EMA update from 3.017578125% to 3.0185546875%, preserving the 4% parameter EMA and all other behavior.
mechanism: Conservative BatchNorm-buffer EMA boundary bisection
evidence_used: The current 3.017578125% rate preserved 9,359 correct with the best cross-entropy; 3.0234375% lowered cross-entropy but lost one prediction, while 3.01953125% timed out. Bisecting the remaining conservative interval is the closest informative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 68.76120012509637, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1938582649230957, "validation_score": 9359.418810184334}

RECENT RESULT
hypothesis: A 3.01904296875% buffer EMA update will preserve 9,359 correct predictions while lowering cross-entropy below 0.1938582649.
change: Increase only the floating-buffer EMA update from 3.0185546875% to 3.01904296875%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: The current 3.0185546875% rate preserved 9,359 correct and improved cross-entropy, while 3.0234375% lost one prediction and the intervening 3.01953125% trial timed out; bisecting the remaining conservative interval is the closest informative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 65.6914023749996, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385809173583984, "validation_score": 9359.418810245088}

RECENT RESULT
hypothesis: A 3.019287109375% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938580917.
change: Increase only the floating-buffer EMA update from 3.01904296875% to 3.019287109375%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: The current 3.01904296875% rate achieved 9,359 correct with the best cross-entropy, while 3.0234375% lost one prediction; bisecting toward the timed-out 3.01953125% trial is the closest conservative boundary refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 72.55989454197697, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385800132751466, "validation_score": 9359.418810276804}

RECENT RESULT
hypothesis: A 3.0194091796875% buffer EMA update will preserve 9,359 correct predictions while lowering validation cross-entropy below 0.1938580013.
change: Increase only the floating-buffer EMA update from 3.019287109375% to 3.0194091796875%, preserving the 4% parameter EMA and all other behavior.
mechanism: Fine-grained BatchNorm-buffer EMA boundary bisection
evidence_used: The current 3.019287109375% rate achieved 9,359 correct with the best cross-entropy, while 3.0234375% lost one prediction; bisecting toward the timed-out 3.01953125% trial is the closest conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 82.31951187504455, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19385795364379882, "validation_score": 9359.418810293531}



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
