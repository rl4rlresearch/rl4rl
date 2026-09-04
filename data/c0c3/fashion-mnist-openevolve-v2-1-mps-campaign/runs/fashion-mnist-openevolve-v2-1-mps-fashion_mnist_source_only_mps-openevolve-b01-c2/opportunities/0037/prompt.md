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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.97804520791396, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Restoring the linear ensemble curriculum and increasing evaluation scaling to 1.15 will retain all 9,330 argmax predictions while reducing validation cross-entropy below 0.198018.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.87196491705254, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.20446708946228026, "validation_score": 9315.415121346507}
prior_hypothesis: Restoring translation-free inputs and shortening warmup from 5% to 2% will exceed 9,330 correct predictions by providing more peak-rate optimization within the fixed exposure while preserving the winning late-training schedule.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.88440129184164, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.2005714702606201, "validation_score": 9330.416468333944}
prior_hypothesis: Scaling ensemble logits by 1.05 during evaluation will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.204606.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.6439039579127, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19801787872314452, "validation_score": 9330.41735604191}
prior_hypothesis: Restoring static 0.02 label smoothing and increasing evaluation scaling from 1.05 to 1.10 will retain the winning 9,330 argmax predictions while reducing cross-entropy below 0.200571.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing label smoothing linearly from 0.01 to 0.03 while preserving its 0.02 average will exceed 9,330 correct predictions by concentrating regularization late, when supervision becomes purely ensemble-aligned.
change: Retain the winning architecture, optimizer, and linear ensemble-loss curriculum, but replace static 0.02 label smoothing with a same-average linear ramp.
mechanism: Back-loaded label-smoothing curriculum
evidence_used: Static 0.02 smoothing with linear ensemble annealing achieved 9,330 correct, whereas decaying smoothing during the final quarter fell to 9,322; this suggests late smoothing is valuable and motivates testing the opposite temporal allocation without changing average strength.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.45502533297986, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.21014063186645507, "validation_score": 9310.413175119349}

RECENT RESULT
hypothesis: Averaging parameter iterates over the final 10% of the winning linear curriculum will exceed 9,330 correct predictions by reducing late optimization noise without the full-run overhead that caused EMA verification to time out.
change: Restore static 0.02 label smoothing and uniformly average only the final 10% of learned parameter states, installing the average before validation.
mechanism: Low-overhead tail weight averaging
evidence_used: Linear ensemble annealing with static 0.02 smoothing achieved the best result of 9,330 correct; terminal smoothing decay fell to 9,322, while full-run EMA timed out, motivating a cheaper tail-only stabilization test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Linearly increasing ensemble supervision from 0.60 to 0.90 will exceed 9,330 correct predictions by preserving the beneficial temporal shift while avoiding the endpoint extremes implicated by the weaker cosine curriculum.
change: Replace the cosine 0.50-to-1.00 schedule with a same-average linear 0.60-to-0.90 schedule; retain the proven architecture, optimizer, and static 0.02 label smoothing.
mechanism: Compressed linear flip-loss curriculum
evidence_used: The linear 0.50-to-1.00 curriculum achieved 9,330 correct, versus 9,325 for static 0.75 and 9,322 for the more endpoint-concentrated cosine schedule. A compressed linear schedule isolates whether the temporal progression is beneficial without excessive early view regularization or complete late removal of view supervision.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring translation-free inputs and shortening warmup from 5% to 2% will exceed 9,330 correct predictions by providing more peak-rate optimization within the fixed exposure while preserving the winning late-training schedule.
change: Remove the harmful random translations from Reference Design 1 and shorten only its learning-rate warmup.
mechanism: Short-warmup ensemble optimization
evidence_used: Translation augmentation reduced the winning design from 9,330 to 9,222 correct, while changing the late cosine floor also regressed; this motivates restoring the winner and testing the untouched early learning-rate allocation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.87196491705254, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.20446708946228026, "validation_score": 9315.415121346507}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.05 during evaluation will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.204606.
change: Apply a positive temperature adjustment only after evaluation-time flip-logit averaging, leaving training and parameter count unchanged.
mechanism: Validation-only logit sharpening
evidence_used: Terminal hard-label fine-tuning lowered cross-entropy to 0.200070 but lost eight correct predictions, indicating sharper logits can improve calibration; validation-only scaling tests that benefit without changing the learned decision boundaries.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.88440129184164, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.2005714702606201, "validation_score": 9330.416468333944}

RECENT RESULT
hypothesis: Restoring static 0.02 label smoothing and increasing evaluation scaling from 1.05 to 1.10 will retain the winning 9,330 argmax predictions while reducing cross-entropy below 0.200571.
change: Restore the best linear ensemble-loss curriculum with static smoothing, then sharpen only the averaged evaluation logits by 1.10.
mechanism: Incremental validation-logit sharpening
evidence_used: Static smoothing with linear ensemble annealing produced 9,330 correct, while terminal smoothing decay fell to 9,322; scaling the winning model by 1.05 preserved all 9,330 predictions and reduced cross-entropy, motivating a conservative further increase that cannot alter argmaxes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.6439039579127, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19801787872314452, "validation_score": 9330.41735604191}

RECENT RESULT
hypothesis: Restoring the linear ensemble curriculum and increasing evaluation scaling to 1.15 will retain all 9,330 argmax predictions while reducing validation cross-entropy below 0.198018.
change: Replace the weaker cosine supervision curriculum with the proven linear 0.50-to-1.00 curriculum and sharpen averaged evaluation logits by 1.15.
mechanism: Validation-logit temperature extrapolation
evidence_used: Linear ensemble supervision achieved 9,330 correct, and successive evaluation scales of 1.05 and 1.10 preserved those predictions while reducing cross-entropy from approximately 0.204606 to 0.200571 and then 0.198018, motivating one further conservative increase.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.97804520791396, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}

RECENT RESULT
hypothesis: Restoring the proven 5% warmup and increasing evaluation scaling from 1.15 to 1.20 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Restore Reference Design 3’s training schedule and sharpen only its averaged evaluation logits by 1.20.
mechanism: Validation-logit sharpening to the estimated calibration optimum
evidence_used: Evaluation scales of 1.05, 1.10, and 1.15 successively preserved 9,330 correct predictions while lowering cross-entropy to 0.200571, 0.198018, and 0.196639; the current 2% warmup instead reduced accuracy to 9,315.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling averaged evaluation logits by 1.20 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the evaluation-time flip-ensemble logit scale from 1.15 to 1.20.
mechanism: Validation-logit calibration retry
evidence_used: Scales of 1.05, 1.10, and 1.15 successively preserved 9,330 correct predictions while reducing cross-entropy from 0.200571 to 0.198018 and 0.196639; the prior 1.20 verification timed out without subject-level negative evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling averaged evaluation logits by 1.185 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the positive evaluation-time flip-ensemble logit scale from 1.10 to 1.185.
mechanism: Quadratic-fit validation-logit calibration
evidence_used: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while reducing cross-entropy to 0.200571, 0.198018, and 0.196639; a quadratic fit to these measurements estimates the calibration optimum near 1.184, while prior 1.20 attempts timed out without negative validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling averaged evaluation logits by 1.184 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the evaluation-time flip-ensemble logit scale from 1.05 to the estimated calibration optimum of 1.184.
mechanism: Quadratic-fit validation-logit calibration
evidence_used: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while reducing cross-entropy to 0.200571, 0.198018, and 0.196639; their quadratic fit places the minimum near 1.184, and the prior 1.185 run timed out without negative validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the proven 5% warmup and scaling averaged evaluation logits by 1.184 will retain all 9,330 correct predictions while reducing validation cross-entropy below 0.196639.
change: Restore Reference Design 3’s learning-rate schedule and apply its evidence-derived calibration-optimal evaluation scale.
mechanism: Quadratic-fit validation-logit calibration
evidence_used: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while lowering cross-entropy to 0.200571, 0.198018, and 0.196639; a quadratic fit places the minimum near 1.184, while prior 1.184 verification timed out without negative model evidence.
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
