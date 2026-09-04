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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 80.91814229195006, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.229253133392334, "validation_score": 9258.406751047785}
prior_hypothesis: Approximating the proven tail-EMA BatchNorm statistics from their midpoint and final values will retain at least 9,255 correct predictions without the runtime cost of updating buffer averages throughout training.

## Recent verification evidence

RECENT RESULT
hypothesis: Applying one-pixel translations to roughly half of training images will exceed 9,258 correct predictions by improving local shift invariance without changing the model, optimizer-step count, or evaluation path.
change: Add vectorized, training-only random crops from replicate-padded images, retaining the original position for half the batch.
mechanism: Mild per-image translation augmentation
evidence_used: The current batch-32 EMA design achieved 9,258 correct, while 0.05 label smoothing outperformed both hard targets and stronger 0.10 smoothing; this motivates another mild, parameter-free regularizer rather than a larger architectural or computational change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will retain exactly 9,258 correct predictions while reducing validation cross-entropy below 0.229253.
change: Increase the evaluation-only logit multiplier from 1.1 to 1.2 without affecting training, parameters, or predicted classes.
mechanism: Evaluation-only logit sharpening
evidence_used: A 1.1 multiplier previously reduced cross-entropy from 0.241946 to 0.226035 without changing argmax predictions; the prior 1.2 verification timed out but provided no contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will retain exactly 9,258 correct predictions while reducing validation cross-entropy below 0.229253.
change: Increase the inference-only logit multiplier from 1.1 to 1.2 without affecting training, parameters, or predicted classes.
mechanism: Evaluation-only logit sharpening
evidence_used: A 1.1 multiplier previously reduced cross-entropy from 0.241946 to 0.226035 without changing argmax predictions; the prior 1.2 attempt timed out and therefore provided no contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging original and horizontally reflected evaluation logits will exceed 9,258 correct predictions by reducing spatial-orientation sensitivity without adding training work.
change: Keep training unchanged, but evaluate each image together with its horizontal reflection and average their logits before applying the existing calibration.
mechanism: Two-view reflection logit ensemble
evidence_used: The current EMA design reached 9,258 correct, while the translation-augmentation attempt identified spatial invariance as promising but timed out; evaluation-only ensembling tests that mechanism without changing the 100,000-example training path.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.15 will preserve exactly 9,258 correct predictions while reducing validation cross-entropy below 0.229253.
change: Increase the inference-only logit multiplier from 1.1 to 1.15 without affecting training, parameters, runtime, or predicted classes.
mechanism: Conservative evaluation-logit calibration
evidence_used: A 1.1 multiplier previously reduced cross-entropy from 0.241946 to 0.226035 without changing argmax predictions; the unverified 1.2 attempts provide no contradictory evidence, motivating a smaller intermediate calibration step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.05 to 0.04 will exceed 9,258 correct predictions by retaining beneficial regularization while slightly reducing target bias, with no added runtime.
change: Change only the cross-entropy label-smoothing coefficient from 0.05 to 0.04.
mechanism: Refined low-strength label smoothing
evidence_used: The 0.05 setting outperformed both hard targets and 0.10 smoothing, establishing a favorable low-strength region; refining it locally is runtime-neutral, unlike recent augmentation and architecture changes that timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing from 0.05 to 0.03 will exceed 9,258 correct predictions by preserving early regularization while reducing late target bias.
change: Replace fixed 0.05 label smoothing with a runtime-neutral linear schedule from 0.05 to 0.03.
mechanism: Mild label-smoothing annealing
evidence_used: Fixed 0.05 outperformed hard targets and 0.10 smoothing; the runtime-neutral 0.04 attempt timed out without contradictory accuracy evidence, motivating a schedule centered near that low-strength region.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.05 to 0.04 will exceed 9,258 correct predictions by slightly reducing target bias while retaining the established benefit of mild regularization.
change: Change only the training cross-entropy label-smoothing coefficient from 0.05 to 0.04.
mechanism: Low-strength label-smoothing refinement
evidence_used: Fixed 0.05 smoothing outperformed both hard targets and 0.10 smoothing; the prior 0.04 attempt timed out, so it supplied no contradictory accuracy evidence and remains the closest runtime-neutral refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the three spatial GELUs with ReLUs will finish more reliably while retaining or exceeding 9,258 correct predictions through faster, canonical BatchNorm-activated convolution blocks.
change: Use ReLU activations in the convolutional feature extractor while preserving the architecture, classifier head, optimizer, EMA, and label smoothing.
mechanism: BatchNorm–ReLU spatial feature extraction
evidence_used: The current design reached 9,258 correct but required 80.92 seconds, while even runtime-neutral variants repeatedly timed out; ReLU reduces activation overhead without narrowing features or changing parameter count, unlike the timed-out depthwise-separable redesign.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 68.65323479217477, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.23810767669677735, "validation_score": 9235.40384209662}

RECENT RESULT
hypothesis: Replacing only the highest-resolution GELU with ReLU will reduce training time while retaining at least 9,258 correct predictions because the deeper smooth activations remain unchanged.
change: Use ReLU after the first convolution while preserving both deeper GELUs, the classifier, optimizer, EMA, and loss.
mechanism: Hybrid ReLU–GELU feature extraction
evidence_used: Replacing all three spatial GELUs with ReLUs reduced training time from 80.92 to 68.65 seconds but lost 23 correct predictions; changing only the most computationally expensive GELU tests whether most of that speedup is available without the accuracy loss caused by removing every smooth activation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 80.26570141711272, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.23115580024719237, "validation_score": 9233.406122441937}

RECENT RESULT
hypothesis: Keeping the first GELU while replacing the two deeper GELUs with ReLU will retain at least 9,258 correct predictions and reduce training time below 75 seconds.
change: Preserve the accuracy-critical first GELU and use ReLU only after the second and third BatchNorm layers.
mechanism: Front-loaded smooth activation with faster deep ReLU blocks
evidence_used: Replacing all GELUs with ReLU reduced runtime to 68.65 seconds, while replacing only the first GELU produced nearly the same accuracy loss as replacing all three (9,233 versus 9,235 correct) and retained the slow 80.27-second runtime. This isolates the first GELU as accuracy-sensitive and the deeper GELUs as the likely runtime cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Approximate GELUs will retain at least 9,258 correct predictions while providing enough runtime headroom to finish reliably, and 1.2× inference sharpening will lower cross-entropy without changing argmax classes.
change: Use PyTorch’s faster tanh GELU approximation throughout the network and raise the evaluation-only logit multiplier from 1.1 to 1.2.
mechanism: Tanh-approximated GELU with calibrated inference logits
evidence_used: Exact GELUs achieved 9,258 correct but took 80.92 seconds, whereas spatial ReLUs finished in 68.65 seconds but lost 23 predictions; tanh GELU preserves the smooth activation closely. Separately, 1.1× sharpening reduced cross-entropy from 0.241946 to 0.226035 without changing predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 68.62904162495397, "validation_accuracy": 0.9226, "validation_correct": 9226, "validation_cross_entropy": 0.22252757110595703, "validation_score": 9226.408988731066}



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
