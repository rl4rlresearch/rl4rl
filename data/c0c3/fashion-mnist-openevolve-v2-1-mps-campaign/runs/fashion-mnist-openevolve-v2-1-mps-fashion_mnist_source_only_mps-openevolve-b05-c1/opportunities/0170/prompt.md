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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 60.546387874986976, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.18585695190429688, "validation_score": 9360.42163601537}
prior_hypothesis: An evaluation-time logit scale of 1.0671112 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.1858569564819336.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the EMA decay cap from 0.98 to 0.99 will exceed 9,360 correct predictions by smoothing more of the low-learning-rate final trajectory while preserving the verified architecture and unaveraged BatchNorm statistics.
change: Change only the parameter EMA decay cap from 0.98 to 0.99.
mechanism: Longer-horizon late-training parameter EMA
evidence_used: The verified 0.98 design achieved 9,360 correct; architecture, loss, probability-TTA, and BatchNorm-buffer changes all performed worse. The sole 0.99 attempt timed out and therefore provides no contrary performance evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 51.68130358308554, "validation_accuracy": 0.9349, "validation_correct": 9349, "validation_cross_entropy": 0.18705469245910644, "validation_score": 9349.421210583789}

RECENT RESULT
hypothesis: Reducing the EMA decay cap from 0.98 to 0.975 will exceed 9,360 correct predictions by keeping averaged parameters better aligned with the unaveraged BatchNorm statistics.
change: Change only the parameter EMA decay cap, preserving the verified architecture, augmentation, schedule, TTA, and calibration.
mechanism: Shorter-horizon late-training parameter EMA
evidence_used: Increasing the cap to 0.99 reduced validation_correct from 9,360 to 9,349, while averaging BatchNorm buffers reduced it to 9,351; a modestly shorter parameter-average horizon directly tests the favorable direction without reintroducing buffer averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 74.66341116698459, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.1856760757446289, "validation_score": 9357.421700336397}

RECENT RESULT
hypothesis: Replacing both phase-sensitive hard max pools with learned channel-wise mixtures of max pooling and trainable low-pass downsampling will exceed 9,360 correct predictions by learning more translation-stable intermediate representations while retaining sharp discriminative responses.
change: Add depthwise low-pass filters initialized to binomial blur kernels, learn a per-channel mixture with ordinary max pooling, and use these modules at both spatial reductions; the expected parameter count is 248,506.
mechanism: Content-adaptive anti-aliased downsampling
evidence_used: Calibration plateaued at 9,360 correct, while redesigned heads fell to 9,326–9,337, indicating the verified prediction head should be preserved. The model nevertheless relies on ten translated evaluation views after two phase-sensitive max-pooling operations, motivating a clean test of learned downsampling rather than another head modification.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 0.03 label smoothing will exceed 9,360 correct predictions by regularizing overconfident class boundaries while preserving the verified architecture, EMA, augmentation, TTA, and calibration.
change: Replace hard-label cross-entropy with mildly smoothed cross-entropy.
mechanism: Mild uniform label smoothing
evidence_used: Confidence-based focal reweighting reduced validation_correct to 9,331, while architecture, EMA, and TTA changes also underperformed; this motivates retaining uniform example weighting and testing a gentler target-level regularizer.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 0.03 label smoothing will exceed 9,360 correct predictions by gently regularizing class boundaries while preserving the verified architecture, EMA, augmentation, and logit-space TTA.
change: Replace hard-target cross-entropy with cross-entropy using 0.03 label smoothing.
mechanism: Mild uniform label smoothing
evidence_used: The only prior verification of this exact low-risk loss change timed out, so it provides no performance evidence against the idea; larger changes to architecture, EMA, focal weighting, and probability-space TTA all reduced validation_correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 67.90299683297053, "validation_accuracy": 0.9332, "validation_correct": 9332, "validation_cross_entropy": 0.20332384605407716, "validation_score": 9332.41551574137}

RECENT RESULT
hypothesis: Recomputing BatchNorm statistics under the final EMA parameters using the latest 6,000 training examples will exceed 9,360 correct predictions by correcting the weight–normalization mismatch more faithfully than averaging stale running statistics.
change: Cache the latest 64 augmented batches, then reset and cumulatively recompute all BatchNorm running statistics after installing the final EMA parameters.
mechanism: Post-EMA BatchNorm recalibration
evidence_used: Averaging BatchNorm buffers reduced validation_correct to 9,351, while leaving them unaveraged reached 9,360; this suggests normalization state matters, but averaged statistics are not valid statistics for the averaged network.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Recomputing BatchNorm statistics from the final 1,536 augmented examples after installing EMA parameters will exceed 9,360 correct predictions by reducing weight–normalization mismatch without the overhead of the timed-out 6,000-example attempt.
change: Cache only the final 16 training batches and use them to reset and cumulatively recompute all BatchNorm running statistics under the final EMA parameters.
mechanism: Efficient post-EMA BatchNorm recalibration
evidence_used: Averaging BatchNorm buffers reduced validation_correct to 9,351, whereas unaveraged buffers reached 9,360; the more principled post-EMA recalibration attempt timed out, so a substantially cheaper retry remains informative.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 73.74421487492509, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1918250473022461, "validation_score": 9319.41952466189}

RECENT RESULT
hypothesis: Adding cross-channel context to the existing channel gate will exceed 9,360 correct predictions while preserving the verified model’s initial behavior and full spatial classifier.
change: Add a 64→16→64 excitation branch, zero-initialize its output layer, and combine it with the existing per-channel gate; parameters increase from 247,546 to 249,674.
mechanism: Identity-initialized low-rank cross-channel recalibration
evidence_used: Channel-projection and pooled-head redesigns underperformed, indicating that channel identities and the verified classifier should be preserved; this adds narrowly targeted capacity without changing either.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 249674, "training_seconds": 58.2336482910905, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1884157241821289, "validation_score": 9328.42072819286}

RECENT RESULT
hypothesis: Redistributing one eighth of the off-center TTA weight to diagonal one-pixel translations will exceed 9,360 correct predictions by reducing two-axis pooling-phase sensitivity while preserving the verified logit-space aggregation and calibration.
change: Extend evaluation TTA with four diagonal translations at 0.125 weight each and reduce cardinal-view weights to 0.875, keeping the total off-center weight and center weight unchanged.
mechanism: Low-weight diagonal translation marginalization
evidence_used: Learned anti-aliased downsampling targeted phase sensitivity but timed out, while probability-space TTA lost one correct prediction; this tests translation stability without changing the verified architecture, training procedure, logit aggregation, or total ensemble weight.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating one eighth of off-center training crops to diagonal one-pixel translations will exceed 9,360 correct predictions by improving two-axis pooling-phase robustness without changing the verified model, evaluation ensemble, or augmentation strength.
change: Preserve the one-third centered-crop probability and two-thirds shifted-crop probability, while replacing one eighth of cardinal shifts with uniformly distributed diagonal shifts.
mechanism: Low-rate diagonal translation augmentation
evidence_used: The diagonal evaluation ensemble and learned anti-aliased pooling both targeted phase sensitivity but timed out, while broader architecture, loss, and EMA changes reduced accuracy; training on sparse diagonal views tests the same mechanism without additional inference cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 58.5462752499152, "validation_accuracy": 0.934, "validation_correct": 9340, "validation_cross_entropy": 0.18374926300048827, "validation_score": 9340.42238674661}

RECENT RESULT
hypothesis: Halving classifier dropout from 0.10 to 0.05 will exceed 9,360 correct predictions by improving optimization of the narrow 30-unit head while retaining mild regularization.
change: Reduce only the classifier-head dropout probability; preserve architecture, parameter count, optimizer, EMA, augmentation, TTA, and calibration.
mechanism: Reduced bottleneck dropout under fixed exposure
evidence_used: Added regularization through label smoothing fell to 9,332 correct and focal reweighting fell to 9,331, while the uniform-loss baseline reached 9,360; this motivates cautiously reducing existing training noise.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 76.93204570794478, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.18807834701538087, "validation_score": 9311.420847666534}

RECENT RESULT
hypothesis: Reducing the EMA decay cap from 0.98 to 0.979 will retain all 9,360 correct predictions while lowering validation cross-entropy, strictly improving validation_score.
change: Shorten only the late-training parameter-averaging horizon; preserve the architecture, loss, augmentation, schedule, TTA, BatchNorm handling, and calibration.
mechanism: Fine-grained EMA horizon interpolation
evidence_used: The 0.975 EMA achieved lower cross-entropy but lost three correct predictions, while 0.99 lost eleven and worsened cross-entropy; testing 0.979 isolates a narrow point between the accuracy-leading 0.98 setting and the better-calibrated 0.975 setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 65.7714123330079, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.18581063003540038, "validation_score": 9358.421652485933}

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
