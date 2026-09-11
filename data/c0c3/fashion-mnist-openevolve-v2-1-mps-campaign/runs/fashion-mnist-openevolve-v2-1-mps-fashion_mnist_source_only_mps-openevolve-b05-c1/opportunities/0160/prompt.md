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
hypothesis: A positive evaluation-time logit scale of 1.067003607749939 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.18585695190429688.
change: Replace only the evaluation-time logit multiplier with the nearest float32 value to the interpolated local minimum.
mechanism: Accuracy-invariant local temperature refinement
evidence_used: Scales 1.0664346, 1.0671112, and 1.0671952 yielded cross-entropies 0.18585700874328615, 0.18585695190429688, and 0.1858569564819336; quadratic interpolation predicts a minimum near 1.0670036. Two attempts at this value timed out but provided no contrary accuracy or loss evidence, and positive scaling preserves argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A positive evaluation-time logit scale of 1.067003607749939 will preserve all 9,360 argmax predictions while reduce validation cross-entropy below 0.18585695190429688.
change: Replace only the evaluation-time logit multiplier with the nearest float32 value to the interpolated local minimum.
mechanism: Accuracy-invariant local temperature refinement
evidence_used: Scales 1.0664346, 1.0671112, and 1.0671952 produced cross-entropies 0.18585700874328615, 0.18585695190429688, and 0.1858569564819336; quadratic interpolation places the minimum near 1.0670036. Three attempts at this value timed out without contrary performance evidence, while positive scaling necessarily preserves class rankings.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 67.55574704194441, "validation_accuracy": 0.936, "validation_correct": 9360, "validation_cross_entropy": 0.1858569549560547, "validation_score": 9360.421636014285}

RECENT RESULT
hypothesis: Replacing the load-bearing 30-unit monolithic flatten bottleneck with a learned 64→20 channel projection and 96-unit spatial head will exceed 9,360 correct predictions by preserving reusable local feature semantics while tripling nonlinear decision capacity.
change: Factor the classifier through a shared 1×1 projection, normalization, and GELU before flattening; widen its hidden layer from 30 to 96 while keeping the expected parameter count at 249,724.
mechanism: Shared channel projection with a widened nonlinear spatial head
evidence_used: Temperature refinement plateaued at 9,360 correct, so calibration no longer improves rankings. The class-specific attention residual fell to 9,337, motivating replacement of the restrictive primary head rather than attaching another unconstrained auxiliary branch.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 249724, "training_seconds": 61.80865866690874, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.18531914443969727, "validation_score": 9326.42182732165}

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics with the same decay as the EMA parameters will exceed 9,360 correct predictions by eliminating the final mismatch between averaged weights and non-averaged normalization state.
change: Track floating-point model buffers during EMA updates and restore their averages alongside parameter averages at the final optimizer step.
mechanism: EMA-consistent BatchNorm state
evidence_used: The widened spatial head reduced accuracy from 9,360 to 9,326, favoring preservation of the current architecture. The current procedure replaces every learned parameter with its EMA value but leaves BatchNorm running statistics from the raw training trajectory, despite already collecting the relevant buffers.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 71.91994012496434, "validation_accuracy": 0.9351, "validation_correct": 9351, "validation_cross_entropy": 0.18566928329467774, "validation_score": 9351.42170275223}

RECENT RESULT
hypothesis: Averaging calibrated class probabilities across the ten evaluation views will exceed 9,360 correct predictions by preventing one unusually confident translated view from dominating the geometric, logit-space ensemble.
change: Preserve the verified architecture, training procedure, center weighting, and temperature while replacing evaluation-time weighted logit averaging with weighted probability averaging followed by conversion back to logits.
mechanism: Mixture-of-views probability ensembling
evidence_used: Temperature refinement plateaued at 9,360 correct, while class-attention and widened-head changes fell to 9,337 and 9,326; this motivates preserving the learned classifier and testing the previously unchanged TTA aggregation rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging calibrated probabilities across the ten evaluation views will exceed 9,360 correct predictions by reducing domination from unusually confident individual views.
change: Replace weighted logit averaging with weighted probability averaging while preserving the verified center weight and temperature.
mechanism: Calibrated probability-space test-time augmentation
evidence_used: Architecture and BatchNorm-state changes reduced accuracy, while the prior probability-ensemble verification only timed out and produced no contrary evidence, making a clean retry the most informative ranking-focused change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 65.33428045781329, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18755425186157226, "validation_score": 9359.42103339634}

RECENT RESULT
hypothesis: Replacing the 30-unit full-resolution bottleneck with 4×4 adaptive pooling and a 91-unit head will exceed 9,360 correct predictions by retaining all 64 learned channels while increasing nonlinear decision capacity within the parameter ceiling.
change: Pool each 64-channel feature map from 7×7 to 4×4 before flattening, then widen the classifier hidden layer from 30 to 91.
mechanism: Channel-preserving spatial compression with a widened nonlinear head
evidence_used: The 64→20 channel-projection head fell to 9,326 correct, suggesting that discarding channel semantics was harmful; this patch instead obtains comparable head capacity by compressing spatial resolution while preserving every backbone channel.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adaptive 4×4 pooling with a 93-unit hidden layer will exceed 9,360 correct predictions by retaining all 64 channel identities while tripling nonlinear head capacity within the parameter ceiling.
change: Insert adaptive average pooling before flattening and widen the classifier bottleneck from 30 to 93 units, producing 249,517 learned parameters.
mechanism: Channel-preserving spatial compression with a widened nonlinear head
evidence_used: The 64→20 projection head fell to 9,326 correct, suggesting that discarding channel semantics is harmful; the prior channel-preserving pooling proposal could not be verified, so this capacity-maximized version remains an informative untested alternative.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reweighting training examples toward low-confidence classifications while retaining half-weight on easy examples will exceed 9,360 correct predictions by improving decision boundaries without changing the verified architecture or evaluation ensemble.
change: Replace uniform cross-entropy with detached confidence-based weights normalized to preserve the average gradient scale.
mechanism: Normalized mild focal reweighting
evidence_used: Architecture changes fell to 9,326–9,337 correct, EMA-consistent BatchNorm fell to 9,351, and probability-space TTA fell to 9,359; this motivates preserving the 9,360-correct model and changing only how its fixed training exposure prioritizes ambiguous examples.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 62.33559845783748, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1905598991394043, "validation_score": 9331.41997046966}

RECENT RESULT
hypothesis: Increasing the EMA decay cap from 0.98 to 0.99 will exceed 9,360 correct predictions by averaging a broader portion of the low-learning-rate trajectory while preserving the verified BatchNorm statistics and architecture.
change: Extend only the parameter-averaging horizon; leave training, BatchNorm buffers, augmentation, TTA, and calibration unchanged.
mechanism: Longer-horizon late-training parameter EMA
evidence_used: Architecture changes fell to 9,326–9,337 correct, focal reweighting fell to 9,331, and averaging BatchNorm state fell to 9,351, motivating a minimal ranking-focused refinement of the otherwise successful parameter EMA.
result: training did not finish within the verification time limit

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
