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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 40.34581370907836, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2150239284515381, "validation_score": 9267.411514529296}
prior_hypothesis: Setting the agreement-only geometric blend to 33.07% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502393417358398.

## Recent verification evidence

RECENT RESULT
hypothesis: Independently sampling the existing uniform full and central crop distributions for every image will exceed 9,267 correct predictions by reducing within-batch augmentation correlation without changing the successful view distribution, architecture, or loss.
change: Replace batch-shared crop offsets with vectorized per-image offsets extracted from the padded 5×5 crop grid.
mechanism: Per-example random crop sampling
evidence_used: Evaluation-matched triangular crop sampling regressed to 9,247 correct, showing that changing the crop distribution is harmful; this patch preserves the baseline’s uniform marginals while making each update cover more of that proven distribution.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 51.792152625042945, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.2153379768371582, "validation_score": 9250.41140819223}

RECENT RESULT
hypothesis: Raising the geometric component to 10% only when the arithmetic and geometric predictors agree will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Use a 10% geometric blend on endpoint-agreement examples and preserve the verified 9.832581520080566% blend elsewhere.
mechanism: Prediction-safe conditional geometric blending
evidence_used: Globally increasing the geometric weight beyond 9.832581520080566% flipped one validation prediction; conditioning the increase on endpoint agreement prevents such argmax changes while testing the locally favorable calibration direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 45.734137249877676, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2154026180267334, "validation_score": 9267.411386311485}

RECENT RESULT
hypothesis: A learned affine warp of the 7×7 feature grid will exceed 9,267 correct predictions by correcting image-specific pose and translation before classification, while retaining the spatial layout that pooling-based designs lost.
change: Replace the fixed-grid assumption with an identity-initialized, bounded spatial transformer over stem features; this adds 11,766 parameters for a total of 245,200.
mechanism: Bounded content-conditioned feature canonicalization
evidence_used: Uniform crop changes and forced rotation augmentation regressed to 9,250 and 9,203, while position-free pooling also regressed, indicating that indiscriminate invariance is harmful and spatial layout is load-bearing. A learned per-image canonicalization can selectively correct nuisance geometry without discarding that layout.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the geometric component from 10% to 11% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2154026180267334.
change: Increase the agreement-only geometric-logit weight to 11%, while preserving the verified 9.832581520080566% blend on disagreement examples.
mechanism: Prediction-safe agreement-stratified blend continuation
evidence_used: Moving agreement examples from the baseline blend to 10% geometric retained 9,267 correct and reduced cross-entropy from 0.21540798034667968 to 0.2154026180267334; because both endpoint predictors have the same argmax on these examples, further convex interpolation cannot change their predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 47.86802029190585, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21537131576538085, "validation_score": 9267.411396906866}

RECENT RESULT
hypothesis: Raising the geometric component from 11% to 12% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21537131576538085.
change: Increase the agreement-only geometric-logit weight to 12%, while preserving the verified 9.832581520080566% blend for disagreement examples.
mechanism: Prediction-safe agreement-stratified blend continuation
evidence_used: Increasing the agreement-only geometric weight from 10% to 11% retained 9,267 correct and reduced cross-entropy from 0.2154026180267334 to 0.21537131576538085; convex interpolation between predictors sharing the same argmax cannot change the predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 42.685781749896705, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2153412998199463, "validation_score": 9267.41140706736}

RECENT RESULT
hypothesis: Raising the geometric component from 12% to 13% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2153412998199463.
change: Increase the agreement-only geometric-logit weight to 13%, while preserving the verified 9.832581520080566% blend for disagreement examples.
mechanism: Prediction-safe agreement-stratified blend continuation
evidence_used: Increasing the agreement-only geometric weight from 10% to 11% and then 12% retained 9,267 correct while successively reducing cross-entropy to 0.21537131576538085 and 0.2153412998199463; convex interpolation between predictors sharing the same argmax cannot change that prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 42.064061417011544, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21531258010864257, "validation_score": 9267.411416789544}

RECENT RESULT
hypothesis: Raising the geometric component from 13% to 14% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21531258010864257.
change: Increase the agreement-only geometric-logit weight to 14%, while preserving the verified 9.832581520080566% blend for disagreement examples.
mechanism: Prediction-safe agreement-stratified blend continuation
evidence_used: Agreement-only increases from 10% through 13% retained 9,267 correct while successively lowering cross-entropy; because both endpoint predictors share the same argmax on these examples, further convex interpolation cannot change their predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 58.58406991697848, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21528516769409178, "validation_score": 9267.41142606961}

RECENT RESULT
hypothesis: Raising the agreement-only geometric component from 14% to 34% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21528516769409178.
change: Use a 34% geometric-logit blend when arithmetic and geometric predictors agree, preserving the verified baseline blend on disagreement examples.
mechanism: Curvature-extrapolated prediction-safe geometric blending
evidence_used: Agreement-only increases from 10% through 14% preserved all 9,267 correct predictions and produced smoothly diminishing cross-entropy improvements; extrapolating those gains places the calibration minimum near 34%, while convex interpolation between predictors sharing the same argmax preserves their predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 51.95969712501392, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21502458267211913, "validation_score": 9267.41151430772}

RECENT RESULT
hypothesis: Raising the agreement-only geometric component from 34% to 34.7% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502458267211913.
change: Use a 34.7% geometric-logit blend when the arithmetic and geometric predictors agree, preserving the verified baseline blend on disagreement examples.
mechanism: Curvature-fitted prediction-safe geometric calibration
evidence_used: Agreement-only weights from 10% through 34% preserved all 9,267 correct predictions, while their diminishing cross-entropy gains imply a calibration minimum near 34.7%; convex interpolation between predictors sharing the same argmax preserves their predicted class.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 47.02417437499389, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2150259391784668, "validation_score": 9267.411513848287}

RECENT RESULT
hypothesis: Reducing the agreement-only geometric blend from 34% to 33% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502458267211913.
change: Use a 33% geometric-logit blend when arithmetic and geometric predictors agree, preserving the verified baseline blend on disagreement examples.
mechanism: Evidence-fitted prediction-safe geometric calibration
evidence_used: Moving from 34% to 34.7% increased cross-entropy to 0.2150259391784668; fitting the observed 14%, 34%, and 34.7% results places the calibration minimum near 33%, while convex interpolation between agreeing predictors preserves their argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 59.7637235422153, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21502393417358398, "validation_score": 9267.411514527357}

RECENT RESULT
hypothesis: Setting the agreement-only geometric blend to 33.07% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502393417358398.
change: Refine the agreement-only blend from 33% to the estimated local cross-entropy minimum of 33.07%, preserving the disagreement blend.
mechanism: Quadratic-fitted prediction-safe geometric calibration
evidence_used: Cross-entropy was 0.21502393417358398 at 33%, 0.21502458267211913 at 34%, and 0.2150259391784668 at 34.7%; a quadratic fit places the minimum near 33.07%, while convex interpolation between agreeing predictors preserves their argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 40.34581370907836, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2150239284515381, "validation_score": 9267.411514529296}

RECENT RESULT
hypothesis: Setting the agreement-only geometric blend to 33.0725% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2150239284515381.
change: Refine the agreement-only blend from 33.07% to the quadratic-fit minimum of approximately 33.0725%, leaving disagreement predictions unchanged.
mechanism: Quadratic-refined prediction-safe geometric calibration
evidence_used: Agreement-only blends preserve argmax; cross-entropies at 33%, 34%, and 34.7% place the fitted minimum near 33.0725%, while 33.07% already improved cross-entropy to 0.2150239284515381.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 52.10554095800035, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21502393188476562, "validation_score": 9267.411514528134}

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
