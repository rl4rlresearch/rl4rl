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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.72992745786905, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21800251846313476, "validation_score": 9258.41050818239}
prior_hypothesis: Scaling evaluation logits by 1.22 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.219309.

## Recent verification evidence

RECENT RESULT
hypothesis: Preserving the exact learned architecture in channels-last format will finish reliably, retain 9,258 correct predictions, and 1.2× evaluation logits will reduce cross-entropy below 0.229253.
change: Use channels-last storage for convolutional computation and increase the evaluation-only logit multiplier from 1.1 to 1.2.
mechanism: Channels-last convolution acceleration with inference calibration
evidence_used: Exact GELUs achieved 9,258 correct but required 80.92 seconds, while faster activation substitutions lost accuracy; prior 1.2× calibration attempts timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing spatial GELUs with channel-wise PReLUs will recover the accuracy lost by fixed ReLUs while retaining their sub-75-second runtime, and 1.2× evaluation sharpening will improve cross-entropy without changing predicted classes.
change: Use learned per-channel negative slopes in all three convolutional blocks, retain the classifier GELU, and raise the evaluation-only logit multiplier from 1.1 to 1.2.
mechanism: Channel-wise parametric rectification with inference sharpening
evidence_used: Fixed spatial ReLUs finished in 68.65 seconds but lost 23 correct predictions versus exact GELUs; PReLU preserves that inexpensive rectified computation while learning the negative response that ReLU removes. Prior 1.1× sharpening improved cross-entropy without changing argmax predictions, while 1.2× attempts timed out without contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Spatial LeakyReLUs with a 0.1 negative slope will finish reliably and recover the accuracy lost by ReLU, retaining negative activations without PReLU’s channel-wise overhead; 1.2× evaluation logits will reduce cross-entropy without changing predicted classes.
change: Replace all three convolutional GELUs with in-place LeakyReLUs while retaining the classifier GELU, and increase inference-only logit scaling from 1.1 to 1.2.
mechanism: Fixed-slope leaky rectification with inference sharpening
evidence_used: Spatial ReLUs finished in 68.65 seconds but scored 9,235 correct, while exact GELUs scored 9,258 but took 80.92 seconds. The channel-wise PReLU attempt timed out, motivating the cheaper fixed-slope form; prior 1.1× sharpening improved cross-entropy without changing argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing evaluation-only logit scaling from 1.1 to 1.15 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.229253.
change: Raise only the positive inference-time logit multiplier, leaving training, parameters, and runtime unchanged.
mechanism: Conservative inference-logit sharpening
evidence_used: A 1.1 multiplier previously lowered cross-entropy from 0.241946 to 0.226035 without changing predictions; the prior 1.15 verification timed out and therefore provided no contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.229253.
change: Increase only the inference-time logit multiplier from 1.1 to 1.2.
mechanism: Evaluation-only logit sharpening
evidence_used: A 1.1 multiplier previously reduced cross-entropy from 0.241946 to 0.226035 without changing predictions; later 1.2 attempts timed out or changed activations, so they provide no contradictory isolated calibration evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 74.94584037479945, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.2193094955444336, "validation_score": 9258.410068158926}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.225 will preserve all argmax predictions while reducing validation cross-entropy below 0.219309.
change: Increase only the evaluation-time logit multiplier from 1.2 to 1.225.
mechanism: Quadratic-guided inference-logit calibration
evidence_used: Cross-entropy fell from 0.241946 at 1.0× to 0.226035 at 1.1× and 0.219309 at 1.2× without changing predictions; a quadratic fit to those measurements places the approximate minimum near 1.223×.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.223 will retain 9,258 correct predictions while reducing validation cross-entropy below 0.219309.
change: Increase only the evaluation-time logit multiplier from 1.2 to 1.223.
mechanism: Quadratic-fit inference calibration
evidence_used: Cross-entropy decreased at every tested scale—0.241946 at 1.0, 0.226035 at 1.1, and 0.219309 at 1.2—without changing predictions; a quadratic fit places the estimated minimum near 1.223, while the previous 1.225 run timed out without contradictory calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging original-image and mirrored-image logits will exceed 9,258 correct predictions while preserving the current training trajectory, parameter count, and training runtime.
change: During evaluation only, average logits from the original and horizontally reflected images before applying the established 1.2× calibration.
mechanism: Horizontal-reflection logit self-ensemble
evidence_used: The exact-GELU model produced the best observed accuracy, while every completed activation substitution lost at least 23 correct predictions; an evaluation-only ensemble tests additional invariance without disturbing that learned model.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training every other batch on horizontally reflected images will exceed 9,258 correct predictions while retaining the single-pass evaluation and finishing within the verification limit.
change: Apply a horizontal flip to alternating training batches without changing the model, optimizer, EMA, or calibrated evaluation logits.
mechanism: Alternating horizontal-reflection augmentation
evidence_used: The original-image/reflection evaluation ensemble targeted useful invariance but timed out because it doubled evaluation inference; training-time reflection introduces that invariance with one forward pass per example and minimal overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.22 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.219309.
change: Increase only the evaluation-time logit multiplier from 1.2 to 1.22.
mechanism: Quadratic-guided inference-logit calibration
evidence_used: Cross-entropy fell from 0.241946 at 1.0× to 0.226035 at 1.1× and 0.219309 at 1.2× without changing predictions; a quadratic fit places the estimated minimum near 1.223×, while prior nearby attempts timed out without contradictory validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.72992745786905, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21800251846313476, "validation_score": 9258.41050818239}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.24 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2180025.
change: Increase only the positive evaluation-time logit multiplier from 1.22 to 1.24.
mechanism: Incremental inference-logit sharpening
evidence_used: Raising the multiplier from 1.20 to 1.22 preserved all 9,258 correct predictions and reduced cross-entropy from 0.2193095 to 0.2180025; the observed improvement indicates the calibration optimum lies above 1.20 and motivates a comparably small upward step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.23 will preserve all 9,258 argmax predictions while reducing validation cross-entropy below 0.2180025.
change: Increase only the evaluation-time logit multiplier from 1.22 to 1.23.
mechanism: Midpoint inference-logit calibration
evidence_used: Increasing the scale from 1.20 to 1.22 preserved 9,258 correct predictions and reduced cross-entropy from 0.2193095 to 0.2180025; the 1.24 attempt timed out without contradictory validation evidence, so 1.23 is an informative midpoint test.
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
