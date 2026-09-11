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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 28.985292833996937, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551945587158203, "validation_score": 9063.398344620384}
prior_hypothesis: Scaling evaluation logits by 1.298 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255194907.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging predictive probabilities across the existing 50 views will exceed 9,063 correct predictions or, if correct counts tie, lower validation cross-entropy by preventing extreme outer-translation logits from dominating the ensemble.
change: Softmax each view’s logits before averaging, then return the log of the averaged class probabilities for exact cross-entropy evaluation.
mechanism: Probability-space transformation ensemble
evidence_used: The uniform 50-view logit ensemble improved correct predictions from 9,055 to 9,063 but worsened cross-entropy from 0.26850 to 0.27379, indicating useful outer-view decisions alongside poorly calibrated logit magnitudes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 30.096750458003953, "validation_accuracy": 0.9058, "validation_correct": 9058, "validation_cross_entropy": 0.2840727714538574, "validation_score": 9058.389386031007}

RECENT RESULT
hypothesis: A 90% full-range and 10% central-range blend will preserve at least 9,063 correct predictions while lowering cross-entropy below 0.273786, improving the tie-broken validation score.
change: Give the central nine translation pairs a modest extra weight while retaining 90% of the successful uniform 50-view ensemble.
mechanism: Light center-weighted multi-radius logit ensemble
evidence_used: The 75%/25% blend lowered cross-entropy to 0.271792 but lost one correct prediction; a smaller 10% center correction tests whether its calibration benefit can be retained without crossing that decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 27.360899458872154, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2729324203491211, "validation_score": 9063.392793829435}

RECENT RESULT
hypothesis: Replacing the coordinate-specific flattened head and 50-view compensation with deeper low-resolution, multi-scale features and learned global pooling will exceed 9,063 correct predictions.
change: Replace the model with a 242,098-parameter dilated bottleneck CNN using image-conditioned channel gates, four spatial-attention heads, coarse spatial pooling, and an 18-view central translation/flip ensemble.
mechanism: Global-context gated residual features with multi-head attentive shape pooling
evidence_used: The 18-view ensemble already reached 9,055 correct with lower cross-entropy than the 50-view version, while the current model spends most of its capacity on an absolute-position linear head; this challenges the old assumption that brute-force translations are preferable to learning global shape aggregation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 15% central-range correction will retain at least 9,063 correct predictions while lowering validation cross-entropy below 0.272932.
change: Increase the central 18-view ensemble weight from 10% to 15%, retaining 85% of the full 50-view ensemble.
mechanism: Conservative center-weighted ensemble interpolation
evidence_used: A 10% correction preserved 9,063 correct and improved cross-entropy from 0.273786 to 0.272932, while 25% lowered cross-entropy further but lost one correct prediction; 15% probes the interval conservatively.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 29.917300625005737, "validation_accuracy": 0.9061, "validation_correct": 9061, "validation_cross_entropy": 0.27253375091552734, "validation_score": 9061.39291688699}

RECENT RESULT
hypothesis: Scaling the successful 90% full-range/10% central ensemble logits by 1.05 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.272932.
change: Apply a small positive logit scale at evaluation time without changing training, parameters, or predicted classes.
mechanism: Decision-preserving logit temperature sharpening
evidence_used: The current ensemble has the best verified correct count, and probability-space averaging worsened cross-entropy, indicating that softening its predictions is harmful; positive scaling preserves every argmax while testing modest sharpening.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 44.233997457893565, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2665633026123047, "validation_score": 9063.394769056524}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.10 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.266563.
change: Increase the successful evaluation-only logit scale from 1.05 to 1.10 without changing training or class decisions.
mechanism: Decision-preserving incremental logit sharpening
evidence_used: Increasing the scale from 1.00 to 1.05 preserved 9,063 correct predictions and reduced cross-entropy from 0.272932 to 0.266563, showing that the ensemble remains underconfident.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 24.205624083057046, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2619290191650391, "validation_score": 9063.396218798685}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.15 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.261929.
change: Increase the evaluation-only logit scale from 1.10 to 1.15 without changing training, parameters, or predicted classes.
mechanism: Decision-preserving continued logit sharpening
evidence_used: Raising the scale from 1.05 to 1.10 preserved all 9,063 correct predictions and reduced cross-entropy from 0.266563 to 0.261929, so the ensemble remains underconfident at the current scale.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 31.188926375005394, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.25870613327026365, "validation_score": 9063.39723330711}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.20 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.258706.
change: Increase the evaluation-only logit scale from 1.15 to 1.20 without changing training, parameters, or predicted classes.
mechanism: Decision-preserving incremental logit sharpening
evidence_used: Each increase from 1.05 through 1.15 preserved all 9,063 correct predictions and successively reduced cross-entropy; the latest 1.15 scale improved it from 0.261929 to 0.258706.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 34.59393691690639, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.25663830490112305, "validation_score": 9063.397886964014}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.25 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.256638.
change: Increase the evaluation-only logit scale from 1.20 to 1.25 without changing training, parameters, or predicted classes.
mechanism: Decision-preserving incremental logit sharpening
evidence_used: Every increase from 1.05 through 1.20 preserved all 9,063 correct predictions and successively reduced cross-entropy; the latest increase lowered it from 0.258706 to 0.256638.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 42.61697470792569, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.25552196884155276, "validation_score": 9063.398240741626}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.30 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255522.
change: Increase the evaluation-only logit scale from 1.25 to 1.30 without changing training, parameters, or class decisions.
mechanism: Decision-preserving near-optimal logit sharpening
evidence_used: Every scale increase from 1.05 through 1.25 preserved all 9,063 correct predictions and reduced cross-entropy; the latest increase improved it from 0.256638 to 0.255522, though the shrinking gains indicate the calibration optimum is approaching.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 46.587361083133146, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551949073791504, "validation_score": 9063.398344509733}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.295 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255195.
change: Reduce the decision-preserving evaluation-only logit scale from 1.30 to 1.295.
mechanism: Quadratic-guided logit temperature calibration
evidence_used: Cross-entropy improvements shrank from 0.001116 at 1.20→1.25 to 0.000327 at 1.25→1.30; quadratic interpolation of those measurements places the calibration minimum near 1.296.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 38.9162491671741, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551960182189941, "validation_score": 9063.398344157202}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.298 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255194907.
change: Reduce the decision-preserving evaluation-only logit scale from 1.30 to 1.298.
mechanism: Local quadratic logit-temperature calibration
evidence_used: The nearby 1.295 and 1.30 results retained identical predictions, with cross-entropies of 0.255196018 and 0.255194907; fitting these with the 1.25 result places the local minimum near 1.2983.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 28.985292833996937, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551945587158203, "validation_score": 9063.398344620384}

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
