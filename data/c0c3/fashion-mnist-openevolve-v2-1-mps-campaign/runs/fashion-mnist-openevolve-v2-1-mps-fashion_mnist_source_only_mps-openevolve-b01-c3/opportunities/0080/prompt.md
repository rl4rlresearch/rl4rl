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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 56.349987000226974, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2175831844329834, "validation_score": 9287.410649560861}
prior_hypothesis: Recency^0.5 tail weighting will exceed 9,287 correct predictions by extending the observed accuracy-favoring shift toward earlier tail iterates while retaining uniform ten-view logit pooling.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.70711100008339, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20961861572265625, "validation_score": 9287.413353426859}
prior_hypothesis: Restoring linear-recency averaging and scaling pooled validation logits by 1.10 will retain exactly 9,287 correct predictions while reducing cross-entropy below 0.2173399.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.4552982088644, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20854735527038573, "validation_score": 9287.41371982473}
prior_hypothesis: Scaling the linear-recency ten-view logits by 1.125 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2096186.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.1797975001391, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.21733987007141115, "validation_score": 9287.410731638955}
prior_hypothesis: Averaging logits across all ten translation/flip views will exceed 9,286 correct predictions by rewarding class evidence that remains consistent across transformations.

## Recent verification evidence

RECENT RESULT
hypothesis: Recency^1.25 tail weighting with uniform ten-view logit pooling will exceed 9,287 correct predictions by interpolating between linear weighting’s higher correct count and recency^1.5 weighting’s lower cross-entropy.
change: Restore ten-view logit averaging and endpoint-sampled half-density parameter averaging, using recency^1.25 weights.
mechanism: Intermediate-recency endpoint-sampled tail quadrature
evidence_used: Linear endpoint quadrature with ten-view logit pooling achieved 9,287 correct, while recency^1.5 achieved 9,286 with slightly lower cross-entropy; an intermediate exponent tests whether modest later-iterate emphasis preserves the former’s decisions while improving margins.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 74.20383870787919, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.21726771659851074, "validation_score": 9286.410755985049}

RECENT RESULT
hypothesis: Trimming the highest and lowest translation logits per class after flip-pair averaging will exceed 9,287 correct predictions by suppressing direction-specific padding artifacts while retaining evidence from three translation positions.
change: Replace the uniform mean over all ten evaluation logits with flip-pair averaging followed by a coordinate-wise trimmed mean across the five translations; training remains unchanged.
mechanism: Flip-paired trimmed translation-logit pooling
evidence_used: Uniform ten-view logit averaging produced the best result at 9,287 correct, while center-prioritized pooling timed out; robust trimming tests the same suspected boundary-artifact mechanism without assuming the center view is always best.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 69.30114916712046, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.21682852630615235, "validation_score": 9276.410904239332}

RECENT RESULT
hypothesis: Preserving four coarse spatial regions alongside global peak evidence will exceed 9,287 correct predictions by distinguishing classes through part arrangement that the current globally pooled representation discards.
change: Replace global mean/max classification with a parameter-neutral 2×2 average-pyramid plus global-max head, and restore the strongest verified linear-recency tail averaging baseline.
mechanism: Quadrant-aware spatial-pyramid classification head
evidence_used: Ten-view pooling with linear endpoint quadrature reached 9,287 correct, while further averaging and view-pooling changes did not improve it; spatial mechanisms remain plausible but prior attention-based versions timed out, motivating this lightweight learned use of coarse layout.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Recency^0.75 weighting will exceed 9,287 correct predictions because correctness declined as the exponent increased from 1.0 to 1.25 and 1.5, suggesting that modestly earlier tail coverage improves decision boundaries.
change: Retain the strongest ten-view logit pooling and half-density quadrature while reducing the parameter-averaging recency exponent from 1.25 to 0.75.
mechanism: Earlier-biased endpoint-sampled tail quadrature
evidence_used: Linear-recency weighting achieved 9,287 correct, versus 9,286 for both recency^1.25 and recency^1.5; testing 0.75 probes the accuracy-favoring direction without adding runtime or parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.93466704222374, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2174412525177002, "validation_score": 9287.410697435269}

RECENT RESULT
hypothesis: Recency^0.5 tail weighting will exceed 9,287 correct predictions by extending the observed accuracy-favoring shift toward earlier tail iterates while retaining uniform ten-view logit pooling.
change: Restore uniform ten-view logit averaging and endpoint-sampled half-density parameter averaging, then reduce the recency exponent from the best-tested 0.75 to 0.5.
mechanism: Square-root-recency endpoint-sampled tail quadrature
evidence_used: Linear weighting achieved 9,287 correct, recency^1.25 and ^1.5 fell to 9,286, and recency^0.75 recovered 9,287; testing 0.5 continues the supported direction without adding parameters or computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 56.349987000226974, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2175831844329834, "validation_score": 9287.410649560861}

RECENT RESULT
hypothesis: Retaining 2×2 average-pooled spatial layout alongside global peak evidence will exceed 9,287 correct predictions by distinguishing classes whose parts differ in arrangement.
change: Expand the classifier input from global mean/max features to four quadrant averages plus global maxima, reducing its hidden width to 24 so the model remains below the parameter ceiling.
mechanism: Parameter-neutral quadrant-aware spatial-pyramid head
evidence_used: Tail-weighting variants plateaued at 9,287 correct and trimmed view pooling regressed to 9,276; the prior quadrant-aware proposal could not be verified, leaving this lightweight spatial mechanism unmeasured.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics with the same recency-weighted tail ensemble as the trainable parameters will exceed 9,287 correct predictions by reducing parameter/statistic mismatch at evaluation.
change: Preserve the current architecture, training, ten-view pooling, and recency-0.75 parameter averaging while also averaging every BatchNorm running mean and variance over the sampled tail iterates.
mechanism: Tail-coupled BatchNorm state averaging
evidence_used: Recency exponents 0.5, 0.75, and 1.0 all plateaued at 9,287 correct, while later-biased variants fell to 9,286; the current code averages BatchNorm affine parameters but retains terminal running statistics, leaving an untested source of tail-ensemble mismatch.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 76.54996758303605, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.2175780204772949, "validation_score": 9278.410651302496}

RECENT RESULT
hypothesis: Concentrating the same mean smoothing strength early and approaching hard labels during the averaged tail will exceed 9,287 correct predictions by regularizing high-learning-rate training while sharpening late decision boundaries.
change: Restore the strongest linear-recency averaging baseline and replace constant 0.02 label smoothing with a cosine decay from 0.04 to 0.
mechanism: Cosine-decayed label smoothing with linear-recency tail averaging
evidence_used: Recency powers 0.5, 0.75, and 1.0 all plateaued at 9,287 correct, with linear recency producing the lowest cross-entropy; this motivates restoring that baseline and testing the previously unchanged loss schedule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 61.22544900001958, "validation_accuracy": 0.9271, "validation_correct": 9271, "validation_cross_entropy": 0.2019432632446289, "validation_score": 9271.41599301339}

RECENT RESULT
hypothesis: Restoring linear-recency averaging and scaling pooled validation logits by 1.10 will retain exactly 9,287 correct predictions while reducing cross-entropy below 0.2173399.
change: Restore the strongest verified linear-recency tail average and mildly sharpen only the final ten-view evaluation logits without changing their argmax.
mechanism: Decision-invariant test-time logit sharpening
evidence_used: Linear recency with ten-view logit pooling achieved 9,287 correct, while cosine-decayed smoothing reduced cross-entropy to 0.20194; evaluation-only scaling can recover some of that confidence benefit while mathematically preserving every predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.70711100008339, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20961861572265625, "validation_score": 9287.413353426859}

RECENT RESULT
hypothesis: Scaling the pooled validation logits by 1.20 will preserve exactly 9,287 correct predictions while reducing validation cross-entropy below 0.2096186.
change: Increase evaluation-only ten-view logit scaling from 1.10 to 1.20 without altering training or predicted classes.
mechanism: Decision-invariant temperature sharpening
evidence_used: Scaling the same linear-recency ten-view logits from 1.00 to 1.10 preserved 9,287 correct predictions and reduced cross-entropy from 0.2173399 to 0.2096186, motivating one further equal sharpening increment.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling pooled validation logits by 1.15 will preserve all 9,287 argmax predictions while reducing cross-entropy below 0.2096186.
change: Restore the verified linear-recency baseline’s evaluation sharpening and increase its logit scale from 1.10 to the untested midpoint 1.15.
mechanism: Decision-invariant intermediate temperature sharpening
evidence_used: A 1.10 scale preserved 9,287 correct predictions and reduced cross-entropy from 0.2173399 to 0.2096186; the 1.20 trial timed out, so 1.15 conservatively probes further sharpening without changing decisions, parameters, or meaningful computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the linear-recency ten-view logits by 1.125 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2096186.
change: Restore the verified linear-recency tail average and apply a conservative evaluation-only scale between the successful 1.10 setting and the unverified 1.15 setting.
mechanism: Decision-invariant fine-grained temperature sharpening
evidence_used: Linear recency with 1.10 scaling achieved 9,287 correct and reduced cross-entropy from 0.2173399 to 0.2096186; positive scaling cannot change predicted classes, and the 1.15 attempt timed out without subject-level evidence against further sharpening.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.4552982088644, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20854735527038573, "validation_score": 9287.41371982473}

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
