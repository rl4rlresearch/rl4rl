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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245104, "training_seconds": 72.59424875001423, "validation_accuracy": 0.9314, "validation_correct": 9314, "validation_cross_entropy": 0.19205170211791991, "validation_score": 9314.419444894136}
prior_hypothesis: Normalizing the 30-feature classifier bottleneck will exceed 9,249 correct predictions by stabilizing its optimization within the fixed two-pass budget while preserving the successful representation and regularization level.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing batch size from 96 to 80 will exceed 9,249 correct predictions by providing 20% more optimizer updates and beneficial gradient noise within the fixed 100,000-example budget.
change: Change only the training batch size, preserving the architecture, learning-rate schedule, augmentation, and calibrated probability-space ensemble.
mechanism: Increased fixed-exposure update frequency
evidence_used: A prior batch-size reduction improved correctness from 9,204 to 9,239, whereas subsequent architectural changes reduced correctness; this directly supports continuing the successful optimization direction conservatively.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing batch size from 96 to 88 will exceed 9,249 correct predictions by providing about 9% more optimizer updates while avoiding the greater runtime risk observed at batch size 80.
change: Change only the training batch size, preserving the model, learning-rate schedule, augmentation, and calibrated inference ensemble.
mechanism: Moderate fixed-exposure update-frequency increase
evidence_used: A prior batch-size reduction improved correctness from 9,204 to 9,239, while the batch-size-80 experiment timed out; 88 is a conservative intermediate test of the same favorable optimization direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 88, "examples_processed": 100000, "optimizer_steps": 1138, "parameters": 245044, "training_seconds": 71.77568862517364, "validation_accuracy": 0.9238, "validation_correct": 9238, "validation_cross_entropy": 0.21693088073730468, "validation_score": 9238.410869678726}

RECENT RESULT
hypothesis: Replacing the shared 30-feature flattened bottleneck with deeper spatial processing and four fixed-position evidence components per class will exceed 9,249 correct predictions by preserving spatial geometry while learning class-specific channel–position templates.
change: Add a residual block at 7×7 resolution and replace the dense head with a rank-four, class-specific spatial evidence head; training and calibrated test-time augmentation remain unchanged.
mechanism: Class-specific factorized spatial evidence
evidence_used: Content-addressed pooling fell to 9,228 correct, showing that input-dependent spatial aggregation was harmful, while widening the fixed-position bottleneck fell to 9,213, showing that more late dense capacity was insufficient. This tests a different mechanism that retains fixed spatial structure but moves representation learning into the convolutional trunk and computes each class from its own distributed evidence templates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final training iterate with a 0.99-decay EMA over the second half of training will exceed 9,249 correct predictions by reducing late optimization noise without changing the successful architecture or exposure budget.
change: Track an exponential moving average of model parameters after halfway through training and install those averaged weights after the final optimizer step.
mechanism: Tail exponential weight averaging
evidence_used: Optimization changes previously improved correctness while multiple architecture changes reduced it; tail averaging targets optimization stability while preserving the 245,044-parameter representation and batch size 96.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 70.77037579100579, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21651254959106445, "validation_score": 9240.411010967513}

RECENT RESULT
hypothesis: Halving classifier dropout from 0.1 to 0.05 will exceed 9,249 correct predictions by reducing optimization noise during the fixed two-pass budget while retaining mild regularization.
change: Reduce only the dropout probability in the existing classifier head; preserve architecture, parameter count, optimizer, augmentation, and calibrated ensemble.
mechanism: Reduced bottleneck dropout
evidence_used: Smaller batches and a higher peak learning rate previously improved correctness, whereas architecture changes and tail EMA regressed, motivating a conservative optimization-focused change within the successful model.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 60.47971620806493, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21235424842834472, "validation_score": 9239.412420710076}

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.10 to 0.15 will exceed 9,249 correct predictions by further reducing feature co-adaptation while preserving the successful architecture and optimization regime.
change: Raise only the classifier-head dropout probability from 0.10 to 0.15.
mechanism: Stronger bottleneck dropout regularization
evidence_used: Reducing dropout to 0.05 lowered correctness from 9,249 to 9,239 despite improving cross-entropy, directly indicating that dropout regularization benefits the primary, correctness-ranked objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing peak learning rate from 3.3e-3 to 3.5e-3 will exceed 9,249 correct predictions by improving convergence within the fixed 100,000-example budget.
change: Raise only `PEAK_LR`, preserving the successful architecture, batch size, augmentation, schedule shape, and calibrated inference ensemble.
mechanism: Conservative peak-learning-rate increase
evidence_used: Prior higher peak learning rates improved correctness, while reducing batch size to 88, tail EMA, and lower dropout all regressed; this isolates a previously favorable optimization direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 69.05167416692711, "validation_accuracy": 0.9221, "validation_correct": 9221, "validation_cross_entropy": 0.22083003540039062, "validation_score": 9221.409557420362}

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.10 to 0.15 will exceed 9,249 correct predictions by reducing bottleneck feature co-adaptation.
change: Raise only the classifier-head dropout probability, preserving all other training and inference behavior.
mechanism: Stronger classifier-head regularization
evidence_used: Reducing dropout to 0.05 lowered correctness from 9,249 to 9,239, indicating that dropout benefits the primary objective; the prior 0.15 trial timed out and provided no contrary accuracy evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 76.40808724984527, "validation_accuracy": 0.9187, "validation_correct": 9187, "validation_cross_entropy": 0.23000600509643554, "validation_score": 9187.40650208042}

RECENT RESULT
hypothesis: Normalizing the 30-feature classifier bottleneck will exceed 9,249 correct predictions by stabilizing its optimization within the fixed two-pass budget while preserving the successful representation and regularization level.
change: Insert BatchNorm1d after the first classifier linear layer, adding only 60 learned parameters.
mechanism: Dense-bottleneck batch normalization
evidence_used: Changing dropout in either direction reduced correctness, and larger architectural changes also regressed; this tests deterministic bottleneck conditioning without altering dropout, augmentation, schedule, or inference fusion.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245104, "training_seconds": 72.59424875001423, "validation_accuracy": 0.9314, "validation_correct": 9314, "validation_cross_entropy": 0.19205170211791991, "validation_score": 9314.419444894136}

RECENT RESULT
hypothesis: Adding channel-wise BatchNorm after the convolutional trunk will exceed 9,314 correct predictions by stabilizing the flattened classifier input while remaining below the parameter ceiling.
change: Append a 64-channel BatchNorm2d layer to the feature extractor, adding 128 learned parameters without otherwise changing training or inference.
mechanism: Pre-classifier spatial feature normalization
evidence_used: BatchNorm on the 30-feature bottleneck improved correctness from 9,249 to 9,314; this applies the same successful conditioning mechanism at the remaining unnormalized trunk-to-classifier boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245232, "training_seconds": 59.027689500013366, "validation_accuracy": 0.9305, "validation_correct": 9305, "validation_cross_entropy": 0.19097607192993163, "validation_score": 9305.419823715845}

RECENT RESULT
hypothesis: Reducing the successful bottleneck BatchNorm momentum from 0.1 to 0.05 will exceed 9,314 correct predictions by producing less noisy evaluation statistics from batch-size-96 training while retaining its optimization benefit.
change: Set the 30-feature BatchNorm1d momentum to 0.05; preserve all architecture, optimization, augmentation, and inference behavior.
mechanism: Smoothed bottleneck running-statistics estimation
evidence_used: Adding bottleneck BatchNorm improved correctness from 9,249 to 9,314, whereas adding trunk BatchNorm reduced it to 9,305; this motivates refining the proven bottleneck normalization rather than adding normalization elsewhere.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245104, "training_seconds": 68.89217404089868, "validation_accuracy": 0.9314, "validation_correct": 9314, "validation_cross_entropy": 0.19218692893981934, "validation_score": 9314.419397317537}

RECENT RESULT
hypothesis: Increasing bottleneck BatchNorm momentum from 0.1 to 0.2 will exceed 9,314 correct predictions by aligning evaluation statistics more closely with the late-training feature distribution.
change: Set only the 30-feature BatchNorm1d momentum to 0.2, preserving all other architecture, optimization, augmentation, and inference behavior.
mechanism: More responsive bottleneck running-statistics estimation
evidence_used: Bottleneck BatchNorm raised correctness from 9,249 to 9,314, while reducing its momentum to 0.05 preserved correctness but worsened cross-entropy from 0.1920517 to 0.1921869; this supports testing the opposite, more-responsive direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245104, "training_seconds": 70.58692362508737, "validation_accuracy": 0.9314, "validation_correct": 9314, "validation_cross_entropy": 0.1922281494140625, "validation_score": 9314.419382817161}

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
