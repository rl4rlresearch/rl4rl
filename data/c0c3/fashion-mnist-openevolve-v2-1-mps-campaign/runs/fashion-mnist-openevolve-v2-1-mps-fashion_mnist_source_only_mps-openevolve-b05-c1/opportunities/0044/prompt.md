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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 61.93403479200788, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21500014228820802, "validation_score": 9249.411522585551}
prior_hypothesis: Scaling fused log-probabilities by 1.104 will preserve all 9,249 argmax predictions while reducing validation cross-entropy below 0.2150005993.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the fused log-probabilities by 1.1 will preserve all 9,249 class decisions while lowering validation cross-entropy below 0.216368.
change: Sharpen the probability-space test-time ensemble after fusion without changing training, parameters, or predicted argmaxes.
mechanism: Post-fusion temperature sharpening
evidence_used: The prior 1.1-scaling attempt measured the same 9,249 correct predictions with cross-entropy reduced from 0.216368 to 0.215002; its timeout was unrelated to this computation-neutral calibration.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 58.263879291014746, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21500194969177247, "validation_score": 9249.411521973381}

RECENT RESULT
hypothesis: Scaling the fused log-probabilities by 1.15 will preserve all 9,249 class decisions while reducing validation cross-entropy below 0.215002.
change: Increase probability-ensemble sharpening from 1.1 to 1.15 without changing training, inference views, parameters, or predicted argmaxes.
mechanism: Conservative post-fusion temperature refinement
evidence_used: Increasing the fusion scale from 1.0 to 1.1 preserved 9,249 correct predictions and reduced cross-entropy from 0.216368 to 0.215002, directly supporting a conservative continuation in the same direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the fused log-probabilities by 1.15 will preserve all 9,249 class decisions while reducing validation cross-entropy below 0.215002.
change: Increase probability-ensemble sharpening from 1.1 to 1.15 without changing training, parameters, inference views, or predicted argmaxes.
mechanism: Conservative post-fusion temperature refinement
evidence_used: Increasing the fusion scale from 1.0 to 1.1 preserved 9,249 correct predictions and reduced cross-entropy from 0.216368 to 0.215002; the prior 1.15 attempt timed out without subject-level contrary evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 66.50631899992004, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21522826766967773, "validation_score": 9249.411445333608}

RECENT RESULT
hypothesis: Scaling fused log-probabilities by 1.106 will preserve all 9,249 argmax predictions while reducing validation cross-entropy below 0.215002.
change: Refine post-fusion sharpening from 1.1 to 1.106 without changing training, inference views, or parameters.
mechanism: Quadratic-interpolated ensemble calibration
evidence_used: Scales 1.0, 1.1, and 1.15 produced cross-entropies 0.216368, 0.215002, and 0.215228 with identical correctness; quadratic interpolation places the minimum near 1.106.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 65.63242979184724, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21500059928894044, "validation_score": 9249.411522430764}

RECENT RESULT
hypothesis: Scaling fused log-probabilities by 1.1063 will preserve all 9,249 predictions while reducing validation cross-entropy below 0.2150005993.
change: Refine the inference-only probability-ensemble sharpening factor from 1.106 to 1.1063.
mechanism: Fine-grained post-fusion temperature calibration
evidence_used: Scales 1.0, 1.1, and 1.15 yielded cross-entropies 0.216368, 0.215002, and 0.215228 with identical correctness; quadratic interpolation places the minimum near 1.10633, slightly above the current best scale.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 75.85554791707546, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21500074577331543, "validation_score": 9249.41152238115}

RECENT RESULT
hypothesis: Scaling fused log-probabilities by 1.104 will preserve all 9,249 argmax predictions while reducing validation cross-entropy below 0.2150005993.
change: Refine the inference-only probability-ensemble sharpening factor from 1.106 to 1.104.
mechanism: Local convex temperature recalibration
evidence_used: Scale 1.1063 worsened cross-entropy to 0.2150007458 relative to 0.2150005993 at 1.106, while 1.1 was also worse at 0.2150019497; local quadratic interpolation of these three identical-correctness results places the minimum near 1.104.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 61.93403479200788, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21500014228820802, "validation_score": 9249.411522585551}

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
