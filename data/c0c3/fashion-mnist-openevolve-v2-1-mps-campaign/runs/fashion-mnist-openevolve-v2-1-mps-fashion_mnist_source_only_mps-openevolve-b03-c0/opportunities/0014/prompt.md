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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.85783979203552, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.22164978942871094, "validation_score": 9316.409282598275}
prior_hypothesis: Increasing label smoothing from 0.02 to 0.04 will exceed 9,291 correct predictions by extending the demonstrated accuracy benefit of soft targets while retaining the successful 249,934-parameter architecture.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the parameter-heavy dense head with a second residual block and a 3×3 pooled classifier will exceed 9,257 correct predictions by learning deeper spatial features while preserving coarse layout.
change: Widen the second stage to 72 channels, add a residual block after the final pooling layer, and replace the 125k-parameter dense head with adaptive pooling and a linear classifier.
mechanism: Spatial capacity reallocation with coarse-grid pooling
evidence_used: The 237,346-parameter residual CNN reached 92.57% accuracy, but over half its parameters are concentrated in the flattening head; reallocating them to residual spatial processing directly extends the mechanism that produced the verified improvement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a second 64-channel residual block while narrowing the dense head will exceed 9,257 correct predictions by improving spatial feature extraction without discarding the 7×7 layout.
change: Add a residual block after final pooling and reduce the classifier hidden width from 40 to 20, yielding 248,390 learned parameters.
mechanism: Layout-preserving spatial capacity reallocation
evidence_used: The verified residual CNN reached 92.57% accuracy despite allocating over half its parameters to the dense head; this conservatively tests the prior spatial-reallocation hypothesis while avoiding the unverified adaptive-pooling design.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248390, "training_seconds": 45.487382917199284, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.22126563682556152, "validation_score": 9250.40941133929}

RECENT RESULT
hypothesis: Adding small random translations and per-image horizontal flips while retaining the verified 237,346-parameter model will exceed 9,257 correct predictions by reducing overfitting to image position and orientation.
change: Augment each training batch with a randomly shifted 28×28 crop from two-pixel replicated padding and independent horizontal flips.
mechanism: Translation-and-reflection invariance through lightweight augmentation
evidence_used: The verified residual model reached 92.57% accuracy, while reallocating capacity to another residual block reduced accuracy to 92.50%; this motivates improving generalization without changing the successful architecture or optimizer.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237346, "training_seconds": 35.85471670795232, "validation_accuracy": 0.9093, "validation_correct": 9093, "validation_cross_entropy": 0.25886057891845704, "validation_score": 9093.397184571804}

RECENT RESULT
hypothesis: Reducing the batch size to 64 while lowering the peak learning rate to 2.0e-3 will exceed 9,257 correct predictions by providing twice as many, less aggressive optimizer updates without altering the best verified architecture.
change: Halve the training batch size and consistently reduce the AdamW and scheduled peak learning rate.
mechanism: Finer-grained optimization with batch-scaled learning rate
evidence_used: The 237,346-parameter model at batch size 128 is the strongest verified design; added capacity and augmentation both regressed, so this isolates optimization granularity while preserving the successful model.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237346, "training_seconds": 58.25952020799741, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2065622714996338, "validation_score": 9290.41440049288}

RECENT RESULT
hypothesis: Halving the batch size to 32 and reducing the peak learning rate to 1.4e-3 will exceed 9,290 correct predictions by extending the verified benefit of finer-grained optimization while moderating noisier gradients.
change: Use batch size 32 and consistently lower the AdamW initial and scheduled peak learning rates to 1.4e-3.
mechanism: Higher-resolution stochastic optimization
evidence_used: Reducing batch size from 128 to 64 with a lower learning rate improved validation correctness from 9,257 to 9,290 without changing the architecture, while architecture expansion and augmentation regressed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the final low-learning-rate trajectory will exceed 9,290 correct predictions by reducing optimizer noise without increasing the batch count or model size.
change: Wrap the successful AdamW configuration with an EMA that begins after 70% of training and installs the averaged weights after the final optimizer step.
mechanism: Late-training exponential weight averaging
evidence_used: Batch size 64 improved correctness to 9,290 through finer-grained optimization, while batch size 32 exceeded the time limit; EMA seeks additional stability using the existing 1,564-step run.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237346, "training_seconds": 57.21400799998082, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20558840255737304, "validation_score": 9286.414735243752}

RECENT RESULT
hypothesis: Averaging predictions from each validation image and its horizontal reflection will exceed 9,290 correct predictions by reducing orientation-sensitive errors while preserving the strongest verified training configuration.
change: Keep training unchanged and, during evaluation only, average logits from the original and horizontally flipped images.
mechanism: Horizontal-reflection test-time logit ensemble
evidence_used: The 237,346-parameter model with batch size 64 is strongest at 9,290 correct; capacity expansion and EMA regressed, while the failed augmentation trial confounded reflection with two-pixel translations, motivating an isolated inference-time test of label-preserving reflection.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237346, "training_seconds": 52.163940666941926, "validation_accuracy": 0.8663, "validation_correct": 8663, "validation_cross_entropy": 0.3828109161376953, "validation_score": 8663.361582335057}

RECENT RESULT
hypothesis: Batch size 48 with a 1.7e-3 peak learning rate will exceed 9,290 correct predictions by gaining 33% more optimizer updates than batch size 64 while avoiding the timeout observed at batch size 32.
change: Reduce the batch size from 64 to 48 and scale both AdamW’s initial learning rate and the scheduled peak learning rate to 1.7e-3.
mechanism: Time-budgeted intermediate-batch optimization
evidence_used: Moving from batch size 128 to 64 improved correctness from 9,257 to 9,290, while batch size 32 with 1.4e-3 failed to finish; batch size 48 directly tests the remaining feasible optimization-granularity range.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 237346, "training_seconds": 72.33287700009532, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20828157196044922, "validation_score": 9279.413810829861}

RECENT RESULT
hypothesis: Independent two-pixel translations without reflection will exceed 9,290 correct predictions by improving position robustness while avoiding the strongly harmful orientation change.
change: Replace unaugmented batches with per-image random crops from two-pixel replicated padding; retain the verified architecture, batch size, optimizer, loss, and schedule.
mechanism: Translation-only spatial augmentation
evidence_used: Horizontal-reflection test-time averaging reduced correctness from 9,290 to 8,663, indicating reflection is not label-preserving for this task; therefore, the earlier combined translation-and-reflection regression does not rule out translation alone.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237346, "training_seconds": 78.12998320814222, "validation_accuracy": 0.9205, "validation_correct": 9205, "validation_cross_entropy": 0.23030412368774414, "validation_score": 9205.406403579711}

RECENT RESULT
hypothesis: Removing 0.02 label smoothing will exceed 9,290 correct predictions by strengthening gradients toward the true class during the fixed two-pass training horizon; if correctness ties, sharper probabilities should also reduce validation cross-entropy.
change: Retain the strongest verified architecture, batch size, optimizer, and schedule while switching training loss to ordinary cross-entropy.
mechanism: Hard-label objective for short-horizon convergence
evidence_used: Batch size 64 produced the best result at 9,290 correct, while architecture expansion, augmentation, EMA, test-time reflection, and finer batches all regressed or timed out; the remaining label smoothing is an untested regularizer that may impede convergence under limited exposure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237346, "training_seconds": 61.00484766601585, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.20345666580200195, "validation_score": 9270.415469882886}

RECENT RESULT
hypothesis: Widening the classifier hidden layer from 40 to 44 units will exceed 9,290 correct predictions by increasing layout-sensitive capacity while remaining below the 250,000-parameter ceiling.
change: Expand the successful flattening classifier to 44 hidden units, yielding 249,934 learned parameters while leaving training and regularization unchanged.
mechanism: Layout-preserving dense-head expansion
evidence_used: Reallocating dense-head parameters into an additional residual block reduced correctness, whereas the 40-unit layout-preserving head produced the strongest 9,290 result; this tests the opposite allocation direction using the remaining parameter budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 60.37565691699274, "validation_accuracy": 0.9291, "validation_correct": 9291, "validation_cross_entropy": 0.20789716110229492, "validation_score": 9291.413942524332}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.02 to 0.04 will exceed 9,291 correct predictions by extending the demonstrated accuracy benefit of soft targets while retaining the successful 249,934-parameter architecture.
change: Increase cross-entropy label smoothing from 0.02 to 0.04; leave the model, batch size, optimizer, and schedule unchanged.
mechanism: Moderate target-distribution regularization
evidence_used: Removing 0.02 label smoothing reduced correctness from 9,290 to 9,270 despite improving cross-entropy, showing that soft-target regularization materially benefits the primary objective; this tests whether a moderately stronger value improves it further.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.85783979203552, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.22164978942871094, "validation_score": 9316.409282598275}



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
