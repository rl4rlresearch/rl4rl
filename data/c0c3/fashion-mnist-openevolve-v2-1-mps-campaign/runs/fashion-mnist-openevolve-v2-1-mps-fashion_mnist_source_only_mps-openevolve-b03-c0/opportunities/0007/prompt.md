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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 237346, "training_seconds": 58.25952020799741, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2065622714996338, "validation_score": 9290.41440049288}
prior_hypothesis: Reducing the batch size to 64 while lowering the peak learning rate to 2.0e-3 will exceed 9,257 correct predictions by providing twice as many, less aggressive optimizer updates without altering the best verified architecture.

## Recent verification evidence

RECENT RESULT
hypothesis: A deeper residual CNN near the 250,000-parameter ceiling, trained with twice as many optimizer updates and a short warmup, will exceed the starting design’s 8,928 correct predictions by learning stronger spatial features within the same 100,000-example exposure.
change: Replace the shallow CNN with two residual stages, reduce batch size to 128, use AdamW with stronger regularization, add a warmup-cosine schedule, and reduce label smoothing.
mechanism: Residual batch-normalized CNN with finer-grained optimization
evidence_used: The starting 105,866-parameter two-stage CNN reaches 89.28% accuracy after only 392 updates, leaving substantial parameter capacity and optimization granularity available under the fixed exposure and 250,000-parameter limit.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237346, "training_seconds": 33.91490879189223, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21309012184143067, "validation_score": 9257.412170531272}

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
