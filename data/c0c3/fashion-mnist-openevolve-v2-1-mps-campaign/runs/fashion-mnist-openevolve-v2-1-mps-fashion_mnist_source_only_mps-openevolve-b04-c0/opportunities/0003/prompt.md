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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216538, "training_seconds": 40.435976000037044, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.21250880165100097, "validation_score": 9283.412368140602}
prior_hypothesis: A deeper 216,538-parameter CNN trained with 128-image batches, mild flip augmentation, and warmup-cosine AdamW will exceed the current 8,928 validation-correct result.

## Recent verification evidence

RECENT RESULT
hypothesis: A deeper 216,538-parameter CNN trained with 128-image batches, mild flip augmentation, and warmup-cosine AdamW will exceed the current 8,928 validation-correct result.
change: Replace the shallow network with four convolutional stages, halve batch size, add horizontal-flip augmentation, reduce label smoothing, and use a warmup-cosine learning-rate schedule with a nonzero floor.
mechanism: Capacity-efficient batch-normalized CNN with higher update density
evidence_used: The starting 105,866-parameter model reaches 89.28% after only 392 optimizer steps, leaving substantial parameter capacity and suggesting that richer spatial features plus twice as many updates can improve the fixed-exposure result.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216538, "training_seconds": 40.435976000037044, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.21250880165100097, "validation_score": 9283.412368140602}

RECENT RESULT
hypothesis: Halving the batch size from 128 to 64 will exceed 9,283 correct predictions by providing about 1,563 optimizer updates within the same 100,000-example exposure.
change: Change only the batch size, preserving the validated architecture, augmentation, loss, and warmup-cosine AdamW schedule.
mechanism: Higher optimizer-update density
evidence_used: The available design reached 92.83% while increasing update density from the earlier 392-step regime to 782 steps; testing 64-image batches directly extends the most promising observed mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216538, "training_seconds": 64.91070549981669, "validation_accuracy": 0.9272, "validation_correct": 9272, "validation_cross_entropy": 0.2075602508544922, "validation_score": 9272.414058014618}



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
