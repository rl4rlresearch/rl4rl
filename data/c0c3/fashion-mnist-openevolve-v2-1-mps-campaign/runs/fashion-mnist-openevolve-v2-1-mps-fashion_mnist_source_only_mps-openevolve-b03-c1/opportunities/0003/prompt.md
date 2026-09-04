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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 244386, "training_seconds": 24.701929665869102, "validation_accuracy": 0.905, "validation_correct": 9050, "validation_cross_entropy": 0.2664461006164551, "validation_score": 9050.394805590035}
prior_hypothesis: A 244,386-parameter batch-normalized CNN trained for roughly twice as many optimizer steps will exceed 8,928 correct predictions, while mild augmentation and flip-averaged evaluation reduce overfitting.

## Recent verification evidence

RECENT RESULT
hypothesis: A near-budget residual CNN trained with twice as many optimizer updates, mild translation/flip augmentation, and warmup-cosine scheduling will exceed the current 8,928 correct predictions.
change: Replace the 105,866-parameter baseline with a roughly 247,000-parameter residual model, use batch size 128, augment training images, reduce label smoothing, and adopt warmup-cosine AdamW.
mechanism: Residual multiscale CNN with exposure-efficient augmentation
evidence_used: The starting design reaches 89.28% accuracy after only 392 optimizer steps, motivating greater model capacity and more updates within the unchanged 100,000-example exposure budget.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 244,386-parameter batch-normalized CNN trained for roughly twice as many optimizer steps will exceed 8,928 correct predictions, while mild augmentation and flip-averaged evaluation reduce overfitting.
change: Use batch size 128, add two convolutional stages within the parameter ceiling, apply mild translation and horizontal-flip augmentation, use low-smoothing AdamW with warmup-cosine decay, and average original/flipped logits during evaluation.
mechanism: Exposure-efficient deep CNN with evaluation-time flip ensembling
evidence_used: The baseline achieves 8,928 correct with only 392 optimizer steps and 105,866 parameters; this motivates safely using the remaining parameter budget and doubling update frequency. The prior near-budget residual attempt produced no verifiable metric, so this patch uses a simpler sequential architecture.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 244386, "training_seconds": 24.701929665869102, "validation_accuracy": 0.905, "validation_correct": 9050, "validation_cross_entropy": 0.2664461006164551, "validation_score": 9050.394805590035}



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
