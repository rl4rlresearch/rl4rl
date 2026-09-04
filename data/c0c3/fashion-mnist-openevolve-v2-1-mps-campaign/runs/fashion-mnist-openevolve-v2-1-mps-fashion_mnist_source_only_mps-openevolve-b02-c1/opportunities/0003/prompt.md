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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 24.309294582810253, "validation_accuracy": 0.9129, "validation_correct": 9129, "validation_cross_entropy": 0.23464069633483886, "validation_score": 9129.404976120975}
prior_hypothesis: Averaging class probabilities for original and horizontally flipped validation images will exceed 9,097 correct predictions because training already enforces flip invariance.

## Recent verification evidence

RECENT RESULT
hypothesis: A deeper 232,962-parameter CNN trained with twice as many optimizer updates and light geometric augmentation will exceed the baseline’s 8,928 correct predictions.
change: Replace the small CNN with a three-stage spatial CNN, reduce batch size to 128, add random ±2-pixel crops and horizontal flips, remove label smoothing, and use warmup-cosine AdamW.
mechanism: High-capacity batch-normalized CNN with translation and reflection invariance
evidence_used: The starting 105,866-parameter model reaches 89.28% after only 392 optimizer steps, leaving substantial parameter and update capacity available under the fixed limits.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 29.360637167003006, "validation_accuracy": 0.9097, "validation_correct": 9097, "validation_cross_entropy": 0.24107790565490722, "validation_score": 9097.402875595257}

RECENT RESULT
hypothesis: Averaging class probabilities for original and horizontally flipped validation images will exceed 9,097 correct predictions because training already enforces flip invariance.
change: Keep training unchanged, but ensemble original and mirrored predictions during evaluation.
mechanism: Flip-ensemble test-time augmentation
evidence_used: The 9,097-correct design trains with random horizontal flips, so paired evaluation can reduce residual orientation-dependent prediction variance without adding parameters or changing exposure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 24.309294582810253, "validation_accuracy": 0.9129, "validation_correct": 9129, "validation_cross_entropy": 0.23464069633483886, "validation_score": 9129.404976120975}



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
