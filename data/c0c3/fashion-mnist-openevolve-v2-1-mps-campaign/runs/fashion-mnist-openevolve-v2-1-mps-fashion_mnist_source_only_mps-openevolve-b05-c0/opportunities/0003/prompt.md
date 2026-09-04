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
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing capacity from 105,866 to 232,682 parameters, doubling optimizer updates via batch size 128, and adding translation/flip invariance will improve beyond 8,928 correct validation predictions.
change: Replace the shallow CNN with a five-convolution BatchNorm network, random crops and flips, evaluation-time flip averaging, and a warmup-cosine AdamW schedule.
mechanism: Augmented deep convolutional model with test-time flip ensembling
evidence_used: The starting model reaches 89.28% accuracy with only two convolution layers and 392 optimizer steps, leaving substantial parameter and update capacity under the 250,000-parameter ceiling.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 234,858-parameter residual BatchNorm CNN trained with twice as many optimizer updates will exceed the starting design’s 8,928 correct predictions; lightweight horizontal-flip training and evaluation ensembling will further improve Fashion-MNIST-style invariance.
change: Replace the shallow CNN with a parameter-compliant residual network, use batch size 128, per-example horizontal flips, probability-space flip ensembling at evaluation, grouped AdamW decay, and a short-warmup cosine schedule.
mechanism: Residual convolutional feature learning with flip-consistent ensembling
evidence_used: The verified starting design reaches 89.28% accuracy with only 105,866 parameters and 392 updates, indicating unused capacity and optimization budget; the prior larger design was not verified, so this patch retains the promising capacity/update changes while using a simpler augmentation path.
result: the implementation could not be verified



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
