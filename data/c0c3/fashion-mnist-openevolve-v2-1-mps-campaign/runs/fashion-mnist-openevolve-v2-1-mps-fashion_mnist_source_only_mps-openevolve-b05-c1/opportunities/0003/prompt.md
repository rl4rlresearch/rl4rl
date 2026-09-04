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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 39.26220383308828, "validation_accuracy": 0.9159, "validation_correct": 9159, "validation_cross_entropy": 0.23834408340454102, "validation_score": 9159.40376500094}
prior_hypothesis: Averaging predictions across the centered image and four one-pixel translations, with their horizontal flips, will exceed 9,134 correct predictions by converting training-time translation robustness into more stable validation decisions.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the 105,866-parameter baseline to a batch-normalized 245,044-parameter residual network, while doubling optimizer updates and adding Fashion-MNIST-compatible augmentation, will exceed 8,928 correct validation predictions within the same 100,000-example budget.
change: Use a deeper residual model, batch size 128, random translations and horizontal flips, unsmoothed cross-entropy, short warmup with cosine decay, and horizontal-flip test-time ensembling.
mechanism: Residual CNN with mild geometric augmentation and flip-ensemble inference
evidence_used: The starting model reaches 8,928 correct with only two convolutional layers and 105,866 parameters, leaving substantial capacity under the 250,000-parameter ceiling and only 392 optimizer steps.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 39.47495845798403, "validation_accuracy": 0.9134, "validation_correct": 9134, "validation_cross_entropy": 0.2425220588684082, "validation_score": 9134.402407342735}

RECENT RESULT
hypothesis: Averaging predictions across the centered image and four one-pixel translations, with their horizontal flips, will exceed 9,134 correct predictions by converting training-time translation robustness into more stable validation decisions.
change: Replace two-view flip inference with a ten-view sequential ensemble over the original image, four cardinal translations, and horizontal flips, while leaving training unchanged.
mechanism: Augmentation-matched translation-and-flip test-time ensemble
evidence_used: The 9,134-correct design trains on random translations up to two pixels but ensembles only horizontal flips at validation; symmetric one-pixel translation views directly exploit that learned invariance without changing exposure, optimization, or parameter count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 39.26220383308828, "validation_accuracy": 0.9159, "validation_correct": 9159, "validation_cross_entropy": 0.23834408340454102, "validation_score": 9159.40376500094}



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
