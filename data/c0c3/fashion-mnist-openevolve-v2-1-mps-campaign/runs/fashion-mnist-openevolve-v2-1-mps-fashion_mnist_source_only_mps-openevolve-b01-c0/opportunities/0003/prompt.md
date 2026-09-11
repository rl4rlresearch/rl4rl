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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.460448458092287, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.25047449645996095, "validation_score": 9167.399848218749}
prior_hypothesis: Averaging predictions across the centered image and four one-pixel translations, each with its horizontal flip, will exceed 9,138 correct predictions because the verified model was explicitly trained for translation and flip invariance.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing useful capacity to 216,346 parameters, doubling optimizer updates, and training translation/flip invariance will exceed the current 8,928 correct predictions after the same 100,000-example exposure.
change: Replace the shallow CNN with a four-convolution network, use batch size 128, per-image random translations and flips, evaluation-time flip averaging, decoupled AdamW parameter groups, and a warmup-cosine schedule.
mechanism: Batch-normalized deeper CNN with per-image geometric augmentation and flip-ensemble inference
evidence_used: The starting 105,866-parameter two-layer CNN reaches 8,928 correct with only 392 optimizer steps; substantial parameter headroom and the short fixed exposure motivate a higher-capacity normalized model with more update opportunities.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 34.70552358403802, "validation_accuracy": 0.9138, "validation_correct": 9138, "validation_cross_entropy": 0.2494143997192383, "validation_score": 9138.40018747992}

RECENT RESULT
hypothesis: Averaging predictions across the centered image and four one-pixel translations, each with its horizontal flip, will exceed 9,138 correct predictions because the verified model was explicitly trained for translation and flip invariance.
change: Replace two-view logit averaging during evaluation with ten-view arithmetic probability averaging over centered, translated, and flipped inputs; training remains unchanged.
mechanism: Translation-flip probability ensemble
evidence_used: The current 216,346-parameter model reached 9,138 correct while training on random translations and flips, but evaluation exploits only flip invariance; translation ensembling directly uses the already-learned augmentation invariance without changing the fixed training exposure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.460448458092287, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.25047449645996095, "validation_score": 9167.399848218749}



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
