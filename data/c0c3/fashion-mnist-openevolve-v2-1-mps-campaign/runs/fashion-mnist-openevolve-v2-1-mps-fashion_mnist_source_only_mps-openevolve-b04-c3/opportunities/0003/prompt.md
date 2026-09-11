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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245040, "training_seconds": 41.259548500180244, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.21916018409729005, "validation_score": 9210.410118380278}
prior_hypothesis: Using the available parameter headroom for batch-normalized residual features, while doubling optimizer updates and removing target smoothing, will exceed the baseline’s 8,928 correct predictions after the same 100,000 examples.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.16630454198457, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.21002658920288086, "validation_score": 9229.413214060303}
prior_hypothesis: Preserving the qualified 245,040-parameter residual model while reducing batch size from 128 to 64 will exceed 9,210 correct predictions because the fixed exposure budget will provide roughly 1,563 optimizer updates instead of 782.

## Recent verification evidence

RECENT RESULT
hypothesis: Using the available parameter headroom for batch-normalized residual features, while doubling optimizer updates and removing target smoothing, will exceed the baseline’s 8,928 correct predictions after the same 100,000 examples.
change: Replace the shallow CNN with a 245,040-parameter residual depthwise model, use batch size 128, apply horizontal-flip augmentation and matching test-time averaging, and train with unsmoothed cross-entropy.
mechanism: Residual multiscale CNN with flip invariance
evidence_used: The baseline reaches 8,928 correct with only 105,866 parameters, one convolution per spatial scale, 392 optimizer steps, and label smoothing; substantial capacity and update-count headroom remains below the 250,000-parameter ceiling.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245040, "training_seconds": 41.259548500180244, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.21916018409729005, "validation_score": 9210.410118380278}

RECENT RESULT
hypothesis: Preserving the qualified 245,040-parameter residual model while reducing batch size from 128 to 64 will exceed 9,210 correct predictions because the fixed exposure budget will provide roughly 1,563 optimizer updates instead of 782.
change: Adopt the qualified residual depthwise architecture, flip augmentation, flip-averaged inference, and unsmoothed loss, while halving its batch size to 64.
mechanism: Smaller-batch residual depthwise learning
evidence_used: Reference Design 1 improved validation correctness from 8,928 to 9,210 while increasing optimizer steps from 392 to 782; this motivates testing whether another update-count increase improves the same established model.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.16630454198457, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.21002658920288086, "validation_score": 9229.413214060303}



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
