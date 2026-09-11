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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 31.57998920790851, "validation_accuracy": 0.8994, "validation_correct": 8994, "validation_cross_entropy": 0.28040900344848635, "validation_score": 8994.390500221924}
prior_hypothesis: A 230,442-parameter CNN that concentrates added capacity at 7×7 resolution, paired with twice as many optimizer updates and mild translation/flip augmentation, will exceed the verified baseline’s 8,928 correct predictions without the verification risk of the prior deeper high-resolution designs.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the under-capacity feature extractor while adding small label-preserving translations will exceed 8,928 correct predictions within the fixed exposure budget and 250,000-parameter ceiling.
change: Use a 232k-parameter five-convolution network, batch size 128, per-image reflected translations and flips, lighter label smoothing, relaxed clipping, and a warmup-cosine AdamW schedule.
mechanism: Deeper batch-normalized CNN with translation augmentation and warmup-cosine optimization
evidence_used: The starting two-convolution, 105,866-parameter model reached 89.28% accuracy, leaving substantial parameter capacity available for learning richer spatial features.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 225,578-parameter residual feature extractor, twice as many optimizer updates, and mild spatial augmentation will exceed the starting design’s 8,928 correct predictions within the same 100,000-example budget.
change: Replace the shallow CNN with a four-convolution residual network, use batch size 128, reflected random crops and horizontal flips, lighter smoothing, relaxed clipping, and warmup-cosine AdamW.
mechanism: Compact residual CNN with translation-and-flip augmentation
evidence_used: The verified 105,866-parameter two-convolution baseline reached 89.28% accuracy, indicating useful headroom below the 250,000-parameter ceiling; this patch tests that capacity hypothesis with a simpler, lower-compute architecture than the unverified five-convolution attempt.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 230,442-parameter CNN that concentrates added capacity at 7×7 resolution, paired with twice as many optimizer updates and mild translation/flip augmentation, will exceed the verified baseline’s 8,928 correct predictions without the verification risk of the prior deeper high-resolution designs.
change: Add a compact 7×7 residual block and wider classifier, use batch size 128, reflected translations and horizontal flips, reduce label smoothing, relax clipping, and apply warmup-cosine AdamW.
mechanism: Compute-aware low-resolution residual refinement
evidence_used: The 105,866-parameter baseline achieved 89.28% accuracy, demonstrating capacity headroom; two larger multi-convolution attempts could not be verified, motivating a simpler architecture that adds most computation only after spatial downsampling.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 31.57998920790851, "validation_accuracy": 0.8994, "validation_correct": 8994, "validation_cross_entropy": 0.28040900344848635, "validation_score": 8994.390500221924}



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
