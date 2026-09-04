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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 25.700520290993154, "validation_accuracy": 0.9151, "validation_correct": 9151, "validation_cross_entropy": 0.2413816478729248, "validation_score": 9151.402777019344}
prior_hypothesis: A wider two-convolution-per-stage network trained with random translations and horizontal flips, then evaluated by averaging original and mirrored logits, will exceed the current 8,928 correct predictions while remaining below 250,000 parameters.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 24.57859691698104, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.2382940196990967, "validation_score": 9162.403781324989}
prior_hypothesis: Extending the qualified 9,151-correct design to average logits over centered and one-pixel-shifted views, with their mirrors, will exceed 9,151 correct predictions or reduce cross-entropy at a tie because its training augmentation already teaches the corresponding translation invariance.

## Recent verification evidence

RECENT RESULT
hypothesis: A wider two-convolution-per-stage network trained with random translations and horizontal flips, then evaluated by averaging original and mirrored logits, will exceed the current 8,928 correct predictions while remaining below 250,000 parameters.
change: Use batch size 128, a 241,426-parameter batch-normalized CNN, vectorized crop-and-flip augmentation, a short-warmup cosine AdamW schedule, lighter label smoothing, and horizontal-flip test-time ensembling.
mechanism: Capacity-scaled convolutional ensemble with geometric invariance
evidence_used: The starting 105,866-parameter two-convolution model reaches 89.28% accuracy after 100,000 examples without augmentation, leaving substantial parameter headroom and no explicit geometric invariance.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 25.700520290993154, "validation_accuracy": 0.9151, "validation_correct": 9151, "validation_cross_entropy": 0.2413816478729248, "validation_score": 9151.402777019344}

RECENT RESULT
hypothesis: Extending the qualified 9,151-correct design to average logits over centered and one-pixel-shifted views, with their mirrors, will exceed 9,151 correct predictions or reduce cross-entropy at a tie because its training augmentation already teaches the corresponding translation invariance.
change: Adopt the qualified 241,426-parameter CNN and training recipe, then replace two-view flip evaluation with a ten-view ensemble spanning the center and four one-pixel translations.
mechanism: Translation-orbit test-time ensembling
evidence_used: Reference Design 1 reached 91.51% accuracy using random translations during training but evaluated only the original and mirrored images, leaving matched translation augmentation unexploited at inference.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 24.57859691698104, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.2382940196990967, "validation_score": 9162.403781324989}



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
