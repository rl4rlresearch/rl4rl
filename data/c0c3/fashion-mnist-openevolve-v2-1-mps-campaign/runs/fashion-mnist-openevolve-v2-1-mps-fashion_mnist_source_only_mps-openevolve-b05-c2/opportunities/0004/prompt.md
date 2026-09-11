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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 65.1923690000549, "validation_accuracy": 0.919, "validation_correct": 9190, "validation_cross_entropy": 0.23743358840942383, "validation_score": 9190.40406208841}
prior_hypothesis: Adding four one-pixel translation views to the best 249,260-parameter residual design’s flip ensemble will exceed 9,177 correct predictions by reducing residual shift sensitivity without changing training exposure or learned-parameter count.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241538, "training_seconds": 36.692127916030586, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.24475890426635744, "validation_score": 9168.40168421233}
prior_hypothesis: Increasing convolutional capacity while training on random translations and flips will exceed 8,928 correct predictions within the fixed exposure and parameter budgets.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.643368375021964, "validation_accuracy": 0.9177, "validation_correct": 9177, "validation_cross_entropy": 0.24093204765319826, "validation_score": 9177.40292294888}
prior_hypothesis: Replacing the reference design’s large dense head with two additional residual convolutions while retaining its augmentation and optimization recipe will exceed 9,168 correct validation predictions within the parameter budget.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing convolutional capacity while training on random translations and flips will exceed 8,928 correct predictions within the fixed exposure and parameter budgets.
change: Use a 241,538-parameter four-convolution network, batch size 128, spatial augmentation, lighter label smoothing, cosine AdamW, and horizontal-flip evaluation averaging.
mechanism: Invariance-trained deeper batch-normalized CNN with flip test-time averaging
evidence_used: The current 105,866-parameter two-convolution model reaches 89.28% accuracy, suggesting additional feature-extraction capacity and explicit spatial invariance are promising under the 250,000-parameter ceiling.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241538, "training_seconds": 36.692127916030586, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.24475890426635744, "validation_score": 9168.40168421233}

RECENT RESULT
hypothesis: Replacing the reference design’s large dense head with two additional residual convolutions while retaining its augmentation and optimization recipe will exceed 9,168 correct validation predictions within the parameter budget.
change: Use a 249,260-parameter six-convolution residual CNN with batch size 128, translation/flip augmentation, flip-averaged evaluation, light label smoothing, and cosine AdamW.
mechanism: Residual convolutional capacity reallocation
evidence_used: The 241,538-parameter augmented four-convolution reference achieved 91.68% accuracy; most of its parameters are concentrated in the dense head, motivating reallocating them to deeper spatial feature extraction with residual shortcuts.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.643368375021964, "validation_accuracy": 0.9177, "validation_correct": 9177, "validation_cross_entropy": 0.24093204765319826, "validation_score": 9177.40292294888}

RECENT RESULT
hypothesis: Adding four one-pixel translation views to the best 249,260-parameter residual design’s flip ensemble will exceed 9,177 correct predictions by reducing residual shift sensitivity without changing training exposure or learned-parameter count.
change: Adopt Reference Design 2’s residual CNN and training recipe, then average validation logits across centered, four cardinally shifted, and horizontally flipped views.
mechanism: Matched translation-and-flip test-time ensembling
evidence_used: Reference Design 2 achieved 9,177 correct with random ±2-pixel translations but evaluates only the centered image and its flip, motivating an evaluation ensemble matched to its learned translation invariance.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 65.1923690000549, "validation_accuracy": 0.919, "validation_correct": 9190, "validation_cross_entropy": 0.23743358840942383, "validation_score": 9190.40406208841}



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
