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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 243178, "training_seconds": 27.024489792063832, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21875486373901368, "validation_score": 9258.410254773027}
prior_hypothesis: A 243,178-parameter batch-normalized CNN trained in roughly 782 optimizer steps will exceed the baseline’s 8,928 correct predictions without relying on augmentation or inference ensembling.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 243178, "training_seconds": 28.976443791994825, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.22301236190795898, "validation_score": 9230.40882661171}
prior_hypothesis: Preserving the verified 243,178-parameter design while adding label-preserving horizontal-flip training and two-view inference will exceed 9,258 correct predictions on the fixed validation set.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing feature capacity while remaining below 250,000 parameters, doubling optimizer updates, and training for flip/translation invariance will exceed the current 8,928 correct predictions and reduce the 0.3359 validation cross-entropy.
change: Replace the small two-layer CNN with a batch-normalized residual CNN, use batch size 128, add horizontal-flip and translation augmentation, average flipped logits during validation, and apply warmup-cosine AdamW optimization.
mechanism: Residual multiscale CNN with light geometric augmentation and flip-ensemble inference
evidence_used: The current 105,866-parameter model reaches 89.28% accuracy after only 392 optimizer steps, leaving substantial parameter and update capacity available under the fixed exposure and 250,000-parameter limits.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 229,098-parameter six-layer CNN trained with twice as many optimizer updates will exceed 8,928 validation-correct predictions; light geometric augmentation and averaged flipped logits should also improve the 0.3359 cross-entropy.
change: Use batch size 128, a deeper batch-normalized CNN, random per-image flips plus batchwise translations, evaluation-time flip averaging, lightly smoothed cross-entropy, and warmup-cosine AdamW.
mechanism: Update-dense batch-normalized CNN with translation/flip augmentation and flip-ensemble inference
evidence_used: The verified 105,866-parameter baseline reaches 8,928 correct with only 392 optimizer steps, leaving substantial capacity and update-frequency headroom; the previous more ambitious residual proposal was unverified, motivating a simpler implementation using only standard PyTorch operations.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 243,178-parameter batch-normalized CNN trained in roughly 782 optimizer steps will exceed the baseline’s 8,928 correct predictions without relying on augmentation or inference ensembling.
change: Reduce batch size to 128, replace the baseline with an efficient four-convolution network, use decoupled parameter-group decay, lower label smoothing, and apply warmup-cosine AdamW.
mechanism: Update-dense efficient CNN with depthwise spatial refinement
evidence_used: The verified 105,866-parameter baseline reaches 89.28% accuracy with only 392 optimizer steps. The recent multi-change designs were unverified, so this patch tests the available capacity and update headroom using only standard deterministic layers and training operations.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 243178, "training_seconds": 27.024489792063832, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21875486373901368, "validation_score": 9258.410254773027}

RECENT RESULT
hypothesis: Preserving the verified 243,178-parameter design while adding label-preserving horizontal-flip training and two-view inference will exceed 9,258 correct predictions on the fixed validation set.
change: Adopt the qualified batch-normalized CNN and optimizer schedule, randomly flip individual training images, and average original and mirrored logits during evaluation.
mechanism: Flip-invariant update-dense CNN with evaluation-time logit averaging
evidence_used: The qualified reference achieved 9,258 correct and 0.21875 cross-entropy without augmentation; retaining that verified core isolates horizontal-flip invariance as the next informative improvement, whereas earlier augmented proposals were not successfully verified.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 243178, "training_seconds": 28.976443791994825, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.22301236190795898, "validation_score": 9230.40882661171}



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
