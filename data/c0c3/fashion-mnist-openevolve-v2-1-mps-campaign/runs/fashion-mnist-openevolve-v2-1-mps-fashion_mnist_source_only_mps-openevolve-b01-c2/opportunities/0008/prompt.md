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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 215550, "training_seconds": 37.8976491659414, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2080647979736328, "validation_score": 9287.41388508368}
prior_hypothesis: Reallocating dense-head parameters to a residual 7×7 convolution while retaining Reference Design 3’s paired-flip loss will exceed 9,276 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 42.01414312492125, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21310321731567383, "validation_score": 9268.412166081882}
prior_hypothesis: Training both horizontal orientations with ensemble-aligned supervision and evaluating an EMA of the final optimization trajectory will exceed the best prior result of 9,276 correct predictions.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 33.73269945802167, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.21099624671936035, "validation_score": 9276.412883195431}
prior_hypothesis: Supervising each horizontal orientation individually while retaining the validation-matched ensemble loss will exceed 9,261 correct predictions.

## Recent verification evidence

RECENT RESULT
hypothesis: Using the unused parameter budget for batch-normalized residual features, while doubling optimizer updates and training for translation/flip invariance, will exceed 8,928 correct validation predictions.
change: Replace the shallow CNN with a 24/48/80-channel residual network, use batch size 128, per-image affine augmentation, decoupled AdamW decay, warmup-cosine scheduling, lighter label smoothing, and horizontal-flip inference averaging.
mechanism: Residual multiscale CNN with affine augmentation and flip ensembling
evidence_used: The current 105,866-parameter model reaches 8,928 correct with only 392 optimizer steps, leaving substantial capacity under the 250,000-parameter ceiling and supporting a higher-capacity, more frequently updated model.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 216k-parameter four-convolution model trained with twice as many optimizer updates and horizontal-flip regularization will exceed 8,928 correct validation predictions.
change: Increase convolutional capacity within the parameter ceiling, reduce batch size to 128, add normalization and dropout, apply per-image horizontal flips, use flip-averaged inference, and adopt warmup-cosine AdamW training.
mechanism: Batch-normalized deep CNN with flip invariance
evidence_used: The verified 105,866-parameter shallow model achieved 8,928 correct in only 392 steps; the broader prior change could not be verified, motivating a simpler capacity-and-update increase without affine sampling or residual machinery.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 30.331214583013207, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21675402641296387, "validation_score": 9247.410929398338}

RECENT RESULT
hypothesis: Adding per-image two-pixel translation augmentation to the verified 216,346-parameter flip-invariant CNN will exceed 9,247 correct validation predictions.
change: Adopt Reference Design 1’s architecture and optimization, then add normalized-background-safe random padded crops during training.
mechanism: Translation-regularized flip-invariant CNN
evidence_used: The reference design achieved 9,247 correct versus the current 8,928; translation augmentation is the most focused untested extension of that verified design.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.832512624794617, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.2465844955444336, "validation_score": 9141.40109595602}

RECENT RESULT
hypothesis: Optimizing the verified 216,346-parameter model on both horizontal orientations of every presented image, using the same averaged logits used at validation, will exceed 9,247 correct predictions.
change: Adopt Reference Design 1, but replace stochastic one-view flipping with paired original/flip training and an ensemble-level loss.
mechanism: Paired-flip orbit training with flip-ensemble inference
evidence_used: Reference Design 1 achieved the best result at 9,247 correct with flip augmentation and flip-averaged inference, while adding translation reduced performance to 9,141; this motivates strengthening the validated flip invariance without introducing translation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.3079188328702, "validation_accuracy": 0.9261, "validation_correct": 9261, "validation_cross_entropy": 0.21184926834106446, "validation_score": 9261.412592566636}

RECENT RESULT
hypothesis: Supervising each horizontal orientation individually while retaining the validation-matched ensemble loss will exceed 9,261 correct predictions.
change: Blend cross-entropy on the averaged logits with cross-entropy on each original and flipped view.
mechanism: Paired-flip deep supervision with ensemble-aligned loss
evidence_used: Paired-flip ensemble training reached 9,261 correct versus 9,247 for stochastic flipping, while translation fell to 9,141; this isolates stronger flip supervision without introducing a harmful augmentation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 33.73269945802167, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.21099624671936035, "validation_score": 9276.412883195431}

RECENT RESULT
hypothesis: Training both horizontal orientations with ensemble-aligned supervision and evaluating an EMA of the final optimization trajectory will exceed the best prior result of 9,276 correct predictions.
change: Replace stochastic flipping with paired-view training, blend per-view and ensemble losses, and maintain a 0.99-decay parameter EMA that is installed after the final optimizer step.
mechanism: Paired-flip deep supervision with terminal weight averaging
evidence_used: Paired-flip deep supervision achieved 9,276 correct, outperforming stochastic flips at 9,247 and translation augmentation at 9,141; EMA directly extends the strongest verified mechanism by reducing late-step parameter noise without changing its augmentation assumptions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 42.01414312492125, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21310321731567383, "validation_score": 9268.412166081882}

RECENT RESULT
hypothesis: Reallocating dense-head parameters to a residual 7×7 convolution while retaining Reference Design 3’s paired-flip loss will exceed 9,276 correct predictions.
change: Remove harmful translation augmentation, restore paired-flip deep supervision, and replace classifier width with a budget-neutral residual convolutional refinement stage.
mechanism: Budget-neutral residual feature refinement with paired-flip supervision
evidence_used: Reference Design 3 achieved the best result—9,276 correct—with paired-flip supervision, while translation reduced performance to 9,141; its 216,346 parameters are concentrated in the dense head, motivating a clean reallocation toward spatial feature extraction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 215550, "training_seconds": 37.8976491659414, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2080647979736328, "validation_score": 9287.41388508368}



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
