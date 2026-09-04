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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 21.138614665949717, "validation_accuracy": 0.917, "validation_correct": 9170, "validation_cross_entropy": 0.2676764091491699, "validation_score": 9170.394422422309}
prior_hypothesis: Averaging the successful model’s final half-training trajectory will exceed 9,168 correct predictions by reducing optimizer noise without introducing the harmful input transformations seen in flip ensembling.

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

RECENT RESULT
hypothesis: A 241,130-parameter residual BatchNorm CNN trained for roughly twice the optimizer updates will exceed the verified baseline’s 8,928 correct predictions.
change: Replace the shallow CNN with a four-block residual network, use batch size 128, add padded crops and per-image horizontal flips, and train with warmup-cosine AdamW.
mechanism: Parameter-efficient residual feature hierarchy with lightweight translation and reflection augmentation
evidence_used: The verified baseline achieves 89.28% accuracy using only 105,866 parameters and 392 updates, leaving substantial unused capacity and update budget; the unverified designs motivate retaining those core changes while omitting evaluation-time ensembling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241130, "training_seconds": 56.532060417113826, "validation_accuracy": 0.8705, "validation_correct": 8705, "validation_cross_entropy": 0.3728164291381836, "validation_score": 8705.364214755438}

RECENT RESULT
hypothesis: A 239,394-parameter three-convolution extension of the verified baseline, trained with batch size 128 for 782 optimizer steps, will exceed 8,928 correct validation predictions.
change: Widen the baseline to 24/48 channels, add one 48-channel convolution, use an 88-unit classifier, and halve the batch size while retaining the verified loss and optimizer schedule.
mechanism: Conservative width-and-depth scaling without augmentation
evidence_used: The 241,130-parameter residual model with crops, flips, and BatchNorm fell to 8,705 correct; this motivates preserving the baseline’s simple unaugmented training path while testing additional capacity and updates without those confounding changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 22.49971158290282, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.2671722816467285, "validation_score": 9168.39457933798}

RECENT RESULT
hypothesis: Preserving the verified training path while averaging original and horizontally flipped validation predictions will exceed 9,168 correct predictions.
change: Leave training unchanged and ensemble original-image and flipped-image class probabilities only in evaluation mode.
mechanism: Probability-space horizontal-reflection test-time ensembling
evidence_used: The current unaugmented CNN achieved 9,168 correct, while the residual model combining architectural and augmentation changes regressed to 8,705; this isolates a parameter-free symmetry improvement without disturbing the successful optimization path.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 18.34922662517056, "validation_accuracy": 0.8889, "validation_correct": 8889, "validation_cross_entropy": 0.4421930076599121, "validation_score": 8889.346694233951}

RECENT RESULT
hypothesis: Averaging the successful model’s final half-training trajectory will exceed 9,168 correct predictions by reducing optimizer noise without introducing the harmful input transformations seen in flip ensembling.
change: Maintain a 0.99-decay EMA of model parameters during the latter half of training and install the averaged weights after the final optimizer step.
mechanism: Tail exponential weight averaging
evidence_used: The unmodified training path reached 9,168 correct, while horizontal-flip evaluation fell to 8,889; weight-space averaging preserves the validated input distribution while providing parameter-free ensembling.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 21.138614665949717, "validation_accuracy": 0.917, "validation_correct": 9170, "validation_cross_entropy": 0.2676764091491699, "validation_score": 9170.394422422309}



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
