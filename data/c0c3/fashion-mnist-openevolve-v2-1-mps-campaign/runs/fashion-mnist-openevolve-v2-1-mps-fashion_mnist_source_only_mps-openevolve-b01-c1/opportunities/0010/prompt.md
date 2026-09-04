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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 27.360899458872154, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2729324203491211, "validation_score": 9063.392793829435}
prior_hypothesis: A 90% full-range and 10% central-range blend will preserve at least 9,063 correct predictions while lowering cross-entropy below 0.273786, improving the tie-broken validation score.

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

RECENT RESULT
hypothesis: Averaging original and horizontally flipped logits will exceed 8,994 correct predictions because the verified model was trained with 50% horizontal flips and already reached 89.94% accuracy.
change: Preserve single-view training while using a two-view logit ensemble whenever the model is in evaluation mode.
mechanism: Evaluation-time flip ensembling
evidence_used: The current 230,442-parameter design achieved 8,994 correct predictions with horizontal-flip augmentation, directly motivating inference-time averaging over the same label-preserving transformation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 23.807681124890223, "validation_accuracy": 0.9039, "validation_correct": 9039, "validation_cross_entropy": 0.2744351547241211, "validation_score": 9039.392330671471}

RECENT RESULT
hypothesis: Averaging logits across nine one-pixel translations and their horizontal flips will exceed 9,039 correct predictions because flip ensembling already added 45 correct predictions and translation augmentation was also used throughout training.
change: Replace two-view evaluation with an 18-view reflected translation-and-flip logit average while leaving training and parameter count unchanged.
mechanism: Translation-and-flip test-time ensemble
evidence_used: Two-view flip ensembling improved validation correct from 8,994 to 9,039; this applies the same successful inference-time averaging idea to the model’s other trained invariance.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 27.713475833181292, "validation_accuracy": 0.9055, "validation_correct": 9055, "validation_cross_entropy": 0.2685044647216797, "validation_score": 9055.394164950858}

RECENT RESULT
hypothesis: Averaging logits across all twenty-five two-pixel translations and their horizontal flips will exceed 9,055 correct predictions because training sampled the same five-by-five reflected translation distribution, while the successful current ensemble covers only its central nine translations.
change: Expand evaluation-time augmentation from 18 views over one-pixel shifts to 50 views over two-pixel shifts, without changing training or parameter count.
mechanism: Training-distribution-matched translation-and-flip test-time ensemble
evidence_used: Expanding flip-only evaluation to nine translations plus flips improved validation correct from 9,039 to 9,055; matching the full translation range used during training is the direct next test of that successful mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 26.29882675013505, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2737864295959473, "validation_score": 9063.392530481078}

RECENT RESULT
hypothesis: Blending the 50-view ensemble with its central 18-view subset will exceed 9,063 correct predictions by retaining useful two-pixel views while reducing their influence on borderline examples.
change: Average 75% of the existing five-by-five translation ensemble with 25% of the better-calibrated central three-by-three ensemble.
mechanism: Center-weighted multi-radius test-time ensemble
evidence_used: Uniform 50-view evaluation improved correct predictions from 9,055 to 9,063 but worsened cross-entropy from 0.26850 to 0.27379, indicating that outer translations add useful decisions yet should receive less weight.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 27.678473208099604, "validation_accuracy": 0.9062, "validation_correct": 9062, "validation_cross_entropy": 0.27179238357543944, "validation_score": 9062.393145930466}

RECENT RESULT
hypothesis: Averaging predictive probabilities across the existing 50 views will exceed 9,063 correct predictions or, if correct counts tie, lower validation cross-entropy by preventing extreme outer-translation logits from dominating the ensemble.
change: Softmax each view’s logits before averaging, then return the log of the averaged class probabilities for exact cross-entropy evaluation.
mechanism: Probability-space transformation ensemble
evidence_used: The uniform 50-view logit ensemble improved correct predictions from 9,055 to 9,063 but worsened cross-entropy from 0.26850 to 0.27379, indicating useful outer-view decisions alongside poorly calibrated logit magnitudes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 30.096750458003953, "validation_accuracy": 0.9058, "validation_correct": 9058, "validation_cross_entropy": 0.2840727714538574, "validation_score": 9058.389386031007}

RECENT RESULT
hypothesis: A 90% full-range and 10% central-range blend will preserve at least 9,063 correct predictions while lowering cross-entropy below 0.273786, improving the tie-broken validation score.
change: Give the central nine translation pairs a modest extra weight while retaining 90% of the successful uniform 50-view ensemble.
mechanism: Light center-weighted multi-radius logit ensemble
evidence_used: The 75%/25% blend lowered cross-entropy to 0.271792 but lost one correct prediction; a smaller 10% center correction tests whether its calibration benefit can be retained without crossing that decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 27.360899458872154, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2729324203491211, "validation_score": 9063.392793829435}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
