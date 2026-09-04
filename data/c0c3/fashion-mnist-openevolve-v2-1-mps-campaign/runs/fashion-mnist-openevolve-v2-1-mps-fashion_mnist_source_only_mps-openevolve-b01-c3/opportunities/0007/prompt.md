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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 31.955160333076492, "validation_accuracy": 0.9111, "validation_correct": 9111, "validation_cross_entropy": 0.25698145141601564, "validation_score": 9111.397778343855}
prior_hypothesis: Extending the proven flip ensemble to five one-pixel translation views will exceed 9,094 correct predictions because training already exposes the model to random translations and horizontal reflections.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 23.78748208307661, "validation_accuracy": 0.9109, "validation_correct": 9109, "validation_cross_entropy": 0.2574250900268555, "validation_score": 9109.397638001632}
prior_hypothesis: Adding the four diagonal one-pixel translations to the best five-position ensemble will exceed 9,111 correct predictions because training exposes the residual model to joint horizontal and vertical translations.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 24.031964333960786, "validation_accuracy": 0.9094, "validation_correct": 9094, "validation_cross_entropy": 0.2598704833984375, "validation_score": 9094.39686619108}
prior_hypothesis: Averaging original and horizontally flipped logits at evaluation will exceed 9,086 correct predictions because the model is already trained with 50% horizontal flips, making both views class-preserving ensemble members.

## Recent verification evidence

RECENT RESULT
hypothesis: A 232,146-parameter three-convolution network, trained with twice as many optimizer updates and mild class-preserving augmentation, will exceed the baseline’s 8,928 correct predictions.
change: Widen and deepen the CNN, add batch normalization and dropout, use batch size 128, apply per-image translations and horizontal flips, reduce label smoothing, and use warmup-cosine AdamW.
mechanism: Wider batch-normalized CNN with translation and reflection invariance
evidence_used: The baseline reaches 8,928 correct with only 105,866 parameters and 392 optimizer steps, leaving substantial parameter capacity and optimization granularity available under the fixed exposure budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232146, "training_seconds": 19.57713604113087, "validation_accuracy": 0.9035, "validation_correct": 9035, "validation_cross_entropy": 0.2694350746154785, "validation_score": 9035.393875992557}

RECENT RESULT
hypothesis: Adding a 32-channel convolution before the second pooling stage while retaining the verified training recipe will exceed 9,035 correct predictions by preserving more 14×14 spatial detail.
change: Reproduce the qualified batch-128 augmented AdamW design, then add one batch-normalized convolution and delay its second pooling operation; the resulting model has approximately 241,426 learned parameters.
mechanism: Mid-resolution convolutional refinement before downsampling
evidence_used: Reference Design 1 achieved 9,035 correct with 232,146 parameters, demonstrating that the wider three-convolution recipe is effective while leaving enough capacity for a 9,280-parameter mid-resolution refinement layer.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 27.946601500036195, "validation_accuracy": 0.9078, "validation_correct": 9078, "validation_cross_entropy": 0.25985166244506835, "validation_score": 9078.396872119873}

RECENT RESULT
hypothesis: Adding a residual 64-channel convolutional block at 7×7 while shrinking the dense layer to remain below 250,000 parameters will exceed the reference design’s 9,078 correct predictions.
change: Reproduce the verified batch-128 augmentation and warmup-cosine recipe, then replace dense capacity with a two-convolution residual refinement stage; estimated parameter count is 240,654.
mechanism: Late-resolution residual convolutional refinement
evidence_used: Reference Design 2 improved from 9,035 to 9,078 correct by adding convolutional refinement before downsampling, motivating another spatial refinement layer while retaining its proven training recipe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 36.23156945896335, "validation_accuracy": 0.9086, "validation_correct": 9086, "validation_cross_entropy": 0.2632421043395996, "validation_score": 9086.395806946493}

RECENT RESULT
hypothesis: Averaging original and horizontally flipped logits at evaluation will exceed 9,086 correct predictions because the model is already trained with 50% horizontal flips, making both views class-preserving ensemble members.
change: Keep training unchanged, but during evaluation average logits from each validation image and its horizontal reflection.
mechanism: Flip-ensemble inference
evidence_used: The current 9,086-correct design already applies random horizontal flips throughout training; flip-averaged inference directly exploits that learned invariance without adding parameters or reducing the fixed training exposure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 24.031964333960786, "validation_accuracy": 0.9094, "validation_correct": 9094, "validation_cross_entropy": 0.2598704833984375, "validation_score": 9094.39686619108}

RECENT RESULT
hypothesis: Extending the proven flip ensemble to five one-pixel translation views will exceed 9,094 correct predictions because training already exposes the model to random translations and horizontal reflections.
change: Use the validated 240,654-parameter residual architecture and, during evaluation, average logits across the centered and four cardinally shifted views plus their horizontal reflections.
mechanism: Translation-and-flip test-time logit ensemble
evidence_used: The residual architecture reached 9,086 correct, and flip averaging raised it to 9,094 without retraining; the same model is also trained on translations of up to two pixels, motivating translation averaging as the next inference-time invariance to exploit.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 31.955160333076492, "validation_accuracy": 0.9111, "validation_correct": 9111, "validation_cross_entropy": 0.25698145141601564, "validation_score": 9111.397778343855}

RECENT RESULT
hypothesis: Adding the four diagonal one-pixel translations to the best five-position ensemble will exceed 9,111 correct predictions because training exposes the residual model to joint horizontal and vertical translations.
change: Replace the current CNN with the validated 240,654-parameter residual architecture and average evaluation logits across all nine one-pixel translation positions and their horizontal reflections.
mechanism: Nine-position translation-and-flip test-time ensemble
evidence_used: The residual model improved from 9,094 to 9,111 correct when centered and cardinal translations were ensembled; diagonal views complete the local 3×3 translation neighborhood already represented by training augmentation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 23.78748208307661, "validation_accuracy": 0.9109, "validation_correct": 9109, "validation_cross_entropy": 0.2574250900268555, "validation_score": 9109.397638001632}



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
