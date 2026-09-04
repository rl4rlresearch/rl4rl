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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 73.77810429083183, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.22463382492065428, "validation_score": 9319.408285309311}
prior_hypothesis: Lowering classifier dropout from 0.10 to 0.05 will exceed 9,316 correct predictions by improving short-horizon feature utilization while retaining the beneficial 0.04 label smoothing.

## Recent verification evidence

RECENT RESULT
hypothesis: Initializing AdamW at the schedule’s 20% warmup rate will exceed 9,319 correct predictions by preventing the first update from occurring at full peak learning rate before abruptly dropping.
change: Change AdamW’s initial learning rate from 2.0e-3 to 4.0e-4; retain the existing warmup, cosine schedule, architecture, and regularization.
mechanism: Warmup-consistent optimizer initialization
evidence_used: The strongest configuration achieved 9,319 correct, while subsequent fine-grained dropout changes regressed. Its schedule specifies a 0.2× starting multiplier, but the optimizer currently performs its first update at the full 2.0e-3 peak, making warmup consistency an untested optimization improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 67.19492674991488, "validation_accuracy": 0.9298, "validation_correct": 9298, "validation_cross_entropy": 0.22133691635131836, "validation_score": 9298.40938744527}

RECENT RESULT
hypothesis: Removing the 5% learning-rate warmup will exceed 9,319 correct predictions by preserving aggressive early optimization throughout the short two-pass training horizon.
change: Replace the warmup-plus-cosine schedule with cosine decay from the full 2.0e-3 learning rate to 3% of peak.
mechanism: Warmup-free cosine decay
evidence_used: Making the first update warmup-consistent reduced correctness from 9,319 to 9,298; this suggests the full-rate initial update is beneficial and motivates testing full-rate early training rather than a 5% low-rate ramp.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.63605229184031, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.22911708641052247, "validation_score": 9275.406796069738}

RECENT RESULT
hypothesis: Retaining 0.04 smoothing for 90% of training and annealing it to zero over the final 10% will exceed 9,319 correct predictions by preserving its verified generalization benefit while sharpening class margins near convergence.
change: Replace constant label smoothing with a late linear decay from 0.04 to 0.0.
mechanism: Terminal label-smoothing annealing
evidence_used: Constant 0.04 smoothing achieved 9,316–9,319 correct, while hard labels produced substantially lower cross-entropy but only 9,270 correct; a terminal anneal tests whether their complementary benefits can be combined.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 74.91917475010268, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.2045325019836426, "validation_score": 9309.415098803209}

RECENT RESULT
hypothesis: Ramping label smoothing from 0 to 0.04 during the first 10% of training will exceed 9,319 correct predictions by strengthening early class-learning gradients while retaining smoothing throughout convergence.
change: Linearly warm label smoothing from 0 to 0.04 over the first 10% of optimizer steps, then hold it at 0.04.
mechanism: Early hard-target smoothing ramp
evidence_used: Constant 0.04 smoothing reached 9,319 correct, while annealing it to zero late fell to 9,309; this indicates smoothing is especially valuable near convergence, motivating the reverse schedule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on centered images half the time and cardinally shifted images otherwise will exceed 9,319 correct predictions by reducing the flattening classifier’s sensitivity to small positional changes without excessive augmentation.
change: Apply a deterministic eight-step augmentation cycle containing four centered batches and one batch for each one-pixel cardinal translation.
mechanism: Controlled one-pixel translation augmentation
evidence_used: The strongest configuration reached 9,319 correct, while repeated local changes to smoothing, dropout, and scheduling failed; this tests an orthogonal input-side improvement while preserving the verified model and optimizer.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging predictions for each image and its horizontal reflection will exceed 9,319 correct predictions by reducing the flattening classifier’s residual sensitivity to left-right orientation without adding training-time augmentation overhead.
change: Keep training unchanged; during evaluation, process original and horizontally flipped images together and average their logits.
mechanism: Evaluation-time horizontal-reflection logit ensemble
evidence_used: Repeated fine-grained changes to smoothing, dropout, and scheduling failed to improve 9,319 correct, while training-time translation augmentation exceeded the time limit; evaluation-only ensembling tests transformation robustness without slowing the fixed 100,000-example training loop.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 53.10311104101129, "validation_accuracy": 0.8458, "validation_correct": 8458, "validation_cross_entropy": 0.4334543411254883, "validation_score": 8458.348807761542}

RECENT RESULT
hypothesis: Replacing the final noisy AdamW weights with a 0.99-decay exponential moving average will exceed 9,319 correct predictions by stabilizing the converged classifier without adding regularization during optimization.
change: Track an EMA of every learned parameter after each optimizer step and copy the averaged weights into the model after the final step.
mechanism: Exponential moving-average weight consolidation
evidence_used: Fine-grained changes to dropout, label smoothing, and learning-rate scheduling all failed to improve the 9,319-correct configuration, motivating an orthogonal endpoint-averaging change that preserves its successful training trajectory.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 55.4709343330469, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.22391337509155274, "validation_score": 9315.408525644196}

RECENT RESULT
hypothesis: Reallocating dense-head parameters to a 3×3 convolution at 7×7 resolution will exceed 9,319 correct predictions by learning stronger local features within the same parameter ceiling.
change: Add a 64-channel post-pooling convolution with batch normalization and SiLU, while reducing the classifier hidden width from 44 to 32; total learned parameters become 249,162.
mechanism: Late spatial feature refinement
evidence_used: Dropout, label-smoothing, scheduling, and EMA refinements all failed to improve the 9,319-correct design, motivating an orthogonal capacity reallocation from the flattening classifier into spatial feature extraction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249162, "training_seconds": 77.61421049991623, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.21766921310424805, "validation_score": 9315.410620548355}

RECENT RESULT
hypothesis: Blending max and average pooling at both downsampling stages will exceed 9,319 correct predictions by reducing one-pixel phase sensitivity while preserving salient local activations.
change: Add parameter-free mixed pooling and replace both max-pooling layers, leaving the verified optimizer, schedule, classifier, dropout, and label smoothing unchanged.
mechanism: Mixed max-average downsampling
evidence_used: Controlled one-pixel translation augmentation targeted the current flattening classifier’s positional sensitivity but exceeded the time limit; mixed pooling addresses the same invariance inside the model without expanding the training examples or approaching the parameter ceiling.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 83.05353287514299, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.22335204696655273, "validation_score": 9277.408713093864}

RECENT RESULT
hypothesis: Adding lightweight channel attention at both feature stages will exceed 9,319 correct predictions by dynamically emphasizing class-relevant feature channels while preserving the strong dropout, smoothing, and schedule configuration.
change: Add identity-centered squeeze-excitation gates after both residual blocks and reduce the classifier width from 44 to 43, keeping the model below the parameter ceiling at approximately 248,435 learned parameters.
mechanism: Identity-centered squeeze-excitation channel attention
evidence_used: Reallocating capacity to additional spatial feature processing reached 9,315 correct with improved cross-entropy, showing feature-stage changes remain competitive; channel attention tests a cheaper dynamic reallocation without the timeout risk of mixed pooling or augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Residualizing the competitive 7×7 post-pooling convolution will exceed 9,319 correct predictions by retaining the original feature path while learning the local refinement that previously reduced cross-entropy to 0.21767.
change: Add a zero-initialized residual 3×3 convolution after final pooling and reduce classifier width from 44 to 32, yielding 249,162 learned parameters.
mechanism: Identity-initialized residual spatial refinement
evidence_used: Plain post-pooling convolution achieved 9,315 correct—only four below the best—and substantially improved cross-entropy; an identity-initialized skip connection targets its accuracy regression without changing its parameter count or computational scale.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249162, "training_seconds": 61.674538916908205, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.23161939239501952, "validation_score": 9242.40596957395}

RECENT RESULT
hypothesis: A depthwise-separable 3×3 refinement at 7×7 resolution will exceed 9,319 correct predictions by retaining the competitive post-pooling convolution’s local feature modeling while preserving a 42-unit classifier head.
change: Add a depthwise-separable convolution, batch normalization, and SiLU after final pooling; reduce classifier width from 44 to 42, yielding 248,440 learned parameters.
mechanism: Parameter-efficient late spatial refinement
evidence_used: The full post-pooling convolution reached 9,315 correct with cross-entropy 0.21767, only four predictions below the best despite shrinking the classifier to 32 units; a separable refinement tests the same promising mechanism with far less parameter reallocation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248440, "training_seconds": 81.24332091701217, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.2155227928161621, "validation_score": 9310.411345639057}



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
