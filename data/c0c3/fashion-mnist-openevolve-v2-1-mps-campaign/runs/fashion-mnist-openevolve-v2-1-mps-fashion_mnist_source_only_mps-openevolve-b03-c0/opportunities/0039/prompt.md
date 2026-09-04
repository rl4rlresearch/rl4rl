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

RECENT RESULT
hypothesis: Raising the warmup floor from 20% to 40% of peak will exceed 9,319 correct predictions by accelerating early optimization while retaining the warmup that outperformed full-rate cosine training.
change: Keep the beneficial full-rate first update, then warm from 40% instead of 20% to the unchanged 2.0e-3 peak over 5% of training.
mechanism: Shallower post-kick learning-rate warmup
evidence_used: Warmup-consistent initialization scored 9,298 and removing warmup scored 9,275, versus 9,319 for the current full-rate kick followed by warmup; this motivates preserving both features while testing a less severe post-kick learning-rate drop.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 56.2615264588967, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.22488417091369628, "validation_score": 9290.40820186257}

RECENT RESULT
hypothesis: Replacing the first max pool with a learned stride-2 convolution while retaining a 41-unit classifier will exceed 9,319 correct predictions by preserving fine local structure during the first resolution reduction.
change: Use a normalized, activated 3×3 stride-2 convolution for the first downsampling stage and reduce the classifier width from 44 to 41, yielding 249,773 learned parameters.
mechanism: Learned early strided downsampling
evidence_used: The full 7×7 refinement convolution reached 9,315 correct and improved cross-entropy to 0.21767, indicating learned spatial processing is competitive; placing that capacity at the earlier downsampling boundary preserves more classifier capacity and directly tests learned rather than fixed pooling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249773, "training_seconds": 71.44511995883659, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.22881120300292968, "validation_score": 9273.406897331972}

RECENT RESULT
hypothesis: Reducing BatchNorm momentum from 0.10 to 0.05 will exceed 9,319 correct predictions by reducing evaluation-time normalization noise without changing training dynamics, parameter count, or runtime materially.
change: Set every BatchNorm layer’s running-statistics momentum to 0.05.
mechanism: Stabilized BatchNorm running statistics
evidence_used: EMA weight consolidation remained competitive at 9,315 correct, suggesting endpoint stability matters; BatchNorm statistics are a distinct, untested source of endpoint noise that can be averaged more reliably at batch size 64.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 76.32178745907731, "validation_accuracy": 0.9317, "validation_correct": 9317, "validation_cross_entropy": 0.2248573600769043, "validation_score": 9317.40821079768}

RECENT RESULT
hypothesis: Combining the dominant flattened classifier with a lightweight translation-invariant global-average branch will exceed 9,319 correct predictions by adding robust whole-image evidence without the runtime cost of augmentation.
change: Reduce the dense hidden width from 44 to 43 and add a 64-to-10 classifier over globally averaged final features; sum both branches’ logits. The model has 247,437 learned parameters.
mechanism: Residual global-average classification branch
evidence_used: Translation augmentation targeted positional sensitivity but timed out, while spatial-refinement models remained competitive at 9,315 correct and improved cross-entropy; a global-average residual branch tests complementary spatial invariance cheaply while preserving the successful original feature path.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding BatchNorm to the dense projection will exceed 9,319 correct predictions by improving short-horizon optimization while retaining nearly all classifier capacity.
change: Reduce the hidden width from 44 to 43 and normalize its activations with BatchNorm1d before SiLU, remaining below the parameter ceiling.
mechanism: Normalized dense-head optimization
evidence_used: Spatial refinements improved cross-entropy but lost accuracy when classifier width fell to 32; this instead preserves the dominant flattened head and adds inexpensive optimization support not tested by prior scheduling, smoothing, or endpoint-averaging changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.95 will preserve exactly 9,319 correct predictions while reducing validation cross-entropy below 0.2246338, producing a strict validation_score improvement.
change: Leave training unchanged and sharpen logits only when the model is in evaluation mode.
mechanism: Evaluation-only temperature sharpening
evidence_used: Prior EMA and spatial-refinement variants lowered cross-entropy but lost correct predictions; temperature scaling targets the tie-break metric without changing argmax decisions, while the existing 0.04 label smoothing motivates mild sharpening.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Horizontally flipping every other training batch will exceed 9,319 correct predictions by teaching label-preserving reflection invariance without increasing the example budget or materially increasing runtime.
change: Apply a horizontal flip to alternating training batches while leaving the model, optimizer, schedule, dropout, and label smoothing unchanged.
mechanism: Deterministic horizontal-reflection augmentation
evidence_used: Translation augmentation targeted positional sensitivity but timed out, while parameter-free architectural invariance changes did not improve accuracy; alternating horizontal flips test a cheaper, label-preserving invariance using only a single tensor operation on half the batches.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 66.90075945784338, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.23663444442749024, "validation_score": 9237.404323203395}



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
