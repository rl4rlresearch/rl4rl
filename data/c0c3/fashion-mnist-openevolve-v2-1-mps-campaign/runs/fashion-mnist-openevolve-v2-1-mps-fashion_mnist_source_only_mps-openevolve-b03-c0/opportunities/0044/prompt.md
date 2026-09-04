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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 76.18434812501073, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20153608093261718, "validation_score": 9319.416133987097}
prior_hypothesis: Reducing evaluation temperature from 0.85 to 0.80 will preserve all 9,319 argmax-correct predictions while lowering validation cross-entropy below 0.20476837.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Dilating the second convolution of the 64-channel residual block will exceed 9,319 correct predictions by capturing broader garment structure while preserving the successful classifier capacity, parameter count, and training cost.
change: Make residual blocks optionally dilated and use dilation 2 only for the second convolution at 14×14 resolution.
mechanism: Multiscale dilated residual context
evidence_used: Late spatial refinement reached 9,315 correct and substantially improved cross-entropy, indicating that additional spatial context is promising; dilation adds that context without shrinking the dense head as prior refinement designs required.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 60.955002875067294, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.22027728385925294, "validation_score": 9300.40974293844}

RECENT RESULT
hypothesis: Sharpening evaluation logits by a temperature of 0.95 will preserve all 9,319 argmax-correct predictions while reducing validation cross-entropy below 0.2246338.
change: Divide logits by 0.95 only during evaluation, leaving training, parameters, and optimizer behavior unchanged.
mechanism: Evaluation-only temperature sharpening
evidence_used: The prior temperature-scaling verification timed out rather than producing negative accuracy evidence; label smoothing can leave logits mildly underconfident, and positive temperature scaling mathematically preserves every argmax prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.49659595801495, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.21638825035095216, "validation_score": 9319.41105296755}

RECENT RESULT
hypothesis: Reducing evaluation temperature from 0.95 to 0.90 will preserve all 9,319 argmax-correct predictions while lowering validation cross-entropy below 0.21638825.
change: Divide evaluation logits by 0.90 instead of 0.95, leaving training and model parameters unchanged.
mechanism: Stronger evaluation-only temperature sharpening
evidence_used: Evaluation-only temperature 0.95 preserved 9,319 correct predictions and improved cross-entropy; this tests whether the calibration optimum lies at a slightly stronger sharpening level.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 82.00251500005834, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20974316787719727, "validation_score": 9319.413310869015}

RECENT RESULT
hypothesis: Reducing evaluation temperature from 0.90 to 0.85 will preserve all 9,319 argmax-correct predictions while lowering validation cross-entropy below 0.20974317.
change: Divide evaluation logits by 0.85 instead of 0.90, leaving training, parameters, and optimizer behavior unchanged.
mechanism: Stronger evaluation-only temperature sharpening
evidence_used: Temperature reductions from 1.00 to 0.95 and then 0.90 each preserved exactly 9,319 correct predictions while lowering cross-entropy from 0.2246338 to 0.21638825 and then 0.20974317, motivating one further calibrated step in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.10361987515353, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20476837387084962, "validation_score": 9319.41501753436}

RECENT RESULT
hypothesis: Reducing evaluation temperature from 0.85 to 0.80 will preserve all 9,319 argmax-correct predictions while lowering validation cross-entropy below 0.20476837.
change: Divide evaluation logits by 0.80 instead of 0.85, leaving training, parameters, and optimizer behavior unchanged.
mechanism: Evaluation-only temperature sharpening
evidence_used: Successive temperatures of 0.95, 0.90, and 0.85 each preserved exactly 9,319 correct predictions while reducing cross-entropy to 0.21638825, 0.20974317, and 0.20476837, respectively, indicating the calibration optimum has not yet been reached.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 76.18434812501073, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20153608093261718, "validation_score": 9319.416133987097}



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
