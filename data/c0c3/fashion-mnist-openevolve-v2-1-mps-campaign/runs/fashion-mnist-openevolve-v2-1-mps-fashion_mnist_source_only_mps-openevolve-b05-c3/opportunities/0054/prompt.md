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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.46581783308648, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.23885296478271484, "validation_score": 9167.403599147125}
prior_hypothesis: Beginning center/cardinal augmentation after 39/64 of training will exceed 9,163 correct predictions by extending the monotonically beneficial inference-aligned phase beyond three-eighths while making a smaller change than the timed-out two-fifths configuration.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249318, "training_seconds": 77.04319570795633, "validation_accuracy": 0.9164, "validation_correct": 9164, "validation_cross_entropy": 0.2412181049346924, "validation_score": 9164.402830089259}
prior_hypothesis: Widening the validated flattened classifier head from 48 to 52 units on the best 39/64 curriculum will exceed 9,167 correct predictions by using the remaining parameter budget to improve position-sensitive class separation.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 75.7961785828229, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.23861026840209962, "validation_score": 9163.403678229348}
prior_hypothesis: Using AdamW β₂=0.99 with the verified 39/64 augmentation transition will exceed 9,167 correct predictions by letting adaptive learning rates respond to terminal inference-aligned gradients instead of retaining second-moment statistics from most of the broad-translation phase.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 78.00768562499434, "validation_accuracy": 0.9172, "validation_correct": 9172, "validation_cross_entropy": 0.24028782081604003, "validation_score": 9172.403132233992}
prior_hypothesis: Moving the broad-to-cardinal transition from 39/64 to 19/32 of training will exceed 9,167 correct predictions by continuing the observed improvement from progressively longer terminal phases.

## Recent verification evidence

RECENT RESULT
hypothesis: Widening the validated flattened classifier head from 48 to 52 units on the best 39/64 curriculum will exceed 9,167 correct predictions by using the remaining parameter budget to improve position-sensitive class separation.
change: Restore the verified 39/64 augmentation transition and widen both classifier-layer dimensions to 52, raising learned parameters from 245,818 to 249,318 with negligible added computation.
mechanism: Parameter-ceiling positional head expansion
evidence_used: Reference Design 1 achieved the best completed result at 9,167 correct using the 39/64 transition, while attention pooling fell to 9,103, supporting retention and modest expansion of the positional flattened head rather than another computationally heavier architectural change.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249318, "training_seconds": 77.04319570795633, "validation_accuracy": 0.9164, "validation_correct": 9164, "validation_cross_entropy": 0.2412181049346924, "validation_score": 9164.402830089259}

RECENT RESULT
hypothesis: Adding lightweight squeeze-excitation to the verified 39/64 curriculum will exceed 9,167 correct predictions by adaptively reweighting convolutional channels without sacrificing the position-sensitive 48-unit head.
change: Restore the best verified augmentation boundary and spend the remaining parameter budget on an identity-initialized 96→16→96 channel gate, producing 249,002 learned parameters.
mechanism: Identity-initialized channel recalibration
evidence_used: The 39/64 curriculum with the 48-unit head achieved the best result of 9,167 correct, while widening that head to 52 units reduced performance to 9,164; this motivates allocating spare capacity to feature recalibration instead of further head widening.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: On the verified 39/64 curriculum, balancing the five inference-aligned translation choices within every terminal minibatch will exceed 9,167 correct predictions by reducing augmentation-distribution noise without reducing view diversity.
change: Restore the best verified 39/64 transition and replace independent terminal direction draws with a rotating, near-uniform assignment across each batch.
mechanism: Stratified terminal translation sampling
evidence_used: Reference Design 1 achieved the best completed result—9,167 correct—with the 39/64 boundary, while later transitions performed worse. The terminal phase and inference both use the same five translation directions, motivating exact minibatch stratification as a focused improvement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Symmetric final-stage pooling will exceed 9,167 correct predictions by improving one-pixel translation consistency without discarding the position-sensitive 3×3 feature grid.
change: Replace the final 2×2 stride-2 pool, which ignores the last feature row and column of the 7×7 map, with overlapping 3×3 stride-2 pooling that covers the complete map.
mechanism: Boundary-complete overlapping max pooling
evidence_used: The 39/64 translation curriculum achieved 9,167 correct, while global attention pooling fell to 9,103; this motivates retaining localized spatial pooling while removing its asymmetric boundary truncation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the 48-unit positional head and widening the final feature map from 96 to 99 channels will exceed 9,167 correct predictions by spending spare parameters on spatial feature extraction rather than the head.
change: Revert the classifier head to the best verified width and widen only the final convolution, producing 249,712 learned parameters.
mechanism: Late convolutional feature-capacity reallocation
evidence_used: The 48-unit reference achieved 9,167 correct, while widening its head to 52 units used comparable capacity but fell to 9,164; this motivates allocating the remaining budget to convolutional representation quality instead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using AdamW β₂=0.99 with the verified 39/64 augmentation transition will exceed 9,167 correct predictions by letting adaptive learning rates respond to terminal inference-aligned gradients instead of retaining second-moment statistics from most of the broad-translation phase.
change: Restore the best verified 39/64 augmentation boundary and shorten AdamW’s second-moment memory from the default 0.999 to 0.99.
mechanism: Short-memory adaptive variance for curriculum shifts
evidence_used: The 39/64 curriculum produced the best completed result at 9,167 correct; its deliberate mid-run distribution change motivates testing an optimizer whose variance estimate adapts substantially faster than the default, whose approximately 693-step half-life spans much of this 1,564-step run.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 75.7961785828229, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.23861026840209962, "validation_score": 9163.403678229348}

RECENT RESULT
hypothesis: Moving the broad-to-cardinal transition from 39/64 to 19/32 of training will exceed 9,167 correct predictions by continuing the observed improvement from progressively longer terminal phases.
change: Use broad ±2 translations for the first 19/32 of training, then center/cardinal one-pixel translations for the remaining 13/32.
mechanism: Extended inference-aligned augmentation curriculum
evidence_used: Extending the inference-aligned phase improved correct predictions monotonically from 9,141 at one quarter to 9,155 at one third, 9,163 at three eighths, and 9,167 at 25/64; the next equal 1/64 extension is the most focused continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 78.00768562499434, "validation_accuracy": 0.9172, "validation_correct": 9172, "validation_cross_entropy": 0.24028782081604003, "validation_score": 9172.403132233992}

RECENT RESULT
hypothesis: Moving the broad-to-cardinal transition from 38/64 to 37/64 of training will exceed 9,172 correct predictions by continuing the observed monotonic improvement from successively longer terminal inference-aligned phases.
change: Use broad ±2 translations for the first 37/64 of training, then center/cardinal one-pixel translations for the remaining 27/64.
mechanism: Further extended inference-aligned augmentation curriculum
evidence_used: Extending the terminal phase from 25/64 to 26/64 raised validation_correct from 9,167 to 9,172, continuing the earlier monotonic gains at terminal phases of one quarter, one third, and three eighths; the next one-step boundary change is the most informative continuation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 75.55717633315362, "validation_accuracy": 0.916, "validation_correct": 9160, "validation_cross_entropy": 0.24011634140014648, "validation_score": 9160.40318797786}

RECENT RESULT
hypothesis: Combining the retained 3×3 max-pooled spatial grid with pre-pool channel averages will exceed 9,172 correct predictions by preserving position-sensitive evidence while recovering diffuse texture and boundary information discarded by the final max pool.
change: Replace the assumption that final max-pooled activations alone contain all useful evidence with a learned head that concatenates local maxima and global channel means; reduce its hidden width to 47 to stay below 250,000 parameters, and use the best verified 19/32 curriculum.
mechanism: Hybrid spatial-grid and global-distribution head
evidence_used: The 19/32 curriculum achieved 9,172 correct, while widening the existing head fell to 9,164 and attention pooling reportedly fell to 9,103. This motivates adding complementary global evidence without discarding the validated positional representation or materially increasing convolutional computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Placing the broad-to-cardinal transition at 75/128 of training will exceed 9,172 correct predictions by refining the bracketed optimum between the successful 76/128 boundary and unsuccessful 74/128 boundary.
change: Restore the validated 48-unit positional head and use broad translations for the first 75/128 of training, followed by inference-aligned center/cardinal translations.
mechanism: Midpoint-refined augmentation curriculum
evidence_used: The 19/32 (76/128) curriculum achieved 9,172 correct, outperforming 39/64 (78/128) at 9,167, while extending the terminal phase to 37/64 (74/128) fell to 9,160; testing their unmeasured midpoint is the most focused boundary refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using a four-times-faster EMA for BatchNorm statistics on the best 19/32 curriculum will exceed 9,172 correct predictions by aligning normalization with the terminal inference-matched distribution while retaining strong weight averaging.
change: Restore default AdamW β₂ and the verified 19/32 transition, then update floating BatchNorm buffers at EMA rate 0.02 while keeping learned parameters at 0.005.
mechanism: Curriculum-responsive BatchNorm EMA
evidence_used: The default-AdamW 19/32 design achieved the best completed result at 9,172 correct, whereas β₂=0.99 achieved 9,163; the terminal distribution shift specifically motivates faster adaptation of normalization statistics.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Moving the broad-to-cardinal transition from 76/128 to 77/128 of training will exceed 9,172 correct predictions by locating a better boundary between the best 76/128 result and the weaker 78/128 result.
change: Use broad ±2 translations for the first 77/128 of training, followed by center/cardinal one-pixel translations for the remaining 51/128.
mechanism: Half-step augmentation-boundary refinement
evidence_used: The 76/128 boundary achieved 9,172 correct versus 9,167 at 78/128, while 74/128 fell to 9,160; 77/128 is the untested half-step on the stronger side of this bracket.
result: training did not finish within the verification time limit



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
