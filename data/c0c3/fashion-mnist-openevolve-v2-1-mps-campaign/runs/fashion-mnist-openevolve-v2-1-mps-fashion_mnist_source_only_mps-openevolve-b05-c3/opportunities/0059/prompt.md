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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}
prior_hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 75.7961785828229, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.23861026840209962, "validation_score": 9163.403678229348}
prior_hypothesis: Using AdamW β₂=0.99 with the verified 39/64 augmentation transition will exceed 9,167 correct predictions by letting adaptive learning rates respond to terminal inference-aligned gradients instead of retaining second-moment statistics from most of the broad-translation phase.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 78.00768562499434, "validation_accuracy": 0.9172, "validation_correct": 9172, "validation_cross_entropy": 0.24028782081604003, "validation_score": 9172.403132233992}
prior_hypothesis: Moving the broad-to-cardinal transition from 39/64 to 19/32 of training will exceed 9,167 correct predictions by continuing the observed improvement from progressively longer terminal phases.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Retaining the best 19/32 augmentation curriculum while disabling label smoothing only for the final eighth of training will exceed 9,172 correct predictions by sharpening decision boundaries at low learning rates without giving up early regularization.
change: Restore the verified 19/32 broad-to-cardinal transition and use hard-label cross-entropy during the final eighth of training.
mechanism: Late hard-label margin refinement
evidence_used: The 19/32 curriculum achieved the best completed result at 9,172 correct, while head widening and AdamW β₂ changes were weaker; the existing augmentation, dropout, weight decay, and EMA permit a focused late loss refinement without architectural cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.
change: Restore the validated 48-unit head and 19/32 transition, and exclude BatchNorm buffers from EMA so evaluation uses statistics accumulated from the terminal inference-aligned distribution.
mechanism: Terminal-aligned BatchNorm statistics with parameter-only EMA
evidence_used: The 48-unit 19/32 design achieved the best completed result at 9,172 correct; the weaker 52-unit head achieved 9,164, and the curriculum shift makes the current 0.005 EMA of BatchNorm buffers a targeted source of stale broad-augmentation statistics.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}

RECENT RESULT
hypothesis: On the verified 19/32 curriculum, increasing only the BatchNorm-buffer EMA rate from 0.005 to 0.01 will exceed 9,172 correct predictions by reducing normalization-statistic lag while retaining the smoothing that outperformed unaveraged final statistics.
change: Restore default AdamW and the best 19/32 transition, keep parameter EMA at 0.005, and use a moderately faster 0.01 EMA for floating BatchNorm buffers.
mechanism: Faster BatchNorm-buffer EMA
evidence_used: The default-AdamW 19/32 design achieved 9,172 correct with all tensors averaged at 0.005, whereas excluding BatchNorm buffers fell to 9,168; this supports retaining buffer averaging while testing a midpoint toward faster terminal-distribution adaptation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final 1/64 of broad ±2 augmentation with dense ±1 translations will exceed 9,172 correct predictions by easing distribution alignment without prematurely imposing the cardinal-only restriction.
change: Use broad ±2 translations through 37/64, all nine ±1 translations from 37/64 to 38/64, then retain the verified center/cardinal terminal phase.
mechanism: Dense-local augmentation bridge
evidence_used: The 38/64 boundary achieved 9,172 correct, while moving cardinal-only augmentation directly to 37/64 fell to 9,160; a mild local bridge tests whether that interval benefits from reduced translation magnitude rather than full terminal restriction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A learnable, max-biased mixture of max and average pooling in the final stage will exceed 9,172 correct predictions by recovering diffuse local evidence while preserving the validated position-sensitive 3×3 representation.
change: Restore the best verified all-tensor EMA and replace only the final max pool with a lightweight per-channel max/average mixture initialized close to max pooling.
mechanism: Per-channel mixed final pooling
evidence_used: The 19/32 curriculum with all-tensor EMA achieved 9,172 correct; the hybrid spatial/global head motivated preserving distributed evidence but timed out, so this 96-parameter pooling change tests that mechanism with negligible computational overhead.
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
