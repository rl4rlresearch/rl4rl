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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.94878062489443, "validation_accuracy": 0.9193, "validation_correct": 9193, "validation_cross_entropy": 0.23912537384033203, "validation_score": 9193.403510419976}
prior_hypothesis: On the verified 19/32 curriculum, residual refinement within each convolutional stage will exceed 9,172 correct predictions by preserving useful early-stage features and improving gradient flow during the fixed two-pass exposure budget.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 78.00768562499434, "validation_accuracy": 0.9172, "validation_correct": 9172, "validation_cross_entropy": 0.24028782081604003, "validation_score": 9172.403132233992}
prior_hypothesis: Moving the broad-to-cardinal transition from 39/64 to 19/32 of training will exceed 9,167 correct predictions by continuing the observed improvement from progressively longer terminal phases.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Resetting the all-tensor EMA at the verified 19/32 augmentation transition will exceed 9,172 correct predictions by excluding residual broad-augmentation trajectory averages from the terminal inference-aligned model.
change: Restore the best 19/32 curriculum, retain EMA for parameters and floating buffers, and reinitialize its averages immediately after the first terminal-phase optimizer step.
mechanism: Curriculum-synchronized EMA reset
evidence_used: The all-tensor-EMA 19/32 design achieved the best completed result at 9,172 correct, while unaveraged BatchNorm buffers fell to 9,168; with EMA rate 0.005, roughly 4% of the pre-transition average survives to evaluation, motivating a targeted reset without sacrificing terminal smoothing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: On the verified 19/32 curriculum, residual refinement within each convolutional stage will exceed 9,172 correct predictions by preserving useful early-stage features and improving gradient flow during the fixed two-pass exposure budget.
change: Replace the plain six-layer convolutional chain with three parameter-neutral residual stages while retaining the validated positional head and TTA; restore default AdamW and the best 19/32 curriculum.
mechanism: Stagewise residual feature refinement
evidence_used: The positional 48-unit head reached 9,172 correct, while widening it fell to 9,164 and attention pooling reportedly fell to 9,103. This suggests preserving the validated prediction head and challenging the load-bearing assumption that each second convolution should completely overwrite its stage representation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.94878062489443, "validation_accuracy": 0.9193, "validation_correct": 9193, "validation_cross_entropy": 0.23912537384033203, "validation_score": 9193.403510419976}

RECENT RESULT
hypothesis: Adding input-conditioned per-channel gates to the residual branches will exceed 9,193 correct predictions by preserving the successful residual initialization while learning which refinements should be amplified or suppressed.
change: Add squeeze-excitation gates to all three residual stages, initialized to reproduce the current network exactly; this adds 4,060 parameters for a total of 249,878.
mechanism: Identity-preserving channel-gated residual refinement
evidence_used: Stagewise residual refinement improved validation_correct from 9,172 to 9,193, while head and pooling changes were weaker or timed out, motivating a focused extension of the newly validated residual mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Parameter-neutral residual stages with initially inactive refinement branches will exceed 9,193 correct predictions by retaining the verified residual topology while allowing each stage to learn refinements gradually from an identity-preserving initialization.
change: Replace the plain convolutional chain with the best verified three-stage residual architecture and zero-initialize each second BatchNorm scale; retain the validated head, curriculum, optimizer, EMA, and TTA.
mechanism: Zero-initialized residual refinement
evidence_used: Stagewise residual refinement improved validation_correct from 9,172 to 9,193 with unchanged parameter count, while the larger squeeze-excitation extension timed out; zero-initialized residual branches isolate a lightweight optimization improvement without added parameters or meaningful compute.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding parameter-free downsampled shortcuts between the successful residual stages will exceed 9,193 correct predictions by preserving earlier-stage features across channel expansion and pooling.
change: Restore the verified residual architecture and all-tensor EMA, then add average-pooled, zero-padded shortcuts around the second and third residual stages without increasing parameter count.
mechanism: Cross-stage residual feature preservation
evidence_used: Stagewise residual refinement improved validation_correct from 9,172 to 9,193 at the same parameter count, while parameter-only EMA fell to 9,168; this motivates retaining all-tensor EMA and extending the validated feature-preservation mechanism across stage boundaries.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Learnable per-channel gains initialized to reproduce the verified residual network will exceed 9,193 correct predictions by adapting each refinement branch’s contribution without the computational cost of squeeze-excitation.
change: Restore the best 19/32 residual design and add one unit-initialized gain per output channel to each residual branch, adding 192 parameters for a total of 246,010.
mechanism: Per-channel residual refinement scaling
evidence_used: Stagewise residual refinement improved validation_correct from 9,172 to 9,193 with unchanged parameter count; the more expensive squeeze-excitation extension timed out, motivating an identity-preserving, negligible-overhead test of channel-selective refinement strength.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing only the final 2×2 max pool with a 3×3 stride-2 max pool will exceed 9,193 correct predictions by covering all of the 7×7 final feature map instead of systematically discarding its last row and column.
change: Parameterize residual-stage pooling and use overlapping 3×3 stride-2 pooling in the third stage while preserving the verified residual architecture, 3×3 classifier input, parameter count, optimizer, curriculum, EMA, and TTA.
mechanism: Coverage-complete overlapping final pooling
evidence_used: Stagewise residual refinement produced the best result at 9,193 correct with 245,818 parameters. Its final 2×2 stride-2 pool maps 7×7 to 3×3 while omitting one boundary row and column, so correcting that asymmetric information loss is a targeted parameter-free extension.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding BatchNorm to the verified residual model’s 48-unit classifier bottleneck will exceed 9,193 correct predictions by improving optimization and regularizing the positional head without the weaker capacity increase of head widening.
change: Restore the parameter-neutral three-stage residual architecture and add BatchNorm1d after the existing 48-unit classifier projection, for 245,914 learned parameters.
mechanism: Residual feature refinement with a normalized positional bottleneck
evidence_used: Stagewise residual refinement achieved the best completed result at 9,193 correct with 245,818 parameters, while widening the classifier head was weaker; bottleneck normalization targets optimization rather than additional capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: On the verified residual architecture, increasing all-tensor EMA interpolation from 0.005 to 0.0075 will exceed 9,193 correct predictions by reducing pre-transition influence while retaining the BatchNorm smoothing lost by parameter-only EMA.
change: Restore the validated three-stage residual network and all-tensor EMA, then use a moderately faster 0.0075 EMA rate.
mechanism: Faster tail-tracking EMA on residual features and normalization statistics
evidence_used: Residual stages with all-tensor 0.005 EMA achieved 9,193 correct, versus 9,168 for parameter-only EMA; at 0.0075, less than 1% of the pre-transition average survives the terminal phase, compared with about 4% at 0.005.
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
