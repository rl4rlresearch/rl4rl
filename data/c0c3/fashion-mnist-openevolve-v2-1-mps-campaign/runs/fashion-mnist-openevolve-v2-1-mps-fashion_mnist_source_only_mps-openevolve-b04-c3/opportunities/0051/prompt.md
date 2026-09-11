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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 246010, "training_seconds": 81.55593858310021, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.21357478485107423, "validation_score": 9230.412005923527}
prior_hypothesis: Adding a complementary global-average classifier to the verified 12.5%-floor design will exceed 9,240 correct predictions without materially increasing runtime.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.82347762514837, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2088966438293457, "validation_score": 9240.413600287959}
prior_hypothesis: A 12.5% terminal learning-rate floor will exceed 9,233 correct predictions by preserving the demonstrated benefit of late optimization while remaining well below the harmful 20% floor.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 50.31158974999562, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20908836936950684, "validation_score": 9233.413534703224}
prior_hypothesis: A 7.5% terminal learning-rate floor will exceed 9,233 correct predictions by retaining useful late optimization while allowing more final convergence than the inferior 20% floor.

## Recent verification evidence

RECENT RESULT
hypothesis: Matching the strongest verified design will increase validation correctness from 9,227 toward the demonstrated 9,240.
change: Replace probability-space flip fusion with arithmetic logit averaging and raise the cosine schedule’s terminal learning-rate floor from 0% to 12.5%.
mechanism: Logit-space flip ensemble with 12.5%-floor cosine decay
evidence_used: Reference Design 2 achieved the best verified result—9,240 correct with 0.208897 cross-entropy—using these settings, while the current probability-fusion, zero-floor design achieved 9,227.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing the fixed-position 4,704-to-38 bottleneck with four learned spatial part summaries plus a global summary will exceed 9,240 correct predictions by preserving channel-rich local evidence while retaining the verified backbone’s runtime profile.
change: Restore the verified depthwise residual backbone, flip training and inference, and 12.5%-floor schedule, but compute predictions through initialized quadrant-attention pools and a 372-unit head; the model has 249,774 learned parameters.
mechanism: Content-adaptive multi-part attention pooling
evidence_used: The fixed-flatten design reached 9,240 correct, while widening its narrow classifier fell to 9,223 and added spatial convolutions timed out; reallocating the same head compute to content-dependent part aggregation cleanly challenges the shared fixed-location bottleneck without increasing convolutional work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the fixed 4,704-to-38 bottleneck with 4×4 adaptive average pooling and a 112-unit head will exceed 9,240 correct predictions by trading redundant pixel-level positions for greater classifier rank without increasing runtime.
change: Pool the 7×7 feature maps to 4×4 and widen the classifier to 112 units, reducing the model from 245,040 to approximately 239,282 parameters while preserving all verified training and inference settings.
mechanism: Coarse spatial pooling with a wider nonlinear head
evidence_used: The 245,040-parameter flat-head design achieved 9,240 correct, whereas merely widening its bottleneck to 39 fell to 9,223 and more computationally involved spatial alternatives timed out; this motivates a low-cost reallocation of dense capacity rather than adding parameters or convolutional work.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 7.5% to 12.5% will exceed the current 9,233 correct predictions and recover the 9,240-result region.
change: Change only the cosine schedule’s terminal floor, preserving the verified architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.
mechanism: Verified 12.5%-floor cosine decay
evidence_used: Reference Design 2 used the otherwise identical implementation and achieved the strongest verified result—9,240 correct with 0.208897 cross-entropy—versus 9,233 correct for the current 7.5% floor; unsuccessful replications provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 15% terminal learning-rate floor with logit-space flip ensembling will exceed 9,240 correct predictions by extending the improvement from 7.5% to 12.5% without approaching the harmful 20% floor.
change: Restore the strongest verified logit-space flip ensemble and increase its cosine terminal learning-rate floor from 12.5% to 15%.
mechanism: Moderately elevated terminal-rate cosine decay
evidence_used: Reference Design 2 achieved 9,240 correct at a 12.5% floor versus 9,233 at 7.5%; prior evidence identifies 20% as harmful, motivating a smaller upward refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.77745412499644, "validation_accuracy": 0.9219, "validation_correct": 9219, "validation_cross_entropy": 0.212547159576416, "validation_score": 9219.412355095677}

RECENT RESULT
hypothesis: An 11.25% terminal learning-rate floor will exceed 9,240 correct predictions by refining the narrow optimum between the weaker 7.5% and 15% schedules.
change: Restore the verified 245,040-parameter residual depthwise model, batch-64 flip training and logit-space flip ensemble, while setting the cosine schedule floor to 11.25%.
mechanism: Intermediate terminal-rate cosine decay
evidence_used: The 12.5% floor achieved 9,240 correct, versus 9,233 at 7.5% and 9,219 at 15%; testing 11.25% targets the most promising unmeasured interval without adding runtime or parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Channel-adaptive residual gating will exceed 9,240 correct predictions by adding lightweight backbone selectivity while preserving the strongest verified schedule, augmentation, and classifier.
change: Add zero-initialized squeeze-excitation gates to both residual depthwise blocks, increasing learned parameters from 245,040 to 248,082 with negligible spatial computation.
mechanism: Identity-initialized squeeze-excitation residual gating
evidence_used: The 245,040-parameter backbone reached 9,240 correct, while classifier reallocations failed and spatial augmentation repeatedly timed out; a low-cost backbone refinement tests an orthogonal source of improvement without changing the verified training procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 10.5% terminal learning-rate floor will exceed 9,240 correct predictions by targeting the estimated optimum between the improving 7.5%-to-12.5% interval and the sharply weaker 15% result.
change: Change only the cosine schedule’s terminal floor from 7.5% to 10.5%, preserving the verified architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.
mechanism: Interpolated terminal-rate cosine decay
evidence_used: The identical model achieved 9,233 correct at 7.5%, 9,240 at 12.5%, and 9,219 at 15%; these results bracket a narrow optimum below 12.5%, motivating a lower interpolation without adding runtime or parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 83.3415296250023, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.21066991958618164, "validation_score": 9232.412994485047}

RECENT RESULT
hypothesis: Adding a complementary global-average classifier to the verified 12.5%-floor design will exceed 9,240 correct predictions without materially increasing runtime.
change: Restore logit-space flip ensembling and the 12.5% cosine floor, then add a zero-initialized 970-parameter global-pooled head alongside the fixed-position classifier.
mechanism: Zero-initialized global-context logit shortcut
evidence_used: The fixed flatten-head design reached 9,240, while widening its bottleneck fell to 9,223 and more computationally intensive spatial heads timed out; a cheap pooled shortcut adds translation-tolerant evidence without disrupting the proven classifier at initialization.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 246010, "training_seconds": 81.55593858310021, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.21357478485107423, "validation_score": 9230.412005923527}

RECENT RESULT
hypothesis: Averaging the final 5% of the verified 12.5%-floor trajectory will exceed 9,240 correct predictions by reducing late-update variance without sacrificing useful terminal optimization.
change: Restore the strongest residual depthwise model, batch-64 flip augmentation, unsmoothed loss, and logit-space flip ensemble, then average model weights over the near-constant-rate final 5% of training.
mechanism: Late-trajectory stochastic weight averaging
evidence_used: Reference Design 1 achieved the best verified result at 9,240 correct; nearby terminal-floor changes and classifier modifications were worse, motivating a low-overhead stabilization of its final trajectory instead of another architectural or schedule change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 2% label smoothing will exceed 9,240 correct predictions by reducing overconfident fitting while preserving nearly all hard-label learning during the fixed two-pass exposure.
change: Add 0.02 label smoothing to the existing cross-entropy loss without changing architecture, runtime-intensive computation, or the strongest verified 12.5%-floor schedule.
mechanism: Mild label-smoothing regularization
evidence_used: The current unsmoothed design is strongest at 9,240 correct; nearby schedule changes fell to 9,232 and 9,219, while architectural additions either regressed or timed out, motivating an isolated low-cost loss regularizer.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.26312066591345, "validation_accuracy": 0.922, "validation_correct": 9220, "validation_cross_entropy": 0.22450960159301758, "validation_score": 9220.408326728797}

RECENT RESULT
hypothesis: Adding a class-specific second-order channel-covariance branch to the verified 12.5%-floor model will exceed 9,240 correct predictions by capturing translation-tolerant feature co-occurrences that neither the narrow flattened head nor the failed first-order global-average shortcut represents.
change: Preserve the proven flattened classifier, add a zero-initialized 22-channel bilinear covariance classifier for complementary logits, and restore the strongest verified 12.5% cosine floor; the model has 249,692 learned parameters.
mechanism: Compact bilinear covariance shortcut
evidence_used: The flattened model reached 9,240 correct, while its global-average shortcut fell to 9,230, challenging the shared assumption that first-order pooled features provide useful complementary context; compact covariance pooling instead retains 253 pairwise channel interactions with negligible spatial computation.
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
