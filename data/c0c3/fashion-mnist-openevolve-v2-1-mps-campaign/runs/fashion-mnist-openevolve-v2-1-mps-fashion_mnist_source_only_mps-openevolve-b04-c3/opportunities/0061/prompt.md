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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.86788470903412, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20905703506469728, "validation_score": 9243.41354542052}
prior_hypothesis: Lowering the power-mean order from 0.75 to 0.60 will retain at least 9,243 correct predictions while reducing cross-entropy below 0.2090745.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 65.20178408315405, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.20935865631103515, "validation_score": 9242.413442279832}
prior_hypothesis: Arithmetic averaging of original and flipped class probabilities will exceed 9,240 correct predictions by preserving confident evidence from either view when their logit scales differ.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 50.31158974999562, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20908836936950684, "validation_score": 9233.413534703224}
prior_hypothesis: A 7.5% terminal learning-rate floor will exceed 9,233 correct predictions by retaining useful late optimization while allowing more final convergence than the inferior 20% floor.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.93156333290972, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.209074520111084, "validation_score": 9243.41353944003}
prior_hypothesis: A 0.625-order power mean will retain 9,242 correct predictions while reducing cross-entropy below the verified 0.75-order ensemble’s 0.209166.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing the classifier dropout while restoring the verified 12.5%-floor design will exceed 9,240 correct predictions by improving limited-exposure fitting without adding runtime or parameters.
change: Remove the regressive global-average shortcut and disable the 10% classifier dropout, preserving the strongest verified backbone, optimizer, flip ensemble, and cosine schedule.
mechanism: Reduced classifier regularization under fixed two-pass exposure
evidence_used: The unsmoothed reference design achieved 9,240 correct, while 2% label smoothing fell to 9,220 and the global shortcut fell to 9,230; both regressions support testing less regularization and restoring the proven single classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Arithmetic averaging of original and flipped class probabilities will exceed 9,240 correct predictions by preserving confident evidence from either view when their logit scales differ.
change: Restore the verified 245,040-parameter residual depthwise model, batch-64 flip training, unsmoothed loss, and 12.5%-floor schedule, while replacing logit averaging with probability-mixture inference.
mechanism: Probability-space horizontal-flip ensembling
evidence_used: Reference Design 1 achieved the best verified result at 9,240 correct using logit-space flip ensembling; nearby schedule, loss, and architectural changes were worse or timed out, so changing only the untested ensemble aggregation is the most isolated next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 65.20178408315405, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.20935865631103515, "validation_score": 9242.413442279832}

RECENT RESULT
hypothesis: A 1.25-order power mean of original and flipped probabilities will exceed 9,242 correct predictions by modestly strengthening the confident-view preservation that made arithmetic probability averaging outperform geometric/logit averaging.
change: Replace equal arithmetic probability mixing with a calibrated 1.25-order probability power mean, leaving training and model parameters unchanged.
mechanism: Mildly confidence-weighted probability power-mean ensembling
evidence_used: Arithmetic probability averaging achieved 9,242 correct versus 9,240 for logit-space geometric averaging; a mild extrapolation beyond the arithmetic mean directly tests whether greater weight on confident per-view evidence continues that improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.03911916702054, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.20954788665771484, "validation_score": 9242.413377597957}

RECENT RESULT
hypothesis: A 0.75-order power mean will retain the arithmetic ensemble’s 9,242 correct predictions while lowering cross-entropy toward the better-calibrated logit ensemble, strictly improving validation_score.
change: Replace logit averaging with an equal 0.75-order generalized mean of original and horizontally flipped class probabilities; training remains unchanged.
mechanism: Sub-arithmetic probability power-mean flip ensemble
evidence_used: Logit/geometric averaging achieved 9,240 correct with 0.208897 cross-entropy, arithmetic probability averaging achieved 9,242 with 0.209359, and order 1.25 retained 9,242 but worsened cross-entropy to 0.209548. This motivates interpolating between geometric and arithmetic aggregation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 81.38509520795196, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.2091661834716797, "validation_score": 9242.41350809081}

RECENT RESULT
hypothesis: A 0.5-order power mean will retain the 0.75-order ensemble’s 9,242 correct predictions while lowering cross-entropy toward geometric/logit averaging, thereby strictly improving validation_score.
change: Restore the verified 12.5% cosine learning-rate floor and replace logit averaging with an equal 0.5-order generalized mean of original and horizontally flipped class probabilities.
mechanism: Half-order probability power-mean flip ensemble
evidence_used: Orders 0.75, 1.0, and 1.25 all produced 9,242 correct, while cross-entropy improved as the order decreased; geometric averaging had still lower cross-entropy but only 9,240 correct. Testing order 0.5 efficiently probes the interval for better calibration without changing training cost or parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.625-order power mean will retain 9,242 correct predictions while reducing cross-entropy below the verified 0.75-order ensemble’s 0.209166.
change: Remove the regressive global-average shortcut, restore the verified 245,040-parameter classifier, and ensemble original and flipped predictions with a 0.625-order probability mean.
mechanism: 0.625-order probability power-mean flip ensemble
evidence_used: Orders 0.75, 1.0, and 1.25 all achieved 9,242 correct while cross-entropy improved as order decreased; geometric averaging had still lower cross-entropy but 9,240 correct, motivating a conservative interpolation below 0.75 after the 0.5-order attempt timed out.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.93156333290972, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.209074520111084, "validation_score": 9243.41353944003}

RECENT RESULT
hypothesis: A 0.5625-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below 0.2090745.
change: Lower only the inference-time probability power-mean order from 0.625 to 0.5625.
mechanism: 0.5625-order probability power-mean flip ensemble
evidence_used: Reducing the order from 0.75 to 0.625 improved correct predictions from 9,242 to 9,243 and cross-entropy from 0.209166 to 0.209075; 0.5625 conservatively continues that trend while remaining above the unverified 0.5-order attempt.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the power-mean order from 0.75 to 0.60 will retain at least 9,243 correct predictions while reducing cross-entropy below 0.2090745.
change: Change only the inference-time probability power-mean order to 0.60, preserving the verified training procedure, architecture, schedule, and parameter count.
mechanism: 0.60-order probability power-mean flip ensemble
evidence_used: Order 0.625 achieved the best result with 9,243 correct and 0.2090745 cross-entropy, improving both metrics over order 0.75; order 0.60 is a conservative continuation toward the lower-cross-entropy geometric ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.86788470903412, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20905703506469728, "validation_score": 9243.41354542052}

RECENT RESULT
hypothesis: A 0.5875-order power mean will retain at least 9,243 correct predictions while reducing validation cross-entropy below 0.209057.
change: Replace arithmetic probability averaging with a 0.5875-order generalized mean of original and horizontally flipped class probabilities, leaving training unchanged.
mechanism: Finely tuned sub-arithmetic probability power-mean flip ensemble
evidence_used: Orders 0.625 and 0.60 both achieved 9,243 correct, while lowering the order to 0.60 reduced cross-entropy from 0.2090745 to 0.2090570; 0.5875 conservatively continues that verified trend.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the fixed 38-unit flattened bottleneck with eight input-dependent spatial attention pools and a 228-unit fusion layer will exceed 9,243 correct predictions by retaining multiple discriminative part descriptors while learning where to gather them.
change: Replace the parameter-dominant flattened classifier with learned positional attention pooling, restore the strongest verified 0.60-order flip ensemble, and restore the verified 12.5% cosine learning-rate floor; the resulting model has 249,342 learned parameters.
mechanism: Multi-query positional part-attention pooling
evidence_used: The best design reached 9,243 correct with the 0.60-order ensemble, but all available designs share a 179,256-parameter flatten-to-38 head. The failed global-average shortcut pooled the same shallow features into one first-order descriptor; eight learned, position-aware pools instead preserve complementary regions and fuse them through a substantially wider learned representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249342, "training_seconds": 78.29844929184765, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.2308595317840576, "validation_score": 9166.406220195797}



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
