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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}
prior_hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.27387729194015, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}
prior_hypothesis: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.21758331069946288.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.942004709038883, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}
prior_hypothesis: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.103174792137, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758494262695313, "validation_score": 9240.410648967883}
prior_hypothesis: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.
change: Restore the proven training recipe and change its evaluation-only logit multiplier from 1.10 to 1.295.
mechanism: Fine-grained post-ensemble confidence calibration
evidence_used: Cross-entropy decreased monotonically from 0.218881 at 1.20 to 0.217864 at 1.25 and 0.217585 at 1.30 while all 9,240 predictions remained unchanged; the sharply diminishing gains indicate an optimum near 1.30, motivating a fine-grained interpolation rather than another 0.05 step.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2964 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217583535.
change: Apply a 1.2964 evaluation-only multiplier to the translation-and-flip-ensembled logits.
mechanism: Local quadratic confidence calibration
evidence_used: Cross-entropy improved from 0.217864218 at scale 1.25 to 0.217583535 at 1.295, then worsened slightly to 0.217584943 at 1.30; local quadratic interpolation places the calibration minimum near 1.2964.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.54287579189986, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758331069946288, "validation_score": 9240.410649518277}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.21758331069946288.
change: Move the evaluation-only multiplier from 1.2964 to the local quadratic optimum inferred from neighboring verified scales.
mechanism: Second-order confidence calibration refinement
evidence_used: Cross-entropy was 0.21758353462219238 at 1.295, 0.21758331069946288 at 1.2964, and 0.21758494262695313 at 1.30; quadratic interpolation places the minimum near 1.296352.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.27387729194015, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.217583535 to approximately 0.217583305.
change: Replace the evaluation-only logit multiplier with the best verified local-quadratic calibration.
mechanism: Second-order confidence calibration refinement
evidence_used: Reference Design 1 used the identical training recipe with scale 1.296352 and achieved the best available score, 9,240.410649520078, improving on the current 1.295 scale while preserving all predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Changing the evaluation-only logit scale from 1.30 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2175849426 to approximately 0.2175833054.
change: Replace the current evaluation-logit multiplier with the best verified calibration while leaving training and class decisions unchanged.
mechanism: Verified second-order confidence calibration
evidence_used: Reference Design 1 is otherwise identical and achieved the best available score, 9240.410649520078, at scale 1.296352 versus the current 9240.410648967883 at 1.30.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Row- and column-aware channel modulation will exceed 9,240 correct predictions by adding global silhouette interactions while preserving the proven position-sensitive classifier.
change: Add 5,968 learned parameters of separable row/column context gating after local refinement, keeping total parameters at 248,514, and use the best verified 1.296352 evaluation calibration.
mechanism: Identity-initialized coordinate-conditioned feature gating
evidence_used: The additive global-moment head fell from 9,240 to 9,227 correct, showing that pooled evidence should not independently vote on classes. This patch instead uses axis-preserving context to modulate the successful spatial path, starts as an exact identity, and cannot inject separate logits.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 248514, "training_seconds": 40.359193958109245, "validation_accuracy": 0.9225, "validation_correct": 9225, "validation_cross_entropy": 0.21815493850708007, "validation_score": 9225.41045681809}

RECENT RESULT
hypothesis: Replacing the final optimizer iterate with a 0.98-decay exponential average of the same model weights will exceed 9,240 correct predictions by reducing late-training parameter noise without disturbing the proven architecture, augmentation, or evaluation calibration.
change: Track an EMA of all learned parameters after each optimizer step and install it after the final step; retain final BatchNorm statistics and the verified 1.296352 logit scale.
mechanism: End-of-training exponential weight averaging
evidence_used: The calibrated baseline consistently reaches 9,240 correct, while adding a global head reduced this to 9,227 and coordinate gating to 9,225; preserving the successful representation and changing only final-weight selection is the clearest orthogonal next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 26.343437500065193, "validation_accuracy": 0.9226, "validation_correct": 9226, "validation_cross_entropy": 0.2205278331756592, "validation_score": 9226.409658826624}

RECENT RESULT
hypothesis: Changing the evaluation-only logit scale from 1.295 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2175835346 to approximately 0.2175833054.
change: Replace only the evaluation-logit multiplier with the best verified calibration, leaving training and class decisions unchanged.
mechanism: Verified second-order confidence calibration
evidence_used: Reference Design 1 used the identical implementation at scale 1.296352 and achieved the best available score, 9240.410649520078; architecture changes and EMA reduced validation correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Changing the evaluation-only logit scale from 1.30 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy to approximately 0.2175833054.
change: Replace only the evaluation-logit multiplier, leaving training and class decisions unchanged.
mechanism: Verified evaluation-logit calibration
evidence_used: Reference Design 1 is otherwise identical and achieved the best available score, 9240.410649520078, at scale 1.296352 versus the current 9240.410648967883 at 1.30.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.
change: Replace only the evaluation-time logit multiplier, leaving training and class decisions unchanged.
mechanism: Verified evaluation-logit calibration
evidence_used: Reference Design 1 used the otherwise identical implementation at scale 1.296352 and achieved the best available score, 9240.410649520078; prior architecture and EMA experiments reduced validation correctness.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.942004709038883, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}

RECENT RESULT
hypothesis: Encouraging original and horizontally flipped views to agree will exceed 9,240 correct predictions by aligning training with the proven flip-averaged evaluation while preserving the successful architecture.
change: Add a gradually ramped Jensen–Shannon-style consistency loss between paired-view predictions; retain the existing supervised loss, optimizer, augmentation, and calibrated evaluation logits.
mechanism: Ramped flip-consistency regularization
evidence_used: Architecture gating and EMA reduced correctness to 9,225 and 9,226, respectively, whereas the baseline repeatedly achieved 9,240; this motivates a parameter-free change targeting the evaluation invariance already encoded by paired training and flip averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 32.66938679199666, "validation_accuracy": 0.9234, "validation_correct": 9234, "validation_cross_entropy": 0.21797940216064454, "validation_score": 9234.410515973515}

RECENT RESULT
hypothesis: Reducing the batch size to 128 will exceed 9,240 correct predictions by providing roughly 50% more optimizer updates over the fixed 100,000-example exposure while retaining the proven architecture, objective, augmentation, and evaluation ensemble.
change: Change only the training batch size from 192 to 128.
mechanism: Smaller-batch higher-frequency optimization
evidence_used: The calibrated baseline achieves 9,240 correct in only 522 optimizer steps, while added gating, EMA, and consistency regularization reduced correctness; increasing optimization frequency preserves the successful representation and tests an orthogonal route to better convergence.
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
