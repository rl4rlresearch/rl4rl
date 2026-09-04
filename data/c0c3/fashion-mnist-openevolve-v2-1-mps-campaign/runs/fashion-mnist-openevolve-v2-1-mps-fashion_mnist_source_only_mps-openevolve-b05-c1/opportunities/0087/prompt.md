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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 75.34807645902038, "validation_accuracy": 0.9345, "validation_correct": 9345, "validation_cross_entropy": 0.18751400756835937, "validation_score": 9345.421047664966}
prior_hypothesis: A per-channel self-conditioned gate will exceed 9,344 correct predictions by retaining the beneficial full-feature-map recalibration signal while avoiding the runtime cost of a squeeze-and-excitation MLP.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging per-view log-probabilities will exceed 9,345 correct predictions by favoring class evidence consistent across translated and flipped views instead of allowing a single confident view to dominate the arithmetic probability mixture.
change: Replace arithmetic softmax averaging during evaluation with the same weighted ensemble over log-softmax outputs; training, parameters, view weights, and evaluation cost remain unchanged.
mechanism: Weighted geometric test-time augmentation ensemble
evidence_used: Raw-mean diagonal gating is the current best at 9,345 correct, while subsequent representation changes failed to improve or exceeded the time limit; refining the existing ten-view aggregation isolates an untested inference decision without disturbing the successful model.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final 64-channel residual block with a cheaper 48-channel bottleneck and widening the flattened classifier from 30 to 45 units will exceed 9,345 correct predictions by preserving more class-specific spatial information within the parameter and runtime limits.
change: Introduce a bottleneck residual block for the 7×7 feature stage and invest its parameter savings in a wider positional classifier, raising parameters from 247,546 to 247,997 while reducing convolutional computation.
mechanism: Compute-neutral spatial-to-classifier capacity reallocation
evidence_used: Direct statistics bypasses improved accuracy around the rank-30 flattened bottleneck, whereas additional positional heads timed out; reallocating capacity inside the existing path tests the same bottleneck limitation while substantially reducing final-stage convolution cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a per-channel bottom-versus-top activation contrast will exceed 9,345 correct predictions by supplying coarse spatial layout directly to the classifier without relying on the rank-30 flattened bottleneck.
change: Extend the normalized statistics residual from mean/std/max to mean/std/max/vertical-contrast, adding 768 parameters for a total of 248,314 while preserving the initial logits.
mechanism: Vertical-contrast statistics bypass
evidence_used: The statistics bypass improved prior results, while attempts to widen positional processing timed out; a single low-frequency spatial statistic targets the same lost spatial information with substantially less computation and parameter cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 248314, "training_seconds": 76.81869908282533, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.18973870925903322, "validation_score": 9331.420260344652}

RECENT RESULT
hypothesis: Reducing the batch size from 96 to 80 will exceed 9,345 correct predictions by providing 1,250 optimizer steps instead of 1,042 and modestly noisier gradients within the same fixed 100,000-example exposure.
change: Set the training batch size to 80 while preserving the architecture, optimizer, schedule shape, augmentation, and inference ensemble.
mechanism: Smaller-batch update densification
evidence_used: Multiple representation additions failed to improve the 9,345-correct result or exceeded the time limit, while every reported run used only 1,042 updates at batch size 96; increasing update density tests whether optimization exposure, rather than missing model capacity, is now the limiting factor.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 0.04 label smoothing will exceed 9,345 correct predictions by reducing overconfident fitting during the fixed two-example-per-training-image exposure without adding parameters or material runtime.
change: Retain the successful architecture, batch size, optimizer, augmentation, and inference ensemble while changing the training objective to mildly smoothed cross-entropy.
mechanism: Mild label-smoothing regularization
evidence_used: Raw-mean gating reached the best 9,345 correct, while subsequent representation changes either regressed or exceeded the time limit; a computationally negligible loss-level intervention tests whether generalization rather than missing capacity is limiting accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A learned 12-dimensional covariance descriptor will exceed 9,345 correct predictions by exposing cross-channel co-occurrence evidence that the flattened bottleneck and independent mean/std/max statistics cannot represent.
change: Add a zero-initialized residual classifier over the upper triangle of a learned low-rank channel covariance matrix, preserving the current predictor at initialization and remaining below the parameter ceiling.
mechanism: Low-rank bilinear channel-covariance pooling
evidence_used: The first-order statistics bypass improved the classifier, but diagonal quadratic enrichment fell to 9,338, indicating that additional per-channel curvature is insufficient; this patch instead tests genuine cross-channel interactions after cheap 7×7 aggregation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Conditioning the identity-initialized channel gate on raw mean, standard deviation, and maximum will exceed 9,345 correct predictions by learning complementary per-channel distribution cues without the harmful normalization or runtime cost seen in prior recalibration variants.
change: Add zero-initialized per-channel scale vectors for the already-computed spatial standard deviation and maximum, then include them in the existing gate.
mechanism: Multi-statistic diagonal channel recalibration
evidence_used: Raw-mean diagonal gating produced the current best 9,345 correct, whereas standardized and running-mean-centered conditioning regressed; extending the successful raw-scale gate with already-available descriptors isolates whether its conditioning signal is under-specified.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training the zero-initialized statistics head and successful diagonal channel gate at 1.5× the backbone learning rate will exceed 9,345 correct predictions by letting these late-starting residual paths specialize within only 1,042 optimizer steps.
change: Place the statistics head and recalibration parameters in a separate AdamW group whose 1.5× learning-rate multiplier is preserved throughout the existing schedule.
mechanism: Accelerated residual-adapter optimization
evidence_used: The statistics bypass and raw-mean diagonal gate produced the best 9,345-correct design, while added representational paths regressed or exceeded the time limit; accelerating only the already-beneficial 2,058 parameters tests optimization underexposure without adding parameters or meaningful computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighted logit averaging will exceed 9,345 correct predictions by favoring evidence consistent across augmented views while avoiding the evaluation overhead of computing ten softmax distributions.
change: Replace arithmetic probability averaging with its computationally cheaper weighted-logit counterpart, preserving view weights and calibration scaling.
mechanism: Softmax-free geometric test-time aggregation
evidence_used: Geometric aggregation was previously motivated by the 9,345-correct raw-mean-gating result but timed out; weighted logit averaging gives the same class decisions as weighted log-softmax averaging while eliminating per-view softmax operations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A learned per-pixel gate will exceed 9,345 correct predictions by complementing successful channel recalibration with dynamic foreground emphasis while leaving the established statistics branch unchanged.
change: Add a zero-initialized 1×1 spatial-attention projection and apply its single-channel gate only to the feature map entering the positional classifier, adding 65 parameters and little computation.
mechanism: Identity-initialized spatial saliency recalibration
evidence_used: Raw-mean channel gating achieved the best result of 9,345, whereas a fixed vertical-contrast descriptor regressed to 9,331; dynamically modulating the full spatial representation tests layout information without compressing it into another statistic.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Moving raw-mean channel gating after statistics BatchNorm will exceed 9,345 correct predictions by preserving recalibration while preventing sample-dependent gates from altering BatchNorm statistics.
change: Normalize the unchanged mean/std/max descriptors before applying the existing channel gate; architecture, parameters, optimizer, and runtime remain effectively unchanged.
mechanism: Post-normalization statistics recalibration
evidence_used: Raw-mean diagonal gating achieved the best 9,345-correct result, while normalized gate conditioning regressed to 9,318 and removing statistics gating was inconclusive due to timeout; this isolates gate placement without changing its successful conditioning signal.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the raw-mean gate’s local response by 1.5× will exceed 9,345 correct predictions by accelerating useful recalibration within 1,042 updates without adding parameters or meaningful runtime.
change: Reparameterize the existing bounded gate to retain its identity initialization and 0–2 range while increasing its initial derivative from 0.5 to 0.75.
mechanism: Higher-sensitivity identity-preserving channel gate
evidence_used: Raw-mean diagonal gating achieved the best result of 9,345 correct; the adapter learning-rate experiment timed out, so this isolates gate optimization through a zero-cost parameterization change.
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
