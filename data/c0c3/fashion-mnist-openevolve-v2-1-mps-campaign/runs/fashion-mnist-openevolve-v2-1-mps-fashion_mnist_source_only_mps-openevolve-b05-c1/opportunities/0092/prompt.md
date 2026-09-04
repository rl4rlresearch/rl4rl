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

RECENT RESULT
hypothesis: Replacing convolutional GELUs with ReLUs will exceed 9,345 correct predictions by providing sparse feature regularization while reducing the training cost that caused multiple otherwise negligible changes to time out.
change: Use in-place ReLU for the stem and residual-block activations while retaining the classifier GELU, architecture, parameters, optimizer, augmentation, and evaluation ensemble.
mechanism: Sparse residual-backbone rectification
evidence_used: The best design required 75.35 seconds, and numerous parameter-free or negligible-cost variants subsequently exceeded the time limit; accelerating the repeatedly applied spatial activations directly targets that runtime fragility without removing capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling diagonal one-pixel translations during training will exceed 9,345 correct predictions by improving local translation invariance without adding parameters, optimizer work, or material runtime.
change: Expand the existing random-crop augmentation from six center/cardinal choices to ten choices comprising two centered crops and all eight one-pixel neighboring translations.
mechanism: Diagonal translation augmentation
evidence_used: Raw-mean gating achieved the best 9,345-correct result, while multiple added representation paths regressed or timed out; the current augmentation covers only center and cardinal translations, so diagonal jitter tests an untried generalization axis at essentially unchanged cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the peak learning rate by approximately 6% will exceed 9,345 correct predictions by extracting more optimization progress from the fixed 1,042 updates without increasing parameters, computation, or evaluation cost.
change: Increase `PEAK_LR` from 3.3e-3 to 3.5e-3 while preserving the architecture, batch size, augmentation, EMA, and cosine schedule.
mechanism: Fixed-budget learning-rate intensification
evidence_used: The raw-mean gated architecture reached the best 9,345-correct result, while added representation paths regressed or timed out; the smaller-batch experiment specifically identified update density as a plausible limitation but increased runtime, motivating a modest zero-cost increase in update magnitude.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the fixed 30-dimensional spatial bottleneck with four learned class-specific queries will exceed 9,345 correct predictions by dynamically selecting discriminative image regions for each class while reducing runtime-sensitive computation.
change: Replace the flattened classifier, handcrafted statistics branch, and diagonal gate with a lightweight cross-attention head that pools the 7×7 feature tokens through 40 learned class-part queries and class-specific readouts.
mechanism: Class-part cross-attention pooling
evidence_used: Raw-mean gating plateaued at 9,345 correct, while further post-hoc descriptors either regressed or timed out; this challenges the shared assumption that predictions should flow through one fixed global bottleneck by making spatial aggregation class-conditional and input-dependent.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing classifier dropout from 0.1 to zero during the final 30% of training will exceed 9,345 correct predictions by preserving early regularization while allowing cleaner convergence within the fixed 1,042-update budget.
change: Keep dropout at 0.1 for the first 70% of updates, then linearly reduce it to zero without changing parameters or computational cost.
mechanism: Late-stage dropout annealing
evidence_used: Raw-mean gating produced the best 9,345-correct result, while many added representation and loss variants timed out; adapting the existing regularizer directly targets limited optimization exposure without adding runtime-sensitive computation.
result: the implementation could not be verified



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
