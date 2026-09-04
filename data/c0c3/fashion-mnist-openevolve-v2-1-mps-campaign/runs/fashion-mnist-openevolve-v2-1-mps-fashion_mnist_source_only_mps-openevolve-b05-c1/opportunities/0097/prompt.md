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

RECENT RESULT
hypothesis: Independently supervising the statistics head early and annealing that auxiliary loss to zero will exceed 9,345 correct predictions by accelerating robust first-order feature learning while preserving late-stage residual specialization.
change: Expose statistics logits during training and add a computationally negligible auxiliary cross-entropy whose weight decreases linearly from 0.15 to zero; inference, parameters, and example accounting remain unchanged.
mechanism: Annealed auxiliary supervision for the statistics branch
evidence_used: The first-order statistics bypass improved the classifier, while the attempted higher learning rate for its late-starting residual parameters timed out; direct annealed supervision tests the same underexposure hypothesis without adding parameters, forward passes, or meaningful runtime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 70.0535058749374, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.19065491104125976, "validation_score": 9307.419936956849}

RECENT RESULT
hypothesis: Lowering classifier dropout from 0.10 to 0.05 will exceed 9,345 correct predictions by improving convergence during the fixed 1,042 updates while retaining mild regularization.
change: Reduce the existing classifier dropout probability to 0.05 without changing parameters, loss, runtime, or interfaces.
mechanism: Reduced classifier stochastic regularization
evidence_used: The best 9,345-correct design still uses 0.10 dropout, while late-stage dropout annealing could not be verified; this safely tests the unresolved underexposure hypothesis without the auxiliary objective that regressed to 9,307 correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 65.7079664580524, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.1902842918395996, "validation_score": 9301.420067712754}

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.10 to 0.15 will exceed 9,345 correct predictions by strengthening the regularization that the reduced-dropout result showed was beneficial.
change: Raise the existing classifier dropout probability to 0.15 without changing parameters, optimizer steps, augmentation, or evaluation cost.
mechanism: Stronger classifier co-adaptation control
evidence_used: Lowering dropout to 0.05 reduced validation correct from 9,345 to 9,301 and worsened cross-entropy, directly motivating a controlled test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 66.52635404118337, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.18776000366210938, "validation_score": 9330.420960462096}

RECENT RESULT
hypothesis: Setting classifier dropout to 0.1125 will exceed 9,345 correct predictions by moving modestly toward stronger regularization without incurring the degradation observed at 0.15.
change: Increase classifier dropout probability from 0.10 to 0.1125; all other architecture, optimization, augmentation, and evaluation behavior remains unchanged.
mechanism: Evidence-guided classifier dropout interpolation
evidence_used: Dropout 0.10 achieved 9,345 correct, while 0.05 fell to 9,301 and 0.15 retained 9,330; the asymmetric degradation indicates that the local optimum may lie slightly above 0.10.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 68.2946737499442, "validation_accuracy": 0.9329, "validation_correct": 9329, "validation_cross_entropy": 0.18800582885742187, "validation_score": 9329.420873355883}

RECENT RESULT
hypothesis: Linearly decaying label smoothing from 0.05 to zero will exceed 9,345 correct predictions by regularizing early decision-boundary formation while restoring exact hard-label optimization near convergence.
change: Replace hard-label cross-entropy with label smoothing that decreases continuously over the fixed training schedule.
mechanism: Annealed target-distribution smoothing
evidence_used: Dropout 0.10 achieved 9,345 correct, while both reducing it to 0.05 and increasing it to 0.15 or 0.1125 regressed, motivating an orthogonal output-space regularizer whose influence vanishes late rather than another persistent dropout change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 247546, "training_seconds": 64.19530616607517, "validation_accuracy": 0.9341, "validation_correct": 9341, "validation_cross_entropy": 0.19130824012756348, "validation_score": 9341.419706657905}



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
