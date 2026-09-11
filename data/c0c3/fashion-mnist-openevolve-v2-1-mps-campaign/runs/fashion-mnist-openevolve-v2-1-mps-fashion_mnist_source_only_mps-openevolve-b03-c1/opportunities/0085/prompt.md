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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 66.76434204191901, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.23899501876831056, "validation_score": 9166.403552873438}
prior_hypothesis: Learning the existing three derivative kernels while constraining each to remain zero-mean will exceed 9,162 correct predictions by adapting edge and texture extraction to the dataset without adding the redundant gradient-magnitude channel that regressed to 9,114.

## Recent verification evidence

RECENT RESULT
hypothesis: Weighted averaging of per-view class probabilities will exceed 9,162 correct predictions by preventing an overconfident shifted or flipped view from dominating the ten-view ensemble.
change: Replace weighted raw-logit averaging during evaluation with weighted arithmetic probability averaging, returned as temperature-scaled log probabilities; training and parameter count remain unchanged.
mechanism: Confidence-normalized test-time augmentation ensemble
evidence_used: Dual-statistic attention reached 9,162 correct, while subsequent attention refinements consistently regressed or timed out, motivating an orthogonal refinement of the existing multi-view decision rule without increasing training cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding gradient magnitude to the existing signed derivative inputs will exceed 9,162 correct predictions by exposing boundary strength that the first linear convolution cannot directly derive from horizontal and vertical responses before its nonlinearity.
change: Add a fixed gradient-magnitude input channel and expand the first convolution from four to five inputs, raising learned parameters from 249,762 to 249,978.
mechanism: Nonlinear orientation-invariant edge-strength channel
evidence_used: Dual-statistic attention reached 9,162 correct, while subsequent attention and fusion refinements regressed or timed out; this motivates an orthogonal, lightweight improvement to the input representation while retaining the best attention mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249978, "training_seconds": 56.50290670897812, "validation_accuracy": 0.9114, "validation_correct": 9114, "validation_cross_entropy": 0.24515600814819336, "validation_score": 9114.40155610761}

RECENT RESULT
hypothesis: Averaging each channel’s two strongest spatial activations will exceed 9,162 correct predictions by retaining the useful localized signal from maximum pooling while reducing sensitivity to a single noisy activation.
change: Replace the maximum-pooled channel descriptor with a parameter-free top-two spatial average while preserving equal dual-descriptor fusion, initialization, and parameter count.
mechanism: Top-two spatial salience channel attention
evidence_used: Adding maximum context to mean-only attention improved correctness from 9,153 to 9,162, while changing fusion weights consistently regressed; this motivates refining the complementary salience descriptor rather than its successful equal weighting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Normalizing the 160-unit hidden classifier representation will exceed 9,162 correct predictions by stabilizing optimization of the parameter-dominant dense head without consuming the remaining parameter budget.
change: Insert non-affine BatchNorm1d between the first classifier linear layer and GELU, preserving the 249,762 learned-parameter count.
mechanism: Parameter-free dense-head batch normalization
evidence_used: Dual-statistic attention reached 9,162 correct, while subsequent attention refinements and the added edge-strength input failed to improve; this motivates an orthogonal optimization change to the currently unnormalized classifier head.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 10% label-aware mixup during the first 75% of optimization, followed by clean low-learning-rate updates, will exceed 9,162 correct predictions by improving generalization without preventing final adaptation to unmixed validation images.
change: Add fixed-strength batch mixup to the early training loss while preserving the existing label smoothing, architecture, parameter count, schedule, and final quarter of clean training.
mechanism: Early weak mixup with clean-tail fine-tuning
evidence_used: Dual-statistic attention achieved 9,162 correct, while repeated attention refinements and an added edge-strength channel failed to improve; this motivates testing an orthogonal, computationally cheap loss-level regularizer.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 71.22526083304547, "validation_accuracy": 0.913, "validation_correct": 9130, "validation_cross_entropy": 0.24390363159179687, "validation_score": 9130.401960398942}

RECENT RESULT
hypothesis: Lowering classifier dropout from 15%/10% to 10%/5% will exceed 9,162 correct predictions by improving head adaptation within the fixed 1,564-step budget while retaining augmentation, label smoothing, weight decay, and EMA regularization.
change: Reduce both classifier dropout rates without changing architecture, parameter count, evaluation, or runtime.
mechanism: Reduced dense-head dropout regularization
evidence_used: Early weak mixup reduced correctness from 9,162 to 9,130, indicating that additional regularization is harmful under the fixed exposure budget; modestly relaxing the model’s existing head regularization is the most direct complementary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 77.4718734160997, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.24069262237548827, "validation_score": 9141.403000703787}

RECENT RESULT
hypothesis: Learning the existing three derivative kernels while constraining each to remain zero-mean will exceed 9,162 correct predictions by adapting edge and texture extraction to the dataset without adding the redundant gradient-magnitude channel that regressed to 9,114.
change: Convert the 27 fixed detail-kernel coefficients into learned parameters and remove each kernel’s spatial mean before convolution, preserving the original computation at initialization and raising the parameter count to 249,789.
mechanism: Zero-DC adaptive detail filter bank
evidence_used: The fixed derivative representation supports the 9,162-correct design, whereas adding a handcrafted gradient-magnitude channel reduced correctness to 9,114; adapting the successful filters is a lightweight alternative to expanding the representation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 66.76434204191901, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.23899501876831056, "validation_score": 9166.403552873438}

RECENT RESULT
hypothesis: Replacing fixed maximum pooling with learned per-channel 80/20 max–mean pooling will exceed 9,166 correct predictions by preserving distributed shape evidence while allowing each feature channel to control its salience–smoothing tradeoff.
change: Add a lightweight mixed-pooling module and use it at all three downsampling stages, adding 144 parameters for a total of 249,933.
mechanism: Channel-adaptive mixed-statistic downsampling
evidence_used: Learning the previously fixed derivative filters improved correctness from 9,162 to 9,166, while repeated channel-attention refinements regressed or timed out; this challenges the shared assumption that fixed maximum-only spatial reduction is optimal.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training the successful zero-DC detail filters at half the backbone learning rate will exceed 9,166 correct predictions by preserving their useful derivative priors while still permitting task-specific adaptation.
change: Place `detail_kernels` in a dedicated AdamW parameter group with a 0.5 learning-rate multiplier; leave all other optimization, architecture, and evaluation behavior unchanged.
mechanism: Conservative adaptive-filter learning rate
evidence_used: Making the fixed detail filters trainable improved correctness from 9,162 to 9,166, while expanding the handcrafted representation regressed to 9,114; this directly tests whether gentler adaptation can retain more of the beneficial prior without adding computation or parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Constraining each zero-DC detail kernel to its initial L2 norm will exceed 9,166 correct predictions by removing scale drift while retaining task-specific filter-shape adaptation.
change: Store the three initial kernel norms and renormalize each mean-centered learned kernel before convolution, without changing initialization, parameter count, or training cost materially.
mechanism: Fixed-energy adaptive detail filters
evidence_used: Learning zero-DC detail filters improved correctness from 9,162 to 9,166, while adding gradient magnitude regressed to 9,114; preserving the successful adaptive representation while eliminating its scale ambiguity is the most direct refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 77.9594605001621, "validation_accuracy": 0.9153, "validation_correct": 9153, "validation_cross_entropy": 0.24091059188842773, "validation_score": 9153.402929915554}

RECENT RESULT
hypothesis: Training the zero-DC detail kernels at 1.5× the backbone learning rate will exceed 9,166 correct predictions by allowing their useful task-specific adaptation to converge further within the fixed 1,564-step budget.
change: Give `detail_kernels` a dedicated AdamW parameter group with a persistent 1.5× learning-rate multiplier while leaving the architecture, regularization, EMA, and backbone schedule unchanged.
mechanism: Accelerated adaptive detail-filter learning
evidence_used: Making the detail filters trainable improved correctness from 9,162 to 9,166, while fixed-energy normalization regressed to 9,153; this motivates faster unconstrained adaptation rather than another restriction on kernel scale.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Expanding the successful zero-DC learned detail kernels from 3×3 to 5×5 will exceed 9,166 correct predictions by learning longer-range edge and texture structure while preserving the verified model exactly at initialization.
change: Zero-pad each learned derivative kernel to 5×5 and increase the replicate padding accordingly, adding 48 learned parameters for an estimated total of 249,837.
mechanism: Identity-initialized expanded-support adaptive detail filters
evidence_used: Learning the original zero-DC 3×3 filters improved correctness from 9,162 to 9,166, whereas fixing their energy regressed to 9,153; this motivates adding unconstrained filter-shape capacity without changing the initial computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249837, "training_seconds": 79.6296627910342, "validation_accuracy": 0.9148, "validation_correct": 9148, "validation_cross_entropy": 0.23975780105590822, "validation_score": 9148.403304580599}



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
