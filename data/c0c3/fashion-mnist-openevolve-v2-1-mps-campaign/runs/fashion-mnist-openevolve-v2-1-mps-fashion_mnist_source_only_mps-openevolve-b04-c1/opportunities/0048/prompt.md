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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.43049825006165, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.1928421325683594, "validation_score": 9316.419166951224}
prior_hypothesis: Jointly classifying the mirrored feature mean and absolute feature disagreement will exceed 9,286 correct predictions by learning when the two views corroborate or conflict, which independent-logit averaging cannot represent.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging normalized class probabilities instead of raw logits will exceed 9,286 correct predictions by preventing an overconfident erroneous view from dominating the mirrored ensemble.
change: Change only evaluation-time fusion to the log of the arithmetic mean of both views’ probabilities, retaining training, architecture, runtime, and calibrated logit scale.
mechanism: Probability-space mirrored-view ensembling
evidence_used: Mirrored-view ensembling raised validation correct from 9,237 to 9,282, while temperature calibration cannot change the current 9,286 predictions; testing a more robust fusion rule directly targets further accuracy gains without additional forward passes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a stable grouped 3×3 residual block at 7×7 resolution will exceed 9,286 correct predictions by refining spatial relationships before the classifier, while remaining below the parameter ceiling.
change: Add an 18,560-parameter, identity-initialized grouped-convolution residual block after the second pooling stage; retain the proven optimizer, loss, augmentation, EMA, and evaluation calibration.
mechanism: Identity-initialized grouped spatial refinement
evidence_used: Widening the classifier reduced correct predictions to 9,254, while temperature calibration has saturated without changing predictions; this tests additional spatial feature processing instead of more classifier capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Mixing 10% individual-view cross-entropy into the mirrored-average objective will exceed 9,286 correct predictions by making both constituent views independently discriminative while retaining direct optimization of their ensemble.
change: Replace the ensemble-only loss with a scale-preserving blend of 90% mirrored-average loss and 10% mean per-view loss; architecture, augmentation, EMA, schedule, and evaluation calibration remain unchanged.
mechanism: Per-view auxiliary supervision for mirrored ensemble
evidence_used: Mirrored-view ensembling previously improved validation correct from 9,237 to 9,282, while temperature calibration has now saturated at 9,286; auxiliary supervision directly tests whether stronger constituent predictions can improve the proven ensemble without additional forward passes or parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 70.92210020800121, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.20264423446655275, "validation_score": 9270.415750548393}

RECENT RESULT
hypothesis: Averaging normalized class probabilities instead of raw logits will exceed 9,286 correct predictions by limiting domination from an overconfident erroneous view.
change: Replace evaluation-time logit averaging with the log of the arithmetic mean of both views’ probabilities, retaining the calibrated output scale.
mechanism: Probability-space mirrored-view ensembling
evidence_used: Mirrored-view ensembling previously improved accuracy substantially; the earlier probability-fusion verification timed out, so its effect on prediction quality remains unmeasured despite adding only negligible evaluation computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Jointly classifying the mirrored feature mean and absolute feature disagreement will exceed 9,286 correct predictions by learning when the two views corroborate or conflict, which independent-logit averaging cannot represent.
change: Replace the assumption that mirrored views should be classified independently with an identity-initialized learned fusion of their invariant and disagreement features before the nonlinear classifier.
mechanism: Joint invariant–disagreement view fusion
evidence_used: Mirrored ensembling raised accuracy substantially, while adding individual-view supervision regressed to 9,270 correct; this supports learning a genuinely joint representation rather than strengthening separate view predictions. The fusion adds only a 1×1 projection and reuses the existing two-view convolutional computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.43049825006165, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.1928421325683594, "validation_score": 9316.419166951224}

RECENT RESULT
hypothesis: A nonlinear residual fusion of invariant and disagreement features will exceed 9,316 correct predictions by modeling interactions that the current single linear projection cannot represent.
change: Replace the linear 128-to-64 fusion with a 96-channel GELU bottleneck whose zero-initialized output learns corrections to the invariant features, remaining under the parameter ceiling.
mechanism: Identity-initialized nonlinear view-fusion correction
evidence_used: Joint invariant–disagreement fusion improved validation correct from 9,286 to 9,316, while widening the general classifier regressed; this motivates adding capacity specifically to the successful fusion mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding elementwise mirrored-feature products will exceed 9,316 correct predictions by distinguishing strong bilateral corroboration from feature disagreement patterns that the current linear mean-and-absolute-difference fusion cannot separate.
change: Add a zero-initialized 64-channel coactivation descriptor to the existing identity-initialized fusion, increasing parameters from 224,442 to 228,538 with negligible additional computation.
mechanism: Second-order mirrored coactivation fusion
evidence_used: Mean-and-disagreement fusion improved validation correct from 9,286 to 9,316, establishing view fusion as valuable; the nonlinear bottleneck timed out, motivating an explicit second-order interaction that preserves the successful initialization and avoids a costly hidden fusion layer.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 228538, "training_seconds": 82.81290491600521, "validation_accuracy": 0.9298, "validation_correct": 9298, "validation_cross_entropy": 0.1967527145385742, "validation_score": 9298.417797255795}

RECENT RESULT
hypothesis: A lightweight sample-dependent gate conditioned on pooled invariant and disagreement features will exceed 9,316 correct predictions by adapting the contribution of fused channels to each image.
change: Add an identity-initialized 16-channel squeeze gate over the existing fusion descriptor and use its bounded output to modulate fused features before classification.
mechanism: Global disagreement-conditioned channel gating
evidence_used: Linear invariant–disagreement fusion improved validation correct from 9,286 to 9,316, while local coactivation fusion fell to 9,298 and the larger nonlinear fusion exceeded the time limit; this tests nonlinear, image-dependent fusion with negligible spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A tanh-bounded fusion correction around the invariant mean will exceed 9,316 correct predictions by learning nonlinear disagreement interactions while preserving the successful invariant representation and baseline training dynamics.
change: Reparameterize the fusion projection as a zero-initialized residual correction and apply it through tanh before adding it to the invariant features; parameter count and runtime remain essentially unchanged.
mechanism: Zero-centered nonlinear residual view fusion
evidence_used: Linear invariant–disagreement fusion improved validation correct from 9,286 to 9,316, while a larger nonlinear bottleneck exceeded the time limit and explicit coactivation fell to 9,298; this isolates nonlinear fusion without added parameters or substantial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing the redundant learned transform of invariant features while retaining a zero-initialized disagreement correction will exceed 9,316 correct predictions by simplifying optimization under the fixed exposure budget.
change: Replace the 128-to-64 joint projection with a zero-initialized 64-to-64 disagreement projection added residually to the invariant mean, reducing parameters and computation while preserving the initial function.
mechanism: Residual disagreement-only view fusion
evidence_used: Mean-and-disagreement fusion improved validation correct from 9,286 to 9,316, whereas added coactivation capacity fell to 9,298; this isolates the successful disagreement mechanism and removes an invariant transform that is expressively redundant with the following linear classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the terminal learning-rate floor from 10% to 2% will exceed 9,316 correct predictions by stabilizing late optimization before EMA averaging.
change: Preserve the successful joint view-fusion model while annealing its learning rate more aggressively near the end of the fixed exposure budget.
mechanism: Lower-floor cosine terminal annealing
evidence_used: Joint fusion reached 9,316 correct, whereas classifier widening fell to 9,254 and added coactivation fell to 9,298; the longer-EMA test timed out, so reducing terminal update noise is a focused, runtime-neutral test that adds no capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.64664945797995, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.19827737312316895, "validation_score": 9294.417265660868}

RECENT RESULT
hypothesis: Removing the redundant invariant-feature transform while retaining a zero-initialized disagreement correction will exceed 9,316 correct predictions by improving optimization within the fixed exposure budget.
change: Replace the 128-to-64 joint projection with a 64-to-64 disagreement projection added residually to the invariant mean, reducing parameters and computation while preserving the initial function.
mechanism: Residual disagreement-only view fusion
evidence_used: Joint invariant–disagreement fusion improved validation correct from 9,286 to 9,316, whereas added coactivation fell to 9,298. The previous residual-only verification timed out without measuring accuracy; its lower computational cost makes it the most informative unresolved fusion simplification to verify.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 220346, "training_seconds": 77.01268608402461, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.1999466983795166, "validation_score": 9295.416685174996}



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
