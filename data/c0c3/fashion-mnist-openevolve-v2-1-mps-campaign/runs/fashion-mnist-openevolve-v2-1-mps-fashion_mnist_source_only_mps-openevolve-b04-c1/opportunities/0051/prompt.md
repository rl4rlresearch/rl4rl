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

RECENT RESULT
hypothesis: Training on independently translated crops of up to two pixels will exceed 9,316 correct predictions by improving positional robustness without altering the successful mirrored-feature fusion.
change: Replace the functionally redundant random horizontal flip with efficient replicate-padded, per-image random translation.
mechanism: Per-example translation augmentation
evidence_used: Joint invariant–disagreement fusion reached 9,316 correct, while added fusion capacity and altered annealing failed to improve it. Because the current fusion is unchanged when its two mirrored views are swapped, the existing random flip adds no training variation; translation supplies a distinct spatial perturbation without parameters or extra model passes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 67.30764829204418, "validation_accuracy": 0.9219, "validation_correct": 9219, "validation_cross_entropy": 0.21961892318725587, "validation_score": 9219.409964121165}

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 10% to 20% will exceed 9,316 correct predictions by sustaining useful optimization under the short fixed-exposure budget.
change: Preserve the successful joint view-fusion model while increasing the cosine schedule’s final learning rate from 2.1e-4 to 4.2e-4.
mechanism: Higher-floor cosine terminal annealing
evidence_used: Lowering the terminal floor from 10% to 2% reduced validation correct from 9,316 to 9,294, indicating that stronger late optimization is more promising than further annealing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the shared 48-dimensional global bottleneck with multi-scale local context features and a direct class-specific spatial readout will exceed 9,316 correct predictions by preserving discriminative spatial evidence until the final logits.
change: Retain the successful invariant–disagreement fusion, but replace its flattened MLP classifier with lightweight depthwise context branches at two receptive-field scales, channel mixing, and direct per-class spatial templates.
mechanism: Multi-scale convolutional evidence readout
evidence_used: Joint view fusion reached 9,316 correct, but widening the general classifier fell to 9,254 and several increasingly elaborate fusion mechanisms failed or regressed. This challenges the shared assumption that fused features should first collapse into one small global hidden vector, testing a different prediction mechanism without materially increasing runtime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 140458, "training_seconds": 77.53983633383177, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.19459481658935546, "validation_score": 9307.418551958419}



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
