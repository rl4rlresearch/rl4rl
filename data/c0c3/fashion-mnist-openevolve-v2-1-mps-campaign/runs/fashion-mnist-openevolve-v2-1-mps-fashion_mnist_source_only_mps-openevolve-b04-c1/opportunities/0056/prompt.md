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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.22494162502699, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19341388931274414, "validation_score": 9328.418966131094}
prior_hypothesis: Holding label smoothing at 0.02 for the first half of training and annealing it to zero during EMA collection will exceed 9,323 correct predictions by removing late soft-target bias while retaining early regularization.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Adding a direct 4×4 spatial readout alongside the successful nonlinear classifier will exceed 9,316 correct predictions by preserving class-specific coarse location evidence without replacing the proven fused-feature pathway.
change: Add a zero-initialized, adaptive-pooled linear logit branch from the fused feature map and sum it with the existing classifier output, increasing parameters from 224,442 to 234,692 with negligible computation.
mechanism: Coarse spatial logit bypass
evidence_used: The multi-scale convolutional evidence readout reached 9,307 correct—close to the 9,316 baseline despite discarding its successful MLP—suggesting direct spatial evidence is useful but insufficient alone; a residual bypass combines that evidence with the stronger existing classifier.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a direct 4×4 spatial readout to the successful fused-feature classifier will exceed 9,316 correct predictions by preserving class-specific location evidence alongside the nonlinear bottleneck.
change: Add a zero-initialized adaptive-pooled linear branch from the fused feature map and sum its logits with the existing classifier, increasing parameters to 234,692 without changing the initial function.
mechanism: Zero-initialized coarse spatial logit bypass
evidence_used: The standalone multi-scale spatial readout reached 9,307 correct despite replacing the stronger 9,316-correct MLP, suggesting spatial evidence is complementary; the prior residual-bypass implementation was not verifiable, so this focused combination remains unresolved.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a direct 4×4 spatial readout alongside the existing fused-feature classifier will exceed 9,316 correct predictions by retaining complementary class-specific location evidence.
change: Add a zero-initialized adaptive-pooled linear branch from the fused feature map and sum its logits with the existing classifier output, raising parameters from 224,442 to 234,692.
mechanism: Zero-initialized coarse spatial logit bypass
evidence_used: The standalone multi-scale spatial readout achieved 9,307 correct despite replacing the stronger 9,316-correct classifier, indicating spatial evidence is promising and potentially complementary; the prior bypass attempts were not verifiable, so the combination remains unresolved.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Holding dropout at 0.15 for the first half of training and annealing it to zero during EMA collection will exceed 9,316 correct predictions by reducing late gradient noise without weakening the learning-rate schedule.
change: Keep the successful architecture and early regularization unchanged, then linearly remove classifier dropout over the final half of the fixed exposure budget.
mechanism: EMA-aligned dropout annealing
evidence_used: Lowering the terminal learning-rate floor reduced validation correct from 9,316 to 9,294, suggesting late learning should remain active; annealing dropout instead preserves update magnitude while making late optimization and EMA snapshots less stochastic.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 67.75635575014167, "validation_accuracy": 0.9323, "validation_correct": 9323, "validation_cross_entropy": 0.192366902923584, "validation_score": 9323.419334014367}

RECENT RESULT
hypothesis: Holding label smoothing at 0.02 for the first half of training and annealing it to zero during EMA collection will exceed 9,323 correct predictions by removing late soft-target bias while retaining early regularization.
change: Reuse the successful dropout-decay schedule to linearly reduce label smoothing from 0.02 to zero over the final half of training.
mechanism: EMA-aligned label-smoothing annealing
evidence_used: Annealing dropout during the final half improved validation correct from 9,316 to 9,323, indicating that removing regularization during late optimization and EMA collection is beneficial; this tests the same principle for the remaining loss regularizer without changing runtime or capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.22494162502699, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19341388931274414, "validation_score": 9328.418966131094}



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
