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

RECENT RESULT
hypothesis: Averaging floating-point BatchNorm buffers with the model parameters will exceed 9,328 correct predictions by reducing the mismatch between EMA weights and final-model normalization statistics.
change: Extend EMA averaging from learned parameters to every floating-point model state, while continuing to copy integer tracking buffers directly.
mechanism: State-consistent EMA for BatchNorm statistics
evidence_used: EMA-aligned dropout and label-smoothing annealing improved validation correct from 9,316 to 9,328; the current implementation averages parameters but pairs them with BatchNorm running statistics copied from the non-EMA model.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging floating-point BatchNorm buffers with the EMA parameters will exceed 9,328 correct predictions by reducing normalization noise and the mismatch between averaged weights and final-model statistics.
change: Extend EMA interpolation to every floating-point model state while continuing to copy integer tracking buffers directly.
mechanism: State-consistent EMA of BatchNorm statistics
evidence_used: EMA-aligned dropout and label-smoothing annealing reached 9,328 correct; the previous state-consistent EMA verification timed out, leaving this low-cost correction unresolved rather than disproven.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting BatchNorm momentum to 0.02 will exceed 9,328 correct predictions by smoothing evaluation statistics over roughly the same 50-step timescale as the parameter EMA.
change: Reduce every BatchNorm layer’s running-statistics momentum from the 0.1 default to 0.02 without adding per-step state processing.
mechanism: EMA-timescale BatchNorm statistics
evidence_used: EMA-aligned regularization reached 9,328 correct, while two attempts to average BatchNorm buffers with EMA parameters timed out; matching BatchNorm’s update rate to the existing 0.02 parameter interpolation tests the same state-consistency hypothesis at negligible runtime cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 10% to 15% will exceed 9,328 correct predictions by increasing useful late optimization after dropout and label smoothing have annealed away.
change: Increase the cosine schedule’s final learning rate from 2.1e-4 to 3.15e-4 while preserving the successful architecture, EMA, and regularization schedules.
mechanism: Moderately elevated terminal cosine floor
evidence_used: Lowering the floor from 10% to 2% reduced validation correct from 9,316 to 9,294, while the later removal of dropout and label smoothing improved the 10%-floor design to 9,328; the unmeasured 20%-floor attempt leaves a conservative increase unresolved.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Classifying each mirrored view independently and combining its logits with a learned content-dependent gate will exceed 9,328 correct predictions by preserving signed spatial evidence that pre-classifier averaging and absolute disagreement discard.
change: Replace invariant–disagreement feature fusion with a shared nonlinear classifier applied separately to both views, followed by a zero-initialized learned gate that selects their class predictions.
mechanism: Learned decision-level canonical-view routing
evidence_used: The invariant–disagreement model reached 9,316 correct and improved to 9,328 through regularization, while a spatial evidence readout still reached 9,307; this suggests spatially oriented evidence is valuable, but the current load-bearing assumption that mirrored features should be symmetrized before class prediction may erase it.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the functionally redundant random flip with one-pixel translations during the first half of training will exceed 9,328 correct predictions by adding spatial regularization while preserving clean late optimization and EMA collection.
change: Cycle deterministically through all nine one-pixel translations during the first half of training, then use unmodified images during the second half.
mechanism: Early cyclic translation augmentation
evidence_used: Annealing dropout and label smoothing during the final half improved validation correct from 9,316 to 9,328, supporting early-only regularization; meanwhile, random flipping is redundant because the model already symmetrically fuses every image with its mirrored view.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on balanced one-pixel translations during the first half will exceed 9,328 correct predictions by improving spatial robustness while leaving late optimization and EMA collection clean.
change: Replace the functionally redundant random horizontal flip with a deterministic cycle through all nine one-pixel translations during the first half of training.
mechanism: Early cyclic translation augmentation
evidence_used: Annealing dropout and label smoothing improved validation correct from 9,316 to 9,328, supporting early-only regularization; the prior translation verification timed out, so this low-cost hypothesis remains unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 66.09472895809449, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.20423024673461915, "validation_score": 9286.415202990753}



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
