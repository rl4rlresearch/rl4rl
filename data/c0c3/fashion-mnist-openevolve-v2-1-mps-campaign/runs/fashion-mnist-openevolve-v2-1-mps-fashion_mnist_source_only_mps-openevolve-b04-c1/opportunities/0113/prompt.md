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
hypothesis: Applying random translations of up to two pixels during the first 75% of training will exceed 9,328 correct predictions by improving positional robustness while the final clean-data phase preserves precise classification boundaries.
change: Add efficient replicate-padded per-image random crops before the existing horizontal flip, disabling translation augmentation for the final quarter of training.
mechanism: Early per-image translation augmentation with clean late fine-tuning
evidence_used: The unchanged early-fusion architecture remains best at 9,328 correct, while structural head and refinement changes failed or timed out; augmentation tests a distinct, parameter-neutral source of generalization and replaces some value lost because the existing flip is largely redundant with symmetric paired-view fusion.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by averaging over more late-training updates and suppressing trajectory variance without changing optimization or runtime materially.
change: Double the effective parameter-averaging horizon during the second half of training.
mechanism: Longer-horizon parameter EMA
evidence_used: Learning-rate-floor changes in both directions failed to improve the 9,328-correct baseline, while the prior runtime-neutral 0.01 EMA attempt timed out and produced no contrary accuracy evidence, leaving this stabilization hypothesis unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 68.9803614160046, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1924131622314453, "validation_score": 9322.41931774643}

RECENT RESULT
hypothesis: Increasing the EMA update rate from 0.02 to 0.03 will exceed 9,328 correct predictions by tracking the improving late-training trajectory more responsively while still averaging roughly 33 optimizer updates.
change: Increase only the parameter-EMA interpolation rate from 0.02 to 0.03.
mechanism: Shorter-horizon parameter EMA
evidence_used: Extending the EMA horizon with a 0.01 update rate reduced validation correct from 9,328 to 9,322, directly motivating a runtime-neutral test in the opposite direction.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 82.7579179580789, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19399065170288085, "validation_score": 9325.418763747679}

RECENT RESULT
hypothesis: Retaining more label smoothing during the second half while still reaching zero at training end will exceed 9,328 correct predictions by preserving useful late regularization without permanently biasing the final classifier toward soft targets.
change: Replace the linear label-smoothing decay with a delayed quadratic decay; keep the dropout schedule and all other training behavior unchanged.
mechanism: Delayed quadratic label-smoothing decay
evidence_used: Ending label smoothing at 80% progress reduced validation correct from 9,328 to 9,316, while the linear full-duration decay remains best, motivating a runtime-neutral test in the opposite direction: slower soft-target removal.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining more label smoothing during late EMA collection while still reaching zero at training end will exceed 9,328 correct predictions.
change: Replace linear label-smoothing decay with quadratic decay while leaving dropout and all other training behavior unchanged.
mechanism: Delayed quadratic label-smoothing decay
evidence_used: Ending smoothing at 80% progress reduced validation correct from 9,328 to 9,316, favoring later regularization; the prior quadratic-decay verification timed out and supplied no contrary accuracy evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 79.12329904199578, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1928966495513916, "validation_score": 9325.41914779473}

RECENT RESULT
hypothesis: Uniformly averaging the final 100 parameter states will exceed 9,328 correct predictions by matching the successful 0.02 EMA’s effective sample size while eliminating its long geometric tail and greater weighting of the noisiest final updates.
change: Replace EMA collection over the second half of training with an exact uniform average of the final 100 optimizer-step parameter states; continue using final normalization buffers.
mechanism: Uniform tail-window parameter averaging
evidence_used: Constant EMA rates of 0.01 and 0.03 reached only 9,322 and 9,325 correct, while 0.02 reached 9,328, indicating that an averaging horizon near 100 states is best; this patch preserves that horizon with a different weighting profile and less averaging overhead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 79.37664154195227, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19483280754089355, "validation_score": 9325.418468589785}

RECENT RESULT
hypothesis: Adding low-weight per-view supervision during early training and annealing it away during EMA collection will exceed 9,328 correct predictions by strengthening the successful invariant feature pathway while preserving late specialization of the fused classifier.
change: Reuse the shared fusion and classifier heads to classify each mirrored view independently, add their cross-entropy at weight 0.15, and linearly remove that auxiliary loss over the second half of training.
mechanism: Annealed single-view auxiliary supervision
evidence_used: Replacing the full fusion with disagreement-only residual fusion fell to 9,290 correct, showing that the transformed invariant pathway is essential; dense-head widening also fell to 9,300, motivating parameter-neutral supervision rather than additional capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 83.7917965000961, "validation_accuracy": 0.9313, "validation_correct": 9313, "validation_cross_entropy": 0.194272106552124, "validation_score": 9313.418665057367}

RECENT RESULT
hypothesis: Efficiently applying translations of up to two pixels during the first 75% of training will exceed 9,328 correct predictions by improving positional robustness while retaining a clean late fine-tuning phase.
change: Replace redundant mirrored convolution passes with exact feature-map flipping, and replace redundant horizontal-flip augmentation with deterministic batchwise replicate-padded translations that add minimal runtime overhead.
mechanism: Equivariance-cached cyclic translation augmentation
evidence_used: Per-image translation augmentation timed out without accuracy evidence, while equivariance-cached feature extraction completed in 62.99 seconds; this tests the same promising invariance mechanism using a cheaper translation implementation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging BatchNorm running means and variances alongside the learned parameters will exceed 9,328 correct predictions by evaluating a normalization state aligned with the successful parameter-EMA trajectory.
change: Extend the existing 0.02 EMA to every floating-point model-state tensor while continuing to copy integer BatchNorm counters directly.
mechanism: EMA-consistent BatchNorm statistics
evidence_used: EMA-rate changes to 0.01 and 0.03 reduced validation correct to 9,322 and 9,325, so the 0.02 parameter horizon should be preserved; the current implementation pairs those averaged parameters with final-step normalization buffers, making state alignment a distinct runtime-neutral target.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 62.15120491711423, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19391162071228027, "validation_score": 9327.418791467748}

RECENT RESULT
hypothesis: Adding content-dependent interactions between all 7×7 feature locations will exceed 9,328 correct predictions by modeling relationships between garment parts that the existing local convolutions and static flattened classifier cannot express efficiently.
change: Preserve the successful mirrored-view fusion, but insert a four-head positional self-attention residual before classification. Its output projection starts at zero so training begins as the verified baseline while learning a genuinely nonlocal prediction mechanism.
mechanism: Zero-initialized positional self-attention over fused spatial features
evidence_used: Dense-head widening reduced performance to 9,300 and disagreement-only fusion reached 9,290, so neither generic classifier capacity nor removing the invariant pathway helped. This keeps that pathway and instead challenges the shared assumption that purely local features followed by fixed dense aggregation are sufficient.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Content-dependent channel gating of the fused 7×7 representation will exceed 9,328 correct predictions by emphasizing class-relevant feature channels without the runtime and optimization costs of spatial refinement or attention.
change: Add a 2,128-parameter squeeze-and-excitation gate after mirrored-view fusion, initialized to preserve the current representation exactly at the start of training.
mechanism: Identity-initialized global channel recalibration
evidence_used: Dense-head widening fell to 9,300 correct and global-max evidence reached only 9,325, while spatial refinement and attention timed out; this motivates a lightweight, structurally targeted use of global context instead of more static head capacity or expensive spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining 0.05 classifier dropout at training end will exceed 9,328 correct predictions by regularizing late hard-label optimization without the soft-target bias that made slower label-smoothing decay underperform.
change: Keep the existing early dropout and label-smoothing schedules, but anneal classifier dropout from 0.15 to 0.05 instead of zero during the second half.
mechanism: Late hard-label dropout floor
evidence_used: Ending label smoothing earlier reduced correct predictions to 9,316, indicating late regularization matters, while retaining more smoothing reached only 9,325; dense-head widening also fell to 9,300, motivating unbiased late classifier regularization rather than more capacity or softer targets.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 73.2502585418988, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.19472210578918456, "validation_score": 9321.418507364664}



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
