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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 61.38587808399461, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21609233474731446, "validation_score": 9247.411152990373}
prior_hypothesis: Ramping flip-pair allocation from 25% early to 50% late will exceed 9,246 correct predictions by preserving individual-view feature learning early and emphasizing validation-aligned flip aggregation near convergence.

## Recent verification evidence

RECENT RESULT
hypothesis: Evaluating a 0.99-decay exponential average of the optimization trajectory will exceed 9,247 correct predictions by reducing endpoint variance without changing examples, augmentation, architecture, or training time materially.
change: Maintain an exponential moving average of every learned parameter after each optimizer step and copy the averaged parameters into the model after the final step.
mechanism: Exponential moving-average checkpoint ensembling
evidence_used: Loss-schedule refinements and added architectural features tied or regressed from the 9,247-correct design, while parameter averaging remains an untested temporal-ensemble axis that preserves the proven training objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 60.523177166003734, "validation_accuracy": 0.9225, "validation_correct": 9225, "validation_cross_entropy": 0.21978302307128905, "validation_score": 9225.40990896786}

RECENT RESULT
hypothesis: Decreasing central-crop allocation from 10% to 7.5% will exceed 9,247 correct predictions by favoring the broader crop distribution after increasing central allocation to 12.5% regressed to 9,239.
change: Reweight validation aggregation and all corresponding training objectives from 90/10 to 92.5/7.5 full-versus-central allocation.
mechanism: Reduced central-crop emphasis
evidence_used: Raising central-crop allocation from 10% to 12.5% reduced validation correctness by eight images, providing directional evidence that the successful ensemble may benefit from less central-crop emphasis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 55.809055167017505, "validation_accuracy": 0.9238, "validation_correct": 9238, "validation_cross_entropy": 0.21665183029174806, "validation_score": 9238.41096391552}

RECENT RESULT
hypothesis: Annealing label smoothing from 0.02 to zero will exceed 9,247 correct predictions by retaining early regularization while sharpening class boundaries near convergence.
change: Apply a progress-normalized cosine decay to label smoothing in every individual, pair, and ensemble cross-entropy term, preserving all other settings.
mechanism: Cosine-decayed label smoothing
evidence_used: The cosine pair curriculum improved fixed supervision from 9,246 to 9,247 correct, showing that temporal loss allocation can help; subsequent crop, architecture, and averaging changes regressed, motivating an orthogonal target-regularization schedule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 48.12137687508948, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2165679069519043, "validation_score": 9243.410992265324}

RECENT RESULT
hypothesis: Adding independent ±8° rotations before the proven translation-and-flip pipeline will exceed 9,247 correct predictions by learning a missing orientation invariance without changing model capacity or view supervision.
change: Apply a vectorized random rotation to each training image before reflected padding and crop construction.
mechanism: Vectorized mild rotation augmentation
evidence_used: Schedule refinements plateaued at or below 9,247 correct, while additional translation microbatching regressed to 9,237; this motivates testing an orthogonal geometric augmentation while preserving the successful crop ensemble and loss curriculum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 53.24910895782523, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.23259494018554688, "validation_score": 9162.405648265865}

RECENT RESULT
hypothesis: Adding one parameter-efficient residual block at 7×7 resolution will exceed 9,247 correct predictions by improving part-to-whole feature composition while preserving the proven fixed-coordinate decoder and training objective.
change: Insert a 48→32→32→48 bottleneck residual block after the existing residual stage, increasing learned parameters from 230,442 to 242,954.
mechanism: Bottleneck spatial residual refinement
evidence_used: Widening the flattened decoder regressed to 9,210 and raw-image shortcuts reached 9,224, suggesting additional decoder capacity or low-level inputs are less useful than further learned spatial feature processing; the broader residual-pyramid attempt was not verifiable, so this isolates a conservative feature-refinement change below the parameter ceiling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 242954, "training_seconds": 72.68157358304597, "validation_accuracy": 0.9204, "validation_correct": 9204, "validation_cross_entropy": 0.21752637176513673, "validation_score": 9204.410668722745}

RECENT RESULT
hypothesis: Applying a 4×4 mean-valued cutout to half of training images will exceed 9,247 correct predictions by discouraging brittle localized-feature reliance while preserving the successful translation, flip, and aggregation geometry.
change: Add independent per-image 4×4 random erasing before reflected padding, with 50% probability and each image’s mean intensity as the neutral fill value.
mechanism: Mean-valued localized cutout
evidence_used: Rotation augmentation sharply regressed to 9,162 correct, and added raw localized inputs and spatial refinement reached only 9,224 and 9,204; this motivates a lightweight appearance regularizer that preserves alignment, architecture, compute, and the 9,247-correct ensemble curriculum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 58.103263542056084, "validation_accuracy": 0.9223, "validation_correct": 9223, "validation_cross_entropy": 0.21688347930908203, "validation_score": 9223.410885683388}

RECENT RESULT
hypothesis: Reducing BatchNorm momentum from 0.1 to 0.03 will exceed 9,247 correct predictions by averaging running statistics across more late-training batches, improving fixed validation inference without altering learned capacity or supervision.
change: Set every BatchNorm layer’s running-statistics momentum to 0.03 while preserving the proven architecture, augmentation, loss curricula, optimizer, and evaluation ensemble.
mechanism: Low-variance BatchNorm statistic tracking
evidence_used: The 9,247-correct design remains strongest while added architecture, geometric augmentation, cutout, and parameter averaging all regressed; isolating inference-statistic estimation tests a low-cost stability axis without disturbing the successful computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 61.57845545792952, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.21616607666015625, "validation_score": 9242.411128060217}

RECENT RESULT
hypothesis: Adding 10% dropout to the parameter-dominant fixed-coordinate classifier will exceed 9,247 correct predictions by reducing feature co-adaptation without corrupting the successful input geometry or increasing compute materially.
change: Insert dropout after normalization in the 80-unit classifier head while preserving the proven architecture, augmentation, objectives, optimizer, and evaluation ensemble.
mechanism: Dense-head dropout regularization
evidence_used: Widening the flattened head regressed to 9,210 correct and adding localized raw inputs reached 9,224, suggesting the decoder is not capacity-limited; unlike the 4×4 input cutout that regressed to 9,223, modest head dropout regularizes its roughly 188,000 dense weights without destroying image evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 51.946427166927606, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.216678466796875, "validation_score": 9236.410954918365}

RECENT RESULT
hypothesis: Replacing both max-pooling operations with space-to-depth rearrangement and learned channel mixing will exceed 9,247 correct predictions by retaining fine spatial phase information before hierarchical feature extraction.
change: Replace the 16/32-channel max-pooling stem with a 20/40-channel stem whose lossless pixel-unshuffle stages learn how to combine each 2×2 neighborhood; the resulting 32×7×7 representation preserves the proven residual decoder and training objective at 239,918 parameters.
mechanism: Learned phase-preserving space-to-depth stem
evidence_used: The localized raw-image shortcut reached only 9,224 correct and added post-pooling residual refinement reached 9,204, indicating that neither restoring low-level evidence late nor further processing already-pooled features repairs the representation. This challenges the shared assumption that fixed max-pooling should discard spatial phase early and instead learns the downsampling computation before information is lost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239918, "training_seconds": 70.56849791691639, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.22164268341064453, "validation_score": 9191.40928497898}

RECENT RESULT
hypothesis: Uniformly averaging the final 24 optimizer iterates will exceed 9,247 correct predictions by reducing endpoint minibatch noise without the trajectory lag that caused the 0.99-decay EMA to regress to 9,225.
change: Accumulate each learned parameter over the final 24 post-update checkpoints and replace the endpoint parameters with their uniform average after the last optimizer step.
mechanism: Short-horizon tail checkpoint averaging
evidence_used: Full-trajectory EMA reduced correctness from 9,247 to 9,225, indicating that broad temporal smoothing is harmful; a tightly localized tail average isolates variance reduction from stale-parameter bias while preserving the proven architecture, supervision, and schedule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 50.48445470887236, "validation_accuracy": 0.9105, "validation_correct": 9105, "validation_cross_entropy": 0.25019538116455076, "validation_score": 9105.399937487799}

RECENT RESULT
hypothesis: Sampling crop offsets independently per image will exceed 9,247 correct predictions by increasing coverage of the proven translation distribution and preventing batch-correlated augmentation without introducing new distortions.
change: Replace batch-wide full and central crop offsets with per-example offsets selected from an unfolded crop bank; preserve opposite-offset pairing, losses, architecture, and evaluation.
mechanism: Per-example translation sampling
evidence_used: Rotation and cutout regressed to 9,162 and 9,223 correct, while architectural changes also underperformed; this motivates improving diversity within the successful ±2-pixel crop geometry rather than adding invariances or capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 58.38767279102467, "validation_accuracy": 0.9219, "validation_correct": 9219, "validation_cross_entropy": 0.21577327575683594, "validation_score": 9219.411260890472}

RECENT RESULT
hypothesis: Expanding the cosine flip-pair allocation from 20% early to 55% late will exceed 9,247 correct predictions by strengthening individual-view learning early and validation-aligned flip aggregation near convergence while preserving the same mean loss allocation.
change: Increase the amplitude of the pair-loss curriculum from ±0.125 to ±0.175, leaving architecture, augmentation, optimizer, and all other objectives unchanged.
mechanism: Widened flip-pair supervision curriculum
evidence_used: The existing 25%-to-50% cosine pair curriculum improved fixed supervision from 9,246 to 9,247 correct, whereas later architecture, augmentation, averaging, and label-smoothing changes regressed; widening that successful curriculum is the most direct local test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 58.02698054094799, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21606791076660156, "validation_score": 9245.411161248128}



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
