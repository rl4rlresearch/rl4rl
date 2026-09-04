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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 74.8249261660967, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19362240524291993, "validation_score": 9358.418892941188}
prior_hypothesis: Reducing EMA decay from 0.97 to 0.96 will exceed 9,356 correct predictions by following the stronger late-training trajectory more responsively while preserving useful weight and BatchNorm averaging.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing total individual-view loss weight from 10% to 20% will exceed 9,341 correct predictions by further reducing orientation-specific errors while keeping the flip-ensemble objective dominant.
change: Weight the ensemble loss at 80% and each constituent-view loss at 10%, preserving all other training and evaluation behavior.
mechanism: Stronger auxiliary per-view supervision
evidence_used: Adding 10% individual-view supervision improved validation_correct from 9,335 to 9,341; increasing that successful auxiliary signal is the most direct test of whether stronger constituent accuracy further improves the ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 66.54733533295803, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.1948249137878418, "validation_score": 9330.418471354447}

RECENT RESULT
hypothesis: Reducing total individual-view supervision from 10% to 7.5% will exceed 9,341 correct predictions by retaining its benefit while moving away from the 20% setting that regressed to 9,330.
change: Weight the ensemble loss at 92.5% and each constituent-view loss at 3.75%, preserving all other behavior.
mechanism: Bracketed auxiliary-loss weight refinement
evidence_used: Total individual-view weights of 0%, 10%, and 20% produced 9,335, 9,341, and 9,330 correct respectively, indicating a local optimum near but slightly below 10%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.84282466699369, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19503475074768067, "validation_score": 9331.418397874779}

RECENT RESULT
hypothesis: Increasing total individual-view supervision from 10% to 12.5% will exceed 9,341 correct predictions by strengthening constituent-view accuracy without approaching the 20% setting that regressed to 9,330.
change: Weight the ensemble loss at 87.5% and each constituent-view loss at 6.25%, preserving all other training and evaluation behavior.
mechanism: Upper-bracket auxiliary-loss weight refinement
evidence_used: Total individual-view weights of 7.5%, 10%, and 20% produced 9,331, 9,341, and 9,330 correct respectively; testing 12.5% most directly brackets the successful 10% setting on its unexplored upper side.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.20164570887573, "validation_accuracy": 0.935, "validation_correct": 9350, "validation_cross_entropy": 0.19345836791992188, "validation_score": 9350.418950516783}

RECENT RESULT
hypothesis: Increasing total individual-view supervision from 12.5% to 15% will exceed 9,350 correct predictions by further strengthening constituent views while remaining below the 20% setting that regressed.
change: Weight the ensemble loss at 85% and each constituent-view loss at 7.5%, preserving all other behavior.
mechanism: Upper-bracket auxiliary-loss refinement
evidence_used: Individual-view weights of 10%, 12.5%, and 20% produced 9,341, 9,350, and 9,330 correct respectively, so 15% directly tests the unexplored interval between the current best and the known-regressive setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 73.16613262495957, "validation_accuracy": 0.9332, "validation_correct": 9332, "validation_cross_entropy": 0.19400927581787109, "validation_score": 9332.418757215815}

RECENT RESULT
hypothesis: Reducing total individual-view supervision from 12.5% to 12% will exceed 9,350 correct predictions by moving toward the estimated peak between the 10% and 15% results.
change: Weight the ensemble loss at 88% and each constituent-view loss at 6%, preserving all other training and evaluation behavior.
mechanism: Local auxiliary-loss weight refinement
evidence_used: Total individual-view weights of 10%, 12.5%, and 15% produced 9,341, 9,350, and 9,332 correct respectively; this brackets the optimum near 12.5%, with the asymmetric decline suggesting a slightly lower weight.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 55.939047374995425, "validation_accuracy": 0.9346, "validation_correct": 9346, "validation_cross_entropy": 0.19392794189453125, "validation_score": 9346.418785742804}

RECENT RESULT
hypothesis: Increasing the second-half EMA decay from 0.99 to 0.995 will exceed 9,350 correct predictions by reducing late-training variance while retaining more of the stable trajectory.
change: Slow both parameter and floating-buffer EMA updates from 1% to 0.5%, preserving the architecture, loss, schedule, exposure, and coherent BatchNorm averaging.
mechanism: Longer-horizon exponential weight and BatchNorm averaging
evidence_used: Parameter EMA improved validation_correct from 9,322 to 9,323, and extending it to BatchNorm buffers improved it further to 9,335; with auxiliary-loss tuning now bracketed around the 9,350 design, EMA horizon is the strongest validated mechanism left for a focused refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.69833358284086, "validation_accuracy": 0.9336, "validation_correct": 9336, "validation_cross_entropy": 0.19619019012451172, "validation_score": 9336.417993730536}

RECENT RESULT
hypothesis: Reducing the second-half EMA decay from 0.99 to 0.98 will exceed 9,350 correct predictions by tracking the better-trained late trajectory more responsively while retaining variance reduction.
change: Increase both parameter and floating-buffer EMA update rates from 1% to 2%, preserving all other behavior.
mechanism: Shorter-horizon coherent weight and BatchNorm averaging
evidence_used: The 0.99 coherent EMA supports the current 9,350-correct design, while lengthening its horizon with 0.995 regressed to 9,336; testing a shorter horizon is the most informative bracket of this validated mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 70.99202558305115, "validation_accuracy": 0.9354, "validation_correct": 9354, "validation_cross_entropy": 0.1932364303588867, "validation_score": 9354.41902844003}

RECENT RESULT
hypothesis: Reducing EMA decay from 0.98 to 0.97 will exceed 9,354 correct predictions by tracking the stronger late-training trajectory more responsively while retaining useful averaging.
change: Increase both parameter and floating-buffer EMA update rates from 2% to 3%, preserving all other behavior.
mechanism: Further-shortened coherent EMA horizon
evidence_used: Shortening coherent EMA decay from 0.99 to 0.98 improved validation_correct from 9,350 to 9,354, whereas lengthening it to 0.995 regressed to 9,336; continuing in the successful direction is the most informative next bracket.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.19535804213956, "validation_accuracy": 0.9356, "validation_correct": 9356, "validation_cross_entropy": 0.19343207778930663, "validation_score": 9356.41895974585}

RECENT RESULT
hypothesis: Reducing EMA decay from 0.97 to 0.96 will exceed 9,356 correct predictions by following the stronger late-training trajectory more responsively while preserving useful weight and BatchNorm averaging.
change: Increase both parameter and floating-buffer EMA update rates from 3% to 4%, preserving all other training and evaluation behavior.
mechanism: Shorter-horizon coherent EMA
evidence_used: Successively shortening coherent EMA decay from 0.99 to 0.98 to 0.97 increased validation_correct from 9,350 to 9,354 to 9,356, so continuing one measured step in the validated direction is the most informative next bracket.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 74.8249261660967, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19362240524291993, "validation_score": 9358.418892941188}

RECENT RESULT
hypothesis: Reducing EMA decay from 0.96 to 0.95 will exceed 9,358 correct predictions by tracking the stronger late-training trajectory more responsively while retaining useful weight and BatchNorm averaging.
change: Increase both parameter and floating-buffer EMA update rates from 4% to 5%, preserving all other training and evaluation behavior.
mechanism: Further-shortened coherent EMA horizon
evidence_used: Successively shortening coherent EMA decay from 0.99 to 0.98, 0.97, and 0.96 increased validation_correct from 9,350 to 9,354, 9,356, and 9,358; continuing one measured step is the most informative test of this monotonic trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.53198633296415, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19378726348876954, "validation_score": 9358.418835093398}

RECENT RESULT
hypothesis: An EMA decay of 0.955 will exceed the 0.96 design’s 9,358 correct predictions—or tie it with lower cross-entropy—by balancing late-trajectory responsiveness against variance reduction.
change: Increase parameter and floating-buffer EMA update rates from 4% to 4.5%, preserving all other behavior.
mechanism: Midpoint coherent EMA horizon refinement
evidence_used: Decay 0.96 achieved 9,358 correct with 0.193622 cross-entropy, while 0.95 tied the correct count but worsened cross-entropy to 0.193787; testing their midpoint directly refines the newly bracketed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 53.14107029209845, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.19370771865844727, "validation_score": 9357.418863003217}

RECENT RESULT
hypothesis: An EMA decay of 0.9625 will exceed the 0.96 design’s 9,358 correct predictions—or tie it with lower cross-entropy—by adding modest variance reduction without returning to the less responsive 0.97 horizon.
change: Reduce parameter and floating-buffer EMA update rates from 4% to 3.75%, preserving all other training behavior.
mechanism: Upper-side coherent EMA horizon refinement
evidence_used: Decay 0.96 achieved the best result at 9,358 correct and 0.193622 cross-entropy; 0.97 fell to 9,356 correct, while the lower-side midpoint 0.955 fell to 9,357, making 0.9625 the most informative untested local bracket.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.60182425007224, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.19357746963500977, "validation_score": 9357.418908711601}



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
