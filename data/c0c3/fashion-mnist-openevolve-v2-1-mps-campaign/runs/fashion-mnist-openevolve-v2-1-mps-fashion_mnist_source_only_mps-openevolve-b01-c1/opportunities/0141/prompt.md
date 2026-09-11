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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.822912541916594, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21194436264038086, "validation_score": 9265.412560192872}
prior_hypothesis: Raising the cosine learning-rate floor from 3% to 5% of the 3e-3 peak will exceed 9,262 correct predictions by preserving useful late-training plasticity during the fixed two-pass exposure budget.

## Recent verification evidence

RECENT RESULT
hypothesis: Widening the proven flattened classifier bottleneck from 64 to 70 units will exceed 9,262 correct predictions by using the remaining parameter budget to learn additional spatially specific garment features.
change: Increase the classifier bottleneck width to 70, bringing the model to 249,976 learned parameters while preserving all training and evaluation behavior.
mechanism: Capacity-maximized coordinate-specific classifier head
evidence_used: The 233,434-parameter coordinate-specific design remains strongest, while position-free pooled alternatives reached only 9,253 and 9,249 correct; conservatively expanding the successful head is therefore more motivated than replacing its spatial representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249976, "training_seconds": 53.16531320801005, "validation_accuracy": 0.9221, "validation_correct": 9221, "validation_cross_entropy": 0.21635280227661133, "validation_score": 9221.411064946835}

RECENT RESULT
hypothesis: Reallocating classifier-head parameters into a second residual block with input-conditioned channel gating will exceed 9,262 correct predictions by learning richer spatial features while retaining the coordinate-specific layout information lost by prior pooling designs.
change: Reduce the flattened bottleneck from 64 to 43 units and use the recovered budget for a 56-channel residual refinement block whose channels are gated by global image context, yielding approximately 233,847 learned parameters.
mechanism: Context-gated residual refinement
evidence_used: Widening the coordinate-specific head to 249,976 parameters regressed to 9,221 correct, showing that additional head capacity is ineffective. Global pooling reached 9,253 but discarded useful spatial layout; using global statistics only to modulate a preserved 7×7 feature map tests context-aware representation without that information loss.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Halving the batch size while scaling both initial and scheduled learning rates proportionally will exceed 9,262 correct predictions by providing twice as many, finer-grained optimizer updates without increasing the learning-rate integral per presented example.
change: Change the batch size from 128 to 64 and scale AdamW’s initial and peak learning rates from 3.0e-4/3.0e-3 to 1.5e-4/1.5e-3.
mechanism: Example-normalized small-batch AdamW
evidence_used: The best 233,434-parameter architecture and 0.020→0.009 smoothing schedule should be preserved because head widening regressed to 9,221 and nearby smoothing refinements regressed to 9,257 and 9,256; optimization granularity remains an untested dimension that does not add model compute or parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the residual representation from 56 to 60 channels while narrowing the classifier bottleneck from 64 to 63 will exceed 9,262 correct predictions by investing capacity in spatial feature extraction without the expensive extra block that timed out.
change: Reallocate parameters from the flattened classifier into the existing residual and shortcut paths, retaining the proven topology and training procedure while increasing the model to 248,825 learned parameters.
mechanism: Compute-efficient feature-channel reallocation
evidence_used: Expanding only the classifier head to 249,976 parameters regressed sharply to 9,221 correct, while adding a second context-gated residual block exceeded the time limit; modestly widening the existing residual block tests richer spatial features with substantially less added computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248825, "training_seconds": 72.78075374988839, "validation_accuracy": 0.9238, "validation_correct": 9238, "validation_cross_entropy": 0.21194111709594726, "validation_score": 9238.412561297695}

RECENT RESULT
hypothesis: Cycling AdamW’s first-moment coefficient from 0.95 to 0.85 and back to 0.95 will exceed 9,262 correct predictions by improving adaptation near peak learning rate and smoothing convergence late in training without adding optimizer steps or compute-heavy model capacity.
change: Preserve the best architecture, augmentation, and 0.020→0.009 smoothing schedule while inversely coupling AdamW beta1 to the existing one-cycle learning-rate schedule.
mechanism: Inverse one-cycle AdamW momentum
evidence_used: The 233,434-parameter design with a 0.009 smoothing floor achieved 9,262 correct, while head widening and feature-channel reallocation regressed and the smaller-batch optimization test timed out; a zero-overhead change to the otherwise fixed optimizer trajectory is therefore the most informative next probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.98 EMA over the final 40% of training will exceed 9,262 correct predictions by reducing late optimizer noise while preserving the proven architecture and smoothing schedule.
change: Replace AdamW with an AdamW subclass that averages learned parameters and floating-point BatchNorm buffers late in training, then installs the averaged state after the final optimizer step.
mechanism: Late-training exponential model averaging
evidence_used: The 233,434-parameter design with 0.020→0.009 smoothing remains strongest at 9,262 correct, while capacity reallocations regressed and smaller-batch training timed out; averaging the existing trajectory tests optimization stability with little added computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 61.71108620800078, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.21338610916137696, "validation_score": 9244.41206998846}

RECENT RESULT
hypothesis: Lowering the cosine schedule’s final learning rate from 9e-5 to 3e-5 will exceed 9,262 correct predictions by reducing late-update noise while preserving the successful early trajectory.
change: Change only the cosine learning-rate floor from 3% to 1% of the 3e-3 peak.
mechanism: Deeper terminal learning-rate annealing
evidence_used: The best architecture and 0.020→0.009 smoothing schedule remain strongest, while capacity changes regressed and late EMA fell to 9,244 correct; a more convergent endpoint directly tests late optimization stability without extra parameters, steps, or runtime-heavy averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 52.30452495883219, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21324563941955565, "validation_score": 9253.412117697979}

RECENT RESULT
hypothesis: Raising the cosine learning-rate floor from 3% to 5% of the 3e-3 peak will exceed 9,262 correct predictions by preserving useful late-training plasticity during the fixed two-pass exposure budget.
change: Change only the cosine schedule’s terminal learning-rate floor from 9e-5 to 1.5e-4.
mechanism: Higher terminal learning-rate floor
evidence_used: Lowering the floor from 3% to 1% regressed from 9,262 to 9,253 correct and worsened cross-entropy from 0.21199 to 0.21325, motivating the informative opposite-direction test while preserving the strongest architecture, optimizer, and smoothing schedule.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.822912541916594, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21194436264038086, "validation_score": 9265.412560192872}

RECENT RESULT
hypothesis: Raising the cosine learning-rate floor from 5% to 7% of the 3e-3 peak will exceed 9,265 correct predictions by sustaining beneficial late-training plasticity during the fixed exposure budget.
change: Change only the cosine schedule’s terminal learning-rate floor from 1.5e-4 to 2.1e-4.
mechanism: Extended terminal learning-rate floor
evidence_used: Increasing the floor from 1% to 3% improved correct predictions from 9,253 to 9,262, and increasing it again to 5% improved them to 9,265 with slightly better cross-entropy; an equal-sized step to 7% is the clearest test of the observed trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.328348458046094, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.21339555015563966, "validation_score": 9252.412066782292}

RECENT RESULT
hypothesis: Lowering the cosine learning-rate floor from 5% to 4% of the 3e-3 peak will exceed 9,265 correct predictions by retaining more late plasticity than the 3% floor while avoiding the instability observed at 7%.
change: Change only the terminal learning-rate floor from 1.5e-4 to 1.2e-4.
mechanism: Refined terminal learning-rate annealing
evidence_used: The 5% floor achieved 9,265 correct versus 9,262 at 3%, while 7% regressed sharply to 9,252; testing 4% is the most promising local refinement on the less aggressive side of the current optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 50.46227295813151, "validation_accuracy": 0.9261, "validation_correct": 9261, "validation_cross_entropy": 0.21175265731811524, "validation_score": 9261.412625461953}

RECENT RESULT
hypothesis: A 4.5% cosine learning-rate floor will exceed 9265.412560 by retaining the 5% floor’s classification accuracy while moving cross-entropy toward the better 0.211753 achieved at 4%.
change: Lower the terminal learning-rate multiplier from 0.05 to 0.045 while preserving the 3e-3 peak and all other behavior.
mechanism: Half-step terminal learning-rate refinement
evidence_used: The 5% floor produced the best result with 9,265 correct; 4% was only four predictions behind and had lower cross-entropy, whereas 7% regressed by thirteen predictions. The midpoint between 4% and 5% is therefore the strongest remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 60.95691333292052, "validation_accuracy": 0.9262, "validation_correct": 9262, "validation_cross_entropy": 0.21171827087402345, "validation_score": 9262.412637171543}

RECENT RESULT
hypothesis: Adding one lightweight self-attention block over the 7×7 feature grid will exceed 9,265 correct predictions by learning image-conditioned long-range relationships between garment parts while retaining coordinate-specific classification.
change: Insert learned positional embeddings and a pre-normalized four-head attention/MLP block before the flattened classifier, narrowing its bottleneck from 64 to 59 to keep the model at approximately 248,097 parameters.
mechanism: Position-aware global token attention
evidence_used: Position-free pooling alternatives regressed to 9,253 and 9,249, showing spatial layout matters, while widening the static flattened head regressed to 9,221. This challenges the load-bearing assumption that local convolutions followed by fixed flattened mixing are sufficient, using content-dependent global mixing without the expensive extra convolutional block that timed out.
result: training did not finish within the verification time limit



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
