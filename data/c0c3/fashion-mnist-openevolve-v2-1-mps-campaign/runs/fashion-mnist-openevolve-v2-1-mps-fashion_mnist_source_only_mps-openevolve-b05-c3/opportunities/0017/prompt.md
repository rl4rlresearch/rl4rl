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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.46581783308648, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.23885296478271484, "validation_score": 9167.403599147125}
prior_hypothesis: Beginning center/cardinal augmentation after 39/64 of training will exceed 9,163 correct predictions by extending the monotonically beneficial inference-aligned phase beyond three-eighths while making a smaller change than the timed-out two-fifths configuration.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.43390987510793, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.24108177490234375, "validation_score": 9163.402874339235}
prior_hypothesis: Beginning center/cardinal one-pixel augmentation after five-eighths of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the timed-out two-fifths configuration.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 74.47288516582921, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.24135459518432617, "validation_score": 9155.402785797016}
prior_hypothesis: Extending center/cardinal one-pixel augmentation from the final quarter to the final third of the validated batch-64 training run will exceed 9,141 correct predictions by increasing the EMA model’s exposure to inference-matched views while retaining broad ±2 translation regularization early in training.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.55882224999368, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.24300272483825683, "validation_score": 9141.402251732847}
prior_hypothesis: Using the validated batch-64 configuration while restricting the final quarter of training to the five center/cardinal one-pixel views used by inference will exceed 9,128 correct predictions by preserving early translation regularization and eliminating late optimization on the diagonal and two-pixel views that are mismatched with the best ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: Giving the original and mirrored center views twice the weight of each one-pixel cardinal view will exceed 9,125 correct predictions by retaining useful translation marginalization while reducing errors from shifted-view artifacts.
change: Replace flip-only evaluation with a ten-view probability ensemble comprising center, horizontal mirror, and four one-pixel cardinal translations with mirrors, using double weight for the two center views.
mechanism: Center-weighted cardinal translation-and-flip ensemble
evidence_used: Equal weighting of these ten views achieved 9,125 correct versus 9,110 for flip-only evaluation, while adding diagonal views reduced performance to 9,119; this suggests translations help but excessive weight on displaced views is harmful.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245818, "training_seconds": 36.542762249941006, "validation_accuracy": 0.9125, "validation_correct": 9125, "validation_cross_entropy": 0.24997038497924803, "validation_score": 9125.40000947703}

RECENT RESULT
hypothesis: Halving the batch size to 64 will exceed 9,125 correct predictions by providing roughly twice as many updates, while scaling AdamW decay and EMA momentum preserves their behavior per training example.
change: Adopt the best validated equal-weight center/cardinal/flip ensemble, use batch size 64, halve weight decay, and slow EMA updates to retain the original example-based averaging horizon.
mechanism: Higher-frequency optimization with exposure-matched EMA and validated ten-view inference
evidence_used: Equal ten-view inference achieved the best known score at 9,125 correct, while diagonal and center-weighted variants did not improve it; the next informative axis is therefore training optimization, and the existing batch-128 run performs only 782 updates.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 71.47676383284852, "validation_accuracy": 0.9128, "validation_correct": 9128, "validation_cross_entropy": 0.24578858108520507, "validation_score": 9128.40135220983}

RECENT RESULT
hypothesis: Reducing batch size from 64 to 32 will exceed 9,128 correct predictions because the previous halving increased correct predictions and reduced cross-entropy, while halving AdamW decay and EMA update rate preserves their behavior per training example.
change: Use batch size 32, halve weight decay to 7.5e-5, and halve EMA interpolation to 0.0025 while retaining the best validated architecture, augmentation, schedule, and ten-view inference.
mechanism: Higher-frequency optimization with exposure-matched regularization and EMA
evidence_used: Moving from batch 128 to 64 increased validation correct from 9,125 to 9,128 and reduced cross-entropy from 0.24994 to 0.24579, directly supporting another controlled increase in optimizer-update frequency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch size 48 will exceed 9,128 correct predictions by adding one-third more optimizer updates than batch 64 while avoiding the verification timeout encountered at batch 32.
change: Use batch size 48 and scale AdamW weight decay and EMA interpolation proportionally to preserve their behavior per training example.
mechanism: Intermediate-frequency optimization with exposure-matched regularization and EMA
evidence_used: Reducing batch size from 128 to 64 improved correct predictions from 9,125 to 9,128 and cross-entropy from 0.24994 to 0.24579, while batch 32 timed out; batch 48 tests an intermediate update/runtime tradeoff.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding residual connections to the best validated batch-64 configuration will exceed 9,128 correct predictions by improving optimization through the six-layer network without increasing parameters or training exposure.
change: Adopt the validated batch-64 regularization, EMA, and equal-weight ten-view inference settings, then replace each convolution pair with an equivalent-parameter residual stage.
mechanism: Parameter-free residual refinement within each convolutional stage
evidence_used: Batch size 64 improved validation correct from 9,125 to 9,128 and reduced cross-entropy to 0.24579, indicating that optimization quality remains consequential; parameter-free residual paths target that limitation while retaining the proven architecture capacity and feasible runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the spatially indexed flatten-MLP head with class-specific attention over local evidence, while reallocating its parameters to 112-channel convolutional features, will exceed 9,128 correct predictions by reducing sensitivity to the final 3×3 grid alignment without discarding localized garment cues.
change: Replace the old assumption that a flattened 3×3 feature grid is the best classifier input with two fully convolutional heads that learn class-specific evidence and attention maps; widen the final feature stages while retaining the validated batch-64 training and ten-view inference.
mechanism: Class-specific spatial attention pooling
evidence_used: Cardinal translation ensembling improved flip-only evaluation from 9,110 to 9,125 correct, showing that spatial alignment remains load-bearing, while diagonal views reduced accuracy; learned attention pooling directly targets that sensitivity instead of adding more displaced inference views.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245300, "training_seconds": 76.35996066592634, "validation_accuracy": 0.9103, "validation_correct": 9103, "validation_cross_entropy": 0.2622627799987793, "validation_score": 9103.39611403261}

RECENT RESULT
hypothesis: Using the validated batch-64 configuration while restricting the final quarter of training to the five center/cardinal one-pixel views used by inference will exceed 9,128 correct predictions by preserving early translation regularization and eliminating late optimization on the diagonal and two-pixel views that are mismatched with the best ensemble.
change: Switch to batch size 64 with exposure-matched AdamW decay and EMA momentum, and replace uniform ±2 translation augmentation during the final training quarter with uniform center/cardinal one-pixel translations while retaining horizontal flips.
mechanism: Late-phase inference-aligned cardinal augmentation
evidence_used: Batch size 64 achieved the best result at 9,128 correct, while adding diagonal inference views reduced correct predictions from 9,125 to 9,119; this motivates retaining the proven optimizer configuration and aligning EMA-dominant late training with the successful center/cardinal ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.55882224999368, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.24300272483825683, "validation_score": 9141.402251732847}

RECENT RESULT
hypothesis: Extending center/cardinal one-pixel augmentation from the final quarter to the final third of the validated batch-64 training run will exceed 9,141 correct predictions by increasing the EMA model’s exposure to inference-matched views while retaining broad ±2 translation regularization early in training.
change: Adopt the validated batch-64 optimizer, EMA, and equal-weight ten-view ensemble, then begin inference-aligned cardinal augmentation after two-thirds rather than three-quarters of training.
mechanism: Extended inference-aligned augmentation phase
evidence_used: Restricting the final quarter to center/cardinal views improved the batch-64 result from 9,128 to 9,141 correct; varying the duration of that successful phase is the most direct next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 74.47288516582921, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.24135459518432617, "validation_score": 9155.402785797016}

RECENT RESULT
hypothesis: Beginning center/cardinal one-pixel augmentation halfway through training will exceed 9,155 correct predictions by further increasing inference-matched optimization while preserving 50,000 early examples of broad ±2 translation regularization.
change: Extend the validated inference-aligned augmentation phase from the final third to the final half of training.
mechanism: Half-run inference-aligned augmentation
evidence_used: Extending this phase from the final quarter to the final third increased validation correct from 9,141 to 9,155 and reduced cross-entropy from 0.24300 to 0.24135, directly supporting another controlled boundary extension.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Beginning center/cardinal one-pixel augmentation after 60% of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the unverified half-run configuration.
change: Run broad ±2 translation augmentation for the first three-fifths of training and center/cardinal augmentation for the final two-fifths.
mechanism: Intermediate-duration inference-aligned augmentation
evidence_used: Extending the cardinal phase from the final quarter to the final third improved validation correct from 9,141 to 9,155 and lowered cross-entropy; the attempted final-half phase timed out, making a two-fifths phase the closest informative intermediate test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Beginning center/cardinal one-pixel augmentation after five-eighths of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the timed-out two-fifths configuration.
change: Preserve the validated batch-64 model, optimizer, EMA, loss, and ten-view ensemble while using broad ±2 translations for the first five-eighths of training and center/cardinal one-pixel translations for the final three-eighths.
mechanism: Three-eighths inference-aligned augmentation phase
evidence_used: Extending the cardinal phase from the final quarter to the final third increased validation correct from 9,141 to 9,155 and reduced cross-entropy from 0.24300 to 0.24135; three-eighths is the untested midpoint between that best completed duration and the timed-out two-fifths attempt.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.43390987510793, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.24108177490234375, "validation_score": 9163.402874339235}

RECENT RESULT
hypothesis: Beginning center/cardinal augmentation after 39/64 of training will exceed 9,163 correct predictions by extending the monotonically beneficial inference-aligned phase beyond three-eighths while making a smaller change than the timed-out two-fifths configuration.
change: Adopt the best validated batch-64 architecture, optimizer, EMA, loss, and ten-view ensemble, then use broad ±2 translations for the first 39/64 of training and center/cardinal translations thereafter.
mechanism: Slightly extended inference-aligned augmentation phase
evidence_used: Extending the cardinal phase from one-quarter to one-third to three-eighths increased validation correct from 9,141 to 9,155 to 9,163; 25/64 is the smallest informative extension toward the unverified two-fifths phase.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.46581783308648, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.23885296478271484, "validation_score": 9167.403599147125}



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
