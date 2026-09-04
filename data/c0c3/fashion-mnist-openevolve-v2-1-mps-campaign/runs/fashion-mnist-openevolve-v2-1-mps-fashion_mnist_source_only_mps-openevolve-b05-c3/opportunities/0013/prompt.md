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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 74.47288516582921, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.24135459518432617, "validation_score": 9155.402785797016}
prior_hypothesis: Extending center/cardinal one-pixel augmentation from the final quarter to the final third of the validated batch-64 training run will exceed 9,141 correct predictions by increasing the EMA model’s exposure to inference-matched views while retaining broad ±2 translation regularization early in training.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 71.47676383284852, "validation_accuracy": 0.9128, "validation_correct": 9128, "validation_cross_entropy": 0.24578858108520507, "validation_score": 9128.40135220983}
prior_hypothesis: Halving the batch size to 64 will exceed 9,125 correct predictions by providing roughly twice as many updates, while scaling AdamW decay and EMA momentum preserves their behavior per training example.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.55882224999368, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.24300272483825683, "validation_score": 9141.402251732847}
prior_hypothesis: Using the validated batch-64 configuration while restricting the final quarter of training to the five center/cardinal one-pixel views used by inference will exceed 9,128 correct predictions by preserving early translation regularization and eliminating late optimization on the diagonal and two-pixel views that are mismatched with the best ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the 105,866-parameter two-layer CNN to a roughly 246k-parameter six-layer CNN, while doubling optimizer updates with batch size 128, will exceed 8,928 correct predictions by learning more discriminative features within the fixed 100,000-example exposure.
change: Replace the baseline with a six-convolution batch-normalized network, random per-image translation and horizontal flipping, AdamW with warmup-cosine scheduling, mild label smoothing, and end-of-training exponential weight averaging.
mechanism: High-capacity batch-normalized CNN with lightweight geometric augmentation
evidence_used: The starting design reaches 8,928/10,000 using only 105,866 of the allowed 250,000 parameters and 392 optimizer steps, leaving substantial capacity and optimization-step budget available.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245818, "training_seconds": 43.330183665966615, "validation_accuracy": 0.9099, "validation_correct": 9099, "validation_cross_entropy": 0.25343275756835937, "validation_score": 9099.398904525975}

RECENT RESULT
hypothesis: Averaging predictions for each image and its horizontal mirror will exceed the reference’s 9,099 correct predictions because the qualified model was trained with random horizontal flips, while preserving its parameter count and exposure budget.
change: Adopt the qualified six-layer CNN and training procedure, then add evaluation-time probability averaging between original and mirrored inputs.
mechanism: Flip-ensemble inference over the EMA-trained convolutional network
evidence_used: Reference Design 1 achieved 9,099/10,000 correct with 245,818 parameters; its 50% horizontal-flip augmentation directly supports a low-variance mirrored-view ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245818, "training_seconds": 48.65767579106614, "validation_accuracy": 0.911, "validation_correct": 9110, "validation_cross_entropy": 0.2512523277282715, "validation_score": 9110.399599656217}

RECENT RESULT
hypothesis: Averaging the center view and four one-pixel translations, each with its horizontal mirror, will exceed Reference Design 2’s 9,110 correct predictions because its training already learned both translation and flip invariance, and flip-only ensembling improved the prior model by 11 correct predictions.
change: Adopt the qualified 245,818-parameter six-layer CNN, batch-128 augmented training, warmup-cosine AdamW, and EMA, then extend evaluation from two-view flip averaging to a ten-view translation-and-flip probability ensemble.
mechanism: Translation-and-flip probability ensemble over an EMA-trained convolutional network
evidence_used: Reference Design 2 achieved 9,110 correct using flip ensembling, while its training also used random translations of up to two pixels; deterministic mild translation views therefore test the same successful augmentation-to-ensemble mechanism without increasing learned parameters or training exposure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245818, "training_seconds": 35.947825708892196, "validation_accuracy": 0.9125, "validation_correct": 9125, "validation_cross_entropy": 0.24993682403564454, "validation_score": 9125.40002021733}

RECENT RESULT
hypothesis: Adding the four one-pixel diagonal translations and their mirrors will exceed 9,125 correct predictions because the independent x/y training offsets explicitly include diagonal shifts, while cardinal translation views already improved flip-only ensembling by 15 predictions.
change: Expand evaluation from ten center/cardinal/flip views to all eighteen combinations of a 3×3 one-pixel translation grid and horizontal mirroring.
mechanism: Full 3×3 translation-and-flip probability ensemble
evidence_used: The current ten-view ensemble achieved 9,125 correct versus 9,110 for flip-only evaluation, showing that inference-time marginalization over training-supported translations improves the primary objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245818, "training_seconds": 34.16083383304067, "validation_accuracy": 0.9119, "validation_correct": 9119, "validation_cross_entropy": 0.2501277496337891, "validation_score": 9119.399959124295}

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
