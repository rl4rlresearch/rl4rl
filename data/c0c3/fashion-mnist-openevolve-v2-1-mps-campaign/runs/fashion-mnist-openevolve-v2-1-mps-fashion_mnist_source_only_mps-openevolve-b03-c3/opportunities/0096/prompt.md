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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.78442587493919, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21246144828796387, "validation_score": 9265.41238424587}
prior_hypothesis: Restoring batch size 192 and shifting 0.02 ensemble weight from radius-1 diagonal views to the unshifted view will exceed 9,268 correct predictions by reducing reliance on more-displaced inputs while preserving total calibration weight.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 49.11529295798391, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21245572166442872, "validation_score": 9268.412386193628}
prior_hypothesis: Restoring the verified 40-local/24-context architecture and redistributing the unchanged 0.10 radius-2 TTA weight toward less-displaced axial views will exceed 9,268 correct predictions.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 51.062351166969165, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246516418457032, "validation_score": 9268.412382982018}
prior_hypothesis: Reallocating eight aggregator channels from the local branch to the successful dilation-2 context branch will exceed 9,265 correct predictions by strengthening broader garment-shape modeling while retaining 40 channels for local detail and keeping parameter count unchanged.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.15950512513518, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21244136581420897, "validation_score": 9268.412391076467}
prior_hypothesis: Favoring vertical over horizontal radius-1 translations while restoring the best radius-2 weights will exceed 9,268 correct predictions because the opposite horizontal bias worsened cross-entropy.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the verified 40-local/24-context model while reducing the batch size to 128 will exceed 9,268 correct predictions because the fixed 100,000-example exposure will provide roughly 782 optimizer steps instead of 522.
change: Restore the best multiscale architecture, mildly distance-stratified TTA, and 1.253 calibration, then reduce only the training batch size from 192 to 128.
mechanism: Higher-update small-minibatch optimization
evidence_used: The 40-local/24-context design reached 9,268 correct, while subsequent architecture, TTA-aggregation, augmentation-alignment, and EMA changes did not improve it; every reported design used batch size 192, leaving optimizer-update density as an untested lever.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245150, "training_seconds": 59.16914387512952, "validation_accuracy": 0.9263, "validation_correct": 9263, "validation_cross_entropy": 0.20832988815307618, "validation_score": 9263.413794283251}

RECENT RESULT
hypothesis: Moving one quarter of the way from the best 0.009375/0.00625/0.003125 radius-2 weighting toward the stronger axial weighting will preserve 9,268 correct predictions while reducing validation cross-entropy below 0.2124557217.
change: Increase each radius-2 axial-view weight to 0.01015625 and decrease each corner-view weight to 0.00234375, preserving the total ensemble weight and all training settings.
mechanism: Conservative outer-shift axial interpolation
evidence_used: Mild axial redistribution preserved 9,268 correct and improved cross-entropy over uniform weighting, while full corner-to-axial redistribution further improved cross-entropy but lost one prediction; a conservative interpolation tests the interval between them.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 58.25444416701794, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21245390014648438, "validation_score": 9267.412386813172}

RECENT RESULT
hypothesis: Batch size 160 will exceed 9,268 correct predictions by gaining more optimizer updates than batch 192 without incurring the accuracy loss observed at batch 128.
change: Restore the best mildly distance-stratified radius-2 TTA and reduce training batch size from 192 to 160; preserve the architecture, optimizer, loss, schedule, and calibration.
mechanism: Intermediate-minibatch update-density interpolation
evidence_used: Batch 192 achieved 9,268 correct, while batch 128 improved cross-entropy from 0.21246 to 0.20833 but lost five predictions; batch 160 tests the unmeasured interval between their optimization regimes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 160, "examples_processed": 100000, "optimizer_steps": 626, "parameters": 245150, "training_seconds": 58.77526895911433, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.21019465866088868, "validation_score": 9243.413156673947}

RECENT RESULT
hypothesis: Restoring the best mildly distance-stratified TTA and explicitly aligning original and flipped predictions will exceed 9,268 correct predictions by reducing view-specific errors that the increasingly dominant ensemble loss can otherwise conceal.
change: Restore the verified 0.009375/0.00625/0.003125 radius-2 weights and add a small Jensen–Shannon consistency penalty between original and horizontally flipped predictions.
mechanism: Flip-view Jensen–Shannon consistency regularization
evidence_used: Mild radius-2 weighting achieved the best 9,268-correct result, while stronger weighting lost one prediction. Architecture, EMA, probability-averaging, and batch-size changes all failed; the existing loss shifts to 95% ensemble supervision late in training, motivating direct agreement regularization without changing capacity or exposure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 38.29053920786828, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21291624221801758, "validation_score": 9256.412229618663}

RECENT RESULT
hypothesis: Restoring batch size 192 and shifting 0.02 ensemble weight from radius-1 diagonal views to the unshifted view will exceed 9,268 correct predictions by reducing reliance on more-displaced inputs while preserving total calibration weight.
change: Restore the best verified batch size, increase the unshifted TTA weight from 0.36 to 0.38, and reduce each radius-1 diagonal weight from 0.045 to 0.040.
mechanism: Center-biased inner-translation ensembling
evidence_used: Batch size 192 achieved 9,268 correct versus 9,263 at 128 and 9,243 at 160; mildly favoring less-displaced radius-2 views preserved all 9,268 predictions and improved cross-entropy, motivating the same conservative distance bias among inner views.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.78442587493919, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21246144828796387, "validation_score": 9265.41238424587}

RECENT RESULT
hypothesis: Adding a radius-2 contrast channel will exceed 9,268 correct predictions by exposing broader silhouette structure without reallocating capacity from the verified local/context aggregator.
change: Expand the input basis from five to six channels by adding replicate-padded 5×5 local contrast; adjust only the first convolution’s input width.
mechanism: Fixed multiscale contrast basis
evidence_used: Dilation-2 context improved correctness to 9,268, while further context-channel allocation failed, motivating a distinct coarse-scale signal rather than additional context capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245366, "training_seconds": 31.667365457862616, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.21205690307617187, "validation_score": 9250.412521886334}

RECENT RESULT
hypothesis: Replacing the absolute-position flattened MLP with lightweight convolutional relation modeling and class-specific learned pooling will exceed 9,268 correct predictions by letting each class select informative garment parts while retaining distributed silhouette statistics.
change: Preserve the verified multiscale tokenizer, add two efficient 4×4 spatial-relation blocks, and compute logits from class-specific evidence maps, learned attention/uniform mixtures, and global activation statistics. Retain the best verified distance-stratified TTA.
mechanism: Class-conditioned spatial evidence pooling
evidence_used: Fixed flattening remains the strongest design, but static residual and pooled fusion reached only 9,217 and 9,236; full class-query attention directly challenged this limitation but timed out. This patch tests content-dependent class pooling without quadratic attention, stays below 250,000 parameters, and uses the radius-2 weighting that preserved 9,268 correct while lowering cross-entropy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 234208, "training_seconds": 51.49365970911458, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.21255784683227538, "validation_score": 9227.412351461257}

RECENT RESULT
hypothesis: Adding a lightweight input-conditioned channel gate to the verified 40-local/24-context model will exceed 9,268 correct predictions by adapting its feature mixture to each garment while preserving the successful spatial classifier and starting from an exact identity transformation.
change: Restore the best mildly distance-stratified radius-2 TTA and add an identity-initialized squeeze-style gate before the multiscale aggregators, raising the parameter count from 245,150 to 249,118.
mechanism: Identity-initialized global channel recalibration
evidence_used: The 40-local/24-context flattened model achieved the best 9,268 correct predictions, whereas channel reallocation and replacement pooling heads regressed; this motivates retaining that topology while using the remaining parameter budget for conditional channel recalibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249094, "training_seconds": 49.76393491681665, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21290788040161132, "validation_score": 9257.412232460585}

RECENT RESULT
hypothesis: Zero-initializing the existing refinement branch’s final BatchNorm scale while restoring the best verified TTA weights will exceed 9,268 correct predictions by preserving the successful topology and preventing random residual perturbations during early optimization.
change: Restore Reference Design 1’s center and radius-1 diagonal TTA weights, and initialize the refinement residual as an exact identity without changing parameters or compute.
mechanism: Identity-initialized residual refinement
evidence_used: The 40-local/24-context model with 0.36/0.045 TTA achieved the best 9,268 correct; replacement heads and added channel gating regressed, motivating an optimization-focused change that retains the verified architecture.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 45.26361512509175, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21151177368164062, "validation_score": 9254.412707503849}

RECENT RESULT
hypothesis: Reversing the unsuccessful center bias by moving 0.02 ensemble weight from the unshifted view to radius-1 diagonal views will exceed 9,268 correct predictions.
change: Reduce the unshifted TTA weight from 0.36 to 0.34 and increase each radius-1 diagonal weight from 0.045 to 0.050, preserving total ensemble weight and all training settings.
mechanism: Diagonal-favoring inner-translation ensembling
evidence_used: Moving the same 0.02 weight in the opposite direction—from diagonal views to the center—reduced correctness from 9,268 to 9,265, indicating that radius-1 diagonal evidence is more valuable than additional center weight.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 32.745428583119065, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246579437255858, "validation_score": 9268.41238276768}

RECENT RESULT
hypothesis: Favoring horizontal over vertical radius-1 translations while retaining the best distance-stratified radius-2 weights will exceed 9,268 correct predictions by reducing harmful averaging across vertically informative garment positions.
change: Preserve total TTA weight, increase each horizontal radius-1 shift from 0.09 to 0.10, decrease each vertical shift to 0.08, and restore the radius-2 weighting that tied the best correct count with lower cross-entropy.
mechanism: Axis-aware translation ensembling
evidence_used: Distance-stratified radius-2 TTA preserved 9,268 correct and improved cross-entropy to 0.2124557217; center/diagonal reallocations did not improve correctness, leaving axial orientation as an informative untested dimension.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 48.1495108329691, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.2124831729888916, "validation_score": 9268.412376856964}

RECENT RESULT
hypothesis: Favoring vertical over horizontal radius-1 translations while restoring the best radius-2 weights will exceed 9,268 correct predictions because the opposite horizontal bias worsened cross-entropy.
change: Increase each vertical radius-1 weight to 0.10, decrease each horizontal weight to 0.08, and restore the verified 0.009375/0.00625/0.003125 radius-2 weighting while preserving total ensemble weight.
mechanism: Vertical-axis-aware translation ensembling
evidence_used: Symmetric axis weights with mildly distance-stratified radius-2 TTA achieved 9,268 correct at 0.2124557 cross-entropy; horizontal bias retained 9,268 but worsened cross-entropy to 0.2124832, motivating the opposite directional test.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.15950512513518, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21244136581420897, "validation_score": 9268.412391076467}



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
