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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 49.11529295798391, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21245572166442872, "validation_score": 9268.412386193628}
prior_hypothesis: Restoring the verified 40-local/24-context architecture and redistributing the unchanged 0.10 radius-2 TTA weight toward less-displaced axial views will exceed 9,268 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245150, "training_seconds": 59.16914387512952, "validation_accuracy": 0.9263, "validation_correct": 9263, "validation_cross_entropy": 0.20832988815307618, "validation_score": 9263.413794283251}
prior_hypothesis: Restoring the verified 40-local/24-context model while reducing the batch size to 128 will exceed 9,268 correct predictions because the fixed 100,000-example exposure will provide roughly 782 optimizer steps instead of 522.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 51.062351166969165, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246516418457032, "validation_score": 9268.412382982018}
prior_hypothesis: Reallocating eight aggregator channels from the local branch to the successful dilation-2 context branch will exceed 9,265 correct predictions by strengthening broader garment-shape modeling while retaining 40 channels for local detail and keeping parameter count unchanged.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 43.22874066699296, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21244972343444823, "validation_score": 9267.412388233784}
prior_hypothesis: Restoring the verified 40-local/24-context model and reallocating radius-2 corner weight to axial shifts will exceed 9,268 correct predictions while preserving total ensemble weight.

## Recent verification evidence

RECENT RESULT
hypothesis: Reallocating eight aggregator channels from the local branch to the successful dilation-2 context branch will exceed 9,265 correct predictions by strengthening broader garment-shape modeling while retaining 40 channels for local detail and keeping parameter count unchanged.
change: Change the late aggregator split from 48 local and 16 dilated-context channels to 40 local and 24 dilated-context channels; preserve the 64-channel output, classifier width, training procedure, TTA, and calibration.
mechanism: Increased dilated-context channel allocation
evidence_used: Replacing the 64-channel local aggregator, which achieved 9,255 correct, with a 48-local/16-context split improved correctness to 9,265 at the identical 245,150-parameter count. This motivates a controlled dose increase of the only architectural addition shown to improve the current best design.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 51.062351166969165, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246516418457032, "validation_score": 9268.412382982018}

RECENT RESULT
hypothesis: Increasing the dilated-context allocation from 24 to 32 channels will exceed 9,268 correct predictions by continuing the verified context-capacity trend while retaining 32 local-detail channels.
change: Replace the single 64-channel aggregator with parallel 32-channel local and dilation-2 branches, concatenate their outputs, and preserve parameter count and all training settings.
mechanism: Balanced local-context late consolidation
evidence_used: Local-only aggregation achieved 9,255 correct, 48-local/16-context achieved 9,265, and 40-local/24-context achieved 9,268 at identical parameter counts; another eight-channel reallocation directly tests whether that dose-response continues.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 37.58013370796107, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2157062843322754, "validation_score": 9243.41128355298}

RECENT RESULT
hypothesis: A 44-local/20-dilated channel split will exceed 9,268 correct predictions by retaining more local detail than the failed 32/32 split while adding more context than the successful 48/16 split.
change: Replace the flat classifier with parallel 44-channel local and 20-channel dilation-2 aggregators feeding the verified 140-unit head; preserve all training, augmentation, TTA, and calibration settings.
mechanism: Interpolated local-context late consolidation
evidence_used: Correctness rose from 9,255 with local-only aggregation to 9,265 at 48/16 and 9,268 at 40/24, then fell sharply to 9,243 at 32/32. Testing 44/20 targets the unmeasured interval around the apparent optimum while keeping 245,150 parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 48.071898540947586, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.21176815185546874, "validation_score": 9229.412620185829}

RECENT RESULT
hypothesis: Adding a learned residual 1×1 fusion after the verified 40-local/24-dilated aggregation will exceed 9,268 correct predictions by enabling nonlinear interaction between complementary local-detail and garment-shape channels before spatial classification.
change: Adopt the best 40/24 multiscale aggregator and 140-unit head, add a residual pointwise fusion block, and retain the verified training procedure and 1.253 evaluation calibration; the model has 249,374 learned parameters.
mechanism: Residual cross-scale channel fusion
evidence_used: The 40-local/24-context design achieved the best result at 9,268 correct, improving over both local-only aggregation (9,255) and 48/16 aggregation (9,265). This motivates preserving that allocation while using the remaining parameter budget to fuse the two proven feature scales.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249374, "training_seconds": 40.81991412490606, "validation_accuracy": 0.9217, "validation_correct": 9217, "validation_cross_entropy": 0.216298983001709, "validation_score": 9217.411083135798}

RECENT RESULT
hypothesis: Replicate-padding the dilation-2 context branch will exceed 9,268 correct predictions by eliminating artificial zero-valued boundaries from its wide receptive field while preserving the verified 40-local/24-context allocation.
change: Change only the context aggregator’s padding mode from zero padding to replicate padding; parameter count and all other architecture, training, augmentation, TTA, and calibration settings remain unchanged.
mechanism: Boundary-consistent dilated context aggregation
evidence_used: Adding dilation-2 context improved correctness from 9,255 to 9,265 and then 9,268, while increasing context capacity further or adding fusion reduced accuracy. This motivates improving the successful context branch itself; replicate padding also matches the translation padding already used during training and evaluation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 65.67841108399443, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2110233612060547, "validation_score": 9252.412873951087}

RECENT RESULT
hypothesis: Restoring the verified 40-local/24-context architecture and redistributing the unchanged 0.10 radius-2 TTA weight toward less-displaced axial views will exceed 9,268 correct predictions.
change: Adopt the best verified multiscale aggregator, then weight radius-2 axial, knight, and corner shifts by 0.009375, 0.00625, and 0.003125 respectively while preserving total ensemble weight and calibration.
mechanism: Distance-stratified outer-shift logit ensembling
evidence_used: The 40/24 aggregator achieved the best result at 9,268 correct, while further architectural changes failed. Radius-1 evaluation already downweights diagonal shifts, but radius-2 currently assigns equal weight despite different displacement magnitudes, motivating this controlled evaluation-only refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 49.11529295798391, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21245572166442872, "validation_score": 9268.412386193628}

RECENT RESULT
hypothesis: Restoring the verified 40-local/24-context model and reallocating radius-2 corner weight to axial shifts will exceed 9,268 correct predictions while preserving total ensemble weight.
change: Adopt the best multiscale aggregator and strengthen its successful TTA distance bias: axial, knight, and corner radius-2 shifts receive 0.0125, 0.00625, and 0 respectively.
mechanism: Stronger distance-stratified outer-shift ensembling
evidence_used: Distance-stratified radius-2 TTA preserved 9,268 correct and reduced cross-entropy from 0.2124652 to 0.2124557; this controlled dose increase further favors the least-displaced outer views.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 43.22874066699296, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21244972343444823, "validation_score": 9267.412388233784}

RECENT RESULT
hypothesis: Replacing the fixed flattened MLP with two-layer self-attention over the verified 40-local/24-context tokens and ten learned class queries will exceed 9,268 correct predictions by modeling content-dependent relationships between garment parts.
change: Restore the best multiscale aggregator, convert its 4×4 output into position-aware tokens, and classify through ten learned class queries in two pre-norm attention blocks; retain the best verified TTA weighting and 1.253 calibration. The resulting model remains below 250,000 parameters.
mechanism: Class-query spatial relation pooling
evidence_used: The 40/24 learned multiscale aggregator reached 9,268 correct, while static residual channel fusion fell to 9,217 and pooled identity fusion fell to 9,236. This challenges the load-bearing assumption that fixed flattening or pointwise fusion is sufficient, while retaining the strongest learned spatial tokenizer.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Matching radius-2 training probabilities to the best verified distance-stratified TTA weights will exceed 9,268 correct predictions while preserving the total exposure to outer translations.
change: Reallocate the unchanged 0.10 radius-2 training probability from corner shifts to axial shifts, using the current evaluation ensemble’s 0.009375/0.00625/0.003125 axial/knight/corner weights.
mechanism: Train–evaluation translation-distribution alignment
evidence_used: Distance-stratified evaluation preserved 9,268 correct while reducing cross-entropy from 0.2124652 to 0.2124557 versus uniform outer-shift weights; the stronger redistribution lost one correct prediction, motivating alignment with the milder best-verified distribution.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 39.630209082970396, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2121722682952881, "validation_score": 9267.412482625678}

RECENT RESULT
hypothesis: Applying a 0.98 EMA over the final 35% of optimization will exceed 9,268 correct predictions by reducing endpoint noise while retaining the verified 40-local/24-context representation.
change: Preserve the best architecture, restore its mildly distance-stratified radius-2 TTA, and evaluate an EMA of parameters and BatchNorm state accumulated during the low-learning-rate training tail.
mechanism: Tail-trajectory exponential weight averaging
evidence_used: The 40/24 model produced the best 9,268-correct result, while capacity, fusion, padding, and augmentation changes lost accuracy; the mild TTA redistribution preserved all 9,268 predictions and improved cross-entropy, motivating an optimization-stability change on that baseline.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 42.428608417045325, "validation_accuracy": 0.9263, "validation_correct": 9263, "validation_cross_entropy": 0.21447130813598633, "validation_score": 9263.411701780562}

RECENT RESULT
hypothesis: Averaging translation predictions in probability space with the best mild distance-stratified weights will exceed 9,268 correct predictions by limiting the influence of overconfident errors from displaced views.
change: Restore the best radius-2 weights and replace translation-logit averaging with calibrated probability averaging; training, flip-logit averaging, architecture, and parameter count remain unchanged.
mechanism: Probability-space translation ensembling
evidence_used: Mild distance-stratified TTA retained 9,268 correct and improved cross-entropy, while stronger axial weighting lost one prediction. This suggests retaining all translated evidence but making aggregation more robust to individual harmful views.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 48.89620795799419, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.21243431663513185, "validation_score": 9259.412393474136}

RECENT RESULT
hypothesis: Restoring the verified 40-local/24-context model while reducing the batch size to 128 will exceed 9,268 correct predictions because the fixed 100,000-example exposure will provide roughly 782 optimizer steps instead of 522.
change: Restore the best multiscale architecture, mildly distance-stratified TTA, and 1.253 calibration, then reduce only the training batch size from 192 to 128.
mechanism: Higher-update small-minibatch optimization
evidence_used: The 40-local/24-context design reached 9,268 correct, while subsequent architecture, TTA-aggregation, augmentation-alignment, and EMA changes did not improve it; every reported design used batch size 192, leaving optimizer-update density as an untested lever.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245150, "training_seconds": 59.16914387512952, "validation_accuracy": 0.9263, "validation_correct": 9263, "validation_cross_entropy": 0.20832988815307618, "validation_score": 9263.413794283251}



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
