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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 51.062351166969165, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246516418457032, "validation_score": 9268.412382982018}
prior_hypothesis: Reallocating eight aggregator channels from the local branch to the successful dilation-2 context branch will exceed 9,265 correct predictions by strengthening broader garment-shape modeling while retaining 40 channels for local detail and keeping parameter count unchanged.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 35.00231925002299, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21682710800170898, "validation_score": 9240.410904718272}
prior_hypothesis: Scaling evaluation logits by 1.285 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216927 by compensating for the greater confidence induced by the new 0.04→0 smoothing schedule.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 47.394402750069275, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21134602966308594, "validation_score": 9255.412763973098}
prior_hypothesis: Replacing absolute 7×7 flattening with a shared learned 3×3 stride-2 aggregation and a wider nonlinear head will exceed 9,240 correct predictions by encoding local garment-part relationships while retaining coarse 4×4 layout.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 37.35932166711427, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21670380477905274, "validation_score": 9240.410946360187}
prior_hypothesis: Scaling evaluation logits by 1.253 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.2167039406.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.285 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216927 by compensating for the greater confidence induced by the new 0.04→0 smoothing schedule.
change: Reduce only the evaluation-logit multiplier from 1.296352 to 1.285.
mechanism: Downward logit-temperature recalibration
evidence_used: The 1.296352 optimum was measured with the reference 0.05→0.01 smoothing schedule, whereas the current 0.04→0 schedule lowers smoothing by 0.01 throughout training and improved cross-entropy to 0.216927 without changing correctness; a modest downward recalibration tests that confidence shift while mathematically preserving every predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 35.00231925002299, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21682710800170898, "validation_score": 9240.410904718272}

RECENT RESULT
hypothesis: Averaging per-shift class probabilities instead of logits will exceed 9,240 correct predictions by limiting the influence of overconfident errors from individual translated views.
change: Adopt the best verified 0.04→0 smoothing schedule and 1.285 calibration, while changing only the 25-shift aggregation from a weighted logit average to a weighted posterior mixture; flip pairs remain logit-averaged.
mechanism: Posterior-mixture shift ensembling
evidence_used: Reference Design 1 achieved the best verified score with 9,240 correct using 0.04→0 smoothing and 1.285 calibration. Added spatial refinement and phase-stable pooling reduced correctness, while the design’s extensive shift evaluation indicates residual view sensitivity, motivating a robust aggregation change without altering learned capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 39.350304041057825, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21693955612182617, "validation_score": 9237.410866749697}

RECENT RESULT
hypothesis: With the verified 0.04→0 smoothing schedule, scaling evaluation logits by 1.255 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216827.
change: Restore the best verified smoothing schedule and lower its evaluation-logit multiplier from 1.285 to the estimated calibration optimum of 1.255.
mechanism: Extrapolated logit-temperature recalibration
evidence_used: Reference Design 1 improved cross-entropy from approximately 0.216927 at scale 1.296352 to 0.216827 at 1.285 without changing correctness; the curvature observed across the original schedule’s calibrated scales indicates that this improvement extrapolates to an optimum near 1.255.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 35.480292916996405, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21670428619384766, "validation_score": 9240.410946197588}

RECENT RESULT
hypothesis: Restoring the verified 0.04→0 smoothing schedule and scaling evaluation logits by 1.252 will preserve all 9,240 predictions while reducing validation cross-entropy below 0.2167043.
change: Restore the best verified smoothing schedule and move the evaluation-logit multiplier from 1.30 to the estimated local optimum of 1.252.
mechanism: Local quadratic logit-temperature calibration
evidence_used: With identical 0.04→0 training, reducing the multiplier from 1.285 to 1.255 preserved 9,240 correct and lowered cross-entropy from 0.2168271 to 0.2167043; together with the approximately 0.216927 result at 1.296352, these measurements place the local calibration optimum near 1.252.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.509501916822046, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21670394058227538, "validation_score": 9240.41094631432}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.253 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.2167039406.
change: Increase only the evaluation-logit multiplier from 1.252 to 1.253.
mechanism: Three-point quadratic logit-temperature calibration
evidence_used: With identical 0.04→0 smoothing, scales 1.285, 1.255, and 1.252 produced cross-entropies 0.2168271080, 0.2167042862, and 0.2167039406 while preserving 9,240 correct; their local quadratic fit places the minimum near 1.253.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 37.35932166711427, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21670380477905274, "validation_score": 9240.410946360187}

RECENT RESULT
hypothesis: Replacing absolute 7×7 flattening with a shared learned 3×3 stride-2 aggregation and a wider nonlinear head will exceed 9,240 correct predictions by encoding local garment-part relationships while retaining coarse 4×4 layout.
change: Challenge the assumption that class prediction should independently weight every 7×7 location. Add a learned late spatial aggregator, widen the classifier bottleneck from 48 to 140, and retain the best verified smoothing and 1.253 calibration; the resulting model has 245,150 learned parameters.
mechanism: Learned late spatial consolidation
evidence_used: A second same-grid refinement block fell to 9,232, while fixed overlapping early pooling fell to 9,208. This motivates reallocating dense-head parameters to learned late downsampling that preserves the successful early feature extractor instead of adding same-resolution processing or destroying high-resolution evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 47.394402750069275, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21134602966308594, "validation_score": 9255.412763973098}

RECENT RESULT
hypothesis: Combining learned stride-2 aggregation with a parameter-free averaged identity path will exceed 9,255 correct predictions by preserving coarse channel evidence that the best verified single-path aggregator may discard.
change: Replace the flat 7×7 classifier with a 48-channel learned late aggregator concatenated with 80-channel 3×3 average-pooled features, use a 76-unit nonlinear head, and adopt the best architecture’s verified 1.253 calibration; the model has 245,054 learned parameters.
mechanism: Dual-path late spatial consolidation
evidence_used: Learned late spatial consolidation improved correctness from 9,240 to 9,255, whereas replacing early pooling reduced it to 9,208. This patch retains the successful learned late downsampling and adds pooling only as a complementary identity path.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245054, "training_seconds": 59.593185500008985, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21982876472473145, "validation_score": 9236.40989359692}

RECENT RESULT
hypothesis: Splitting the successful 64-channel late aggregator into 48 local and 16 dilated-context channels will exceed 9,255 correct predictions by combining garment-part detail with broader spatial relationships without reducing head width or increasing parameters.
change: Adopt the best verified architecture, smoothing, and calibration, but replace its single 3×3 stride-2 aggregator with parallel local and dilation-2 learned branches whose concatenated output retains 64 channels and 245,150 parameters.
mechanism: Multiscale learned late spatial consolidation
evidence_used: Learned late spatial consolidation reached 9,255 correct, while another same-grid refinement reached only 9,232 and the pooled dual-path design reached 9,236; this motivates adding broader context specifically within the proven learned consolidation stage while preserving its classifier capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 43.63737245905213, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21112456283569336, "validation_score": 9265.412839451319}

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
