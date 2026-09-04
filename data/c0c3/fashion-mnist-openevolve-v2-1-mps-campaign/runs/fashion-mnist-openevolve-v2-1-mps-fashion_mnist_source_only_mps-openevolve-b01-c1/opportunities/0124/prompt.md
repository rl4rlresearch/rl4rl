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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 68.66242979187518, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.2118162010192871, "validation_score": 9258.412603825216}
prior_hypothesis: Decaying label smoothing from 0.02 to 0.01 will exceed 9,257 correct predictions by gaining the late boundary sharpening seen with decay-to-zero while retaining enough regularization to avoid its three-prediction regression.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,257 correct predictions by retaining early regularization while allowing late training to sharpen class boundaries for exact top-1 accuracy.
change: Compute a progress-dependent smoothing coefficient and use it consistently in every individual, paired-view, and ensemble cross-entropy term.
mechanism: Cosine-decayed label smoothing
evidence_used: The 233,434-parameter architecture remains best while architectural, augmentation, pooling, averaging, and batch-size changes regressed; the loss schedule is comparatively unexplored, and exact correct count is prioritized over calibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 76.14576787501574, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2124201316833496, "validation_score": 9254.412398299017}

RECENT RESULT
hypothesis: Averaging normalized class probabilities across translated and flipped views will exceed 9,257 correct predictions by preventing an overconfident misaligned crop from dominating the ensemble.
change: Replace validation-time arithmetic logit averaging with probability averaging, then return the log-probability mixture with the existing calibration scale.
mechanism: Probability-space test-time augmentation pooling
evidence_used: Center-weighted crop aggregation improved the best design, showing that predictions vary meaningfully across geometric views; probability pooling directly targets that variation without changing the proven training procedure or parameter count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Aligning each translated and flipped prediction with their detached probability ensemble will exceed 9,257 correct predictions by reducing harmful view disagreement without changing the proven transformations or architecture.
change: Retain the existing supervised losses and add a cosine-ramped KL consistency loss between all six view predictions and their 0.9/0.1 full/central probability mixture.
mechanism: Ramped teacher-view consistency regularization
evidence_used: Center-weighted aggregation showed that predictions vary meaningfully across geometric views, while changing the crop distribution or adding rotation regressed; this targets disagreement among the existing successful views instead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Context-dependent channel reweighting will exceed 9,257 correct predictions by exploiting global garment context while preserving the proven coordinate-specific classifier.
change: Add a 960-parameter global-average channel gate after the residual stage, initialized to an exact identity so optimization begins from the current model.
mechanism: Identity-initialized squeeze-and-excitation channel gating
evidence_used: A global-average classification branch reached 9,253 correct, suggesting pooled context contains useful but insufficient standalone evidence; using it to modulate spatial features retains the stronger flattened head.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Mixing each training image with a randomly paired example using Beta(0.2, 0.2) weights will exceed 9,257 correct predictions by regularizing class boundaries without introducing the harmful geometric distortions observed with rotation.
change: Apply batch-level Mixup before generating the existing six views and replace each supervised cross-entropy term with the corresponding mixed-target loss while retaining fixed 0.02 label smoothing.
mechanism: Low-strength batch Mixup
evidence_used: Rotation regressed to 9,201 and evaluation-matched crop sampling reached only 9,255, while the unchanged architecture remains best; this motivates testing non-geometric vicinal augmentation while preserving the proven view pipeline and fixed smoothing.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Randomly masking a small 6×6 region in half of the training images will exceed 9,257 correct predictions by adding mild non-geometric occlusion robustness without the label ambiguity of the unverified Mixup attempt or the harmful distortions observed with rotation.
change: Apply independently positioned, mean-filled 6×6 Cutout masks to 50% of examples before generating the existing translated and flipped views.
mechanism: Per-example mean-filled Cutout
evidence_used: Geometric augmentation changes have plateaued or regressed, while Mixup could not be verified; Cutout tests the still-unresolved value of non-geometric regularization while preserving the proven architecture, labels, view pipeline, and compute profile.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 67.88836845802143, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.21568697471618653, "validation_score": 9227.411290085687}

RECENT RESULT
hypothesis: Replacing the coordinate-specific flattened head with learned classification over multiscale regional means and maxima will exceed 9,257 correct predictions by preserving coarse garment layout and salient local features while reducing sensitivity to exact downsampling phase.
change: Challenge the assumption that all 7×7 coordinates require independent weights: encode each channel with average and maximum statistics over 1×1, 2×2, and 4×4 spatial grids, then classify them through a wider 80-unit bottleneck. The model has 246,186 learned parameters.
mechanism: Dual-statistic spatial-pyramid classifier
evidence_used: The global-average branch reached 9,253 correct and covariance pooling reached 9,249, indicating that position-free pooled evidence is insufficient, while center-weighted crop evaluation showed that spatial alignment matters. A spatial pyramid retains coarse geometry that those alternatives discarded without retaining the current head’s rigid per-coordinate representation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Evaluating a recent-history exponential average of the unchanged model will exceed 9,257 correct predictions by reducing optimizer noise without weakening the proven training objective or architecture.
change: Track EMA copies of learned parameters and floating-point normalization buffers after every optimizer step, then install the averaged state after the final step.
mechanism: Bias-corrected exponential weight averaging
evidence_used: The 233,434-parameter design remains best, while architectural changes and decayed smoothing regressed; weight averaging preserves that design and its successful fixed-smoothing trajectory while targeting generalization variance directly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 58.436724208062515, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21514585876464845, "validation_score": 9255.411473237054}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to 0.01 will exceed 9,257 correct predictions by gaining the late boundary sharpening seen with decay-to-zero while retaining enough regularization to avoid its three-prediction regression.
change: Add a cosine smoothing schedule spanning 0.02 to 0.01 and apply it consistently to every individual, paired-view, and ensemble cross-entropy term.
mechanism: Cosine-decayed label smoothing with a nonzero floor
evidence_used: Decaying smoothing from 0.02 to zero reduced cross-entropy but scored 9,254 correct versus the current 9,257, indicating that late sharpening helps calibration but removing smoothing entirely is slightly too aggressive.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 68.66242979187518, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.2118162010192871, "validation_score": 9258.412603825216}

RECENT RESULT
hypothesis: Ending label smoothing at 0.013 will exceed 9,258 correct predictions by balancing the extra boundary sharpening of the 0.01 floor against the stronger late regularization of fixed 0.02 smoothing.
change: Preserve the 0.02 initial smoothing value while changing the cosine schedule’s final value from 0.01 to 0.013.
mechanism: Cosine-decayed label smoothing with an interpolated 0.013 floor
evidence_used: Decay to 0.01 improved from 9,257 to 9,258 correct, while decay to zero fell to 9,254; the results suggest an interior nonzero floor, and 0.013 probes between the best verified 0.01 endpoint and fixed 0.02 smoothing.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 63.5899620000273, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21196748390197753, "validation_score": 9258.41255232227}



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
