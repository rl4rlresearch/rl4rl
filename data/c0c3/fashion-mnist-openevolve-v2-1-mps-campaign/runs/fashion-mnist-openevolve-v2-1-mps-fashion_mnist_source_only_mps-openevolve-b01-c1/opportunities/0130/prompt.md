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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 54.930013791890815, "validation_accuracy": 0.9262, "validation_correct": 9262, "validation_cross_entropy": 0.21199396362304687, "validation_score": 9262.412543308801}
prior_hypothesis: Ending smoothing at 0.009 will exceed 9,259 correct predictions by retaining slightly more late regularization than the regressed 0.007 floor while remaining closer to the best 0.008 result than the 0.010 floor.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Ending smoothing at 0.008 will exceed 9,258 correct predictions by adding modest late boundary sharpening while retaining substantially more regularization than the zero-floor schedule that regressed to 9,254.
change: Lower the cosine schedule’s final smoothing from 0.01 to 0.008 while preserving its initial value of 0.02.
mechanism: Cosine-decayed label smoothing with a 0.008 floor
evidence_used: The 0.01 floor improved to 9,258 correct and beat the 0.013 floor on cross-entropy, whereas decay to zero lost four correct predictions; 0.008 probes the promising interval immediately below the current best without approaching complete smoothing removal.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 62.26420575007796, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.21177862854003907, "validation_score": 9259.412616618434}

RECENT RESULT
hypothesis: Ending smoothing at 0.007 will exceed 9,259 correct predictions by slightly sharpening late decision boundaries while remaining near the cross-entropy optimum implied by the 0.008, 0.010, and 0.013 results.
change: Lower only the cosine smoothing schedule’s final value from 0.008 to 0.007, preserving the initial value of 0.02 and all other training behavior.
mechanism: Cosine-decayed label smoothing with a locally optimized 0.007 floor
evidence_used: The 0.008 floor achieved the best result at 9,259 correct and lower cross-entropy than 0.010 and 0.013, while a zero floor regressed sharply; the observed cross-entropy trend places the most informative next probe just below 0.008 rather than near zero.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 57.636269292095676, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21188313484191895, "validation_score": 9256.412581036591}

RECENT RESULT
hypothesis: Ending smoothing at 0.009 will exceed 9,259 correct predictions by retaining slightly more late regularization than the regressed 0.007 floor while remaining closer to the best 0.008 result than the 0.010 floor.
change: Raise the cosine schedule’s final smoothing from 0.008 to 0.009 while preserving its initial value of 0.020.
mechanism: Cosine-decayed label smoothing with a 0.009 floor
evidence_used: The 0.008 floor achieved 9,259 correct, while 0.007 fell to 9,256 and 0.010 reached 9,258; 0.009 is the nearest untested value on the better-performing, more-regularized side of the current optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 54.930013791890815, "validation_accuracy": 0.9262, "validation_correct": 9262, "validation_cross_entropy": 0.21199396362304687, "validation_score": 9262.412543308801}

RECENT RESULT
hypothesis: Ending smoothing at 0.0085 will exceed 9,262 correct predictions by preserving the stronger classification behavior of the 0.009 endpoint while moving toward the lower cross-entropy observed at 0.008.
change: Change the cosine label-smoothing schedule from 0.020→0.009 to 0.020→0.0085, leaving all other training behavior unchanged.
mechanism: Half-step refinement of the label-smoothing endpoint
evidence_used: The 0.009 endpoint achieved the best result with 9,262 correct, while 0.008 was the better adjacent endpoint than 0.010 in both correct predictions and cross-entropy; 0.0085 is the most informative untested refinement on that side.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 70.50187616702169, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21185400238037108, "validation_score": 9257.412590954866}

RECENT RESULT
hypothesis: Ending smoothing at 0.0095 will exceed 9,262 correct predictions by staying near the best 0.009 endpoint while adding less late regularization than the regressed 0.010 endpoint.
change: Change the cosine label-smoothing schedule from 0.020→0.009 to 0.020→0.0095, preserving its initial value and all other training behavior.
mechanism: Half-step refinement on the regularized side of the smoothing optimum
evidence_used: The 0.009 endpoint achieved the best result at 9,262 correct, while 0.010 achieved 9,258 and the tested 0.0085 half-step achieved 9,257; 0.0095 is the remaining untested half-step directly adjacent to the current optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 67.96725658304058, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2118368278503418, "validation_score": 9256.412596802234}

RECENT RESULT
hypothesis: Widening the proven flattened classifier bottleneck from 64 to 70 units will exceed 9,262 correct predictions by using the remaining parameter budget to learn additional spatially specific garment features.
change: Increase the classifier bottleneck width to 70, bringing the model to 249,976 learned parameters while preserving all training and evaluation behavior.
mechanism: Capacity-maximized coordinate-specific classifier head
evidence_used: The 233,434-parameter coordinate-specific design remains strongest, while position-free pooled alternatives reached only 9,253 and 9,249 correct; conservatively expanding the successful head is therefore more motivated than replacing its spatial representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249976, "training_seconds": 53.16531320801005, "validation_accuracy": 0.9221, "validation_correct": 9221, "validation_cross_entropy": 0.21635280227661133, "validation_score": 9221.411064946835}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
