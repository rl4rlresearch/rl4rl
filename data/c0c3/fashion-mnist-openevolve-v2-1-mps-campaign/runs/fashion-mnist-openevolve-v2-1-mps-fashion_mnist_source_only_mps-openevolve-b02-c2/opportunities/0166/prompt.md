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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.77373074996285, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19214456176757813, "validation_score": 9328.419412222338}
prior_hypothesis: Restoring constant 0.02 label smoothing will recover 9,328 correct predictions, while sharpening fused validation logits from temperature 0.90 to 0.85 will preserve those argmax predictions and reduce cross-entropy below 0.194148.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.56534891715273, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922137680053711, "validation_score": 9320.419387876082}
prior_hypothesis: Mildly favoring the more decisive horizontal view will finish within the reliable hard-maximum runtime and exceed 9,322 correct predictions by resolving flip disagreements more accurately.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.40186375007033, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914721824645996, "validation_score": 9328.419648907762}
prior_hypothesis: Restoring the verified hard-maximum, constant-0.02-smoothing design will recover 9,328 correct predictions, while temperature 0.80 will preserve its argmax predictions and reduce validation cross-entropy below 0.192145.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 81.10383258294314, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19414760208129883, "validation_score": 9328.41870870831}
prior_hypothesis: Sharpening the fused validation logits with temperature 0.90 will preserve all 9,328 argmax predictions while lowering cross-entropy below 0.201899, producing a strict validation-score improvement.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging maxima from two interleaved checkerboard lattices will finish within the reliable runtime envelope and exceed 9,322 correct predictions by providing two robust salient gradients without `topk` or four separate lattice reductions.
change: Replace single hard-max channel evidence with the mean maxima of the two flattened checkerboard parities, and fuse the shared average/salient attention computation into one convolution call.
mechanism: Two-color checkerboard peak attention
evidence_used: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard-max attention reliably finished near 9,320; the four-lattice approximation also timed out, motivating a cheaper two-lattice compromise.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.32541258307174, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.19233039932250975, "validation_score": 9309.419346852419}

RECENT RESULT
hypothesis: Training each paired flip view under a deterministic 3×3 translation cycle will exceed 9,322 correct predictions while remaining within the hard-maximum model’s reliable runtime envelope.
change: Add zero-padded one-pixel translations before the existing random flip and paired-view supervision, covering all nine offsets without extra model parameters.
mechanism: Cyclic one-pixel translation augmentation
evidence_used: Hard-maximum attention reliably reaches 9,320, while numerous saliency replacements regressed or timed out; the current training path augments only horizontal orientation, making low-cost translation invariance the most informative untested axis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.76328500011005, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.2097536651611328, "validation_score": 9232.41330728263}

RECENT RESULT
hypothesis: Applying 0.02 label smoothing to the reliable paired-view objective will exceed 9,322 correct predictions by reducing brittle class-boundary fitting, while the verified 0.10 confidence-adaptive fusion improves the cross-entropy tie-breaker.
change: Add mild label smoothing without forcing flip agreement, and replace mean-logit evaluation with the best verified confidence-adaptive probability ensemble.
mechanism: Mild per-view target smoothing with confidence-adaptive flip fusion
evidence_used: Hard-max attention reliably reached 9,320 while attention approximations, translation augmentation, and paired-mixture supervision regressed or timed out; coefficient-0.10 probability fusion retained all 9,320 correct and improved cross-entropy from 0.192650 to 0.192214.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 70.16063670790754, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.2018985595703125, "validation_score": 9328.416008485923}

RECENT RESULT
hypothesis: Holding 0.02 smoothing through the first half of training and annealing it to zero will exceed 9,328 correct predictions by retaining the verified regularization benefit while allowing late hard-target boundary refinement; if correct counts tie, it will lower cross-entropy.
change: Replace constant 0.02 label smoothing with a schedule that remains at 0.02 for half the optimizer trajectory, then cosine-decays to zero.
mechanism: Late cosine decay of label smoothing
evidence_used: Fixed 0.02 smoothing improved the reliable hard-max design from 9,320 to 9,328 correct, but increased cross-entropy from about 0.1922 to 0.2019, motivating preservation of early smoothing with less late confidence suppression.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.50203983299434, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.18954639472961426, "validation_score": 9325.420328288343}

RECENT RESULT
hypothesis: Holding label smoothing at 0.025 throughout training will exceed 9,328 correct predictions by strengthening the regularization benefit observed at 0.02 without the late hard-target transition that reduced accuracy.
change: Restore the runtime-reliable hard-maximum channel descriptor and 0.10 confidence-adaptive flip fusion, then train paired views with constant 0.025 label smoothing.
mechanism: Slightly stronger constant label smoothing with reliable hard-max attention
evidence_used: Constant 0.02 smoothing improved hard-max attention from 9,320 to 9,328 correct, while annealing smoothing toward zero reduced the result to 9,325; this motivates a small upward search around the best verified constant setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 79.86398470797576, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.20622329635620118, "validation_score": 9307.414516948487}

RECENT RESULT
hypothesis: Constant 0.018 label smoothing will exceed 9,328 correct predictions by retaining the regularization benefit of 0.02 while avoiding the over-regularization observed at 0.025.
change: Train both paired flip views with constant 0.018 label smoothing, leaving the reliable architecture, optimizer, schedule, and confidence-adaptive evaluation fusion unchanged.
mechanism: Slightly reduced constant label smoothing
evidence_used: Constant 0.02 smoothing improved the hard-max baseline from 9,320 to 9,328 correct, while increasing it to 0.025 fell to 9,307 and decaying it toward zero reached 9,325; this motivates a narrow search immediately below the best verified constant value.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.47984649986029, "validation_accuracy": 0.9313, "validation_correct": 9313, "validation_cross_entropy": 0.20002394332885742, "validation_score": 9313.416658353177}

RECENT RESULT
hypothesis: Holding the verified 0.02 smoothing for 75% of training and tapering only to 0.01 will retain or exceed 9,328 correct predictions while improving cross-entropy over constant smoothing.
change: Use confidence-adaptive probability fusion and cosine-taper label smoothing from 0.02 to 0.01 during the final quarter.
mechanism: Conservative terminal label-smoothing taper
evidence_used: Constant 0.02 smoothing achieved 9,328 correct, while an earlier, complete decay to zero retained 9,325 and substantially lowered cross-entropy; a later partial taper targets that calibration benefit without discarding most of the accuracy-producing regularization.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.514016791014, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19503734855651855, "validation_score": 9326.418396965253}

RECENT RESULT
hypothesis: Sharpening the fused validation logits with temperature 0.90 will preserve all 9,328 argmax predictions while lowering cross-entropy below 0.201899, producing a strict validation-score improvement.
change: Divide the confidence-adaptive fused log-probabilities by 0.90 during evaluation; training remains unchanged.
mechanism: Evaluation-only temperature sharpening
evidence_used: Constant 0.02 label smoothing achieved the best count, 9,328, but raised cross-entropy to 0.201899 versus roughly 0.1922 without smoothing; late smoothing decay lowered cross-entropy to 0.189546. This indicates underconfidence that post-hoc sharpening can address without changing predicted classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 81.10383258294314, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19414760208129883, "validation_score": 9328.41870870831}

RECENT RESULT
hypothesis: Restoring constant 0.02 label smoothing will recover 9,328 correct predictions, while sharpening fused validation logits from temperature 0.90 to 0.85 will preserve those argmax predictions and reduce cross-entropy below 0.194148.
change: Restore the best-count constant smoothing regime and apply temperature 0.85 to the existing confidence-adaptive flip ensemble.
mechanism: Stronger evaluation-only temperature sharpening
evidence_used: Constant 0.02 smoothing produced 9,328 correct, and temperature 0.90 preserved all 9,328 while lowering cross-entropy from 0.201899 to 0.194148; a modest additional sharpening directly tests whether calibration can improve further without affecting predicted classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.77373074996285, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19214456176757813, "validation_score": 9328.419412222338}

RECENT RESULT
hypothesis: Restoring the verified hard-maximum, constant-0.02-smoothing design will recover 9,328 correct predictions, while temperature 0.80 will preserve its argmax predictions and reduce validation cross-entropy below 0.192145.
change: Restore hard-maximum channel attention, confidence-adaptive flip fusion, constant 0.02 label smoothing, and sharpen fused evaluation logits from the best verified temperature of 0.85 to 0.80.
mechanism: Further evaluation-only temperature sharpening
evidence_used: Temperature 0.85 preserved all 9,328 correct predictions and improved cross-entropy from 0.194148 at temperature 0.90 to 0.192145, motivating one further step in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.40186375007033, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914721824645996, "validation_score": 9328.419648907762}

RECENT RESULT
hypothesis: Restoring constant 0.02 label smoothing and temperature-0.80 evaluation will recover 9,328 correct predictions and validation cross-entropy near 0.19147.
change: Train paired flip views with 0.02 label smoothing and divide confidence-adaptive fused log-probabilities by 0.80 during evaluation.
mechanism: Constant label smoothing with calibrated flip-ensemble sharpening
evidence_used: Reference Design 2 achieved the best available score, preserving 9,328 correct predictions while lowering cross-entropy to 0.191472; the current unsmoothed design reached only 9,320 correct.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 83.27676204103045, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914721824645996, "validation_score": 9328.419648907762}

RECENT RESULT
hypothesis: Temperature 0.805 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472, as the improvements from 0.90 to 0.85 to 0.80 place the estimated calibration optimum near 0.805.
change: Slightly relax evaluation-only sharpening from temperature 0.80 to 0.805; training remains unchanged.
mechanism: Interpolated evaluation-temperature calibration
evidence_used: Temperatures 0.90, 0.85, and 0.80 preserved exactly 9,328 correct while cross-entropy decreased from 0.194148 to 0.192145 to 0.191472; quadratic interpolation in inverse temperature estimates a minimum just above 0.80.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.01654874999076, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914772232055664, "validation_score": 9328.419647132368}



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
