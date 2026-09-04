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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.59234541701153, "validation_accuracy": 0.9346, "validation_correct": 9346, "validation_cross_entropy": 0.1877129035949707, "validation_score": 9346.420977155747}
prior_hypothesis: Favoring vertical shifts will exceed 9,344 correct predictions because the equal-and-opposite horizontal-favoring perturbation reduced correctness to 9,339.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.23315116693266, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877075828552246, "validation_score": 9348.420979041657}
prior_hypothesis: Increasing the vertical-shift preference slightly beyond the best 1.25/0.75 weighting will exceed 9,346 correct predictions or retain 9,346 while lowering cross-entropy below 0.18771290.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 71.9894648338668, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18783146209716797, "validation_score": 9344.420935137648}
prior_hypothesis: A 0.9825-decay full-state EMA will exceed 9,344 correct predictions by interpolating between the near-best 0.985 horizon and the best 0.98 horizon.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.50179904093966, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18781963577270508, "validation_score": 9344.420939328616}
prior_hypothesis: A 0.98-decay full-state EMA will exceed 9,343 correct predictions by tracking the final low-learning-rate solution more closely than the successful 0.985 EMA while retaining its averaging of BatchNorm statistics.

## Recent verification evidence

RECENT RESULT
hypothesis: A centered-crop weight of 2.875 with the verified 0.98 EMA will retain 9,344 correct predictions while lowering cross-entropy below 0.18781964.
change: Restore the best-observed EMA decay and test the midpoint between centered-crop weights 2.75 and 3.0.
mechanism: Near-center translation-ensemble boundary refinement
evidence_used: With decay 0.98, weight 3.0 achieved 9,344 correct, while weight 2.75 lowered cross-entropy but lost two predictions; 2.875 directly refines that unresolved accuracy-calibration boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 74.87305241706781, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18780846786499023, "validation_score": 9343.420943286335}

RECENT RESULT
hypothesis: A centered-crop weight of 2.9375 will retain 9,344 correct predictions while lowering cross-entropy below the weight-3.0 result.
change: Move the centered TTA crop weight halfway between the 2.875 and 3.0 settings, retaining the verified 0.98 EMA.
mechanism: Near-optimal center-weight boundary refinement
evidence_used: Weight 3.0 achieved 9,344 correct, while weight 2.875 achieved 9,343 with lower cross-entropy; their untested midpoint directly refines the accuracy boundary and can improve the tie-breaker without changing training.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.2068837499246, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18781390228271486, "validation_score": 9343.42094136046}

RECENT RESULT
hypothesis: A spatially aligned shallow-feature skip will exceed 9,344 correct predictions by preserving fine boundaries and textures lost in the deep-only representation while providing a shorter supervised path.
change: Replace the assumption that the final 64×7×7 tensor alone is sufficient with a 96-channel hypercolumn combining deep features and a learned max/average summary of 14×14 shallow features; resize the dense bottleneck to remain below 250,000 parameters and restore the best-observed TTA center weight.
mechanism: Adaptive multiscale hypercolumn classification
evidence_used: Global pooling fell to 9,085 correct and post-pooling attention reached only 9,300, while the spatially organized baseline reached 9,344; the four-block spatial alternative timed out, motivating a computationally cheap multiscale path that preserves position without adding deep residual computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248136, "training_seconds": 65.88003662507981, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.19143194274902345, "validation_score": 9309.419663081087}

RECENT RESULT
hypothesis: Weighting horizontal shifts more than vertical shifts will exceed 9,344 correct predictions because paired horizontal-flip supervision makes horizontal views more reliable, while the large regressions from removing spatial organization show that vertical alignment remains class-relevant.
change: Preserve the proven center weight, EMA, and total TTA weight while reallocating 0.25 weight from each vertical crop to each horizontal crop.
mechanism: Orientation-aware translation ensembling
evidence_used: The weight-3 symmetric ensemble achieved the best 9,344-correct result, whereas changing its overall center/shift balance regressed; this motivates holding that balance fixed and testing orientation-specific reliability instead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 74.7513949170243, "validation_accuracy": 0.9339, "validation_correct": 9339, "validation_cross_entropy": 0.18806511077880858, "validation_score": 9339.420852355199}

RECENT RESULT
hypothesis: Restoring EMA decay 0.98 will retain 9,344 correct predictions while lowering validation cross-entropy below the current 0.18783146.
change: Change only the full-state EMA decay from 0.9825 to 0.98.
mechanism: Best-observed full-state EMA horizon
evidence_used: Reference Design 3 used decay 0.98 and achieved 9,344 correct with lower cross-entropy of 0.18781964, strictly outperforming the current equal-correct design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 0.75-order power mean with the verified center weight of 3.0 will exceed 9,344 correct predictions by suppressing classes supported by only a few augmented views while preserving the proven crop balance.
change: Restore the best-observed center weight and replace arithmetic probability averaging with a conservative power mean between arithmetic and geometric aggregation.
mechanism: Consensus-favoring power-mean test-time ensemble
evidence_used: The weight-3 arithmetic ensemble achieved the best observed 9,344 correct, while crop-weight changes moved correctness as low as 9,338; this shows augmented-view aggregation affects boundary decisions and motivates testing consensus sensitivity without changing training.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.22029741713777, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18791718940734864, "validation_score": 9343.420904760414}

RECENT RESULT
hypothesis: A 1.25-order power mean with the verified center weight of 3.0 will exceed 9,344 correct predictions by reversing the harmful consensus bias observed with the 0.75-order mean and giving stronger augmented-view evidence modestly more influence.
change: Restore the best-observed center crop weight and replace arithmetic probability averaging with a conservative 1.25-order power mean.
mechanism: Confidence-favoring power-mean test-time ensemble
evidence_used: The weight-3 arithmetic ensemble achieved 9,344 correct, while shifting toward consensus with a 0.75-order power mean reduced correctness to 9,343 and worsened cross-entropy; testing the opposite side of arithmetic is the most direct directional follow-up.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.75660904101096, "validation_accuracy": 0.9342, "validation_correct": 9342, "validation_cross_entropy": 0.18780102005004884, "validation_score": 9342.420945925756}

RECENT RESULT
hypothesis: Reducing EMA decay from 0.98 to 0.9775 will exceed 9,344 correct predictions by tracking the final low-learning-rate solution more closely while retaining full-state averaging.
change: Shorten only the full-state EMA horizon, preserving the best architecture, training procedure, and center-weight-3 arithmetic TTA.
mechanism: Shorter-horizon full-state EMA
evidence_used: Lowering decay from 0.9825 to 0.98 retained 9,344 correct while improving cross-entropy from 0.18783146 to 0.18781964; an equal-sized decrement directly tests whether the favorable shorter-horizon trend continues.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 68.20272891689092, "validation_accuracy": 0.9341, "validation_correct": 9341, "validation_cross_entropy": 0.18784501953125, "validation_score": 9341.420930333317}

RECENT RESULT
hypothesis: An EMA decay of 0.98125 will exceed 9,344 correct predictions by interpolating within the accuracy-preserving region between 0.98 and 0.9825, without entering the degraded short-horizon regime observed at 0.9775.
change: Change only the full-state EMA decay from 0.9825 to the untested midpoint 0.98125.
mechanism: Mid-horizon full-state EMA refinement
evidence_used: Decays 0.98 and 0.9825 both achieved 9,344 correct, while 0.9775 fell to 9,341; testing their midpoint most directly refines the remaining favorable EMA interval.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Favoring vertical shifts will exceed 9,344 correct predictions because the equal-and-opposite horizontal-favoring perturbation reduced correctness to 9,339.
change: Restore the proven center weight of 3.0 and, while preserving total ensemble weight, assign each vertical crop weight 1.25 and each horizontal crop weight 0.75.
mechanism: Vertical-favoring translation ensemble
evidence_used: Symmetric weight-3 TTA achieved 9,344 correct, whereas reallocating 0.25 from each vertical crop to each horizontal crop achieved only 9,339; testing the opposite direction is the most informative follow-up.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.59234541701153, "validation_accuracy": 0.9346, "validation_correct": 9346, "validation_cross_entropy": 0.1877129035949707, "validation_score": 9346.420977155747}

RECENT RESULT
hypothesis: Increasing the vertical-shift preference slightly beyond the best 1.25/0.75 weighting will exceed 9,346 correct predictions or retain 9,346 while lowering cross-entropy below 0.18771290.
change: Restore the proven center weight of 3.0 and shift another 0.0625 of weight from each horizontal crop to each vertical crop, preserving total ensemble weight.
mechanism: Fine-grained vertical-favoring translation ensemble
evidence_used: Symmetric 1.0/1.0 shift weights achieved 9,344 correct, horizontal-favoring 0.75/1.25 fell to 9,339, and vertical-favoring 1.25/0.75 improved to 9,346; a small continuation tests whether the optimum lies just beyond the best observed asymmetry.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.23315116693266, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877075828552246, "validation_score": 9348.420979041657}

RECENT RESULT
hypothesis: Increasing each vertical-crop weight to 1.375 and decreasing each horizontal-crop weight to 0.625 will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy below 0.18770758.
change: Shift another 0.0625 of weight from each horizontal crop to each vertical crop while preserving the center weight and total ensemble weight.
mechanism: Continued vertical-favoring translation ensemble
evidence_used: Correctness improved monotonically from 9,344 at symmetric 1.0/1.0 weights to 9,346 at 1.25/0.75 and 9,348 at 1.3125/0.6875, motivating one more equal-sized step in the successful direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.59405312500894, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18771077651977539, "validation_score": 9347.420977909676}

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
