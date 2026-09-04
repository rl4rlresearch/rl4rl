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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.22494162502699, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19341388931274414, "validation_score": 9328.418966131094}
prior_hypothesis: Holding label smoothing at 0.02 for the first half of training and annealing it to zero during EMA collection will exceed 9,323 correct predictions by removing late soft-target bias while retaining early regularization.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by stabilizing the final parameter-EMA window without reducing the peak learning rate.
change: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5 while preserving its peak and decay timing.
mechanism: Lower-noise terminal cosine annealing
evidence_used: Increasing the EMA update rate to 0.04 reduced correctness from 9,328 to 9,326, consistent with late-update noise being harmful; the complementary lower-floor intervention remains unmeasured because prior verification attempts did not complete.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 4,928-parameter near-identity spatial refinement branch will exceed 9,328 correct predictions by learning local part relationships without the runtime cost of the timed-out larger bottleneck.
change: Add a depthwise-separable residual block after view fusion, initialized at 0.1 output strength; total learned parameters become 229,370.
mechanism: Low-cost depthwise-separable spatial residual refinement
evidence_used: Dense-head widening reached 9,300 and global pooling reached 9,290, while the larger spatial bottleneck timed out; this tests the remaining spatial-processing hypothesis with substantially less computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the cosine learning-rate floor from 0.10 to 0.02 will exceed 9,328 correct predictions by reducing late-update noise during the final parameter-EMA window.
change: Lower the schedule’s terminal learning rate from 2.1e-4 to 4.2e-5 while preserving its peak rate, decay onset, architecture, and EMA behavior.
mechanism: Lower-noise terminal cosine annealing
evidence_used: Increasing the EMA update rate to 0.04 reduced correctness to 9,326, consistent with excessive sensitivity to late updates; the complementary lower-floor intervention remains unmeasured because its prior verification attempts did not complete.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing exact-position flattening with multiscale regional average and maximum statistics will exceed 9,328 correct predictions by adding local shift tolerance and activation-presence cues while retaining the coarse spatial layout that global pooling discarded.
change: Replace the 7×7 flat dense representation with 1×1, 2×2, and 4×4 average/max spatial-pyramid features, widening the hidden layer to 56 while remaining under the parameter ceiling.
mechanism: Multiscale average-max spatial pyramid prediction
evidence_used: Global pooling fell to 9,290, showing that spatial layout is load-bearing, while dense-head widening reached only 9,300 because it retained the same exact-position projection. This tests a different representation that preserves regional layout instead of either erasing it or memorizing every feature location independently.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise during the final half of training.
change: Double the effective parameter-averaging horizon while retaining terminal BatchNorm buffers and all training dynamics.
mechanism: Longer-horizon terminal parameter EMA
evidence_used: Increasing the EMA rate to 0.04 reduced correctness to 9,326 versus the 9,328 baseline, providing directional evidence that shorter averaging is harmful; the complementary 0.01 rate remains unmeasured because prior verification attempts did not complete.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Ramping a 0.15 target-logit margin during EMA collection will exceed 9,328 correct predictions by maintaining pressure on borderline examples after label smoothing reaches zero.
change: Leave the first half unchanged, then progressively subtract up to 0.15 from each target logit only inside the training loss.
mechanism: Late additive target-margin training
evidence_used: The verified 9,328 design benefits from annealing label smoothing to zero in the second half, while added capacity and auxiliary supervision regressed; a small runtime-neutral margin directly tests whether stronger late decision-boundary optimization is more useful than further architectural complexity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the parameter-EMA update rate from 0.02 to 0.015 will exceed 9,328 correct predictions by suppressing late optimizer noise without the stronger lag of the unresolved 0.01 setting.
change: Increase the effective second-half parameter-averaging horizon from roughly 50 to 67 optimizer steps while continuing to copy BatchNorm buffers directly.
mechanism: Moderately longer-horizon terminal parameter EMA
evidence_used: Increasing the EMA rate to 0.04 reduced correctness from 9,328 to 9,326, indicating that a shorter averaging horizon is harmful; 0.015 is a conservative directional test because the more aggressive 0.01 setting was never successfully verified.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 81.70995358400978, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.19295934219360353, "validation_score": 9321.41912576759}

RECENT RESULT
hypothesis: Reducing the terminal learning-rate multiplier from 0.10 to 0.05 will exceed 9,328 correct predictions by limiting late optimizer noise without the stronger convergence slowdown of the repeatedly unresolved 0.02 setting.
change: Lower the cosine schedule floor from 2.1e-4 to 1.05e-4 while preserving the peak rate, decay timing, EMA, and runtime.
mechanism: Moderately lower-noise terminal cosine annealing
evidence_used: A faster 0.04 EMA scored 9,326 and a slower 0.015 EMA scored 9,321 versus the 9,328 baseline, indicating sensitivity to late parameter motion around the current averaging horizon; a moderate learning-rate-floor reduction tests stabilization without changing that verified horizon.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 78.62982287514023, "validation_accuracy": 0.9302, "validation_correct": 9302, "validation_cross_entropy": 0.19563438758850096, "validation_score": 9302.418188039079}

RECENT RESULT
hypothesis: Increasing the EMA update rate from 0.02 to 0.03 will exceed 9,328 correct predictions by reducing parameter lag while avoiding the excess late-step sensitivity observed at 0.04.
change: Shorten the second-half EMA horizon from roughly 50 to 33 optimizer steps without changing training dynamics or BatchNorm-buffer handling.
mechanism: Intermediate-horizon terminal parameter EMA
evidence_used: EMA rates bracket the current setting: 0.015 scored 9,321, 0.02 scored 9,328, and 0.04 scored 9,326. This motivates testing 0.03 between the two strongest settings.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 66.29846533294767, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19399065170288085, "validation_score": 9325.418763747679}

RECENT RESULT
hypothesis: An EMA update rate of 0.024 will exceed 9,328 correct predictions by reducing parameter lag slightly while avoiding the late-step sensitivity observed at 0.03 and 0.04.
change: Increase the second-half parameter-EMA rate from 0.02 to 0.024 without changing training dynamics, BatchNorm handling, or runtime.
mechanism: Finely bracketed terminal parameter EMA
evidence_used: The 0.02 baseline scored 9,328, while 0.03 scored 9,325, 0.04 scored 9,326, and 0.015 scored 9,321; these results bracket a narrow optimum near the current setting and motivate a conservative interpolation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 68.79543212498538, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19368563194274901, "validation_score": 9327.418870753421}

RECENT RESULT
hypothesis: Training with ±2-pixel translations early, reducing to ±1 and then clean images, will exceed 9,328 correct predictions by improving local shift tolerance without sacrificing terminal alignment to the fixed validation distribution.
change: Add inexpensive per-example translated crops during the first 75% of training, annealing their radius from two pixels to one before disabling them for the final quarter.
mechanism: Annealed per-image translation augmentation
evidence_used: Global pooling reduced correctness to 9,290, showing that spatial layout must be preserved, while added spatial refinement timed out; annealed translation regularizes the existing position-sensitive representation without adding learned parameters or inference computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 63.698572624940425, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.20894704208374024, "validation_score": 9270.413583045902}

RECENT RESULT
hypothesis: Raising the terminal learning-rate multiplier from 0.10 to 0.15 will exceed 9,328 correct predictions by preserving useful late learning during the fixed two-exposure budget.
change: Raise the cosine schedule’s terminal learning rate from 2.1e-4 to 3.15e-4 while preserving its peak, decay timing, EMA, and all other training dynamics.
mechanism: Moderately elevated terminal cosine floor
evidence_used: Lowering the terminal multiplier to 0.05 reduced correctness sharply from 9,328 to 9,302, indicating that stronger late annealing caused underfitting rather than beneficial stabilization and motivating a conservative test in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 224442, "training_seconds": 74.45238470798358, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.19542431182861328, "validation_score": 9309.418261528608}



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
