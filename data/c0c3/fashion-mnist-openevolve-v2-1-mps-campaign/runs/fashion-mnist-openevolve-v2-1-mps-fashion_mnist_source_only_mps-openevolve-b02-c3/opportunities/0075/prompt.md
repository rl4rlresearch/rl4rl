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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.05553245800547, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18791449165344237, "validation_score": 9343.42090571629}
prior_hypothesis: A computationally batched 0.985-decay full-state EMA will finish within the time limit and exceed 9,334 correct predictions by tracking late low-learning-rate convergence more closely than the successful 0.99 EMA.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.70533933397382, "validation_accuracy": 0.934, "validation_correct": 9340, "validation_cross_entropy": 0.18788887405395507, "validation_score": 9340.420914793396}
prior_hypothesis: A 0.975-decay full-state EMA will exceed 9,344 correct predictions by tracking final low-learning-rate convergence more closely than the successful 0.98 EMA.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 71.9894648338668, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18783146209716797, "validation_score": 9344.420935137648}
prior_hypothesis: A 0.9825-decay full-state EMA will exceed 9,344 correct predictions by interpolating between the near-best 0.985 horizon and the best 0.98 horizon.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.50179904093966, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18781963577270508, "validation_score": 9344.420939328616}
prior_hypothesis: A 0.98-decay full-state EMA will exceed 9,343 correct predictions by tracking the final low-learning-rate solution more closely than the successful 0.985 EMA while retaining its averaging of BatchNorm statistics.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.975-decay full-state EMA will exceed 9,344 correct predictions by tracking final low-learning-rate convergence more closely than the successful 0.98 EMA.
change: Restore the proven equal mean/max refinement descriptor and foreach-accelerated full-state EMA, shortening its decay from 0.98 to 0.975.
mechanism: Short-horizon full-state exponential weight consolidation
evidence_used: Shortening full-state EMA decay from 0.99 to 0.985 improved correctness from 9,334 to 9,343, and 0.98 further improved it to 9,344; this motivates one localized continuation of the successful trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.70533933397382, "validation_accuracy": 0.934, "validation_correct": 9340, "validation_cross_entropy": 0.18788887405395507, "validation_score": 9340.420914793396}

RECENT RESULT
hypothesis: A 0.9825-decay full-state EMA will exceed 9,344 correct predictions by interpolating between the near-best 0.985 horizon and the best 0.98 horizon.
change: Restore the proven pooled mean/max refinement gate and add foreach-accelerated averaging of all floating-point parameters and BatchNorm buffers from halfway through training, using decay 0.9825.
mechanism: Mid-horizon full-state exponential weight consolidation
evidence_used: Full-state EMA improved from 9,334 correct at decay 0.99 to 9,343 at 0.985 and peaked at 9,344 at 0.98 before regressing to 9,340 at 0.975, motivating a focused midpoint test in the bracket containing the observed optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 71.9894648338668, "validation_accuracy": 0.9344, "validation_correct": 9344, "validation_cross_entropy": 0.18783146209716797, "validation_score": 9344.420935137648}

RECENT RESULT
hypothesis: A 0.98125-decay full-state EMA will exceed the current validation_score by selecting an averaging horizon between the two 9,344-correct endpoints, potentially improving correctness or tie-breaking cross-entropy.
change: Retain the proven architecture and training procedure while changing only the EMA decay from 0.98 to 0.98125.
mechanism: Fine-bracket full-state exponential weight consolidation
evidence_used: Decays 0.98 and 0.9825 both achieved the best observed 9,344 correct, while neighboring 0.975 and 0.985 achieved 9,340 and 9,343; testing their midpoint is the most focused refinement of the observed optimum plateau.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.96864345786162, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18781935539245606, "validation_score": 9343.420939427979}

RECENT RESULT
hypothesis: Restoring the empirically superior 0.98 EMA decay will retain 9,344 correct predictions while lowering cross-entropy below the current 0.18783146.
change: Change only the full-state EMA decay from 0.9825 to 0.98.
mechanism: Best-observed full-state EMA horizon
evidence_used: The 0.98 reference achieved the best observed validation_score, matching the current design’s 9,344 correct predictions with lower cross-entropy (0.18781964 versus 0.18783146); the intermediate 0.98125 test also lost one correct prediction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Changing the EMA decay to 0.98 will reproduce the reference’s 9,344 correct predictions and exceed the current 0.985-decay design’s validation_score.
change: Shorten only the foreach-accelerated full-state EMA decay from 0.985 to 0.98.
mechanism: Best-observed full-state EMA horizon
evidence_used: Reference Design 3’s 0.98 decay achieved the highest observed validation_score, with 9,344 correct and lower cross-entropy than the tied 0.9825 design; the prior restoration attempt was unverifiable and supplied no evidence against the setting.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the verified 0.98 EMA decay will recover 9,344 correct predictions and exceed the current 0.975-decay design’s validation_score.
change: Change only the foreach-accelerated full-state EMA decay from 0.975 to 0.98.
mechanism: Best-observed full-state EMA horizon
evidence_used: The verified 0.98 design achieved the highest observed validation_score with 9,344 correct and 0.18781964 cross-entropy, while the current 0.975 design regressed to 9,340 correct; prior restoration attempts were unverifiable and provide no contrary performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 0.97875-decay full-state EMA will exceed the current validation_score by selecting an averaging horizon between the best 0.98 setting and the regressed 0.975 setting.
change: Change only the foreach-accelerated full-state EMA decay from 0.98 to 0.97875.
mechanism: Lower-bracket full-state EMA horizon refinement
evidence_used: Decay 0.98 achieved the highest observed score with 9,344 correct, while 0.975 fell to 9,340 and the upper midpoint 0.98125 fell to 9,343; testing the unmeasured lower midpoint is the most focused remaining refinement around the observed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.187969374936074, "validation_accuracy": 0.9341, "validation_correct": 9341, "validation_cross_entropy": 0.18782910919189452, "validation_score": 9341.420935971455}

RECENT RESULT
hypothesis: Reallocating the oversized nonlinear dense head into four spatial residual bottlenecks will exceed 9,344 correct predictions by learning richer cross-channel part relationships while retaining the spatial organization that global pooling lost.
change: Replace the assumption that a shallow feature extractor needs a 176K-parameter shared dense bottleneck with four local/contextual residual reasoning blocks and a direct class-specific 7×7 spatial readout; retain the proven refinement gate, training procedure, TTA, and EMA.
mechanism: Deep residual part reasoning with position-sensitive class readout
evidence_used: Global pooling fell to 9,085 correct and post-pooling attention reached only 9,300, showing that spatial organization and pre-prediction feature formation are load-bearing; this design preserves position explicitly while moving capacity from the current dense head into learned spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Changing the EMA decay from 0.985 to the verified optimum of 0.98 will recover 9,344 correct predictions and exceed the current 9,343-correct validation score.
change: Shorten only the foreach-accelerated full-state EMA decay from 0.985 to 0.98.
mechanism: Best-observed full-state EMA horizon
evidence_used: Reference Design 3 achieved the highest observed validation score with decay 0.98: 9,344 correct and 0.18781964 cross-entropy. Both adjacent refinements, 0.97875 and 0.98125, lost correct predictions, while prior restoration failures supplied no contradictory performance result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring decay 0.98 will recover the verified 9,344 correct predictions and exceed the current 0.975-decay design’s validation score.
change: Change only the foreach-accelerated full-state EMA decay from 0.975 to 0.98.
mechanism: Best-observed full-state EMA horizon
evidence_used: The verified 0.98 design achieved the highest observed score with 9,344 correct and 0.18781964 cross-entropy; 0.975 regressed to 9,340, while unsuccessful restoration attempts produced no contradictory validation result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the unshifted crop weight from 3 to 4 will exceed 9,344 correct predictions by reducing boundary errors introduced by replicate-padded shifted views while retaining their ensemble benefit.
change: Change only the five-crop TTA weights so the clean centered view receives half of the total crop weight instead of three-sevenths.
mechanism: Center-biased translation test-time ensembling
evidence_used: The current position-sensitive model achieves the best observed result at 9,344 correct, while removing spatial organization performed substantially worse; this motivates a conservative inference-only refinement that gives the spatially aligned view slightly more influence without disturbing the verified training and 0.98 EMA procedure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 57.40940608386882, "validation_accuracy": 0.9338, "validation_correct": 9338, "validation_cross_entropy": 0.18793966255187988, "validation_score": 9338.420896797845}

RECENT RESULT
hypothesis: Reducing the centered-crop weight from 3 to 2 will exceed 9,344 correct predictions by giving more influence to shifted views, reversing the direction that degraded accuracy.
change: Change only the five-crop TTA weights so the centered view receives one-third of total crop weight instead of three-sevenths.
mechanism: Shift-favoring translation test-time ensembling
evidence_used: Increasing the center weight from 3 to 4 reduced correctness from 9,344 to 9,338, providing direct evidence that emphasizing the aligned view is harmful and motivating the opposite localized adjustment.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 67.65227025002241, "validation_accuracy": 0.9343, "validation_correct": 9343, "validation_cross_entropy": 0.18779218101501466, "validation_score": 9343.420949058254}



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
