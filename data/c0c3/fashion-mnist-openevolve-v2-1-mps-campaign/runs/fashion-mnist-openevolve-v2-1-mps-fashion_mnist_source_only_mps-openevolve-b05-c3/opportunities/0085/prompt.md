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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 75.69645908288658, "validation_accuracy": 0.9201, "validation_correct": 9201, "validation_cross_entropy": 0.23597998046875, "validation_score": 9201.40453729664}
prior_hypothesis: Increasing the terminal-phase all-tensor EMA rate from 0.01 to 0.015 on the qualified residual design will exceed 9,200 correct predictions by further reducing stale parameter and BatchNorm state after the augmentation transition without abandoning late-training smoothing.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.46581783308648, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.23885296478271484, "validation_score": 9167.403599147125}
prior_hypothesis: Beginning center/cardinal augmentation after 39/64 of training will exceed 9,163 correct predictions by extending the monotonically beneficial inference-aligned phase beyond three-eighths while making a smaller change than the timed-out two-fifths configuration.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}
prior_hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 79.16737045813352, "validation_accuracy": 0.92, "validation_correct": 9200, "validation_cross_entropy": 0.23637798385620118, "validation_score": 9200.404407071728}
prior_hypothesis: Increasing the EMA update rate from 0.005 to 0.01 only during the cardinal-augmentation phase will exceed 9,193 correct predictions by reducing broad-phase parameter and BatchNorm lag while retaining late-training smoothing.

## Recent verification evidence

RECENT RESULT
hypothesis: Resetting the all-tensor EMA at the 19/32 augmentation transition will exceed 9,193 correct predictions by removing stale broad-shift averaging while preserving tail smoothing of parameters and BatchNorm statistics.
change: Reinitialize every EMA tensor from the live model exactly when cardinal augmentation begins, then continue the validated 0.005 EMA through the terminal phase.
mechanism: Curriculum-boundary EMA restart
evidence_used: The residual 19/32 design with all-tensor EMA achieved the best completed result of 9,193 correct, while parameter-only EMA fell to 9,168; restarting rather than narrowing the EMA preserves the beneficial normalization-statistic averaging while aligning it with the validated terminal curriculum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing probability averaging with logit averaging on the verified residual architecture will exceed 9,193 correct predictions by emphasizing transform-consistent class evidence while adding no training cost.
change: Restore the best three-stage residual network and aggregate its ten evaluation views in logit space instead of probability space.
mechanism: Geometric-mean TTA consensus on residual features
evidence_used: Stagewise residual refinement achieved the best completed result at 9,193 correct; unlike the timed-out confidence sharpening experiment, logit-space TTA can improve top-1 decisions while preserving the validated training procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified residual topology and EMA of BatchNorm buffers will reproduce approximately 9,193 correct predictions, exceeding the current parameter-only plain model’s 9,168.
change: Replace each plain two-convolution stage with the parameter-neutral verified residual stage and include floating-point model buffers in EMA.
mechanism: Stagewise residual refinement with all-tensor EMA
evidence_used: Reference Design 2 achieved the best completed result—9,193 correct with 245,818 parameters—while the current parameter-only EMA design achieved 9,168; subsequent speculative extensions did not complete verification.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the qualified residual 19/32 design will reproduce approximately 9,193 correct predictions and exceed the current plain model’s 9,167.
change: Replace each plain two-convolution stage with its parameter-neutral residual equivalent and begin cardinal augmentation at 19/32 of training.
mechanism: Stagewise residual refinement with inference-aligned curriculum
evidence_used: Reference Design 2 achieved the best completed result—9,193 correct with 245,818 parameters—while subsequent speculative residual extensions timed out and provide no evidence of improvement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Initializing each refinement BatchNorm scale to zero will exceed 9,193 correct predictions by starting every stage as a stable single-convolution path while allowing residual refinements to emerge during training.
change: Zero-initialize the learned scale of each stage’s second BatchNorm without changing architecture, parameter count, runtime, curriculum, EMA, or TTA.
mechanism: Zero-initialized residual refinement
evidence_used: Residual stages improved the best completed result from 9,172 to 9,193 correct, supporting feature preservation and gradient flow; zero-initialization strengthens that successful identity bias specifically at initialization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the qualified residual architecture will reproduce approximately 9,193 correct predictions and exceed the current plain model’s 9,172.
change: Replace each plain two-convolution stage with a parameter-neutral residual stage while preserving the verified 19/32 curriculum, all-tensor EMA, optimizer, and TTA.
mechanism: Stagewise residual feature refinement
evidence_used: Reference Design 3 achieved the best completed result—9,193 correct with 245,818 parameters—versus 9,172 for the current plain architecture; later speculative modifications did not produce verified improvements.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the qualified residual architecture and EMA of BatchNorm buffers will exceed the current 9,168 correct predictions and reproduce approximately 9,193.
change: Replace each plain convolution pair with a parameter-neutral residual stage and include floating-point model buffers in EMA while preserving the verified 19/32 curriculum.
mechanism: Stagewise residual feature refinement with all-tensor EMA
evidence_used: Reference Design 2 achieved the best completed result of 9,193 correct with 245,818 parameters; the current plain, parameter-only EMA design achieved 9,168.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding input-conditioned channel recalibration to the verified residual 19/32 design will exceed 9,193 correct predictions by emphasizing image-specific shape and texture channels without sacrificing residual feature preservation.
change: Replace fixed serial convolution stages with parameter-neutral residual stages followed by lightweight squeeze-excitation gates, and use the best verified 19/32 curriculum. The old assumption was that every image should use the same channel mixture; the new approach predicts image-dependent channel importance. Total parameters become 249,618.
mechanism: Identity-centered residual channel attention
evidence_used: Residual refinement produced the best completed result of 9,193 correct versus 9,172 for the comparable plain network. This patch preserves that verified mechanism while testing previously unexplored input-conditioned feature selection with little additional computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the EMA update rate from 0.005 to 0.01 only during the cardinal-augmentation phase will exceed 9,193 correct predictions by reducing broad-phase parameter and BatchNorm lag while retaining late-training smoothing.
change: Preserve the verified residual architecture, 19/32 curriculum, and all-tensor EMA, but shorten the EMA averaging horizon after the curriculum transition.
mechanism: Curriculum-synchronized faster all-tensor EMA
evidence_used: The current all-tensor residual design achieved the best completed result of 9,193 correct; this coefficient-only change targets the stale-state concern behind the unverified EMA-restart experiment without adding meaningful computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 79.16737045813352, "validation_accuracy": 0.92, "validation_correct": 9200, "validation_cross_entropy": 0.23637798385620118, "validation_score": 9200.404407071728}

RECENT RESULT
hypothesis: Increasing the terminal-phase all-tensor EMA rate from 0.01 to 0.015 on the qualified residual design will exceed 9,200 correct predictions by further reducing stale parameter and BatchNorm state after the augmentation transition without abandoning late-training smoothing.
change: Restore the qualified parameter-neutral residual topology and use EMA rate 0.005 during broad augmentation and 0.015 during cardinal augmentation.
mechanism: Finer curriculum-synchronized EMA tracking
evidence_used: On the same residual 19/32 curriculum, raising the terminal EMA rate from 0.005 to 0.01 improved validation_correct from 9,193 to 9,200 and cross-entropy from 0.239125 to 0.236378; a smaller continuation tests whether faster inference-aligned tracking remains beneficial.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 75.69645908288658, "validation_accuracy": 0.9201, "validation_correct": 9201, "validation_cross_entropy": 0.23597998046875, "validation_score": 9201.40453729664}

RECENT RESULT
hypothesis: Increasing the residual model’s terminal EMA rate from 0.015 to 0.02 will exceed 9,201 correct predictions by further reducing parameter and BatchNorm-state lag during inference-aligned training.
change: Restore the verified residual topology and BatchNorm-buffer averaging, then use EMA rate 0.005 before the 19/32 curriculum transition and 0.02 afterward.
mechanism: Faster curriculum-synchronized all-tensor EMA
evidence_used: On the same residual curriculum, increasing terminal EMA from 0.005 to 0.01 improved correct predictions from 9,193 to 9,200, and increasing it to 0.015 improved them to 9,201 while also lowering cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A terminal EMA rate of 0.0175 will exceed 9,201 correct predictions by further reducing residual-model and BatchNorm-state lag while making a smaller extrapolation than the unverified 0.02 attempt.
change: Restore the verified residual topology and 19/32 augmentation transition, then use EMA rate 0.005 before the transition and 0.0175 afterward.
mechanism: Finer curriculum-synchronized all-tensor EMA tracking
evidence_used: On the same residual 19/32 design, raising terminal EMA from 0.005 to 0.01 improved correct predictions to 9,200, and 0.015 further improved them to 9,201 while lowering cross-entropy; 0.0175 tests the remaining interval toward the timed-out 0.02 configuration.
result: training did not finish within the verification time limit



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
