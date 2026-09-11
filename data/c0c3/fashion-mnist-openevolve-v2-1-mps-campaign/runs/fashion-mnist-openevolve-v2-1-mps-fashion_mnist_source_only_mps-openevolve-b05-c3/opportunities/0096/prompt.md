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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}
prior_hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 65.12440524995327, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.23588926696777343, "validation_score": 9206.404566989426}
prior_hypothesis: A 0.0325 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by refining the interval between the best reported 0.03 rate and the slightly worse 0.04 rate while preserving the validated 0.015 parameter smoothing.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 79.74243187485263, "validation_accuracy": 0.9205, "validation_correct": 9205, "validation_cross_entropy": 0.2359372703552246, "validation_score": 9205.404551276179}
prior_hypothesis: Using a 0.015 terminal EMA rate for learned parameters and 0.04 for BatchNorm buffers will exceed 9,206 correct predictions by further reducing normalization-state lag while preserving the validated parameter smoothing.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.0530132080894, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.23077224349975586, "validation_score": 9206.406249005566}
prior_hypothesis: Applying 1.05× sharpening to the qualified split-EMA model’s ensemble logits will retain 9,206 correct predictions while lowering cross-entropy below 0.235889, improving validation_score.

## Recent verification evidence

RECENT RESULT
hypothesis: A terminal EMA rate of 0.0175 will exceed 9,201 correct predictions by further reducing residual-model and BatchNorm-state lag while making a smaller extrapolation than the unverified 0.02 attempt.
change: Restore the verified residual topology and 19/32 augmentation transition, then use EMA rate 0.005 before the transition and 0.0175 afterward.
mechanism: Finer curriculum-synchronized all-tensor EMA tracking
evidence_used: On the same residual 19/32 design, raising terminal EMA from 0.005 to 0.01 improved correct predictions to 9,200, and 0.015 further improved them to 9,201 while lowering cross-entropy; 0.0175 tests the remaining interval toward the timed-out 0.02 configuration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining the validated 0.015 terminal EMA for learned parameters while increasing BatchNorm-buffer EMA to 0.03 will exceed 9,201 correct predictions by adapting normalization statistics faster to the cardinal-augmentation distribution without sacrificing parameter smoothing.
change: Track whether each EMA tensor is a model buffer, then use terminal EMA rates of 0.015 for parameters and 0.03 for floating-point buffers; preserve the 0.005 pre-transition rate.
mechanism: Faster curriculum-synchronized BatchNorm-state EMA
evidence_used: The all-tensor residual design reached 9,201 correct at terminal rate 0.015, whereas parameter-only EMA reached 9,168; this shows BatchNorm-state averaging is valuable and motivates independently reducing its post-transition lag.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 80.77500774990767, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.2358737476348877, "validation_score": 9206.404572069725}

RECENT RESULT
hypothesis: Using a 0.015 terminal EMA rate for learned parameters and 0.04 for BatchNorm buffers will exceed 9,206 correct predictions by further reducing normalization-state lag while preserving the validated parameter smoothing.
change: Mark EMA tensors that are model buffers, then apply terminal EMA rates of 0.015 to parameters and 0.04 to buffers while retaining the 0.005 pre-transition rate.
mechanism: Faster post-transition BatchNorm-state tracking
evidence_used: Separating the terminal rates at 0.015 for parameters and 0.03 for buffers reported 9,206 correct versus 9,201 when all tensors used 0.015, indicating that faster BatchNorm-buffer adaptation is the promising axis to continue.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 79.74243187485263, "validation_accuracy": 0.9205, "validation_correct": 9205, "validation_cross_entropy": 0.2359372703552246, "validation_score": 9205.404551276179}

RECENT RESULT
hypothesis: A 0.035 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by refining the interval between the improving 0.03 rate and the slightly worse 0.04 rate while retaining the validated 0.015 parameter rate.
change: Restore the parameter-neutral residual topology, include floating-point BatchNorm buffers in EMA, and use terminal EMA rates of 0.015 for parameters and 0.035 for buffers after the 19/32 curriculum transition.
mechanism: Interpolated post-transition BatchNorm-state tracking
evidence_used: Raising the buffer rate from 0.015 to 0.03 reported 9,206 correct, while 0.04 produced 9,205; testing their midpoint is the most direct local refinement of the strongest observed mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.0325 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by refining the interval between the best reported 0.03 rate and the slightly worse 0.04 rate while preserving the validated 0.015 parameter smoothing.
change: Restore the residual 19/32 curriculum, distinguish BatchNorm buffers from parameters in EMA, and use terminal EMA rates of 0.015 for parameters and 0.0325 for buffers.
mechanism: Interpolated post-transition BatchNorm-state tracking
evidence_used: Separating EMA rates produced the strongest reported result—9,206 correct at a 0.03 buffer rate—while 0.04 reported 9,205; 0.0325 is the closest untested local refinement above the observed optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 65.12440524995327, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.23588926696777343, "validation_score": 9206.404566989426}

RECENT RESULT
hypothesis: A 0.0275 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by testing the unexplored lower side of the best observed 0.03 rate while preserving 0.015 parameter smoothing.
change: Change only the terminal BatchNorm-buffer EMA rate from 0.0325 to 0.0275.
mechanism: Lower-side BatchNorm EMA refinement
evidence_used: A 0.03 buffer rate achieved 9,206 correct with 0.235874 cross-entropy, while 0.0325 tied the correct count but worsened cross-entropy and 0.04 fell to 9,205; 0.0275 is the nearest untested point that brackets 0.03 from below.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a zero-initialized mean-and-standard-deviation prediction branch to the qualified 0.0325-buffer-EMA design will exceed 9,206 correct predictions by complementing coordinate-sensitive spatial templates with translation-stable channel-distribution evidence.
change: Replace the old assumption that flattened 3×3 features alone should determine each class with an additive classifier that also learns from global first- and second-order feature moments; retain the strongest qualified EMA setting.
mechanism: Spatial-template and global-moment logit fusion
evidence_used: Reference Design 1 achieved 9,206 correct, while repeated EMA refinements plateaued at 9,205–9,206; the residual model’s advantage over the plain network indicates that a genuinely different learned representation mechanism is more promising than further coefficient tuning.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247748, "training_seconds": 73.07314416696317, "validation_accuracy": 0.9201, "validation_correct": 9201, "validation_cross_entropy": 0.23701165161132812, "validation_score": 9201.404199911414}

RECENT RESULT
hypothesis: Applying 1.05× sharpening to the qualified split-EMA model’s ensemble logits will retain 9,206 correct predictions while lowering cross-entropy below 0.235889, improving validation_score.
change: Restore the qualified 0.015 parameter/0.0325 BatchNorm-buffer terminal EMA rates and sharpen only evaluation-time ensemble logits without changing their argmax.
mechanism: Decision-preserving TTA temperature sharpening
evidence_used: Reference Design 1 achieved 9,206 correct with 0.235889 cross-entropy, while the added moment branch fell to 9,201; fixed positive logit scaling preserves the stronger model’s predictions and directly targets the cross-entropy tie-breaker.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.0530132080894, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.23077224349975586, "validation_score": 9206.406249005566}

RECENT RESULT
hypothesis: Increasing evaluation-time sharpening from 1.05× to 1.10× will retain exactly 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Restore the qualified residual architecture and split parameter/BatchNorm EMA, then scale the final ensemble log-probabilities by 1.10 during evaluation.
mechanism: Decision-preserving stronger TTA temperature sharpening
evidence_used: The 1.05× sharpening design preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772; a larger positive scale preserves argmax decisions and tests whether the ensemble remains underconfident.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing evaluation-time sharpening from 1.05× to 1.075× will preserve exactly 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Scale the final ensemble log-probabilities by 1.075 instead of 1.05, leaving training, EMA, architecture, and argmax predictions unchanged.
mechanism: Decision-preserving intermediate TTA temperature sharpening
evidence_used: Moving from unsharpened logits to 1.05× sharpening preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772; 1.075× is an informative midpoint toward the timed-out 1.10× test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation-time ensemble log-probabilities by 1.075 will preserve exactly 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Sharpen only the final evaluation-time ensemble output by 1.075; training, architecture, EMA, and predicted classes remain unchanged.
mechanism: Intermediate decision-preserving TTA temperature sharpening
evidence_used: The verified 1.05× scale preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772. The prior 1.075× attempt timed out, so it supplied no performance evidence against this intermediate calibration step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the qualified 0.0325 BatchNorm-buffer EMA and scaling ensemble log-probabilities by 1.075 will retain 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Use the strongest qualified split-EMA training configuration and increase evaluation-only sharpening from 1.05× to 1.075× without changing predicted classes.
mechanism: Decision-preserving intermediate TTA temperature sharpening
evidence_used: The 1.05× design retained 9,206 correct and reduced cross-entropy from 0.235889 to 0.230772; prior 1.075× attempts only timed out and supplied no performance evidence against the calibration change.
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
