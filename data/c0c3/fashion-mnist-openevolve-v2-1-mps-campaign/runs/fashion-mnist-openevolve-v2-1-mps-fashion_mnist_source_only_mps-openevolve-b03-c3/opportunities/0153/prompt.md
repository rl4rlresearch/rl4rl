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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 37.97503695799969, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813371810913087, "validation_score": 9284.413861472869}
prior_hypothesis: Vertical/horizontal radius-one weights of 0.10875/0.07125 will retain at least 9,284 correct predictions while lowering cross-entropy below 0.208136808, improving validation_score.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.0333381248638, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813083267211915, "validation_score": 9283.413862461315}
prior_hypothesis: Increasing the validated vertical-over-horizontal radius-one weighting contrast from 0.02 to 0.04 while preserving total ensemble weight will exceed 9,284 correct predictions.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 51.86431899992749, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20815952377319336, "validation_score": 9284.413852633003}
prior_hypothesis: Replacing hard max pooling with a 90%-max learnable mixture will exceed 9,283 correct predictions by reducing translation aliasing while preserving the baseline’s edge-selective behavior.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 43.2198099589441, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20814357986450197, "validation_score": 9284.413858094627}
prior_hypothesis: Vertical/horizontal radius-one weights of 0.105/0.075 with the validated distance-sensitive outer weights will exceed 9,284 correct predictions by refining the apparent optimum between the 0.10/0.08 and 0.11/0.07 designs.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a lightweight translation-stable global-average readout to the 9,284-correct mixed-pooling model will exceed 9,284 correct predictions by complementing the spatial classifier with shape-level evidence and a shorter optimization path.
change: Add an 810-parameter linear head over globally averaged refined features, zero-initialize it to preserve the baseline’s initial logits, and add its logits at 0.25 strength to the existing classifier.
mechanism: Zero-initialized global-evidence residual head
evidence_used: Scalar mixed pooling achieved the best result at 9,284 correct, while widening the shared classifier fell to 9,239 and squeeze-excitation fell to 9,263; this motivates adding narrowly targeted invariant supervision without altering the proven representation or expanding its main bottleneck.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the 9,284-correct pooling learning rate and averaging TTA predictions in probability space will exceed 9,284 correct by preventing an overconfident shifted or flipped view from disproportionately controlling the ensemble.
change: Restore full-rate optimization for both pooling gates and replace geometric logit averaging across the 50 evaluation views with an arithmetic probability ensemble converted back to calibrated logits.
mechanism: Probability-space transformation marginalization
evidence_used: Independent 90%-max pooling at full learning rate achieved the best result of 9,284 correct, while half-rate gates fell to 9,281 and architectural additions regressed or timed out; this preserves the best training design and targets the still-untested inference aggregation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 45.640353832859546, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20851180801391603, "validation_score": 9280.413731993916}

RECENT RESULT
hypothesis: Restoring independent scalar 90%-max pooling and weighting validation translations exactly like the training augmentation distribution will exceed 9,284 correct predictions by removing an untrained directional and distance-dependent inference bias while preserving logit-space averaging.
change: Replace the regressed channelwise pooling gates with the best verified stage-specific scalar gates, then use 0.09 for all radius-one cardinal shifts, 0.045 for diagonals, and 0.00625 for every radius-two shift.
mechanism: Training-distribution-matched logit TTA
evidence_used: Independent scalar mixed pooling achieved the best result at 9,284 correct, while channelwise pooling fell to 9,281 and probability-space TTA fell to 9,280; this motivates restoring the proven representation and testing only whether exact augmentation-matched logit weights improve inference.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.5811313749291, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20820398025512696, "validation_score": 9283.413837405084}

RECENT RESULT
hypothesis: Independent 90%-max pooling with equal 0.09 weights for horizontal and vertical one-pixel TTA shifts will exceed 9,284 correct predictions by removing unsupported directional bias while retaining the distance-sensitive radius-two weighting absent from the 9,283-correct fully matched TTA design.
change: Restore the best verified independent pooling gates and symmetrize only radius-one cardinal TTA weights, leaving the original distance-weighted outer shifts unchanged.
mechanism: Cardinal-symmetric near-shift logit TTA
evidence_used: Independent pooling with the original TTA achieved 9,284 correct, while simultaneously symmetrizing cardinal shifts and flattening all radius-two weights achieved 9,283; isolating cardinal symmetry tests whether it is beneficial without discarding useful distance-based reliability weighting.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 55.68721762509085, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20820101928710938, "validation_score": 9283.413838419285}

RECENT RESULT
hypothesis: Uniformly weighting radius-two translations at 0.00625 while preserving the 9,284-correct model’s vertical/horizontal radius-one asymmetry will exceed 9,284 correct predictions by matching the training distribution for weak outer shifts without removing the apparently useful near-shift directional bias.
change: Replace distance-dependent radius-two logit weights with equal 0.00625 weights; leave architecture, training, radius-one TTA, and total ensemble weight unchanged.
mechanism: Uniform radius-two TTA with retained near-shift asymmetry
evidence_used: The current asymmetric, distance-weighted TTA reached 9,284 correct; symmetrizing radius-one alone reached 9,283, while symmetrizing radius-one and flattening radius-two also reached 9,283. Their equal correct counts motivate isolating outer-weight flattening while retaining the best design’s near-shift asymmetry.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.817101791035384, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.20816199760437012, "validation_score": 9282.413851785597}

RECENT RESULT
hypothesis: Increasing the validated vertical-over-horizontal radius-one weighting contrast from 0.02 to 0.04 while preserving total ensemble weight will exceed 9,284 correct predictions.
change: Weight vertical radius-one shifts at 0.11 and horizontal shifts at 0.07; retain diagonal, radius-two, architecture, and training settings.
mechanism: Amplified cardinal-direction TTA contrast
evidence_used: The 0.10/0.08 asymmetric design achieved 9,284 correct versus 9,283 for equal 0.09 weights, while flattening radius-two weights regressed, motivating a focused extrapolation of the beneficial cardinal asymmetry.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.0333381248638, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813083267211915, "validation_score": 9283.413862461315}

RECENT RESULT
hypothesis: Vertical/horizontal radius-one weights of 0.105/0.075 with the validated distance-sensitive outer weights will exceed 9,284 correct predictions by refining the apparent optimum between the 0.10/0.08 and 0.11/0.07 designs.
change: Restore distance-dependent radius-two weighting and test the midpoint between the best 0.10/0.08 cardinal contrast and the regressed 0.11/0.07 contrast, preserving total ensemble weight.
mechanism: Midpoint cardinal-asymmetry TTA
evidence_used: The 0.10/0.08 design achieved the best 9,284 correct, while both 0.09/0.09 and 0.11/0.07 achieved 9,283; uniform radius-two weights also regressed to 9,282, motivating a narrow cardinal-weight search with the validated outer weighting restored.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 43.2198099589441, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20814357986450197, "validation_score": 9284.413858094627}

RECENT RESULT
hypothesis: Full-rate pooling with vertical/horizontal radius-one weights of 0.1075/0.0725 will improve validation_score by retaining at least 9,284 correct predictions while lowering cross-entropy relative to the 0.105/0.075 design.
change: Restore the validated full learning rate for both pooling gates and test the midpoint between the 9,284-correct 0.105/0.075 TTA setting and the lower-cross-entropy but 9,283-correct 0.11/0.07 setting.
mechanism: Fine-grained cardinal-asymmetry TTA interpolation
evidence_used: Increasing cardinal contrast monotonically reduced cross-entropy from 0.2081595 at 0.10/0.08 to 0.2081436 at 0.105/0.075 and 0.2081308 at 0.11/0.07, but the last setting lost one correct prediction; meanwhile half-rate pooling regressed from 9,284 to 9,281 correct, so full-rate pooling should be restored before refining the apparent decision-boundary optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.81182725005783, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.208136808013916, "validation_score": 9284.413860414386}

RECENT RESULT
hypothesis: Vertical/horizontal radius-one weights of 0.10875/0.07125 will retain at least 9,284 correct predictions while lowering cross-entropy below 0.208136808, improving validation_score.
change: Move halfway from the current 0.1075/0.0725 TTA weights toward the lower-cross-entropy 0.11/0.07 setting, preserving total ensemble weight and all training behavior.
mechanism: Decision-boundary cardinal-asymmetry refinement
evidence_used: Cross-entropy improved monotonically as cardinal contrast increased from 0.10/0.08 through 0.105/0.075 to 0.1075/0.0725, all retaining 9,284 correct, while 0.11/0.07 lowered cross-entropy further but lost one prediction; this tests the midpoint nearest that apparent classification boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 37.97503695799969, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813371810913087, "validation_score": 9284.413861472869}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
