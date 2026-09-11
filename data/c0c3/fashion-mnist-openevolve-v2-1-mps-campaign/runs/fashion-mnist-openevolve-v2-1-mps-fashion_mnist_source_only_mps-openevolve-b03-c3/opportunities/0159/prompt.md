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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 51.86431899992749, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20815952377319336, "validation_score": 9284.413852633003}
prior_hypothesis: Replacing hard max pooling with a 90%-max learnable mixture will exceed 9,283 correct predictions by reducing translation aliasing while preserving the baseline’s edge-selective behavior.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.0333381248638, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813083267211915, "validation_score": 9283.413862461315}
prior_hypothesis: Increasing the validated vertical-over-horizontal radius-one weighting contrast from 0.02 to 0.04 while preserving total ensemble weight will exceed 9,284 correct predictions.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 37.97503695799969, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813371810913087, "validation_score": 9284.413861472869}
prior_hypothesis: Vertical/horizontal radius-one weights of 0.10875/0.07125 will retain at least 9,284 correct predictions while lowering cross-entropy below 0.208136808, improving validation_score.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.45434991712682, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813335342407227, "validation_score": 9284.413861597797}
prior_hypothesis: Reducing radius-one horizontal weight to 0.0709375 while transferring its total mass to the center view will retain at least 9,284 correct predictions and lower cross-entropy below 0.208133718.

## Recent verification evidence

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

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Vertical/horizontal radius-one weights of 0.109375/0.070625 will retain 9,284 correct predictions while lowering cross-entropy below 0.208133718.
change: Move halfway from the best 0.10875/0.07125 TTA weights toward the lower-cross-entropy but 9,283-correct 0.11/0.07 setting, preserving total ensemble weight and all training behavior.
mechanism: Boundary-focused cardinal-asymmetry interpolation
evidence_used: Increasing cardinal contrast steadily lowered cross-entropy through 0.10875/0.07125 while retaining 9,284 correct, whereas 0.11/0.07 lost one correct prediction; bisecting that narrow interval directly probes the apparent decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.86034054099582, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.2081322525024414, "validation_score": 9283.413861974932}

RECENT RESULT
hypothesis: Vertical/horizontal radius-one weights of 0.1090625/0.0709375 will retain 9,284 correct predictions while lowering cross-entropy below 0.208133718.
change: Bisect the interval between the best 0.10875/0.07125 setting and the 9,283-correct 0.109375/0.070625 setting, preserving total ensemble weight and all other behavior.
mechanism: Boundary-bisected cardinal-asymmetry TTA
evidence_used: The 0.10875/0.07125 design achieved 9,284 correct with 0.208133718 cross-entropy, while 0.109375/0.070625 reduced cross-entropy to 0.208132253 but lost one correct prediction; their midpoint is the most direct probe of that decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 53.9830630000215, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813297843933107, "validation_score": 9283.413861726252}

RECENT RESULT
hypothesis: Vertical/horizontal radius-one weights of 0.10890625/0.07109375 will retain 9,284 correct predictions while lowering cross-entropy below 0.208133718.
change: Bisect the interval between the best 0.10875/0.07125 setting and the lower-cross-entropy but 9,283-correct 0.1090625/0.0709375 setting, preserving total ensemble weight and all other behavior.
mechanism: Further boundary-bisected cardinal-asymmetry TTA
evidence_used: The current 0.10875/0.07125 design achieved 9,284 correct at 0.208133718 cross-entropy, while 0.1090625/0.0709375 reduced cross-entropy to 0.208132978 but lost one prediction; their midpoint is the next direct probe of the classification boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 46.44506787508726, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813334884643556, "validation_score": 9283.413861599365}

RECENT RESULT
hypothesis: Reducing radius-one horizontal weight to 0.0709375 while transferring its total mass to the center view will retain at least 9,284 correct predictions and lower cross-entropy below 0.208133718.
change: Restore the best validated 0.10875 vertical weight, reduce horizontal weight without increasing vertical influence, and raise the center weight to preserve total ensemble mass.
mechanism: Center-compensated horizontal-vote pruning
evidence_used: The 0.10875/0.07125 design achieved 9,284 correct, while the coupled 0.1090625/0.0709375 change lost one; isolating the horizontal reduction tests whether the loss came from increased vertical influence while favoring the reliable center prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.45434991712682, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813335342407227, "validation_score": 9284.413861597797}



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
