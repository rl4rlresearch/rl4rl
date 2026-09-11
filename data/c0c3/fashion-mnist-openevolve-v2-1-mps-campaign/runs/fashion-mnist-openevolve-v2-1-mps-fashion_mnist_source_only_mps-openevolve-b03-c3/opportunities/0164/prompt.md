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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 41.0333381248638, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20813083267211915, "validation_score": 9283.413862461315}
prior_hypothesis: Increasing the validated vertical-over-horizontal radius-one weighting contrast from 0.02 to 0.04 while preserving total ensemble weight will exceed 9,284 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.3833637079224, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813265075683593, "validation_score": 9284.413861838504}
prior_hypothesis: Reducing radius-one horizontal weight from 0.070625 to 0.0703125 while transferring its total mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132999.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 56.234569374937564, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813197937011718, "validation_score": 9284.413862068497}
prior_hypothesis: Reducing each radius-one horizontal weight to 0.0696875 and transferring the combined mass to the center will retain 9,284 correct predictions while lowering cross-entropy below 0.208132310.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.70478770881891, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2081323101043701, "validation_score": 9284.4138619552}
prior_hypothesis: Holding vertical weight at 0.10875 while reducing each horizontal radius-one weight to 0.07000 and transferring their combined mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132651.

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

RECENT RESULT
hypothesis: Keeping vertical weight at 0.10875 while reducing horizontal weight to 0.070625 and transferring the removed mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208133353.
change: Apply the next 0.0003125 horizontal-weight reduction beyond Reference Design 3, increasing the center weight by 0.000625 to preserve total ensemble mass.
mechanism: Further center-compensated horizontal-vote pruning
evidence_used: Reference Design 3 retained 9,284 correct and improved cross-entropy to 0.208133353 by reducing horizontal influence while holding vertical influence fixed; this continues that isolated direction without the vertical increase associated with prior one-prediction regressions.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.316511125070974, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813299865722656, "validation_score": 9284.413861719328}

RECENT RESULT
hypothesis: Replacing layer-wide pooling ratios with content-conditioned per-channel gates, initialized to reproduce the current 90% max-pooling behavior, will improve validation_score beyond 9284.413861719328 when combined with the best validated TTA weights.
change: Give every channel its own baseline pooling logit and learned response to local peakiness, allowing edge and texture channels to downsample differently; restore Reference Design 1’s best-known center/vertical/horizontal TTA weighting. The model remains below the parameter ceiling at 245,294 learned parameters.
mechanism: Contrast-conditioned channelwise mixed pooling
evidence_used: TTA refinements plateaued at 9,284 correct and increasingly small weight changes crossed a sharp boundary back to 9,283, indicating that fixed-view calibration is exhausted. All available designs assume one input-independent pooling ratio per layer; content-conditioned channelwise pooling instead changes the learned representation while starting exactly from the validated pooling behavior.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245294, "training_seconds": 45.81469799997285, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.2083370018005371, "validation_score": 9280.413791847188}

RECENT RESULT
hypothesis: Reducing radius-one horizontal weight from 0.070625 to 0.0703125 while transferring its total mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132999.
change: Decrease each horizontal radius-one TTA vote by 0.0003125 and increase the center vote by 0.000625, preserving total ensemble weight and all training behavior.
mechanism: Continued center-compensated horizontal-vote pruning
evidence_used: Two consecutive center-compensated horizontal reductions retained 9,284 correct while improving cross-entropy from 0.208133718 to 0.208133353 and then 0.208132999; continuing the same isolated direction is the closest evidence-backed probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.3833637079224, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813265075683593, "validation_score": 9284.413861838504}

RECENT RESULT
hypothesis: Holding vertical weight at 0.10875 while reducing each horizontal radius-one weight to 0.07000 and transferring their combined mass to the center will retain 9,284 correct predictions and lower cross-entropy below 0.208132651.
change: Advance one 0.0003125 step beyond Reference Design 2 along the validated horizontal-pruning direction, setting the center weight to 0.3625 while preserving total ensemble mass and all training behavior.
mechanism: Continued center-compensated horizontal-vote pruning
evidence_used: Three consecutive isolated horizontal-weight reductions retained 9,284 correct and monotonically improved cross-entropy from 0.208133718 to 0.208133353, 0.208132999, and 0.208132651; the regressions occurred when vertical influence was also increased.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.70478770881891, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2081323101043701, "validation_score": 9284.4138619552}

RECENT RESULT
hypothesis: Reducing each radius-one horizontal weight to 0.0696875 and transferring the combined mass to the center will retain 9,284 correct predictions while lowering cross-entropy below 0.208132310.
change: Advance one 0.0003125 step beyond Reference Design 3 along the validated horizontal-pruning direction, increasing the center weight to 0.363125 while preserving total ensemble mass and all training behavior.
mechanism: Continued center-compensated horizontal-vote pruning
evidence_used: Four consecutive center-compensated horizontal reductions retained 9,284 correct while monotonically lowering cross-entropy, most recently from 0.208132651 at weight 0.0703125 to 0.208132310 at 0.07000.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 56.234569374937564, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20813197937011718, "validation_score": 9284.413862068497}



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
