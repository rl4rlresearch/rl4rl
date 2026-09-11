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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.05727233295329, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}
prior_hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below the current 0.19625152359008788.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.47602354199626, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}
prior_hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150985717774.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.65318641695194, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962260711669922, "validation_score": 9331.41798119273}
prior_hypothesis: Increasing the verified bounded adjustment from 0.02 to 0.025 will exceed 9,331 correct predictions or retain 9,331 while reducing cross-entropy below 0.19623047790527343.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.40315091586672, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515151977539, "validation_score": 9331.417972302352}
prior_hypothesis: A 0.5085078125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151824951173.

## Recent verification evidence

RECENT RESULT
hypothesis: Linearly removing label smoothing during the final quarter of training, from the best verified inference fusion baseline, will exceed 9,331 correct predictions or retain 9,331 while lowering cross-entropy below 0.19625150680541992.
change: Restore the best verified complementary flip-fusion coefficients and anneal label smoothing from 0.02 to zero over the final 25% of optimization.
mechanism: Late hard-target annealing
evidence_used: The 0.5085009765625 flipped-view weight achieved the best score twice, while fusion, calibration, EMA, probability fusion, and translation experiments failed to improve correctness; late target annealing tests an untried training-objective dimension while preserving the established representation-learning regime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 62.104978542076424, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.1954039577484131, "validation_score": 9321.418268650325}

RECENT RESULT
hypothesis: Raising constant label smoothing from 0.02 to 0.025 will exceed 9,331 correct predictions by improving generalization, while restoring the best verified flip-fusion coefficients.
change: Restore the independently verified inference fusion weight and slightly increase label smoothing for both ensemble and individual-view losses.
mechanism: Slightly stronger uniform target regularization
evidence_used: Annealing smoothing toward zero reduced correctness from 9,331 to 9,321 while lowering cross-entropy, indicating that weaker target regularization trades away the primary ranking metric; a modest increase tests the favorable direction without changing runtime or architecture.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 84.68959354120307, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19586089096069337, "validation_score": 9325.418108831704}

RECENT RESULT
hypothesis: Slightly increasing temperature for high-margin predictions and decreasing it for low-margin predictions will preserve all 9,331 argmax predictions while lowering validation cross-entropy below 0.19625150680541992.
change: Replace the fixed evaluation scale with a batch-centered, margin-conditioned positive scale while retaining the best verified flip-fusion weights and unchanged training.
mechanism: Rank-preserving inverse-margin calibration
evidence_used: Both adjacent global scales worsened cross-entropy without changing correctness, indicating that 1.184 is locally optimal globally; the prior confidence-adaptive experiment timed out, leaving conditional calibration untested, while inverse-margin scaling specifically softens costly confident errors and strengthens uncertain correct predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adjusting fusion only when the original and flipped views disagree, slightly favoring the view with the larger top-two margin, will exceed 9,331 correct predictions while leaving all agreeing-view decisions and training unchanged.
change: Retain the best verified global fusion weights and calibration, but add a bounded per-image fusion adjustment based on the difference between each view’s top-two logit margin.
mechanism: Disagreement-gated margin-adaptive view fusion
evidence_used: Fixed fusion and calibration have saturated at 9,331 correct, while probability-space fusion fell to 9,329; this tests per-example view reliability without normalizing away logit-scale evidence or perturbing training.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: On images where the original and flipped views predict different classes, modestly favoring the view with a substantially larger top-two margin will exceed 9,331 correct predictions while preserving the best verified fusion exactly for all other images.
change: Restore the independently verified global fusion coefficients, then apply a bounded two-percentage-point weight adjustment only for disagreeing views whose margin difference exceeds 0.5 logits.
mechanism: High-disparity disagreement-gated view fusion
evidence_used: The best fixed fusion reached 9,331 correct twice, whereas probability-space fusion fell to 9,329; this motivates preserving raw-logit fusion and changing only high-confidence disagreements. The prior adaptive-fusion verification timed out, so its accuracy effect remains unmeasured.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.31558241695166, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19623047790527343, "validation_score": 9331.417979652948}

RECENT RESULT
hypothesis: Increasing the verified disagreement-gated adjustment from 0.02 to 0.03 will exceed 9,331 correct predictions, or retain 9,331 while lowering cross-entropy below 0.19623047790527343.
change: Restore the best verified base fusion weight and apply a three-percentage-point bounded adjustment toward the higher-margin view only when view predictions disagree and their margin difference exceeds 0.5 logits.
mechanism: Stronger high-disparity margin-adaptive view fusion
evidence_used: Reference Design 2’s 0.02 gated adjustment retained 9,331 correct while improving cross-entropy from 0.19625150680541992 to 0.19623047790527343, establishing a favorable direction and motivating a controlled increase in its magnitude.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the verified bounded adjustment from 0.02 to 0.025 will exceed 9,331 correct predictions or retain 9,331 while reducing cross-entropy below 0.19623047790527343.
change: Preserve the verified fusion threshold and base weight while moving halfway toward the unmeasured 0.03 adjustment.
mechanism: Intermediate-strength disagreement-gated view fusion
evidence_used: The 0.02 adjustment retained 9,331 correct and improved cross-entropy over fixed fusion; the 0.03 run timed out, so 0.025 is a controlled interpolation along the established favorable direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.65318641695194, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962260711669922, "validation_score": 9331.41798119273}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Raising the verified bounded adjustment from 0.025 to 0.0275 will exceed 9,331 correct predictions or retain 9,331 while reducing validation cross-entropy below 0.1962260711669922.
change: Restore the best verified adaptive fusion rule and move halfway from its successful 0.025 adjustment toward the unmeasured 0.03 setting.
mechanism: Intermediate extrapolation of disagreement-gated margin-adaptive view fusion
evidence_used: Increasing the adjustment from 0.02 to 0.025 retained 9,331 correct while lowering cross-entropy from 0.19623047790527343 to 0.1962260711669922; the 0.03 verification timed out without contrary accuracy evidence.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 83.10226362501271, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962239959716797, "validation_score": 9331.417981917837}

RECENT RESULT
hypothesis: Raising the adjustment from 0.025 to 0.03 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962260711669922.
change: Increase only the bounded adjustment toward the higher-margin view for decisive original/flip disagreements.
mechanism: Stronger disagreement-gated margin-adaptive view fusion
evidence_used: Increasing the adjustment from 0.02 to 0.025 improved cross-entropy from 0.19623047790527343 to 0.1962260711669922, and the reported 0.0275 run preserved 9,331 correct while further lowering it to 0.1962239959716797; the earlier 0.03 run timed out without contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the verified gate threshold from 0.5 to 0.4 will exceed 9,331 correct predictions, or retain 9,331 while reducing cross-entropy below 0.1962260711669922.
change: Restore the verified 0.025 adaptive fusion rule while extending its small higher-margin-view adjustment to original/flip disagreements with margin disparities between 0.4 and 0.5 logits.
mechanism: Moderately broadened disagreement-gated margin-adaptive flip fusion
evidence_used: The verified 0.025 rule at threshold 0.5 retained 9,331 correct and improved cross-entropy to 0.1962260711669922; the reported 0.0275 result further reduced it to 0.1962239959716797 without changing correctness, supporting cautious expansion of the same view-reliability signal.
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
