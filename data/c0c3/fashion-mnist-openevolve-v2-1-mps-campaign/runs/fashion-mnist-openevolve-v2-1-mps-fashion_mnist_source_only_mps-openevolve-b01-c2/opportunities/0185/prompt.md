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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.65318641695194, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962260711669922, "validation_score": 9331.41798119273}
prior_hypothesis: Increasing the verified bounded adjustment from 0.02 to 0.025 will exceed 9,331 correct predictions or retain 9,331 while reducing cross-entropy below 0.19623047790527343.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.47602354199626, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}
prior_hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150985717774.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 79.2271575408522, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962260711669922, "validation_score": 9331.41798119273}
prior_hypothesis: The qualified 0.025 disagreement-gated adjustment will retain 9,331 correct predictions while reducing validation cross-entropy from 0.19625150680541992 to approximately 0.1962260711669922.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.40315091586672, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515151977539, "validation_score": 9331.417972302352}
prior_hypothesis: A 0.5085078125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151824951173.

## Recent verification evidence

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

RECENT RESULT
hypothesis: An adjustment of 0.02625 will retain 9,331 correct predictions while reducing validation cross-entropy below 0.1962260711669922.
change: Restore the verified adaptive fusion rule and increase its bounded higher-margin-view adjustment from 0.025 to 0.02625.
mechanism: Conservative interpolation of disagreement-gated margin-adaptive flip fusion
evidence_used: Increasing the adjustment from 0.02 to 0.025 improved cross-entropy while preserving 9,331 correct; the reported 0.0275 run further reduced cross-entropy to 0.1962239959716797 with the same correctness, motivating a conservative interpolation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.0275 disagreement-gated adjustment will retain 9,331 correct predictions while reducing validation cross-entropy below the best qualified result of 0.1962260711669922.
change: Replace fixed flip fusion with the reported 0.0275 bounded adjustment toward the higher-margin view for decisive original/flip disagreements.
mechanism: Stronger high-disparity margin-adaptive flip fusion
evidence_used: The reported 0.0275 run retained 9,331 correct and reached 0.1962239959716797 cross-entropy, improving on the qualified 0.025 rule’s 0.1962260711669922; its timeout was associated with training duration rather than degraded validation results.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the gate threshold from 0.5 to 0.45 will exceed 9,331 correct predictions or retain 9,331 while reducing cross-entropy below 0.1962260711669922.
change: Extend the verified 0.025 higher-margin-view adjustment to disagreements with margin disparities between 0.45 and 0.5 logits.
mechanism: Moderately broadened high-disparity view fusion
evidence_used: The 0.5-threshold rule improved cross-entropy while preserving 9,331 correct; the 0.4-threshold run timed out without contrary validation evidence, motivating a conservative midpoint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.025 bounded adjustment toward the higher-margin view on decisive original/flip disagreements will retain 9,331 correct predictions while reducing validation cross-entropy below 0.19625150680541992.
change: Replace fixed evaluation-time flip fusion with the best qualified disagreement-gated adaptive fusion rule; training remains unchanged.
mechanism: Verified high-disparity margin-adaptive flip fusion
evidence_used: Reference Design 1 verified 9,331 correct with 0.1962260711669922 cross-entropy, improving on the current fixed fusion’s 0.19625150680541992 at identical correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The qualified 0.025 disagreement-gated adjustment will retain 9,331 correct predictions while reducing validation cross-entropy from 0.19625150680541992 to approximately 0.1962260711669922.
change: Replace fixed evaluation-time flip weighting with the best verified adaptive rule, shifting weight toward the higher-margin view only for decisive original/flip disagreements.
mechanism: Verified high-disparity margin-adaptive flip fusion
evidence_used: Reference Design 2 achieved the same 9,331 correct predictions as the current design while improving cross-entropy to 0.1962260711669922 and validation_score to 9331.41798119273.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 79.2271575408522, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962260711669922, "validation_score": 9331.41798119273}

RECENT RESULT
hypothesis: The reported 0.0275 adaptive adjustment will retain 9,331 correct predictions, reduce cross-entropy below 0.1962260711669922, and complete verification when the equivalent per-view losses are consolidated.
change: Apply the 0.0275 higher-margin-view adjustment during decisive flip disagreements and compute both per-view losses with one cross-entropy call.
mechanism: Efficient stronger disagreement-gated flip fusion
evidence_used: A prior 0.0275 run reported 9,331 correct and 0.1962239959716797 cross-entropy; its failure was attributed to runtime, while combining the two equal-sized view losses is mathematically equivalent and reduces training overhead.
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
