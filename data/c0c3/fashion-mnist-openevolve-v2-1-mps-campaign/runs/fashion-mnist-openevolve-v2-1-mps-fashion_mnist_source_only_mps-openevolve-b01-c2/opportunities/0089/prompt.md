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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.4581887088716, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19624986610412598, "validation_score": 9330.41797287855}
prior_hypothesis: Scaling flip-ensemble logits by 1.184 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.72896141698584, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Restoring the verified 5% warmup and 1.15 evaluation scale will increase validation correctness from 9,315 to 9,330 and reduce cross-entropy toward 0.196639.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.43989566690288, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19624986610412598, "validation_score": 9330.41797287855}
prior_hypothesis: Scaling evaluation logits by 1.184 will preserve all 9,330 argmax predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196250.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.97804520791396, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Restoring the linear ensemble curriculum and increasing evaluation scaling to 1.15 will retain all 9,330 argmax predictions while reducing validation cross-entropy below 0.198018.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.99-decay EMA of learned parameters will exceed 9,330 correct predictions by smoothing the late cosine-schedule trajectory without adding learned parameters or model forwards.
change: Maintain a parameter EMA after each optimizer step and substitute the averaged weights when the model enters evaluation mode.
mechanism: Tail-weight exponential averaging
evidence_used: The verified 249,393-parameter design reaches 9,330 correct, while architecture and augmentation changes repeatedly timed out; EMA preserves that exact training procedure and adds only lightweight parameter averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Linearly reducing label smoothing from 0.02 to zero will exceed 9,330 correct predictions by retaining early regularization while aligning late updates with the hard-label accuracy objective, without increasing runtime.
change: Reuse training progress to anneal label smoothing in both ensemble and per-view cross-entropy losses.
mechanism: Annealed label smoothing
evidence_used: The current fixed-smoothing design verified 9,330 correct in 77.44 seconds, while added augmentation, consistency, architecture, and EMA work repeatedly timed out; this isolates a zero-overhead loss-curriculum change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the positive scale applied to evaluation-time symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 independently verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current design’s tie-breaker without changing argmax predictions, parameters, or training computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve all 9,330 argmax-correct predictions while reducing validation cross-entropy to approximately 0.196250.
change: Increase only the positive scale applied to evaluation-time symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1 and 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current tie-breaker without changing training or predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Using hard labels for the increasingly dominant ensemble loss while retaining 0.02 smoothing for individual views will exceed 9,330 correct predictions without adding computation.
change: Remove label smoothing only from the ensemble loss, so the existing curriculum progressively shifts training toward the hard-label validation objective.
mechanism: Curriculum-coupled hard ensemble supervision
evidence_used: The best verified design reached 9,330 correct with ensemble weight rising from 0.5 to 1.0. Annealed smoothing targeted the same late hard-label alignment but timed out without contrary validation evidence; this change realizes that idea through the existing loss mixture.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.02 to 0.01 will exceed 9,330 correct predictions by improving hard-label class separation while retaining mild regularization.
change: Halve label smoothing for both ensemble and individual-view cross-entropy losses without changing computation, architecture, or scheduling.
mechanism: Reduced uniform label smoothing
evidence_used: The best verified design reaches 9,330 correct with fixed 0.02 smoothing; prior hard-label and annealed-smoothing experiments timed out rather than producing contrary validation evidence, motivating a conservative midpoint test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the positive evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current design’s tie-breaker without changing training, parameters, or argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy to approximately 0.196250.
change: Increase only the positive evaluation-time scale applied to symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1 and 2 verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current design’s tie-breaker without changing training cost, parameters, or argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Initializing the refinement branch’s final BatchNorm scale to zero will stabilize early optimization and increase validation_correct from 9,330 to at least 9,331 without increasing parameters or runtime.
change: Start the residual refinement block as an identity mapping while allowing it to learn progressively during the fixed training budget.
mechanism: Zero-initialized residual refinement
evidence_used: The best verified design reaches 9,330 correct, while added-computation architecture, augmentation, consistency, and EMA experiments repeatedly timed out; this standard residual initialization changes neither computation nor parameter count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a 0.05-weight margin penalty to the ensemble logits will increase validation correctness beyond 9,330 by directly correcting low-margin decisions while preserving the verified training procedure and calibrated evaluation scale.
change: Supplement the existing smoothed cross-entropy mixture with a hinge penalty requiring the true-class ensemble logit to exceed the strongest competing logit by 0.5.
mechanism: Low-weight hardest-rival margin supervision
evidence_used: The current design already achieves the best verified calibration at 9,330 correct; further temperature scaling cannot change its argmax predictions, while prior hard-label experiments timed out without contrary accuracy evidence. A lightweight margin term targets the primary correctness metric without additional model forwards.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy to approximately 0.196250.
change: Increase only the positive evaluation-time scale applied to symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current score without changing training computation, parameters, or argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the positive evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1 and 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, the best available validation score and strictly better than the current 1.15 scale at identical correctness.
result: the implementation could not be verified



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
