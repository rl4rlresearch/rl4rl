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
hypothesis: Raising the initial ensemble-loss weight from 0.5 to 0.75 will exceed 9,330 correct predictions by aligning more of the fixed training budget with the flip-ensemble used for validation, without increasing runtime.
change: Change the ensemble-loss curriculum from 0.5→1.0 to 0.75→1.0 while retaining individual-view supervision early in training.
mechanism: Front-loaded ensemble-objective curriculum
evidence_used: The best verified design reaches 9,330 correct as ensemble supervision becomes dominant; this conservatively tests earlier validation-objective alignment, while recent more complex loss changes provided no completed contrary result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying one-pixel translations to approximately 47% of training examples will exceed 9,330 correct predictions by improving local translation invariance without adding model forwards or parameters.
change: Randomly select identity or one of eight one-pixel crops for each training image, with identity retained approximately 53% of the time.
mechanism: Conservative per-image translation augmentation
evidence_used: The best verified design already achieves 9,330 correct with calibrated flip ensembling, while subsequent evaluation-fusion and loss modifications produced no completed contrary metrics; a lightweight spatial augmentation targets a distinct remaining invariance without changing the verified architecture or objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the positive evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 independently achieved the best reported validation score with this exact change; recent verification failures produced no contrary metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the evaluation-time scale applied to symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1 and 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, the best available score; later attempts produced no contrary validation metrics.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Softly favoring the more confident orientation at evaluation will correct at least one borderline disagreement and increase validation_correct above 9,330 without changing training or parameters.
change: Replace equal flip-logit averaging with symmetric confidence-weighted averaging while retaining the verified 1.184 calibration scale.
mechanism: Confidence-adaptive symmetric flip fusion
evidence_used: The current equal-weight ensemble verifies 9,330 correct, while temperature scaling cannot change argmax predictions; prior alternative fusion attempts produced no completed metrics, so adaptive view weighting is an untested, training-cost-free way to target correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Halving label smoothing from 0.02 to 0.01 will exceed 9,330 correct predictions by strengthening class-boundary gradients while retaining mild regularization.
change: Use 0.01 label smoothing for both ensemble and individual-view cross-entropy losses, with no added computation.
mechanism: Reduced label-smoothing bias
evidence_used: The current design verifies 9,330 correct at 0.02 smoothing, while evaluation-temperature changes cannot alter correctness and hard-label attempts produced no completed contrary evidence; this tests a conservative intermediate objective at identical runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 independently verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, the best available validation score; subsequent non-completions provide no contrary metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy to approximately 0.196250.
change: Increase only the evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1 and 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, outperforming the current design’s tie-breaker; subsequent attempts yielded no contrary validation metrics.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 0.995-decay EMA of the trained parameters will exceed 9,330 correct predictions by reducing variance in the final low-learning-rate iterates without changing the architecture, examples, or model-forward workload.
change: Maintain an EMA inside the optimizer and install the averaged parameters after the final optimizer step; retain the verified training objective and 1.184 evaluation calibration.
mechanism: Late-iterate exponential weight averaging
evidence_used: The current/reference design already verifies the best available score at 9,330 correct with a cosine-decayed learning rate; averaging its late optimization trajectory is a distinct, low-overhead way to improve correctness after evaluation-fusion and loss tweaks yielded no completed contrary metrics.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to 0 over training will exceed 9,330 correct predictions by retaining early regularization while strengthening late class-boundary fitting.
change: Compute training progress once and linearly anneal label smoothing for both ensemble and individual-view losses.
mechanism: Linear label-smoothing decay
evidence_used: The best verified design uses constant 0.02 smoothing and reaches 9,330 correct; the fixed 0.01 experiment did not complete, so a zero-cost curriculum tests reduced late-stage smoothing without abandoning the verified early objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reduce validation cross-entropy to approximately 0.196250.
change: Increase only the evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 independently verified 1.184 at 9,330 correct and 0.196249866 cross-entropy, the best reported validation score; later non-completions provide no contrary metric evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits to 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the evaluation-time scale applied to the symmetric flip-ensemble logits.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1 and 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, the highest available validation score; subsequent verification non-completions provide no contrary metric evidence.
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
