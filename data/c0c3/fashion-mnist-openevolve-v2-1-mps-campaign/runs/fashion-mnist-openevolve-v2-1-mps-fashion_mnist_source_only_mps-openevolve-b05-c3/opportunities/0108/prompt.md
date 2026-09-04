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
hypothesis: Scaling the qualified split-EMA ensemble logits by 1.0625 will preserve 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Restore the verified residual topology and split parameter/BatchNorm EMA, then increase evaluation-only sharpening from 1.05× to 1.0625×.
mechanism: Conservative decision-preserving TTA temperature refinement
evidence_used: The qualified 1.05× design preserved 9,206 correct and lowered cross-entropy from 0.235889 to 0.230772; 1.0625× is a conservative untested step toward the timed-out 1.075× trials.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble log-probabilities by 1.10 will preserve exactly 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Increase only the evaluation-time sharpening factor from 1.05 to 1.10.
mechanism: Stronger decision-preserving TTA temperature sharpening
evidence_used: The verified 1.05× scale preserved 9,206 correct and reduced cross-entropy from 0.235889 to 0.230772; the prior 1.10× attempt timed out and provided no performance evidence against the stronger calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation-time ensemble log-probabilities by 1.055 will preserve 9,206 correct predictions while reducing validation cross-entropy below 0.230772.
change: Apply an untested 1.055× evaluation-only sharpening factor while leaving training, architecture, EMA, and predicted classes unchanged.
mechanism: Conservative decision-preserving logit sharpening
evidence_used: The verified 1.05× factor preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772; 1.055× is a smaller unexplored refinement than the timed-out 1.0625× and 1.075× attempts.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Combining the reported 0.03 BatchNorm-buffer EMA rate with 1.05× evaluation sharpening will retain 9,206 correct predictions while reducing cross-entropy below 0.230772.
change: Change terminal BatchNorm-buffer EMA from 0.04 to 0.03 and sharpen the final ensemble log-probabilities by 1.05.
mechanism: Best-observed BatchNorm EMA with decision-preserving TTA sharpening
evidence_used: A 0.03 buffer rate reportedly achieved 9,206 correct with lower unsharpened cross-entropy than 0.0325, while 1.05× sharpening preserved 9,206 correct and reduced cross-entropy from 0.235889 to 0.230772.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding image-conditioned channel recalibration after each residual stage will exceed 9,206 correct predictions by using global context to reshape intermediate spatial features, while identity initialization preserves the qualified residual model’s starting behavior.
change: Restore the strongest residual architecture, split parameter/BatchNorm EMA, and 1.05× sharpening; add lightweight squeeze-excitation modules within the 250,000-parameter limit.
mechanism: Identity-initialized hierarchical channel attention
evidence_used: The residual design reached 9,206 correct versus the current plain network’s 9,168. The late global-moment branch fell to 9,201, suggesting global statistics are more informative when they modulate hierarchical feature extraction instead of acting as an additive final classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Disabling label smoothing during the final 13/32 of training will exceed 9,206 correct predictions by preserving early regularization while strengthening true-class margins during the cleaner augmentation and faster-EMA phase.
change: Retain 0.02 label smoothing during broad-shift training, then switch to ordinary cross-entropy when the terminal augmentation phase begins.
mechanism: Terminal hard-label fine-tuning
evidence_used: The qualified model remained underconfident—1.05× evaluation sharpening preserved 9,206 predictions while lowering cross-entropy from 0.235889 to 0.230772—and repeated EMA/calibration refinements did not increase correct count. Terminal hard-label optimization targets learned decision margins while keeping the validated early regularization and evaluation ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the ensemble log-probabilities by 1.05 will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.235889 to approximately 0.230772.
change: Apply the verified 1.05× evaluation-only sharpening factor without changing training or predicted classes.
mechanism: Decision-preserving TTA temperature sharpening
evidence_used: Reference Design 3 used this exact change and retained 9,206 correct predictions while improving validation_score from 9206.404567 to 9206.406249.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the fully verified Reference Design 3 will increase correct predictions from 9,205 to 9,206 and reduce validation cross-entropy from 0.235937 to approximately 0.230772.
change: Use the validated 0.0325 terminal BatchNorm-buffer EMA rate and multiply evaluation-time ensemble log-probabilities by 1.05.
mechanism: Qualified split-EMA with decision-preserving TTA sharpening
evidence_used: Reference Design 3 verified this exact combination at 9,206 correct and 0.230772 cross-entropy, outperforming the current design’s 9,205 correct and 0.235937 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the fully verified Reference Design 3 will improve the current model from 9,168 to approximately 9,206 correct predictions and reduce cross-entropy from 0.239279 to approximately 0.230772.
change: Replace the plain convolutional stack with the qualified residual architecture, EMA both parameters and BatchNorm buffers at validated phase-specific rates, and sharpen evaluation ensemble log-probabilities by 1.05×.
mechanism: Residual feature refinement with split parameter/BatchNorm EMA and calibrated TTA
evidence_used: Reference Design 3 verified this exact implementation at 9,206 correct and 0.230772 cross-entropy; subsequent unsuccessful attempts produced no completed verification evidence against it.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Linearly reducing label smoothing from 0.02 to zero during the final 13/32 training phase will exceed 9,206 correct predictions by strengthening true-class margins while retaining early regularization.
change: Keep 0.02 label smoothing during broad translation training, then anneal it to zero alongside the cleaner terminal augmentation phase.
mechanism: Terminal label-smoothing annealing
evidence_used: The verified 1.05× sharpening retained 9,206 correct while reducing cross-entropy from 0.235889 to 0.230772, indicating underconfidence; the prior abrupt hard-label trial timed out and provided no contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the qualified ensemble log-probabilities by 1.05 will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.235889 to approximately 0.230772.
change: Multiply only the evaluation-time ensemble log-probabilities by 1.05, leaving training and predicted classes unchanged.
mechanism: Decision-preserving TTA temperature sharpening
evidence_used: Reference Design 3 verified this exact change at 9,206 correct and 0.230772 cross-entropy, outperforming the current design’s tied correct count and 0.235889 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring Reference Design 3 will increase validation correctness from 9,205 to approximately 9,206 and reduce cross-entropy from 0.235937 to approximately 0.230772.
change: Set terminal BatchNorm-buffer EMA to 0.0325 and sharpen ensemble log-probabilities by 1.05×.
mechanism: Qualified split-EMA with decision-preserving TTA sharpening
evidence_used: Reference Design 3 verified this exact combination at 9,206 correct and 0.230772 cross-entropy, the best completed validation_score reported.
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
