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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.40000991616398, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22445868225097657, "validation_score": 9206.408343709141}
prior_hypothesis: The verified per-view sharpening and 0.0325 BatchNorm EMA will recover 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.10× will reduce cross-entropy below 0.227083.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 65.12440524995327, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.23588926696777343, "validation_score": 9206.404566989426}
prior_hypothesis: A 0.0325 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by refining the interval between the best reported 0.03 rate and the slightly worse 0.04 rate while preserving the validated 0.015 parameter smoothing.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}
prior_hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble log-probabilities by 1.10–1.20 according to the top-two probability margin will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.
change: Replace fixed 1.10× outer sharpening with stronger sharpening for high-agreement predictions and retain 1.10× for ambiguous predictions.
mechanism: Margin-adaptive decision-preserving calibration
evidence_used: Increasing fixed outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and lowered cross-entropy from 0.227083 to 0.224459; margin conditioning tests further sharpening while limiting its effect on error-prone ambiguous examples.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing only the post-ensemble sharpening from 1.05× to 1.10× will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.227083 to approximately 0.224459.
change: Apply the strongest verified 1.10× outer calibration while retaining the current architecture, training procedure, EMA, and 1.05× per-view sharpening.
mechanism: Verified decision-preserving ensemble calibration
evidence_used: Reference Design 3 verified this exact configuration at 9,206 correct and 0.224459 cross-entropy, outperforming the current design’s tied correctness and 0.227083 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Per-view 1.05× sharpening will retain the verified 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.15× will reduce cross-entropy below 0.224459.
change: Sharpen each of the ten TTA views by 1.05× before probability averaging, then multiply the ensemble log-probabilities by 1.15×.
mechanism: Confidence-weighted TTA with stronger decision-preserving calibration
evidence_used: Reference Design 2 verified 9,206 correct and 0.224459 cross-entropy with 1.05× per-view and 1.10× outer sharpening; increasing the outer factor from 1.05× to 1.10× improved cross-entropy without changing predictions, while prior 1.15× attempts only timed out and supplied no contrary calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the strongest verified residual design’s classifier bottleneck from 48 to the maximum fitting width of 52 will exceed 9,206 correct predictions by improving separation of its 864-dimensional residual features without materially increasing runtime.
change: Restore Reference Design 2’s residual stages, phase-specific parameter/BatchNorm EMA, and calibrated TTA, while using 3,500 of the 4,182 remaining allowed parameters to widen the classifier to 52 units.
mechanism: Capacity-complete residual split-EMA ensemble
evidence_used: Reference Design 2 achieved the best completed score—9,206 correct with 0.224459 cross-entropy—at 245,818 parameters; its residual training system is therefore the qualified baseline, and the unused parameter budget supports a focused capacity test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing per-view sharpening from 1.05× to 1.10× will exceed 9,206 correct predictions by giving decisive TTA views more influence, while the argmax-invariant 1.10× outer calibration preserves cross-entropy gains.
change: Sharpen every original, flipped, and shifted TTA member by 1.10× before probability averaging; leave training and outer calibration unchanged.
mechanism: Stronger confidence-weighted probability TTA
evidence_used: The verified 1.05× per-view design achieved 9,206 correct with 0.224459 cross-entropy. The only 1.10× per-view attempt timed out, so its decision-changing effect remains untested and supplies no contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing only the outer sharpening to 1.10× will preserve all 9,206 predictions while reducing validation cross-entropy to approximately 0.224459.
change: Multiply the averaged TTA log-probabilities by 1.10 instead of 1.05.
mechanism: Verified decision-preserving ensemble calibration
evidence_used: Reference Design 3 verified this exact configuration at 9,206 correct and 0.224459 cross-entropy, improving upon the current design’s tied correctness and 0.227083 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying 1.05× per-view sharpening and 1.10× post-ensemble sharpening will preserve 9,206 correct predictions while reducing cross-entropy from 0.235889 toward the verified 0.224459.
change: Sharpen each of the ten TTA views before probability averaging, then sharpen the ensemble log-probabilities without changing their argmax.
mechanism: Calibrated confidence-weighted TTA
evidence_used: Reference Design 2 verified this exact residual, EMA, and TTA configuration at 9,206 correct and 0.224459 cross-entropy, outperforming the current tied-correctness result of 0.235889 cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the strongest completed reference configuration will increase validation correctness from 9,168 to approximately 9,206 and reduce cross-entropy from 0.239279 toward 0.224459.
change: Replace the plain convolutional stack with residual stages, apply phase-specific EMA to parameters and BatchNorm buffers, and use verified 1.05× per-view plus 1.10× post-ensemble sharpening.
mechanism: Residual split-EMA ensemble with calibrated probability TTA
evidence_used: Reference Design 2 completed within 72.4 seconds with 9,206 correct predictions and 0.224459 cross-entropy, the best verified score supplied; the current design lacks all three of its qualified changes and achieved only 9,168 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing post-ensemble sharpening from 1.10× to 1.125× will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.
change: Apply a midpoint calibration between the verified 1.10× factor and the repeatedly timed-out 1.15× proposal, leaving training and TTA aggregation unchanged.
mechanism: Conservative decision-preserving ensemble calibration
evidence_used: Raising outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459; the 1.15× attempts produced no performance evidence because verification timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A zero-initialized class-specific attention head over the final 3×3 feature map will exceed 9,206 correct predictions by learning where each class’s evidence occurs, while verified 1.10× outer calibration lowers tied-count cross-entropy.
change: Preserve the qualified residual training system, add a low-cost attention-pooled class-evidence path alongside the shared flattened classifier, and use the verified 1.10× post-ensemble sharpening.
mechanism: Parallel class-conditioned spatial evidence pooling
evidence_used: The 9,206-correct designs assume every prediction passes through one shared 48-unit flattened bottleneck. The late global-moment branch reached only 9,201, indicating uniform global summaries are insufficient; class-conditioned pooling retains spatial selection. Unlike the timed-out hierarchical attention design, this operates on only nine final tokens and adds just 1,940 parameters and one 1×1 convolution.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 10% hard view-vote consensus into the verified probability ensemble will exceed 9,206 correct predictions by correcting close decisions dominated by a few confident dissenting views.
change: Retain the verified 1.05× per-view and 1.10× outer sharpening, while accumulating top-class votes from the same ten TTA passes and blending them into the probability mean without adding parameters or model evaluations.
mechanism: Consensus-regularized probability TTA
evidence_used: The unsharpened current design and Reference Design 2 both achieved exactly 9,206 correct despite per-view sharpening, showing confidence scaling alone did not improve decisions; Reference Design 2’s 0.224459 cross-entropy supports retaining its calibration while testing complementary view agreement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the exact best completed reference configuration will increase validation correctness from 9,168 toward 9,206 and reduce cross-entropy toward 0.224459.
change: Replace the plain stack with residual stages, EMA both parameters and BatchNorm buffers at phase-specific rates, and apply verified 1.05× per-view plus 1.10× post-ensemble sharpening.
mechanism: Verified residual split-EMA ensemble with calibrated probability TTA
evidence_used: Reference Design 2 completed in 72.4 seconds with the highest supplied score: 9,206 correct and 0.224459 cross-entropy; later alternatives either timed out or could not be verified.
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
