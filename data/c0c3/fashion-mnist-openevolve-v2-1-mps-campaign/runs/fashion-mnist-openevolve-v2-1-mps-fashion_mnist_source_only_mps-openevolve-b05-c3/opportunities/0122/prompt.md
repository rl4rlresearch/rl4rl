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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 65.12440524995327, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.23588926696777343, "validation_score": 9206.404566989426}
prior_hypothesis: A 0.0325 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by refining the interval between the best reported 0.03 rate and the slightly worse 0.04 rate while preserving the validated 0.015 parameter smoothing.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22455829195678, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.23927925605773925, "validation_score": 9168.403460315789}
prior_hypothesis: On the verified 19/32 curriculum, retaining final-phase BatchNorm statistics while averaging only learned parameters will exceed 9,172 correct predictions by eliminating normalization-statistic lag across the augmentation shift.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.40000991616398, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22445868225097657, "validation_score": 9206.408343709141}
prior_hypothesis: The verified per-view sharpening and 0.0325 BatchNorm EMA will recover 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.10× will reduce cross-entropy below 0.227083.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing fixed max pooling with locally gated interpolation between mean and maximum responses will exceed 9,206 correct predictions by preserving distributed stroke evidence without sacrificing salient features; verified 1.05× sharpening will improve cross-entropy if correctness ties.
change: Challenge the shared assumption that one hard-coded pooling statistic suits every feature and location. Each residual stage instead learns a depthwise, image-conditioned 2×2 pooling gate, initialized near max pooling; retain the validated evaluation sharpening.
mechanism: Content-adaptive mixed-statistic downsampling
evidence_used: Residual refinement improved correctness from 9,168 to 9,206, while the late global-moment branch reached only 9,201, indicating that richer information should be integrated within the spatial hierarchy. The extensive translation augmentation and shifted-view ensemble further identify downsampling sensitivity as load-bearing. Reference Design 3 verified that 1.05× sharpening preserves argmax predictions while lowering cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the fully verified Reference Design 3 will increase validation correctness from 9,205 to approximately 9,206 and reduce cross-entropy from 0.235937 to approximately 0.230772.
change: Set the terminal BatchNorm-buffer EMA rate to 0.0325 and sharpen evaluation-time ensemble log-probabilities by 1.05×.
mechanism: Qualified split-EMA with decision-preserving TTA sharpening
evidence_used: Reference Design 3 verified this exact combination at 9,206 correct and 0.230772 cross-entropy, the strongest completed validation_score; subsequent attempts produced no completed contrary evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring verified Reference Design 3 will raise validation correctness from 9,168 to approximately 9,206 and reduce cross-entropy to approximately 0.230772.
change: Replace the plain stack with the qualified residual architecture, EMA learned parameters and BatchNorm buffers at validated phase-specific rates, and apply 1.05× evaluation sharpening.
mechanism: Residual feature refinement with split-rate terminal EMA and calibrated TTA
evidence_used: Reference Design 3 achieved the strongest completed validation_score, with 9,206 correct predictions and 0.230772 cross-entropy; later unverified attempts provide no contrary performance evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.
change: Apply 1.05× sharpening independently to every original, flipped, and shifted TTA member before averaging their probabilities.
mechanism: Confidence-weighted probability TTA
evidence_used: Post-ensemble 1.05× sharpening preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772, demonstrating underconfidence; moving the same signal inside the ensemble can additionally alter ambiguous decisions toward confident views.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}

RECENT RESULT
hypothesis: Retaining verified 1.05× per-view sharpening and increasing only the decision-preserving post-ensemble factor to 1.10× will preserve 9,206 correct predictions while reducing validation cross-entropy below 0.227083.
change: Sharpen every TTA member by 1.05× before probability averaging, then sharpen the ensemble log-probabilities by 1.10×.
mechanism: Two-stage TTA confidence sharpening
evidence_used: Per-view plus post-ensemble 1.05× sharpening preserved 9,206 correct and improved cross-entropy from 0.235889 to 0.227083; the remaining underconfidence motivates a conservative increase to the argmax-invariant outer factor.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The verified per-view sharpening and 0.0325 BatchNorm EMA will recover 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.10× will reduce cross-entropy below 0.227083.
change: Apply 1.05× sharpening before averaging each TTA view, sharpen the ensemble log-probabilities by 1.10×, and use the validated 0.0325 terminal BatchNorm-buffer EMA rate.
mechanism: Two-stage TTA confidence sharpening with split-rate EMA
evidence_used: Reference Design 3 achieved 9,206 correct and 0.227083 cross-entropy with 1.05× per-view and outer sharpening. Increasing only the outer factor preserves every ensemble argmax; its prior verification timed out and supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.40000991616398, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22445868225097657, "validation_score": 9206.408343709141}

RECENT RESULT
hypothesis: Retaining the verified 1.05× per-view sharpening while increasing the argmax-invariant outer factor from 1.10× to 1.15× will preserve 9,206 correct predictions and reduce validation cross-entropy below 0.224459.
change: Restore the qualified residual architecture and phase-specific parameter/BatchNorm EMA, apply 1.05× sharpening to each TTA view, and sharpen the averaged log-probabilities by 1.15×.
mechanism: Residual split-EMA with stronger ensemble calibration
evidence_used: Reference Design 2 preserved 9,206 correct while increasing outer sharpening from 1.05× to 1.10× reduced cross-entropy from 0.227083 to 0.224459; another conservative 0.05 increment tests the still-improving calibration direction without changing ensemble argmaxes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the outer sharpening factor from 1.10× to 1.15× will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.
change: Increase only the post-ensemble log-probability multiplier; training and per-view TTA remain unchanged.
mechanism: Stronger decision-preserving ensemble calibration
evidence_used: Raising outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459; the prior 1.15× attempt timed out and supplied no contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing only the outer sharpening factor from 1.05× to 1.15× will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.
change: Sharpen the averaged TTA log-probabilities by 1.15×; training and per-view sharpening remain unchanged.
mechanism: Stronger decision-preserving ensemble calibration
evidence_used: Raising the outer factor from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459. Prior 1.15× attempts timed out during training, which provides no contrary calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing per-view sharpening from the verified 1.05× to 1.10× will exceed 9,206 correct predictions by giving decisive TTA views more influence, while 1.10× outer sharpening will improve cross-entropy without changing the ensemble argmax.
change: Sharpen every TTA member by 1.10× before probability averaging and apply the verified 1.10× post-ensemble calibration.
mechanism: Stronger confidence-weighted probability TTA
evidence_used: Reference Design 2 verified that 1.05× per-view sharpening with 1.10× outer sharpening preserves 9,206 correct predictions and improves cross-entropy to 0.224459; repeated 1.15× outer-only attempts timed out, so changing the per-view factor directly tests a remaining route to higher correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Image-conditioned channel and spatial gating of residual refinements will exceed 9,206 correct predictions by suppressing irrelevant texture and emphasizing class-defining shape regions, while retaining the qualified split EMA and calibrated TTA.
change: Replace the static convolutional stack with residual stages whose refinement paths receive learned channel-and-spatial attention, initialized to reproduce ordinary residual addition; retain nearly all classifier capacity with a 47-unit head and use the best verified EMA and TTA calibration.
mechanism: Baseline-preserving channel-spatial residual routing
evidence_used: Residual refinement improved correctness from 9,168 to 9,206, but all qualified designs assume every image uses the same refinement computation. The late global-moment branch reached only 9,201, suggesting global context is more useful when it modulates features inside the spatial hierarchy; the zero-initialized lightweight gates test that alternative without the runtime cost of the failed adaptive-pooling design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble log-probabilities by 1.10–1.20 according to the top-two probability margin will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.
change: Replace fixed 1.10× outer sharpening with stronger sharpening for high-agreement predictions and retain 1.10× for ambiguous predictions.
mechanism: Margin-adaptive decision-preserving calibration
evidence_used: Increasing fixed outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and lowered cross-entropy from 0.227083 to 0.224459; margin conditioning tests further sharpening while limiting its effect on error-prone ambiguous examples.
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
