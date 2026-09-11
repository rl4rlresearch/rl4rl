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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.44132479187101, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656433105468, "validation_score": 9348.42097940267}
prior_hypothesis: An additional 1/1048576 vertical-weight transfer beyond Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.27154212491587, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877065631866455, "validation_score": 9348.420979403076}
prior_hypothesis: An additional 1/524288 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656623840332.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.49035570817068, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656509399414, "validation_score": 9348.420979402401}
prior_hypothesis: A half-step vertical-weight transfer from Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 70.80946620786563, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770658378601074, "validation_score": 9348.420979395774}
prior_hypothesis: An additional 1/32768 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066032409668.

## Recent verification evidence

RECENT RESULT
hypothesis: Learning a separate max-versus-average pooling mixture for each feature channel will exceed 9,348 correct predictions by reducing translation-sensitive aliasing while retaining the spatial grid required by the classifier.
change: Replace both hard max-pooling layers with channel-wise learned mixtures initialized near max pooling. This challenges the old assumption that one fixed downsampling statistic suits every feature channel; the new approach lets edge channels preserve peaks while broader shape channels learn smoother aggregation.
mechanism: Channel-adaptive mixed-statistic downsampling
evidence_used: Translation TTA improved correctness from 9,344 to 9,348, indicating that one-pixel shift sensitivity remains load-bearing. Global pooling fell to 9,085, so this patch targets downsampling instability without discarding spatial organization. Its 96 additional learned parameters bring the verified 249,881-parameter model to 249,977, below the ceiling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249977, "training_seconds": 50.456144500058144, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.188493701171875, "validation_score": 9330.420700588911}

RECENT RESULT
hypothesis: An additional 1/16384 vertical-weight transfer beyond Reference Design 3 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066421508789.
change: Restore the best verified crop ensemble and transfer 1/16384 weight from the first vertical crop to the second, preserving center, horizontal, and total weights.
mechanism: Conservative directional vertical-shift TTA extrapolation
evidence_used: Successive directional transfers of 1/1024, 1/4096, and 1/8192 all retained 9,348 correct while monotonically lowering cross-entropy; halving the next increment is the lowest-risk continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 60.31418062490411, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877066032409668, "validation_score": 9348.420979388879}

RECENT RESULT
hypothesis: An additional 1/32768 vertical-weight transfer beyond Reference Design 3 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066032409668.
change: Restore the best verified crop ensemble and transfer 1/32768 weight from the first vertical crop to the second, preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA micro-extrapolation
evidence_used: Successive directional transfers of 1/1024, 1/4096, 1/8192, and 1/16384 retained 9,348 correct while monotonically lowering cross-entropy; halving the latest increment is the lowest-risk continuation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: An additional 1/32768 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877066032409668.
change: Transfer 1/32768 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA micro-extrapolation
evidence_used: Successive directional transfers through 1/16384 retained 9,348 correct and monotonically lowered cross-entropy; the prior 1/32768 attempt timed out and therefore supplied no contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 70.80946620786563, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770658378601074, "validation_score": 9348.420979395774}

RECENT RESULT
hypothesis: An additional 1/65536 vertical-weight transfer beyond Reference Design 3 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770658378601074.
change: Restore the best verified crop weights, then transfer 1/65536 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA nano-extrapolation
evidence_used: Successive directional transfers through the latest 1/32768 increment retained 9,348 correct and monotonically reduced cross-entropy; halving that latest successful increment is the lowest-risk continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.55262949992903, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770657272338867, "validation_score": 9348.420979399696}

RECENT RESULT
hypothesis: An additional 1/131072 vertical-weight transfer beyond Reference Design 2 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770657272338867.
change: Restore the best verified crop weights, then transfer 1/131072 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA pico-extrapolation
evidence_used: Successive directional transfers through 1/65536 retained 9,348 correct and monotonically lowered cross-entropy; Reference Design 2 is the best verified result, so halving its latest successful increment is the lowest-risk continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 67.86351233301684, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656814575196, "validation_score": 9348.420979401319}

RECENT RESULT
hypothesis: An additional 1/262144 vertical-weight transfer beyond Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656814575196.
change: Restore the best verified crop ensemble, then transfer 1/262144 weight from the first vertical crop to the second while preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA femto-extrapolation
evidence_used: Successive directional transfers through 1/131072 retained 9,348 correct and monotonically lowered cross-entropy; Reference Design 1 is the best verified result, making a halved continuation the lowest-risk next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 54.36882104212418, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656623840332, "validation_score": 9348.420979401995}

RECENT RESULT
hypothesis: An additional 1/524288 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656623840332.
change: Transfer 1/524288 ensemble weight from the first vertical crop to the second while preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA half-step extrapolation
evidence_used: Successive directional transfers through 1/262144 retained 9,348 correct and monotonically lowered cross-entropy; the current design is the best verified result, so halving the latest successful increment is the lowest-risk continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.27154212491587, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877065631866455, "validation_score": 9348.420979403076}

RECENT RESULT
hypothesis: An additional 1/1048576 vertical-weight transfer beyond Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.
change: Restore the best verified crop ensemble, then transfer 1/1048576 weight from the first vertical crop to the second while preserving center, horizontal, and total weights.
mechanism: Directional vertical-shift TTA micro-weight extrapolation
evidence_used: Successive halved directional transfers through 1/524288 retained 9,348 correct and monotonically lowered cross-entropy; Reference Design 1 is the best verified result, so another halved increment is the lowest-risk continuation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.44132479187101, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656433105468, "validation_score": 9348.42097940267}

RECENT RESULT
hypothesis: A half-step vertical-weight transfer from Reference Design 1 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.
change: Set the vertical crop weights halfway between Reference Designs 1 and 2, preserving their combined and total ensemble weight.
mechanism: Bracketed directional TTA weight refinement
evidence_used: Reference Design 1 is the best verified result; a full 1/1048576 transfer in Reference Design 2 slightly worsened cross-entropy, so the midpoint is the most informative local refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.49035570817068, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656509399414, "validation_score": 9348.420979402401}

RECENT RESULT
hypothesis: Replacing global channel gating with spatial self-attention will exceed 9,348 correct predictions by routing shape evidence between related image regions while preserving the 7×7 spatial grid.
change: Replace the refinement block’s globally pooled channel gate with learned query-key attention over all 49 feature positions and a channel-wise residual scale; parameter count decreases by seven.
mechanism: Content-adaptive nonlocal spatial refinement
evidence_used: Translation TTA raised correctness from 9,344 to 9,348, exposing residual shift sensitivity, while global pooling fell to 9,085 because it discarded spatial organization. This challenges the old assumption that global context should only rescale channels; the new mechanism exchanges content between positions without collapsing layout.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249874, "training_seconds": 76.71225066599436, "validation_accuracy": 0.9299, "validation_correct": 9299, "validation_cross_entropy": 0.18864999923706055, "validation_score": 9299.420645270115}

RECENT RESULT
hypothesis: Transferring 1/2097152 weight from the second vertical crop back to the first will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.
change: Symmetrically probe the untested side of the current best crop weighting while preserving the combined vertical and total ensemble weights.
mechanism: Bracketed reverse-direction vertical TTA refinement
evidence_used: The current design is best; both the 1/1048576 forward transfer in Reference Design 1 and its half-step in Reference Design 2 worsened cross-entropy, motivating an equal-distance half-step in the reverse direction.
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
