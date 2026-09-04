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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.3754118331708, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656051635742, "validation_score": 9348.420979404023}
prior_hypothesis: Raising the positive temperature by one float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656127929689.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.27154212491587, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877065631866455, "validation_score": 9348.420979403076}
prior_hypothesis: An additional 1/524288 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770656623840332.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.29558779206127, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656204223632, "validation_score": 9348.420979403481}
prior_hypothesis: Raising the best verified ensemble temperature by one additional float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656051635742.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.11387237510644, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877065616607666, "validation_score": 9348.420979403618}
prior_hypothesis: Favoring reflected predictions by the smallest symmetric float32 increment will preserve 9,348 correct predictions while reducing cross-entropy below 0.18770656051635742.

## Recent verification evidence

RECENT RESULT
hypothesis: Raising the positive temperature by one float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656127929689.
change: Keep the best verified crop weights and change the temperature from float32 0.753170013427734375 to its immediately higher representable value.
mechanism: One-ULP reverse ensemble temperature calibration
evidence_used: The current design has the lowest verified cross-entropy, while lowering its temperature by one ULP worsened cross-entropy without changing correctness; the symmetric higher-temperature probe is therefore the most informative remaining local calibration test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.3754118331708, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656051635742, "validation_score": 9348.420979404023}

RECENT RESULT
hypothesis: Replacing the narrow absolute-position flattening head with a wider multiscale regional representation will exceed 9,348 correct predictions by reducing translation sensitivity without discarding image layout.
change: Pool the 7×7 feature map into 1×1, 2×2, and 4×4 regional descriptors, concatenate them, and classify through a 130-unit head; this keeps the model below 250,000 parameters with similar classifier compute.
mechanism: Layout-preserving spatial-pyramid classification head
evidence_used: Translation TTA improved correctness from 9,344 to 9,348, indicating sensitivity to exact spatial alignment, while global pooling fell to 9,085 because it erased layout. The old assumption is that all 49 absolute feature positions should feed a narrow 56-unit bottleneck; the new approach preserves a 4×4 layout while adding coarse regional context and a wider learned readout.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the best verified ensemble temperature by one additional float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656051635742.
change: Restore Reference Design 2’s optimal crop weights and increase its temperature from 0.753170073032379150390625 to the next representable float32 value, 0.75317013263702392578125.
mechanism: Second-ULP ensemble temperature softening
evidence_used: The first higher-temperature ULP produced the lowest verified cross-entropy without changing correctness, while the symmetric lower-temperature probe worsened cross-entropy; the smallest continuation in the successful direction is the most informative calibration test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.29558779206127, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656204223632, "validation_score": 9348.420979403481}

RECENT RESULT
hypothesis: A temperature approximately 0.136 float32 ULP above Reference Design 2’s value will preserve all 9,348 predictions while reducing validation cross-entropy below 0.18770656051635742.
change: Restore the best calibration region and perform temperature scaling in float64 before returning float32 logits, enabling a temperature between adjacent float32 values.
mechanism: Sub-ULP ensemble temperature interpolation
evidence_used: Reference Design 2 outperformed the immediately lower and higher temperature settings; quadratic interpolation of those three verified cross-entropies places the estimated minimum about 0.136 ULP above its temperature.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Multiplying by the rounded float32 reciprocal will preserve all 9,348 predictions while reducing validation cross-entropy below 0.18770656051635742.
change: Replace temperature division with float32 reciprocal multiplication, producing an effective temperature about 0.105 ULP higher without returning float64 logits.
mechanism: Float32 reciprocal-multiply sub-ULP temperature calibration
evidence_used: Adjacent float32 temperatures bracket the current optimum, and quadratic interpolation estimated the minimum about 0.136 ULP higher; reciprocal multiplication closely approximates that sub-ULP shift while retaining float32 output.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 46.14843645901419, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656280517578, "validation_score": 9348.420979403212}

RECENT RESULT
hypothesis: Transferring one float32 ULP of center-crop weight equally to the two vertical crops while restoring the best verified temperature will preserve all 9,348 correct predictions and reduce validation cross-entropy below 0.18770656051635742.
change: Restore Reference Design 2’s temperature, lower the center weight by one float32 ULP, and raise each vertical weight by one float32 ULP, preserving total ensemble weight and the established horizontal weighting.
mechanism: Center-to-vertical TTA group micro-reweighting
evidence_used: Reference Design 2 is the best verified calibration; adjacent temperature and relative horizontal/vertical probes were worse, leaving center-versus-shifted group weighting as an untested orthogonal refinement. The vertical pair is the more heavily weighted shifted group, motivating the smallest exact transfer toward it.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 57.91775800008327, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656204223632, "validation_score": 9348.420979403481}

RECENT RESULT
hypothesis: Transferring one float32 ULP of center-crop weight equally to the horizontal crops will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18770656051635742.
change: Restore Reference Design 1’s best crop weights and temperature, then lower the center weight by one ULP and raise each horizontal weight by two ULPs, preserving the exact total ensemble weight.
mechanism: Center-to-horizontal TTA group micro-reweighting
evidence_used: Reference Design 1 has the lowest verified cross-entropy; temperature, within-pair horizontal, and vertical weighting have been bracketed, while center-to-vertical transfer did not improve, making the untested center-versus-horizontal group direction the most informative orthogonal probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 67.76614504191093, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656433105468, "validation_score": 9348.42097940267}

RECENT RESULT
hypothesis: Restoring Reference Design 2’s best temperature and transferring one float32 ULP of total weight from the horizontal crops to the center crop will preserve 9,348 correct predictions while lowering cross-entropy below 0.18770656051635742.
change: Raise the center weight by one ULP, lower each horizontal weight by two ULPs to preserve total weight, and restore the best verified temperature.
mechanism: Reverse center–horizontal TTA group reweighting
evidence_used: The opposite center-to-horizontal transfer worsened cross-entropy without changing correctness; probing the reverse direction around Reference Design 2 is the most informative remaining group-weight test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.89876729203388, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656204223632, "validation_score": 9348.420979403481}

RECENT RESULT
hypothesis: Raising the center-crop weight by one float32 ULP while lowering each vertical-crop weight by one ULP will preserve all 9,348 correct predictions and reduce cross-entropy below 0.18770656051635742.
change: Preserve the total ensemble weight and best verified temperature while transferring the smallest exactly balanced weight increment from the vertical crop pair to the center crop.
mechanism: Reverse center–vertical TTA group micro-reweighting
evidence_used: The opposite center-to-vertical transfer did not improve cross-entropy, while within-pair crop weights and temperature have already been bracketed; this makes the untested reverse center–vertical direction the most informative remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 65.66141283302568, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656433105468, "validation_score": 9348.42097940267}

RECENT RESULT
hypothesis: Restoring Reference Design 2’s best temperature and transferring the smallest symmetric float32 weight increment from reflected predictions to native-orientation predictions will retain all 9,348 correct predictions while reducing cross-entropy below 0.18770656051635742.
change: Restore the best verified temperature, then slightly favor each native crop over its horizontal reflection while preserving their exact combined orientation weight.
mechanism: Native-orientation TTA micro-reweighting
evidence_used: Reference Design 2 remains the lowest-cross-entropy implementation; temperature and crop-weight directions have been bracketed without changing correctness, while the currently equal native/reflected weighting is an untested orthogonal calibration dimension.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.785942958900705, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656394958496, "validation_score": 9348.420979402807}

RECENT RESULT
hypothesis: Favoring reflected predictions by the smallest symmetric float32 increment will preserve 9,348 correct predictions while reducing cross-entropy below 0.18770656051635742.
change: Restore Reference Design 2’s optimal crop weights and temperature, then transfer orientation weight from every native crop prediction to its reflection while preserving total ensemble weight.
mechanism: Reflected-orientation TTA micro-reweighting
evidence_used: Favoring native orientations worsened cross-entropy to 0.18770656394958496 without changing correctness; the reverse direction is the most informative remaining orthogonal calibration probe around Reference Design 2.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.11387237510644, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877065616607666, "validation_score": 9348.420979403618}

RECENT RESULT
hypothesis: Reallocating parameters from the 56-unit dense head into deeper 97-channel features and 704 learned channel–spatial templates will exceed 9,348 correct predictions by learning richer part detectors while retaining full 7×7 layout.
change: Replace the load-bearing assumption that image layout requires an unrestricted flattening matrix with a wider separable mechanism: pointwise semantic projection followed by learned per-channel 7×7 templates. Add a high-level convolution and widen spatial refinement; expected parameter count is 249,793.
mechanism: Wide factorized spatial-template classifier
evidence_used: Global pooling fell to 9,085 correct because it erased layout, while translation TTA raised correctness to 9,348, showing that spatial organization and alignment matter. The proposed head preserves all 7×7 positions but trades the narrow 56-unit dense bottleneck for 704 nonlinear spatial templates and stronger high-level features.
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
