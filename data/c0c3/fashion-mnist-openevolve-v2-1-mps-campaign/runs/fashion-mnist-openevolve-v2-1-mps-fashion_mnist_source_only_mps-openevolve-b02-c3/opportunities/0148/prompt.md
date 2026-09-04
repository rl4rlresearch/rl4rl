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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.11387237510644, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877065616607666, "validation_score": 9348.420979403618}
prior_hypothesis: Favoring reflected predictions by the smallest symmetric float32 increment will preserve 9,348 correct predictions while reducing cross-entropy below 0.18770656051635742.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.40722645795904, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656127929689, "validation_score": 9348.420979403752}
prior_hypothesis: A 0.25-ULP interpolation will preserve all 9,348 predictions while lowering validation cross-entropy below 0.18770656051635742 by crossing more float32 rounding thresholds while remaining inside the quadratic estimate’s improving interval.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 62.3754118331708, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656051635742, "validation_score": 9348.420979404023}
prior_hypothesis: Raising the positive temperature by one float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656127929689.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 74.30202249996364, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656051635742, "validation_score": 9348.420979404023}
prior_hypothesis: Interpolating 13.6% from Reference Design 2’s temperature-scaled logits toward the next higher float32 temperature will preserve 9,348 correct predictions while reducing validation cross-entropy below 0.18770656051635742.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Favoring reflected predictions by one quarter of the previously tested symmetric increment will preserve all 9,348 correct predictions while reducing validation cross-entropy below 0.18770656051635742.
change: Keep the best crop weights and temperature, but lower the native-orientation coefficient to the immediate float32 predecessor of 1.0 while retaining unit reflected weight and exact coefficient-aware normalization.
mechanism: Quarter-step reflected-orientation TTA calibration
evidence_used: Full reflected preference worsened cross-entropy by only 1.14e-9, whereas the equal-magnitude native preference worsened it by 3.43e-9; this asymmetric response estimates a shallow optimum about one quarter-step toward reflected predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 55.528667541919276, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656204223632, "validation_score": 9348.420979403481}

RECENT RESULT
hypothesis: Adding diagonally shifted views with separable product weights will exceed 9,348 correct predictions by reducing residual two-axis alignment sensitivity while preserving the best verified calibration.
change: Restore Reference Design 2’s equal orientation weighting and temperature, then extend its axial crop ensemble to all nine one-pixel translations; derive each diagonal weight as the product of its vertical and horizontal axial weights divided by the center weight.
mechanism: Separable 3×3 translation TTA
evidence_used: Axial translation TTA increased correctness from 9,344 to 9,348, whereas subsequent temperature and micro-weight probes only changed cross-entropy. A separable 3×3 ensemble is the smallest untested extension of that successful mechanism and adds diagonal alignment coverage without changing training or parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 65.25140862516128, "validation_accuracy": 0.9335, "validation_correct": 9335, "validation_cross_entropy": 0.18770665435791015, "validation_score": 9335.420979370761}

RECENT RESULT
hypothesis: Interpolating 13.6% from Reference Design 2’s temperature-scaled logits toward the next higher float32 temperature will preserve 9,348 correct predictions while reducing validation cross-entropy below 0.18770656051635742.
change: Restore Reference Design 2 as the lower interpolation endpoint and return a float32 linear interpolation between logits produced by the two adjacent temperatures.
mechanism: Float32 adjacent-temperature logit interpolation
evidence_used: Reference Design 2 has the lowest verified cross-entropy, while both adjacent float32 temperatures were worse; quadratic interpolation estimated the optimum 0.136 ULP higher. The earlier float64 interpolation could not be verified, and reciprocal multiplication did not reproduce the desired arithmetic.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 74.30202249996364, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656051635742, "validation_score": 9348.420979404023}

RECENT RESULT
hypothesis: A 0.25-ULP interpolation will preserve all 9,348 predictions while lowering validation cross-entropy below 0.18770656051635742 by crossing more float32 rounding thresholds while remaining inside the quadratic estimate’s improving interval.
change: Restore the best verified asymmetric crop weights and interpolate 25% from its temperature-scaled logits toward the next higher float32 temperature.
mechanism: Quantization-aware adjacent-temperature interpolation
evidence_used: Reference Design 2 achieved the lowest verified cross-entropy, but its theoretically optimal 0.136 interpolation tied the lower endpoint, indicating float32 quantization suppressed the intended change; 0.25 remains below the estimated break-even point of approximately 0.272.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.40722645795904, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770656127929689, "validation_score": 9348.420979403752}

RECENT RESULT
hypothesis: Sampling training translations in the 9:4:4:2:2 ratio implied by the best TTA weights will exceed 9,348 correct predictions by improving invariance to the more valuable vertical shifts while reducing exposure to less reliable horizontal shifts.
change: Replace the current 3:1:1:1:1 translation distribution with a 21-way discrete approximation of the verified evaluation weights, leaving architecture, runtime, and example accounting unchanged.
mechanism: Evaluation-aligned anisotropic translation sampling
evidence_used: Axial translation TTA raised correctness from 9,344 to 9,348, and its best weights scale almost exactly to 9:4:4:2:2; diagonal translations subsequently reduced correctness to 9,335, showing that matching the useful translation directions is consequential.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 52.56239566579461, "validation_accuracy": 0.9338, "validation_correct": 9338, "validation_cross_entropy": 0.18753530426025392, "validation_score": 9338.421040114097}

RECENT RESULT
hypothesis: Sampling translations in a 9:2:2:4:4 center/up/down/left/right ratio will exceed 9,348 correct predictions by reversing the vertical oversampling that previously reduced correctness.
change: Keep the center-translation probability unchanged while transferring training exposure from vertical shifts to horizontal shifts; architecture, evaluation ensemble, and calibration remain unchanged.
mechanism: Reverse-anisotropic translation sampling
evidence_used: Changing the original scaled 9:3:3:3:3 distribution to 9:4:4:2:2 reduced validation correctness from 9,348 to 9,338. Because the center probability was identical, this supplies directional evidence that favoring vertical over horizontal shifts is harmful and motivates testing the symmetric reverse allocation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 65.03316524997354, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1865619785308838, "validation_score": 9322.421385489377}

RECENT RESULT
hypothesis: Blending 12.5% geometric consensus into the successful axial probability ensemble will exceed 9,348 correct predictions by suppressing isolated high-confidence crop errors while preserving the proven view set.
change: Retain the five axial crops and equal reflection weighting, blend their arithmetic and geometric probability ensembles, and restore the best verified 0.136 adjacent-temperature interpolation.
mechanism: Arithmetic–geometric TTA consensus blending
evidence_used: Axial translation TTA improved correctness from 9,344 to 9,348, while adding diagonal views reduced it to 9,335; changing how the proven axial views are fused is therefore a more informative accuracy probe than adding views or continuing temperature-only refinements.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 61.16849345783703, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18773939895629882, "validation_score": 9348.420967764847}



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
