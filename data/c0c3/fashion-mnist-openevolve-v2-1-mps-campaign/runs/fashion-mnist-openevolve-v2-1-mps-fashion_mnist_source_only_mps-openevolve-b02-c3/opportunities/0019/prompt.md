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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247378, "training_seconds": 37.38582762493752, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21007430572509767, "validation_score": 9268.41319776615}
prior_hypothesis: A single 5×5 depthwise-separable residual block will exceed 9,250 correct predictions by matching the receptive field of two stacked 3×3 blocks while retaining the 56-unit spatial head and avoiding the extra normalization and nonlinearity that accompanied the unsuccessful second block.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 35.64705799985677, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.21234450149536133, "validation_score": 9273.41242402583}
prior_hypothesis: Replicate-padding the best 7×7 depthwise refinement block will exceed 9,270 correct predictions by preventing its whole-map kernel from mixing late features with artificial zero borders.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 246354, "training_seconds": 47.314268292160705, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.21731825561523438, "validation_score": 9250.410738931823}
prior_hypothesis: A depthwise-separable residual block at 7×7 resolution, while preserving the 56-unit location-sensitive head and using matched axial augmentation, will exceed 9,247 correct predictions.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 33.79385079094209, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.21668187713623047, "validation_score": 9274.410953766466}
prior_hypothesis: Replicate padding plus weighted probability averaging will exceed 9,273 correct predictions by retaining the best boundary treatment while preventing overconfident individual views from dominating the translation-and-flip ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: A center weight of 3.25 will exceed 9,168 correct predictions by refining the apparent optimum between weight 3, which achieved 9,168, and weight 4, which declined to 9,166.
change: Increase the centered crop’s ensemble weight from 3.0 to 3.25 while retaining the four axial crops and mirrored views.
mechanism: Local interpolation of center-dominant test-time ensembling
evidence_used: Center weights 1, 2, 3, and 4 produced 9,162, 9,163, 9,168, and 9,166 correct predictions respectively, locating the best observed region near weight 3 with diminishing performance before weight 4.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 21.77573512494564, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.23886156997680663, "validation_score": 9166.403596343705}

RECENT RESULT
hypothesis: A center weight of 2.75 will exceed 9,168 correct predictions by refining the unexplored interval below weight 3, while weights above 3 reduced accuracy.
change: Reduce the centered crop’s ensemble weight from 4.0 to 2.75 while retaining the four axial crops and mirrored views.
mechanism: Local interpolation of center-dominant axial ensembling
evidence_used: Weight 3 achieved the best result at 9,168 correct; weight 2 reached 9,163, while weights 3.25 and 4 both declined to 9,166, locating the observed optimum at or just below 3.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 28.88043558388017, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.2387437759399414, "validation_score": 9167.403634722297}

RECENT RESULT
hypothesis: Training on the same center-weighted axial translation distribution used by the best inference ensemble will exceed 9,168 correct predictions by avoiding the less reliable diagonal and two-pixel translations.
change: Use the qualified center weight of 3.0 and replace uniform independent ±2-pixel training translations with center-three-times-plus-four-axial ±1-pixel sampling.
mechanism: Reliability-matched axial augmentation
evidence_used: Center weight 3 achieved the best observed 9,168 correct, while adding diagonal inference views reduced accuracy to 9,159; the current training sampler nevertheless emphasizes diagonal and two-pixel transformations absent from the successful ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241426, "training_seconds": 27.252630000002682, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21906672744750977, "validation_score": 9245.410149820958}

RECENT RESULT
hypothesis: Reallocating the same parameter budget from the location-specific dense head into deeper residual, channel-gated feature extraction will exceed 9,245 correct predictions under the proven axial augmentation recipe.
change: Replace the shallow four-convolution/flattened-spatial classifier with a nine-convolution residual network using squeeze-excitation gates, learned strided downsampling, and global average pooling; retain training and test-time augmentation to isolate the representational change.
mechanism: Residual channel-attentive global feature aggregation
evidence_used: The 9,245-correct design devotes 176,242 of 241,426 parameters to its flattened 7×7 head, while matched augmentation produced the largest recent gain. This motivates testing whether deeper, globally aggregated features can learn translation-stable class evidence directly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241282, "training_seconds": 75.08010537503287, "validation_accuracy": 0.9085, "validation_correct": 9085, "validation_cross_entropy": 0.27277201385498046, "validation_score": 9085.392843332944}

RECENT RESULT
hypothesis: Adding one 64-channel convolution at 7×7 resolution while retaining a location-sensitive dense head and adopting the proven matched axial augmentation will exceed 9,245 correct predictions.
change: Use the qualified center-weighted axial training sampler, add a post-pooling convolutional refinement layer, and reduce the dense width to 46 to remain below 250,000 parameters.
mechanism: Spatially preserving late-stage feature refinement
evidence_used: Matched axial augmentation achieved 9,245 correct, while replacing spatial dense aggregation with a residual global-pooling network fell to 9,085; this motivates deeper feature extraction without discarding spatial layout.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 246948, "training_seconds": 30.746769957942888, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21743479385375977, "validation_score": 9247.410699614078}

RECENT RESULT
hypothesis: A depthwise-separable residual block at 7×7 resolution, while preserving the 56-unit location-sensitive head and using matched axial augmentation, will exceed 9,247 correct predictions.
change: Add a 4,928-parameter late residual refinement block, retain the wider dense head, and adopt the proven center-weight-three axial training and inference distribution.
mechanism: Parameter-efficient residual spatial refinement
evidence_used: A full late 64-channel convolution improved accuracy from 9,245 to 9,247 despite forcing the dense head from 56 to 46 units; residual depthwise-separable refinement tests the same useful late processing while retaining spatial-head capacity and staying below 250,000 parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 246354, "training_seconds": 47.314268292160705, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.21731825561523438, "validation_score": 9250.410738931823}

RECENT RESULT
hypothesis: A second depthwise-separable residual block at 7×7 resolution will exceed 9,250 correct predictions by extending the late spatial refinement that improved the previous design, while sacrificing only one dense-head unit and remaining below 250,000 parameters.
change: Stack two SpatialRefinement blocks and reduce the location-sensitive classifier width from 56 to 55, yielding 248,135 learned parameters.
mechanism: Stacked parameter-efficient residual spatial refinement
evidence_used: Adding the first depthwise-separable residual block improved validation correctness from 9,247 to 9,250 while preserving the spatial dense head; this directly motivates testing another residual refinement stage with minimal loss of head capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248135, "training_seconds": 51.08871691697277, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.2164805507659912, "validation_score": 9245.411021778922}

RECENT RESULT
hypothesis: A single 5×5 depthwise-separable residual block will exceed 9,250 correct predictions by matching the receptive field of two stacked 3×3 blocks while retaining the 56-unit spatial head and avoiding the extra normalization and nonlinearity that accompanied the unsuccessful second block.
change: Replace the full late convolution with one 5×5 depthwise-separable residual block and widen the location-sensitive classifier from 46 to 56 units, yielding 247,378 learned parameters.
mechanism: Wide-receptive-field residual spatial refinement
evidence_used: One 3×3 SpatialRefinement block with a 56-unit head achieved the best result of 9,250 correct, whereas stacking two blocks fell to 9,245; a single wider-kernel block isolates increased spatial context from harmful additional depth.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247378, "training_seconds": 37.38582762493752, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21007430572509767, "validation_score": 9268.41319776615}

RECENT RESULT
hypothesis: A single 7×7 depthwise-separable residual block will exceed 9,268 correct predictions by extending the successful single-block receptive-field trend across the entire late 7×7 feature map while preserving the 56-unit spatial head.
change: Add one 7×7 depthwise-separable residual refinement block after the final pooling layer, producing 248,914 learned parameters.
mechanism: Whole-map depthwise residual refinement
evidence_used: Expanding the single refinement block from 3×3 to 5×5 improved correctness from 9,250 to 9,268, whereas stacking two 3×3 blocks fell to 9,245; this supports increasing spatial context without adding depth.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 36.11516833305359, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.21227009506225586, "validation_score": 9270.412449339497}

RECENT RESULT
hypothesis: Replicate-padding the best 7×7 depthwise refinement block will exceed 9,270 correct predictions by preventing its whole-map kernel from mixing late features with artificial zero borders.
change: Adopt the qualified 7×7 residual spatial-head design and matched axial training/TTA recipe, changing the refinement block’s depthwise convolution to replicate padding.
mechanism: Boundary-consistent whole-map residual refinement
evidence_used: The single 7×7 block achieved the best result at 9,270 correct, extending the improvement from 3×3 to 5×5, while its large kernel makes boundary padding especially influential; the successful augmentation recipe already uses replicate-padded translations.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 35.64705799985677, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.21234450149536133, "validation_score": 9273.41242402583}

RECENT RESULT
hypothesis: Replicate-padding every 3×3 backbone convolution will exceed 9,273 correct predictions by eliminating artificial zero-border features throughout the network while preserving the qualified architecture and augmentation recipe.
change: Add replicate padding to all four 3×3 feature-extraction convolutions; leave parameters, optimizer, training augmentation, and inference ensemble unchanged.
mechanism: Backbone-wide boundary-consistent convolution
evidence_used: Changing only the 7×7 refinement convolution from zero to replicate padding improved correctness from 9,270 to 9,273 at identical parameter count, while training translations already use replicate padding.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 60.465859290910885, "validation_accuracy": 0.9271, "validation_correct": 9271, "validation_cross_entropy": 0.210849959564209, "validation_score": 9271.41293307734}

RECENT RESULT
hypothesis: Replicate padding plus weighted probability averaging will exceed 9,273 correct predictions by retaining the best boundary treatment while preventing overconfident individual views from dominating the translation-and-flip ensemble.
change: Adopt the qualified replicate-padded 7×7 refinement block and average per-view softmax probabilities instead of raw logits, returning their logarithms as ten-class logits.
mechanism: Arithmetic posterior view ensembling
evidence_used: Replicate padding produced the best result at 9,273 correct, and prior center-weight experiments showed prediction accuracy is sensitive to inference aggregation; posterior averaging directly tests a more robust aggregation rule without changing training.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 33.79385079094209, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.21668187713623047, "validation_score": 9274.410953766466}



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
