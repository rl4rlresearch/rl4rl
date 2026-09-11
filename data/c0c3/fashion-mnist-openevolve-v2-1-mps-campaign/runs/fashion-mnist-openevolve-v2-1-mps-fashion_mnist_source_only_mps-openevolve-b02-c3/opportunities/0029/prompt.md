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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 57.16008404199965, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19197460327148438, "validation_score": 9315.41947202451}
prior_hypothesis: Sharpening the paired-view posterior ensemble from temperature 0.9 to 0.8 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1971655.

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

RECENT RESULT
hypothesis: A post-ensemble temperature of 0.9 will retain the probability ensemble’s 9,274 correct predictions while lowering validation cross-entropy below 0.2166819, strictly improving validation_score.
change: Adopt the qualified replicate-padded 7×7 refinement and probability averaging, then sharpen the returned log probabilities without changing their argmax.
mechanism: Temperature-sharpened posterior view ensembling
evidence_used: Probability averaging improved correctness from 9,273 to 9,274 but worsened cross-entropy from 0.2123445 to 0.2166819; positive temperature scaling preserves every predicted class while testing whether the averaged posterior is underconfident.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 39.52013020799495, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.20702460021972657, "validation_score": 9274.414241764343}

RECENT RESULT
hypothesis: On the qualified 7×7 replicate-padded and probability-ensemble foundation, fusing independent nonlinear embeddings of the 14×14 stem features and deepest 7×7 features will exceed 9,274 correct predictions by recovering fine-grained cues lost in the strictly serial backbone.
change: Replace the single deepest-feature classifier with spatially flattened deep and shallow branches, fuse their embeddings for class prediction, and adopt the qualified 7×7 refinement and temperature-sharpened probability ensemble. The resulting model has 247,456 learned parameters.
mechanism: Dual-scale location-sensitive feature fusion
evidence_used: Global pooling reduced correctness to 9,085 while location-sensitive heads reached 9,274, showing that spatial layout is load-bearing. However, every successful design assumes only the final 7×7 map should reach that head; the proposed bypass preserves spatial coordinates while exposing pre-downsampling detail.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247456, "training_seconds": 50.736577624920756, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.21210200271606444, "validation_score": 9243.412506537305}

RECENT RESULT
hypothesis: Training every sampled axial view together with its horizontal counterpart will exceed 9,274 correct predictions by reducing augmentation variance and directly matching the flip-paired inference ensemble.
change: Concatenate each augmented batch with its horizontal flip and train both views jointly under the existing smoothed cross-entropy loss; preserve the qualified architecture, optimizer, and inference aggregation.
mechanism: Paired horizontal-view supervision
evidence_used: Reliability-matched augmentation produced the largest observed improvement, reaching 9,245 correct, and flip-paired probability ensembling further raised the best architecture to 9,274; explicit paired supervision applies that successful invariance throughout training.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 54.81602695793845, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19716547088623046, "validation_score": 9315.417653208482}

RECENT RESULT
hypothesis: Jointly supervising every augmented image and its horizontal counterpart, plus 0.9 inference temperature scaling, will exceed the current 9,274 correct predictions.
change: Train on concatenated original-and-flipped augmented batches and sharpen the probability-averaged inference logits by a factor of 1/0.9.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 made these changes on the same architecture and augmentation foundation and achieved 9,315 correct predictions with cross-entropy 0.197165, the strongest available result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Matching each augmented training view with its horizontal flip and using 0.9-temperature probability averaging at inference will exceed the current 9,273 correct predictions, consistent with Reference Design 3’s 9,315 correct.
change: Train jointly on augmented images and their horizontal counterparts, then replace weighted logit averaging with temperature-sharpened weighted probability averaging.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 used these exact changes on the same architecture and achieved the strongest verified result: 9,315 correct with 0.197165 validation cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing Reference Design 3’s 7×7 replicate-padded refinement, paired flip supervision, and temperature-sharpened probability ensemble will exceed the current 9,250 correct predictions.
change: Expand the refinement kernel to 7×7 with replicate padding, train each augmented image alongside its horizontal flip, and use weighted probability averaging with 0.9 inference temperature.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 used this exact combination and achieved the strongest verified result: 9,315 correct with 0.197165 validation cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharpening the paired-view posterior ensemble from temperature 0.9 to 0.8 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1971655.
change: Change only the positive inference temperature from 0.9 to 0.8, leaving training and prediction classes unchanged.
mechanism: Stronger post-ensemble temperature calibration
evidence_used: Temperature 0.9 preserved 9,274 correct predictions while reducing cross-entropy from 0.2166819 to 0.2070246; the current paired-view model retains the same probability-ensemble mechanism and has reached 9,315 correct.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248914, "training_seconds": 57.16008404199965, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19197460327148438, "validation_score": 9315.41947202451}

RECENT RESULT
hypothesis: Reproducing Reference Design 3’s paired flip supervision and 0.8-temperature probability ensemble will exceed the current 9,274 correct predictions; the identical qualified design achieved 9,315 correct.
change: Train each augmented image jointly with its horizontal flip and sharpen the probability-averaged inference logits using temperature 0.8.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 achieved the strongest verified result—9,315 correct and 0.1919746 cross-entropy—using the current architecture plus exactly these two changes.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing Reference Design 3 will exceed the current 9,273 correct predictions because the identical qualified implementation achieved 9,315 correct with 0.1919746 cross-entropy.
change: Train every augmented image jointly with its horizontal flip, then use weighted probability averaging across translated and flipped inference views with temperature 0.8.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 is the strongest verified design at 9,315 correct; paired flip supervision raised correctness from 9,274 to 9,315, and temperature 0.8 subsequently reduced cross-entropy while preserving all argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing Reference Design 3 will exceed the current 9,250 correct predictions; the identical qualified design achieved 9,315 correct with 0.1919746 cross-entropy.
change: Use a replicate-padded 7×7 refinement kernel, jointly supervise each augmented image and its horizontal flip, and apply 0.8-temperature probability averaging across translated and flipped inference views.
mechanism: Paired horizontal-view supervision with calibrated posterior ensembling
evidence_used: Reference Design 3 is the strongest verified implementation: paired flip supervision raised correctness to 9,315, while temperature 0.8 preserved its argmax predictions and reduced cross-entropy to 0.1919746.
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
