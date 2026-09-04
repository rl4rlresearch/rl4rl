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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 53.020769792143255, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20671802101135253, "validation_score": 9281.414347006752}
prior_hypothesis: Evaluation scale 1.20485 will preserve all 9,281 argmax predictions while reducing validation cross-entropy below 0.20671812667846678.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 49.8065586250741, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20682972106933595, "validation_score": 9281.414308656202}
prior_hypothesis: Label smoothing of 0.023 will exceed 9,279 correct predictions by probing near the empirical accuracy maximum implied by the verified 0.01, 0.02, and 0.03 results.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 74.43629279197194, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20210640487670897, "validation_score": 9288.415936557672}
prior_hypothesis: Normalizing the 128-unit classifier representation will exceed 9,281 correct predictions by improving optimization of the successful feature extractor while remaining under the 250,000-parameter ceiling.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 78.69068962498568, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20120860176086425, "validation_score": 9284.416247435514}
prior_hypothesis: BatchNorm1d with the feature extractor’s verified 0.00623359375 momentum will exceed 9,288 correct predictions by retaining the successful classifier conditioning while producing less noisy evaluation statistics than the default 0.1 momentum.

## Recent verification evidence

RECENT RESULT
hypothesis: Global channel gating will raise validation_correct above 9,281 by adding input-dependent cross-channel interaction without disturbing the successful pairwise spatial refinement.
change: Add a parameter-efficient channel gate after pairwise refinement, narrow the dense classifier to 124 units to remain below 250,000 parameters, and use the best-verified evaluation scale of 1.20485.
mechanism: Identity-centered squeeze-and-excitation channel gating
evidence_used: Expanding the grouped 5×5 convolution reduced validation_correct to 9,268, suggesting additional local spatial mixing is unhelpful; global channel recalibration tests a distinct interaction mechanism while preserving the best 0.023-smoothed training procedure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248014, "training_seconds": 64.08124987501651, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.20357024993896483, "validation_score": 9256.415430673886}

RECENT RESULT
hypothesis: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while evaluation scale 1.20512 will preserve every argmax and reduce validation cross-entropy below 0.20671802101135253.
change: Restore the best-verified training smoothing and set the evaluation-only multiplier to the quadratic optimum estimated from the three closest calibration results.
mechanism: Local convex logit recalibration
evidence_used: Scales 1.19, 1.20485, and 1.206 produced cross-entropies 0.2067525513, 0.2067180210, and 0.2067181267 with the same 9,281 correct predictions; local quadratic interpolation places the minimum near 1.20512.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 45.43381454190239, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2067180076599121, "validation_score": 9281.414347011336}

RECENT RESULT
hypothesis: Evaluation scale 1.20514 will preserve all 9,281 argmax predictions while reducing validation cross-entropy below 0.2067180076599121.
change: Change only the evaluation-time logit multiplier from 1.20512 to the refined local optimum 1.20514.
mechanism: Updated local quadratic logit recalibration
evidence_used: Scales 1.20485, 1.20512, and 1.206 produced cross-entropies 0.2067180210, 0.2067180077, and 0.2067181267 with identical correct counts; incorporating the new 1.20512 observation places the local quadratic minimum near 1.20514.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 51.22513133403845, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20671800689697264, "validation_score": 9281.414347011598}

RECENT RESULT
hypothesis: Depthwise 5×5 refinement plus a 130-unit classifier will exceed 9,281 correct predictions by reducing harmful refinement-stage channel interaction while preserving cross-channel mixing in the preceding convolution and classifier.
change: Change refinement groups from 64 to 128, widen the classifier from 128 to 130 units, and use the best-verified evaluation scale; the model has 247,552 learned parameters.
mechanism: Channel-isolated spatial refinement with denser classification
evidence_used: Increasing local interaction with 32 groups reduced validation_correct to 9,268, and global channel gating reduced it to 9,256, versus 9,281 with 64 groups. Testing the opposite direction directly probes whether channel-isolated spatial filtering is more effective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247552, "training_seconds": 30.18217191589065, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20733812713623048, "validation_score": 9279.414134192206}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.20514 will preserve all 9,281 correct predictions while lowering validation cross-entropy from 0.2067525513 to approximately 0.2067180069.
change: Replace the current evaluation-only multiplier of 1.19 with the best-verified multiplier of 1.20514.
mechanism: Evaluation-only logit recalibration
evidence_used: Reference Design 2 achieved 9,281 correct with cross-entropy 0.2067180069 at scale 1.20514, outperforming the otherwise identical current design at scale 1.19; positive scaling cannot change argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 57.13115804083645, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20671800689697264, "validation_score": 9281.414347011598}

RECENT RESULT
hypothesis: Training on balanced one-pixel translations during the second exposure will exceed 9,281 correct predictions by improving spatial robustness without changing the successful architecture or first-pass optimization.
change: Preserve the first exposure unchanged, apply balanced replicate-padded 3×3 crops during the second exposure, and restore the best-verified evaluation scale of 1.20514.
mechanism: Second-pass deterministic translation augmentation
evidence_used: Channel-mixing, gating, and depthwise architectural changes all underperformed the 9,281-correct baseline, motivating an orthogonal input-invariance test while retaining its verified smoothing and calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Flipping half of each second-exposure batch will exceed 9,281 correct predictions by improving left-right invariance while avoiding the runtime cost of padded translation augmentation.
change: Preserve the first exposure unchanged, then deterministically flip alternating images horizontally during the second exposure; retain the best architecture, smoothing, schedule, and evaluation calibration.
mechanism: Low-overhead second-pass horizontal-flip augmentation
evidence_used: Architectural channel-mixing variants underperformed the 9,281-correct baseline, while second-pass translation augmentation timed out; horizontal flipping tests the same orthogonal invariance strategy with substantially less preprocessing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging 25% horizontally flipped logits with 75% original logits will exceed 9,281 correct predictions by adding left-right robustness without the training-time overhead that caused flip augmentation to time out.
change: Apply a parameter-free, evaluation-only weighted horizontal-flip ensemble while preserving the verified architecture, training procedure, and logit scale.
mechanism: Conservative horizontal-flip test-time ensembling
evidence_used: Training-time horizontal-flip augmentation was motivated as an orthogonal alternative after several architecture variants underperformed, but timed out before yielding accuracy evidence; evaluation-only ensembling tests that invariance without increasing training time.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 63.57808212493546, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21591361618041993, "validation_score": 9266.41121342285}

RECENT RESULT
hypothesis: Averaging the final 64 low-learning-rate parameter states will improve validation_score beyond 9281.414347011598 by reducing endpoint noise while preserving the successful training trajectory.
change: Uniformly average the last 64 post-update model states, install the average after the final step, and use the best-verified evaluation scale of 1.20514.
mechanism: Final-window Polyak weight averaging
evidence_used: The baseline reached 9,281 correct, while broader, narrower, gated, and augmented variants failed to improve it; final-iterate averaging tests an orthogonal variance-reduction mechanism without changing the verified architecture, loss, or example exposure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 57.3805572912097, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.20648623428344726, "validation_score": 9266.414426609928}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.20514 will preserve all 9,281 correct predictions while lowering validation cross-entropy to approximately 0.2067180069.
change: Replace the current evaluation-only multiplier with the best-verified calibration scale.
mechanism: Evaluation-only logit recalibration
evidence_used: The otherwise identical reference design achieved 9,281 correct with 0.2067180069 cross-entropy at 1.20514, improving on the current 0.2068297211; positive scaling preserves argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Normalizing the 128-unit classifier representation will exceed 9,281 correct predictions by improving optimization of the successful feature extractor while remaining under the 250,000-parameter ceiling.
change: Add BatchNorm1d between the classifier’s first linear layer and GELU; retain the verified architecture, loss, schedule, dropout, and evaluation calibration.
mechanism: Dense-representation batch normalization
evidence_used: Spatial channel-mixing, gating, depthwise refinement, augmentation, ensembling, and weight averaging all underperformed the 9,281-correct baseline; this tests a distinct, lightweight classifier-conditioning change while preserving the best feature extractor and adding only 256 learned parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 74.43629279197194, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20210640487670897, "validation_score": 9288.415936557672}

RECENT RESULT
hypothesis: BatchNorm1d with the feature extractor’s verified 0.00623359375 momentum will exceed 9,288 correct predictions by retaining the successful classifier conditioning while producing less noisy evaluation statistics than the default 0.1 momentum.
change: Add BatchNorm1d after the first classifier linear layer, using the same low running-statistics momentum as the convolutional normalization layers.
mechanism: Slowly accumulated dense-feature batch normalization
evidence_used: Default-momentum dense BatchNorm achieved the best available result at 9,288 correct, while every BatchNorm2d layer in that successful feature extractor uses momentum 0.00623359375; applying that verified statistic-accumulation rate isolates whether more stable dense running statistics improve generalization.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 78.69068962498568, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20120860176086425, "validation_score": 9284.416247435514}



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
