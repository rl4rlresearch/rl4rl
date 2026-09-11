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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 47.86802029190585, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21537131576538085, "validation_score": 9267.411396906866}
prior_hypothesis: Raising the geometric component from 10% to 11% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2154026180267334.

## Recent verification evidence

RECENT RESULT
hypothesis: A residual four-head attention layer over the 7×7 feature grid will exceed 9,267 correct predictions by learning content-dependent, garment-wide relationships while preserving the spatial layout required by the existing classifier.
change: Insert a pre-normalized, position-aware self-attention block before the flattened classifier, with a learnable residual scale initialized conservatively; total learned parameters become 249,059.
mechanism: Position-aware global self-attention residual
evidence_used: Static dilated wider-context extraction regressed to 9,238 correct, while position-free pooling produced 9,253/9,249, indicating that spatial layout is load-bearing but fixed receptive-field expansion is insufficient. Dense input-conditioned attention tests a distinct global interaction mechanism without pooling away positions or replacing the successful flattened head.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 9.832581818103790283203125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Increase the geometric-logit contribution to the midpoint between the highest verified 9,267-correct blend and the lowest verified 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.8325817584991455078125% blend retained 9,267 correct, while 9.83258187770843505859375% lost one prediction; their midpoint is the most informative remaining probe of the decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 42.99846224999055, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798034667968, "validation_score": 9267.41138449647}

RECENT RESULT
hypothesis: Averaging the final 20% of the optimization trajectory with decay 0.98 will exceed 9,267 correct predictions by reducing late-step parameter noise without changing the successful representation or inference ensemble.
change: Track a bias-corrected EMA of all floating model parameters and normalization buffers during the final training phase, then install the averaged state after the last optimizer step.
mechanism: Late-training exponential weight averaging
evidence_used: Geometric-blend refinements have exhausted the float-level decision boundary without exceeding 9,267 correct, while wider-context architectural changes regressed or timed out; late weight averaging tests a distinct generalization mechanism while preserving the best architecture and TTA.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 70.23605033312924, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21561995849609375, "validation_score": 9254.411312759803}

RECENT RESULT
hypothesis: A lightweight squeeze-excitation gate will exceed 9,267 correct predictions by adding input-dependent global context while preserving the spatially flattened representation and avoiding the timed-out attention layer’s quadratic cost.
change: Add a 14-unit channel gate after the residual block, initialized to an exact identity mapping and adding 1,638 learned parameters.
mechanism: Identity-initialized global channel recalibration
evidence_used: Full spatial attention timed out, while position-free pooling regressed; channel recalibration uses global context only to modulate intact 7×7 feature maps, testing a cheaper interaction mechanism without discarding load-bearing spatial layout.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 235072, "training_seconds": 59.707203374942765, "validation_accuracy": 0.9224, "validation_correct": 9224, "validation_cross_entropy": 0.21966347122192384, "validation_score": 9224.409949147283}

RECENT RESULT
hypothesis: Training the dominant ensemble loss with the same arithmetic/geometric aggregation used during validation will exceed 9,267 correct predictions by eliminating the mismatch between average-logit training and predominantly average-probability inference.
change: Replace the geometric-only sampled ensemble loss with the validation-time calibrated arithmetic/geometric logit blend while preserving the existing augmentation distribution, architecture, and loss schedule.
mechanism: Inference-aligned arithmetic ensemble supervision
evidence_used: The best design assigns 90.1674% of its inference logits to arithmetic probability averaging, while its ensemble training loss uses only averaged raw logits; repeated geometric-weight refinements exhausted the current decision boundary without exceeding 9,267 correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 56.82175154192373, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2205922119140625, "validation_score": 9237.409637219638}

RECENT RESULT
hypothesis: Sampling central training crops with the validation ensemble’s exact 1:2:1 per-axis distribution will exceed 9,267 correct predictions by concentrating updates on the most heavily weighted evaluation views.
change: Replace uniform sampling over the central 3×3 offsets with independent triangular sampling produced by summing two Bernoulli draws per axis.
mechanism: Evaluation-matched triangular crop sampling
evidence_used: Geometric-blend refinements plateaued at 9,267 correct, while wholesale inference-aligned supervision regressed to 9,237; this surgical change preserves the successful loss and architecture while correcting the remaining central-crop distribution mismatch.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 54.94590404210612, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.2151459373474121, "validation_score": 9247.411473210446}

RECENT RESULT
hypothesis: Applying per-image rotations up to ±7.5° on half of training examples will exceed 9,267 correct predictions by learning a missing pose invariance without altering the successful architecture or validation ensemble.
change: Add probability-0.5 bilinear random rotation in `prepare_training_batch`, using reflection padding and preserving the existing batch interface.
mechanism: Mild stochastic rotation augmentation
evidence_used: Matching training to the existing validation crop distribution reduced accuracy to 9,247, and inference-aligned supervision reduced it to 9,237; this motivates testing a distinct, semantically safe augmentation axis rather than further refining the exhausted translation/ensemble formulation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 60.80900979205035, "validation_accuracy": 0.9203, "validation_correct": 9203, "validation_cross_entropy": 0.2229315963745117, "validation_score": 9203.408853611667}

RECENT RESULT
hypothesis: A zero-initialized 7×7 residual bottleneck will exceed 9,267 correct predictions by adding local feature depth while preserving the successful spatial representation and initial network behavior.
change: Add a 36-channel bottleneck residual block after the existing residual stage, bringing the model to 249,386 learned parameters.
mechanism: Identity-initialized local bottleneck refinement
evidence_used: Global channel gating regressed to 9,224 and spatial attention timed out, while prior pooling experiments showed spatial layout is load-bearing; a cheap local residual block adds capacity without pooling, global interactions, or quadratic computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249386, "training_seconds": 62.58914400008507, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.21506749076843262, "validation_score": 9243.411499775773}

RECENT RESULT
hypothesis: Independently sampling the existing uniform full and central crop distributions for every image will exceed 9,267 correct predictions by reducing within-batch augmentation correlation without changing the successful view distribution, architecture, or loss.
change: Replace batch-shared crop offsets with vectorized per-image offsets extracted from the padded 5×5 crop grid.
mechanism: Per-example random crop sampling
evidence_used: Evaluation-matched triangular crop sampling regressed to 9,247 correct, showing that changing the crop distribution is harmful; this patch preserves the baseline’s uniform marginals while making each update cover more of that proven distribution.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 51.792152625042945, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.2153379768371582, "validation_score": 9250.41140819223}

RECENT RESULT
hypothesis: Raising the geometric component to 10% only when the arithmetic and geometric predictors agree will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Use a 10% geometric blend on endpoint-agreement examples and preserve the verified 9.832581520080566% blend elsewhere.
mechanism: Prediction-safe conditional geometric blending
evidence_used: Globally increasing the geometric weight beyond 9.832581520080566% flipped one validation prediction; conditioning the increase on endpoint agreement prevents such argmax changes while testing the locally favorable calibration direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 45.734137249877676, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2154026180267334, "validation_score": 9267.411386311485}

RECENT RESULT
hypothesis: A learned affine warp of the 7×7 feature grid will exceed 9,267 correct predictions by correcting image-specific pose and translation before classification, while retaining the spatial layout that pooling-based designs lost.
change: Replace the fixed-grid assumption with an identity-initialized, bounded spatial transformer over stem features; this adds 11,766 parameters for a total of 245,200.
mechanism: Bounded content-conditioned feature canonicalization
evidence_used: Uniform crop changes and forced rotation augmentation regressed to 9,250 and 9,203, while position-free pooling also regressed, indicating that indiscriminate invariance is harmful and spatial layout is load-bearing. A learned per-image canonicalization can selectively correct nuisance geometry without discarding that layout.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Raising the geometric component from 10% to 11% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2154026180267334.
change: Increase the agreement-only geometric-logit weight to 11%, while preserving the verified 9.832581520080566% blend on disagreement examples.
mechanism: Prediction-safe agreement-stratified blend continuation
evidence_used: Moving agreement examples from the baseline blend to 10% geometric retained 9,267 correct and reduced cross-entropy from 0.21540798034667968 to 0.2154026180267334; because both endpoint predictors have the same argmax on these examples, further convex interpolation cannot change their predicted class.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 47.86802029190585, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21537131576538085, "validation_score": 9267.411396906866}



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
