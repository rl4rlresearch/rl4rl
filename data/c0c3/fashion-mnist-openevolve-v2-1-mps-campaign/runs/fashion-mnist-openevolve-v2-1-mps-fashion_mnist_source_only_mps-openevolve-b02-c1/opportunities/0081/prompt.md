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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 79.54549562511966, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19609205169677735, "validation_score": 9286.418028026597}
prior_hypothesis: Temperature 0.912 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960927105.

## Recent verification evidence

RECENT RESULT
hypothesis: Temperature 0.91213 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960920517.
change: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.91213.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.925 yielded cross-entropies of 0.1960927105, 0.1960920517, and 0.1961161629 with identical predictions; quadratic interpolation estimates the minimum near 0.91213, and its only prior verification timed out without testing the claim.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A deeper residual backbone with explicit feature reuse will exceed 9,286 validation-correct predictions while remaining under the parameter and runtime limits.
change: Replace the assumption that every stage must overwrite its input through full convolutions with identity-preserving bottleneck refinements at 14×14 and 7×7; retain the proven training, augmentation, TTA, and classifier head.
mechanism: Parameter-reallocated residual bottleneck refinement
evidence_used: Calibration changes plateaued at 9,286 correct, while the relational-token and squeeze-excitation alternatives timed out. The proposed residual mechanism directly changes representation learning, fits in 239,594 parameters, and uses fewer convolution MACs than the current backbone despite being deeper.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.91213 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960920517.
change: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.91213.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.925 yielded cross-entropies of 0.1960927105, 0.1960920517, and 0.1961161629 with identical predictions; quadratic interpolation estimates the minimum near 0.91213, while prior attempts timed out without contradicting the hypothesis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 78.49580687494017, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19609205322265624, "validation_score": 9286.418028026064}

RECENT RESULT
hypothesis: Adding a lightweight residual block at 3×3 resolution will exceed 9,286 correct predictions while remaining below 250,000 parameters and avoiding the runtime cost that caused earlier deeper residual designs to time out.
change: Add a zero-initialized 96-channel depthwise-separable residual block after the final pooling layer, increasing the model from 232,682 to 243,146 learned parameters.
mechanism: Identity-initialized low-resolution depthwise residual refinement
evidence_used: Temperature calibration has plateaued at 9,286 correct, so further improvement requires changing representation learning. The earlier residual design timed out; concentrating one refinement block at 3×3 resolution tests the same promising feature-reuse mechanism with negligible spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.912028 will preserve all 9,286 predictions while reducing validation cross-entropy below 0.19609205169677735.
change: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.912028.
mechanism: Local quadratic temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.91213 produced cross-entropies of 0.1960927105, 0.1960920517, and 0.1960920532 with identical predictions; quadratic interpolation of these closest measurements estimates the minimum near 0.912028.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.912028 will preserve all 9,286 predictions while reducing validation cross-entropy below 0.19609205169677735.
change: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.912028.
mechanism: Local quadratic post-ensemble temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.91213 produced cross-entropies of 0.1960927105, 0.1960920517, and 0.1960920532 with identical predictions; quadratic interpolation estimates the minimum near 0.912028, and its prior verification timed out without testing the hypothesis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.912028 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.19609205169677735.
change: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.912028.
mechanism: Local quadratic post-ensemble temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.91213 produced cross-entropies of 0.1960927105, 0.1960920517, and 0.1960920532 with identical predictions; quadratic interpolation estimates the minimum near 0.912028, while previous attempts timed out without testing the hypothesis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing each centered view’s weight from 2.0 to 2.25 will exceed 9,286 correct predictions by reducing boundary-artifact influence from translated views without increasing runtime or parameters.
change: Give the original and horizontally flipped centered views slightly more ensemble weight, update the normalization constant, and retain the proven temperature.
mechanism: Center-biased test-time augmentation fusion
evidence_used: Temperature calibration has plateaued at 9,286 correct, while representation-changing attempts timed out; reweighting the existing 12-view ensemble is the lowest-cost mechanism that can change argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting both centered views at 2.25 will exceed 9,286 correct predictions by reducing boundary artifacts from translated views without changing training or inference cost materially.
change: Increase the original and flipped centered-view weights from 2.0 to 2.25 and normalize the 12-view log-probability mixture by its new total weight of 12.5.
mechanism: Center-biased test-time augmentation fusion
evidence_used: Temperature calibration plateaued at 9,286 correct; the prior center-weighting attempt timed out, so this argmax-changing, parameter-free hypothesis remains untested rather than contradicted.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting both centered views at 2.25 will exceed 9,286 correct predictions by reducing translated-view boundary artifacts.
change: Increase centered-view weights from 2.0 to 2.25 and normalize the probability ensemble by its new total weight of 12.5.
mechanism: Center-biased test-time augmentation fusion
evidence_used: Temperature calibration plateaued at 9,286 correct; two prior verifications of this argmax-changing, parameter-free hypothesis timed out without testing or contradicting it.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using full batches of 50 will reduce optimizer steps from 2,084 to 2,000 and avoid timeout, while weighting both centered views at 2.25 will change ensemble decisions and exceed 9,286 validation-correct predictions.
change: Increase batch size from 48 to 50, increase both centered-view ensemble weights from 2.0 to 2.25, and normalize by the resulting total weight of 12.5.
mechanism: Split-aligned batching with center-biased TTA fusion
evidence_used: Three center-weighted TTA attempts timed out without testing the hypothesis, while the verified baseline took 79.5 seconds and 2,084 steps; batch size 50 exactly divides both the 50,000-image split and 100,000-example exposure budget, producing 2,000 full updates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fusing compressed 7×7 contour features with 3×3 semantic features will exceed 9,286 correct predictions while remaining under 250,000 parameters.
change: Replace the single endpoint classifier with learned fine-detail and deep-semantic branches, concatenating their embeddings for class prediction; retain the proven training and TTA procedures.
mechanism: Multi-resolution contour–semantic late fusion
evidence_used: Temperature calibration plateaued at 9,286 correct, while deeper residual refinements timed out. This challenges the load-bearing assumption that all evidence must pass through three pooling operations, adding a distinct representation path with only modest low-cost computation (243,898 parameters).
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
