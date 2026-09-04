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
hypothesis: Temperature 0.95 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.
change: Divide the unchanged sequential TTA ensemble logits by 0.95 instead of 0.97.
mechanism: Conservative post-ensemble temperature sharpening
evidence_used: Moving from temperature 1.03 to 0.97 improved cross-entropy from 0.197943 to 0.196560 with identical predictions; 0.95 continues that measured direction conservatively, while the more aggressive 0.925 attempts produced only timeouts.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.925 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.
change: Sharpen the unchanged 12-view ensemble by changing its evaluation temperature from 0.97 to 0.925.
mechanism: Evidence-fitted post-ensemble temperature sharpening
evidence_used: Temperature 1.03 worsened cross-entropy to 0.197943, while 0.97 improved it to 0.196560 with identical predictions; this directly supports further sharpening, and previous 0.925 attempts timed out without producing contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.23899808316492, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19611616287231445, "validation_score": 9286.418019600036}

RECENT RESULT
hypothesis: Temperature 0.91 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196116.
change: Sharpen the unchanged 12-view ensemble by changing its evaluation temperature from 0.925 to 0.91.
mechanism: Evidence-fitted post-ensemble temperature calibration
evidence_used: Cross-entropy improved from 0.197943 at temperature 1.03 to 0.196560 at 0.97 and 0.196116 at 0.925 with identical predictions; a quadratic interpolation of these measured points places the local minimum near 0.91.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.91 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196116.
change: Change the unchanged 12-view ensemble’s evaluation temperature from 0.925 to 0.91.
mechanism: Evidence-fitted post-ensemble temperature sharpening
evidence_used: Cross-entropy improved monotonically from 0.197943 at temperature 1.03 to 0.196560 at 0.97 and 0.196116 at 0.925 with identical predictions; quadratic interpolation places the estimated minimum near 0.91, and the previous 0.91 attempt timed out without testing the hypothesis.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 77.0140182080213, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.1960927104949951, "validation_score": 9286.418027796351}

RECENT RESULT
hypothesis: Temperature 0.912 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960927105.
change: Change only the 12-view ensemble’s evaluation temperature from 0.91 to 0.912.
mechanism: Quadratic-fit post-ensemble temperature calibration
evidence_used: Temperatures 0.97, 0.925, and 0.91 produced cross-entropies of 0.1965603333, 0.1961161629, and 0.1960927105 with identical predictions; quadratic interpolation of these local measurements estimates the minimum near 0.9119.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 79.54549562511966, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19609205169677735, "validation_score": 9286.418028026597}

RECENT RESULT
hypothesis: Temperature 0.91213 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.1960920517.
change: Change only the 12-view ensemble’s evaluation temperature from 0.912 to 0.91213.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: Temperatures 0.91, 0.912, and 0.925 produced cross-entropies of 0.1960927105, 0.1960920517, and 0.1961161629 with identical predictions; quadratic interpolation of these local measurements estimates the minimum near 0.91213.
result: training did not finish within the verification time limit

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
