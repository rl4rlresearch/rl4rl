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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.23899808316492, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19611616287231445, "validation_score": 9286.418019600036}
prior_hypothesis: Temperature 0.925 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.

## Recent verification evidence

RECENT RESULT
hypothesis: Temperature 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Divide the final evaluation ensemble logits by 1.03; training and predicted classes remain unchanged.
mechanism: Mild post-ensemble temperature calibration
evidence_used: Stronger tail averaging reduced cross-entropy to 0.197021 but lost predictions, so calibration should target the tie-break metric without moving decision boundaries; the prior 1.03 attempt timed out and remains untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Vectorizing the unchanged 12-weight TTA ensemble and applying temperature 1.03 will preserve all 9,286 argmax predictions, reduce validation cross-entropy below 0.197145, and avoid the recurrent verification timeouts.
change: Evaluate the ten augmented views in two batched backbone calls instead of ten sequential calls, then divide the ensemble logits by 1.03.
mechanism: Vectorized temperature-calibrated test-time augmentation
evidence_used: The current design has the best observed correct count, while prior temperature-1.03 attempts timed out without testing accuracy; positive temperature scaling preserves argmax exactly, and batching the existing views reduces evaluation overhead without changing their weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Pairwise TTA batching will avoid the timeout of five-view batches while preserving the 9,286 predictions, and temperature 1.03 will lower cross-entropy without changing argmax classes.
change: Evaluate the unchanged weighted 12-view ensemble in batches of at most two views, then divide its logits by 1.03.
mechanism: Memory-bounded pairwise TTA with temperature calibration
evidence_used: The sequential ensemble achieved the best 9,286 correct, while the prior five-view vectorization timed out; smaller batches retain its computation with lower peak memory, and positive temperature scaling preserves argmax exactly.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending BatchNorm running statistics with their sparse tail averages will exceed 9,286 correct predictions by aligning evaluation normalization with the already beneficial 50:50 tail-averaged parameters.
change: Track BatchNorm running means and variances at every existing tail sample and blend them 50:50 with their tail averages at the final step.
mechanism: Tail-consistent BatchNorm state averaging
evidence_used: Half-strength parameter averaging improved validation_correct from 9,280 to 9,286, but the current implementation leaves BatchNorm running statistics at their final-iterate values, creating a normalization mismatch with the blended weights.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Apply temperature 1.03 to the existing sequential 12-weight TTA ensemble without changing training or evaluation workload.
mechanism: Mild post-ensemble temperature calibration
evidence_used: The current design has the best observed correct count; stronger tail averaging reduced cross-entropy but changed decision boundaries, while prior temperature attempts timed out and therefore provide no accuracy or calibration evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 79.8308060830459, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19794324645996095, "validation_score": 9286.41738204333}

RECENT RESULT
hypothesis: Temperature 0.97 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Divide the existing sequential 12-weight TTA ensemble logits by 0.97 without changing training, parameters, or evaluation workload.
mechanism: Mild post-ensemble logit sharpening
evidence_used: Temperature 1.03 preserved 9,286 correct but worsened cross-entropy from 0.197145 to 0.197943, directly indicating that the locally favorable calibration direction is sharpening rather than softening.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.97 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.
change: Divide the existing sequential TTA ensemble logits by 0.97 without changing training, parameters, or evaluation workload.
mechanism: Mild post-ensemble logit sharpening
evidence_used: Temperature 1.03 preserved 9,286 correct but worsened cross-entropy from 0.197145 to 0.197943, directly indicating that sharpening is the favorable local calibration direction; the prior 0.97 run timed out and did not test this claim.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 67.59501712513156, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19656033325195313, "validation_score": 9286.417864428651}

RECENT RESULT
hypothesis: Dynamically reweighting each convolutional stage from its image-wide channel context will exceed 9,286 validation-correct predictions while remaining within the parameter and runtime limits.
change: Add lightweight squeeze-excitation gates after the 32-, 64-, and 96-channel stages; zero-initialize their final projections so the network begins as the proven static backbone and learns image-dependent channel routing.
mechanism: Identity-initialized input-conditioned channel routing
evidence_used: Calibration and tail-averaging changes plateaued at 9,286 correct, while the relational-token head timed out. This challenges the shared assumption of static image-independent channel processing with only 7,408 additional parameters and negligible spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.925 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.
change: Sharpen the unchanged sequential TTA ensemble by dividing its logits by 0.925 instead of 0.97.
mechanism: Evidence-guided post-ensemble temperature calibration
evidence_used: Temperature 1.03 worsened cross-entropy to 0.197943, while 0.97 improved it to 0.196560 with identical predictions; these measured points indicate the local optimum lies further in the sharpening direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.925 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.
change: Divide the unchanged sequential TTA ensemble logits by 0.925 instead of 0.97.
mechanism: Evidence-guided post-ensemble temperature sharpening
evidence_used: Temperature 1.03 worsened cross-entropy to 0.197943, whereas 0.97 improved it to 0.196560 with identical predictions, indicating the local optimum lies further in the sharpening direction; the previous 0.925 verification timed out and produced no contrary result.
result: training did not finish within the verification time limit

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
