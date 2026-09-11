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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.05727233295329, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}
prior_hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below the current 0.19625152359008788.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.47602354199626, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}
prior_hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150985717774.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.48744987486862, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515079498291, "validation_score": 9331.417972304886}
prior_hypothesis: A 0.50850048828125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150680541992.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.40315091586672, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515151977539, "validation_score": 9331.417972302352}
prior_hypothesis: A 0.5085078125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151824951173.

## Recent verification evidence

RECENT RESULT
hypothesis: The immediately lower representable float32 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Set evaluation fusion to exactly complementary coefficients 0.49149906635284424 and 0.5085009336471558 while preserving calibration and training behavior.
mechanism: Float32-adjacent fusion refinement
evidence_used: The best verified weight is bracketed by worse same-accuracy neighbors, and their cross-entropies interpolate to a minimum slightly below it; prior attempts at this adjacent value timed out without contrary performance evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.01238483306952, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150718688966, "validation_score": 9331.417972305151}

RECENT RESULT
hypothesis: Restoring the best verified flipped-view weight will retain 9,331 correct predictions and reduce cross-entropy from 0.1962515079498291 to approximately 0.19625150680541992.
change: Replace only the evaluation-fusion coefficients with the best verified complementary weights.
mechanism: Verified evaluation-fusion optimum
evidence_used: Reference Designs 1 and 2 achieved the highest reported validation score at flipped-view weight 0.5085009765625, while the current 0.50850048828125 weight produced higher cross-entropy with the same accuracy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The immediately higher representable float32 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Test the unexplored upper float32 neighbor of the best verified fusion weight using exactly complementary coefficients.
mechanism: Float32-adjacent fusion search
evidence_used: Weight 0.5085009765625 achieved the best verified cross-entropy; its immediate lower neighbor was slightly worse, while the more distant upper weight in the current design is also worse, making the immediate upper neighbor the most informative remaining refinement.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 85.09141491702758, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150756835938, "validation_score": 9331.417972305018}

RECENT RESULT
hypothesis: Increasing the positive evaluation-logit scale by one float32 step will preserve all 9,331 argmax predictions while reducing validation cross-entropy below 0.19625150680541992.
change: Replace the 1.184 evaluation calibration with its immediate float32 successor, leaving training and the best verified fusion weights unchanged.
mechanism: Float32-adjacent confidence calibration
evidence_used: The current design has the best verified 9,331-correct result; positive scaling cannot change argmax predictions, and prior attempts at this adjacent scale timed out without contrary performance evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.37858879193664, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515079498291, "validation_score": 9331.417972304886}

RECENT RESULT
hypothesis: Decreasing the positive evaluation-logit scale by one float32 step will preserve all 9,331 argmax predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Replace the 1.184 evaluation calibration with its immediate float32 predecessor, leaving training and the best verified fusion weights unchanged.
mechanism: Float32-adjacent confidence calibration
evidence_used: The immediate higher float32 scale preserved 9,331 correct but worsened cross-entropy to 0.1962515079498291, indicating the local calibration gradient favors a lower scale.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.55424012499861, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515106201172, "validation_score": 9331.417972303952}

RECENT RESULT
hypothesis: Training the ensemble objective with the best verified 0.5085009765625 flipped-view weighting will produce a validation score above 9331.417972305286 by aligning learned fusion with inference.
change: Restore the best verified evaluation coefficients and use the same asymmetric mixture for ensemble logits during training.
mechanism: Evaluation-aligned asymmetric view training
evidence_used: The 0.5085009765625 flipped-view evaluation weight achieved the best verified result twice, while tested neighboring weights were worse; this indicates a consistent view-quality asymmetry that the current equal-weight training ensemble ignores.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.76690320903435, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.19695564041137695, "validation_score": 9307.417726424539}

RECENT RESULT
hypothesis: Using the independently verified flipped-view weight 0.5085009765625 will retain 9,331 correct predictions while reducing validation cross-entropy below the current 0.1962515151977539.
change: Replace only the inference-time fusion coefficients with the best verified complementary weights.
mechanism: Verified evaluation-fusion restoration
evidence_used: Reference Designs 1 and 3 both achieved the highest reported score with these coefficients; tested adjacent and more distant weights had higher cross-entropy at the same accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A positive scale that increases slightly for above-average-confidence images and decreases for below-average-confidence images will preserve exactly 9,331 argmax predictions while reducing validation cross-entropy below 0.19625150680541992.
change: Retain the best verified flip fusion and global scale, then add a small batch-centered, per-image confidence adjustment that cannot change predicted classes.
mechanism: Confidence-adaptive positive logit calibration
evidence_used: Both adjacent float32 global scales worsened cross-entropy at the same 9,331 correct, indicating global calibration is locally optimized; confidence-conditional scaling tests an orthogonal calibration dimension while mathematically preserving every argmax.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Evaluating a 0.99-decay EMA of learned parameters will exceed 9,331 correct predictions by reducing late-training weight noise while preserving the verified architecture, loss, and inference fusion.
change: Maintain an EMA after every optimizer step and replace model parameters with it after the final step.
mechanism: End-of-training exponential weight averaging
evidence_used: Adjacent calibration and fusion refinements preserved 9,331 correct but did not improve the score, while asymmetric fusion training reduced correctness to 9,307; parameter averaging tests an orthogonal generalization improvement without changing the successful objective or inference rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A computationally sparse approximation of the previously unverified 0.99-decay EMA will exceed 9,331 correct predictions without breaching the training-time limit.
change: Restore the best verified flip-fusion coefficients and update an equivalent 0.99-per-step EMA every 16 steps, installing the averaged parameters after the final optimizer step.
mechanism: Periodic end-of-training exponential weight averaging
evidence_used: Fine-grained fusion and calibration changes have saturated at 9,331 correct, while per-step EMA timed out without performance evidence; periodic updates test the same orthogonal generalization mechanism with substantially less overhead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 65.66402799985372, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.1967038215637207, "validation_score": 9326.417814325476}

RECENT RESULT
hypothesis: Arithmetic probability fusion of the original and flipped views will exceed 9,331 correct predictions, or retain 9,331 while reducing cross-entropy below 0.19625150680541992.
change: Restore the best verified view weights and combine normalized view probabilities instead of raw logits, preserving training, parameters, and inference cost.
mechanism: Probability-space test-time view ensemble
evidence_used: Neighboring logit-fusion weights and calibration scales have saturated at 9,331 correct, while evaluation-aligned training and EMA reduced correctness; probability fusion is an orthogonal inference-only ensemble rule that limits domination by an overconfident disagreeing view.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 72.84868233394809, "validation_accuracy": 0.9329, "validation_correct": 9329, "validation_cross_entropy": 0.19681876029968262, "validation_score": 9329.417774199892}

RECENT RESULT
hypothesis: Applying one-pixel translations to half of each batch during the first 75% of training, followed by clean-only optimization, will exceed 9,331 correct predictions by improving spatial robustness without creating a persistent train-validation mismatch.
change: Deterministically cycle through eight one-pixel translation directions for alternating examples, preserving half of every batch unchanged and disabling augmentation for the final quarter of training.
mechanism: Conservative translation augmentation with clean late training
evidence_used: Inference-only fusion and calibration refinements have saturated at 9,331 correct, while asymmetric training, EMA, and probability fusion reduced correctness; this motivates an orthogonal, low-overhead augmentation that leaves the verified model, objective, and inference rule intact.
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
