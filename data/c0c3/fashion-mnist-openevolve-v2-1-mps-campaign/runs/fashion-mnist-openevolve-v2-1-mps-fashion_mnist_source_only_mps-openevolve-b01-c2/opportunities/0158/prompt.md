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
change: Set evaluation fusion to complementary float32 coefficients 0.49149906635284424 and 0.5085009336471558, preserving calibration and all training behavior.
mechanism: Float32-adjacent fusion refinement
evidence_used: The best verified weight, 0.5085009765625, beats same-accuracy neighbors on both sides, and their cross-entropies indicate a minimum slightly below it. Prior attempts at this adjacent value timed out without supplying contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The immediately lower representable float32 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Set evaluation fusion to complementary coefficients 0.49149906635284424 and 0.5085009336471558 while preserving calibration and training behavior.
mechanism: Float32-adjacent fusion refinement
evidence_used: The best verified weight, 0.5085009765625, beats same-accuracy neighbors on both sides, whose cross-entropies interpolate to a minimum slightly below it; prior attempts at this adjacent value timed out without performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below the current 0.19625152359008788.
change: Replace the current evaluation-fusion coefficients with the best verified complementary weights while preserving calibration and training behavior.
mechanism: Verified evaluation-fusion optimum
evidence_used: Reference Design 1 achieved the highest reported validation score, with 9,331 correct predictions and 0.19625150680541992 cross-entropy.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.05727233295329, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}

RECENT RESULT
hypothesis: Moving one representable float32 step below the best verified flipped-view weight will retain 9,331 correct predictions while reducing validation cross-entropy below 0.19625150680541992.
change: Replace the evaluation-fusion coefficients with exactly complementary float32 weights immediately below the current effective flipped-view weight.
mechanism: Float32-adjacent fusion refinement
evidence_used: The current weight outperformed verified same-accuracy neighbors on both sides, whose cross-entropies interpolate to a minimum slightly below it; prior attempts at this adjacent value timed out without providing contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: One representable float32 increase in the calibration scale will preserve all 9,331 argmax predictions while reducing validation cross-entropy below 0.19625150680541992.
change: Increase only the positive evaluation-logit scale from its current float32 value to its immediate successor, leaving fusion and training unchanged.
mechanism: Float32-adjacent confidence calibration
evidence_used: The best verified design already fixes the primary objective at 9,331 correct; positive logit rescaling cannot change its argmax predictions, while label smoothing and view averaging motivate testing slightly stronger confidence calibration as an orthogonal tie-break refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the best verified flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy from 0.1962515079498291 to approximately 0.19625150680541992.
change: Replace the current evaluation-fusion coefficients with the best verified complementary weights while preserving calibration and training behavior.
mechanism: Verified evaluation-fusion optimum
evidence_used: Reference Designs 1 and 2 independently achieved the highest reported score with flipped-view weight 0.5085009765625; the current lower neighboring weight produced slightly higher cross-entropy at the same accuracy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the best verified flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below the current 0.1962515151977539.
change: Replace only the evaluation-fusion coefficients with the best verified complementary weights.
mechanism: Verified evaluation-fusion optimum
evidence_used: Reference Designs 1 and 3 achieved the highest reported score using flipped-view weight 0.5085009765625, with 9,331 correct and cross-entropy 0.19625150680541992.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the evaluation-logit scale by one float32 step will preserve all 9,331 argmax predictions while reducing validation cross-entropy below 0.19625150680541992.
change: Replace the 1.184 evaluation calibration with its immediate float32 successor, leaving fusion and training unchanged.
mechanism: Float32-adjacent confidence calibration
evidence_used: The best verified design has 9,331 correct predictions; positive logit scaling preserves those predictions, while the prior attempt at this stronger calibration timed out and supplied no contrary performance evidence.
result: training did not finish within the verification time limit

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
