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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.48744987486862, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515079498291, "validation_score": 9331.417972304886}
prior_hypothesis: A 0.50850048828125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150680541992.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.47602354199626, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}
prior_hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150985717774.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.46174675016664, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625152359008788, "validation_score": 9331.41797229942}
prior_hypothesis: A 0.50853125 flipped-view weight will retain 9,331 correct predictions while lowering cross-entropy below 0.196251542.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.40315091586672, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515151977539, "validation_score": 9331.417972302352}
prior_hypothesis: A 0.5085078125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151824951173.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.50850390625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.1962515151977539.
change: Shift evaluation fusion to 0.49149609375 original-view and 0.50850390625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of flipped-view fusion plateau
evidence_used: The verified 0.5085078125 weight retained 9,331 correct and improved cross-entropy over every higher tested weight; bisecting toward the unresolved 0.5085 point is the smallest informative refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.50850390625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.1962515151977539.
change: Shift evaluation fusion to 0.49149609375 original-view and 0.50850390625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of flipped-view fusion plateau
evidence_used: The verified 0.5085078125 weight retained 9,331 correct and achieved the lowest observed cross-entropy; the previous 0.50850390625 attempt timed out and supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.70088449981995, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625151023864745, "validation_score": 9331.417972304085}

RECENT RESULT
hypothesis: A 0.508501953125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151023864745.
change: Shift evaluation fusion to 0.491498046875 original-view and 0.508501953125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of flipped-view fusion plateau
evidence_used: Successive reductions through the verified 0.50850390625 weight retained 9,331 correct while lowering cross-entropy; bisecting toward the unresolved 0.5085 point is the smallest informative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.62667120806873, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150985717774, "validation_score": 9331.417972304218}

RECENT RESULT
hypothesis: A 0.5085009765625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150985717774.
change: Shift evaluation fusion to 0.4914990234375 original-view and 0.5085009765625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of the flipped-view fusion plateau
evidence_used: Every verified reduction from 0.509 through 0.508501953125 retained 9,331 correct while monotonically lowering cross-entropy; bisecting the remaining interval toward 0.5085 is the smallest conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.47602354199626, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150680541992, "validation_score": 9331.417972305286}

RECENT RESULT
hypothesis: A 0.50850048828125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150680541992.
change: Shift evaluation fusion to 0.49149951171875 original-view and 0.50850048828125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of the flipped-view fusion plateau
evidence_used: The verified 0.5085009765625 weight retained 9,331 correct and achieved the lowest observed cross-entropy; bisecting its remaining interval toward 0.5085 is the smallest informative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.48744987486862, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515079498291, "validation_score": 9331.417972304886}

RECENT RESULT
hypothesis: A 0.508500732421875 flipped-view weight will retain 9,331 correct predictions and reduce cross-entropy below 0.19625150680541992.
change: Set evaluation fusion to 0.491499267578125 original-view and 0.508500732421875 flipped-view weighting while retaining the verified 1.184 calibration and all training behavior.
mechanism: Bracketed fusion-weight refinement
evidence_used: The best verified weight is 0.5085009765625; neighboring 0.50850048828125 and 0.508501953125 weights both have higher cross-entropy while retaining 9,331 correct, so bisecting the narrower interval toward the lower neighbor directly refines the apparent local optimum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.508500732421875 flipped-view weight will retain 9,331 correct predictions and reduce cross-entropy below 0.19625150680541992.
change: Set evaluation fusion to 0.491499267578125 original-view and 0.508500732421875 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Bracketed fusion-weight refinement
evidence_used: The best verified weight is 0.5085009765625; the lower neighbor 0.50850048828125 has higher cross-entropy with the same 9,331 correct predictions, so their midpoint directly refines the bracket. The prior midpoint verification timed out and provided no contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.508500732421875 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Set evaluation fusion to 0.491499267578125 original-view and 0.508500732421875 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Bracketed fusion-weight refinement
evidence_used: The best verified weight is 0.5085009765625, while the lower neighbor 0.50850048828125 has higher cross-entropy with the same 9,331 correct predictions. Their midpoint remains the most direct unresolved refinement; its two prior verifications timed out without contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.50850146484375 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Set evaluation fusion to 0.49149853515625 original-view and 0.50850146484375 flipped-view weighting while retaining the verified 1.184 calibration and all training behavior.
mechanism: Upper-half bracketed fusion-weight refinement
evidence_used: The best verified weight, 0.5085009765625, lies between worse same-accuracy neighbors at 0.50850048828125 and 0.508501953125. Three lower-midpoint attempts timed out without performance evidence, so bisecting the untested upper half is the most informative remaining refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.87354149995372, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625150833129884, "validation_score": 9331.417972304751}

RECENT RESULT
hypothesis: Moving exactly one representable float32 step below the best verified fusion weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625150680541992.
change: Use complementary evaluation-fusion coefficients 0.49149906635284424 and 0.5085009336471558, retaining calibration and training behavior.
mechanism: Float32-adjacent fusion refinement
evidence_used: The best weight, 0.5085009765625, is bracketed by worse same-accuracy results at 0.50850048828125 and 0.50850146484375. Their cross-entropies place the interpolated minimum slightly below the center, motivating the immediately lower representable float32 coefficient.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Moving one representable float32 step below the best verified flipped-view weight will retain 9,331 correct predictions and lower validation cross-entropy below 0.19625150680541992.
change: Replace the current evaluation-fusion coefficients with complementary weights 0.49149906635284424 and 0.5085009336471558 while preserving calibration and training behavior.
mechanism: Float32-adjacent fusion refinement
evidence_used: The best verified weight, 0.5085009765625, outperformed same-accuracy neighbors on both sides; interpolation favors a slightly lower weight, and the sole prior verification of this adjacent value timed out without contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The immediately lower representable float32 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.
change: Replace evaluation fusion with exactly complementary float32 coefficients one representable step below the current effective flipped-view weight.
mechanism: Float32-adjacent fusion refinement
evidence_used: The current weight outperformed verified same-accuracy neighbors on both sides, and their cross-entropies interpolate to a minimum slightly below it. Two attempts at this adjacent value timed out but supplied no contrary performance evidence.
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
