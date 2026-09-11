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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.70088449981995, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625151023864745, "validation_score": 9331.417972304085}
prior_hypothesis: A 0.50850390625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.1962515151977539.

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
hypothesis: A 0.50875 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251798 by moving toward equal fusion.
change: Change evaluation fusion from 0.490/0.510 to 0.49125/0.50875 while retaining the verified 1.184 calibration.
mechanism: Conservative left-edge refinement of flipped-view fusion
evidence_used: Weights 0.509 and 0.510 both achieved 9,331 correct, with 0.509 producing lower cross-entropy; the unresolved 0.5085 trial motivates testing the conservative midpoint between it and the verified 0.509 design.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.1360278329812, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962516471862793, "validation_score": 9331.417972256235}

RECENT RESULT
hypothesis: A 0.508625 flipped-view weight will retain 9,331 correct predictions while lowering cross-entropy below 0.196251647 by reducing asymmetry relative to the best verified 0.50875 fusion.
change: Replace equal evaluation-time fusion with 0.491375 original-view and 0.508625 flipped-view weighting, retaining the verified 1.184 calibration.
mechanism: Midpoint refinement of flipped-view fusion plateau
evidence_used: Flipped weight 0.50875 achieved 9,331 correct with lower cross-entropy than 0.509, while equal fusion achieved only 9,330; 0.508625 conservatively bisects the interval between the best verified point and the unresolved 0.5085 trial.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 66.21577908284962, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625157890319825, "validation_score": 9331.417972280095}

RECENT RESULT
hypothesis: A 0.5085625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251579 by moving closer to equal fusion.
change: Use 0.4914375 original-view and 0.5085625 flipped-view evaluation weights, retaining the verified 1.184 calibration.
mechanism: Left-edge bisection of flipped-view fusion
evidence_used: Weight 0.508625 achieved 9,331 correct and improved cross-entropy over 0.50875 and 0.509; 0.5085625 is the conservative midpoint toward the unresolved 0.5085 trial.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.5085625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251579.
change: Shift evaluation fusion from 0.491375/0.508625 to 0.4914375/0.5085625, retaining the verified 1.184 calibration.
mechanism: Left-edge bisection of flipped-view fusion
evidence_used: The verified 0.508625 weight retained 9,331 correct and improved cross-entropy over 0.50875 and 0.509; the prior 0.5085625 attempt timed out and supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 63.67797491583042, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625154190063476, "validation_score": 9331.417972293022}

RECENT RESULT
hypothesis: A 0.50853125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251542 by moving slightly closer to equal fusion.
change: Shift evaluation fusion to 0.49146875 original-view and 0.50853125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of flipped-view fusion plateau
evidence_used: The verified 0.5085625 weight retained 9,331 correct and improved cross-entropy over 0.508625, 0.50875, and 0.509; bisecting toward the unresolved 0.5085 point conservatively probes the plateau’s left edge.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.50853125 flipped-view weight will retain 9,331 correct predictions while lowering cross-entropy below 0.196251542.
change: Shift evaluation fusion to 0.49146875 original-view and 0.50853125 flipped-view weighting, retaining the verified 1.184 calibration.
mechanism: Left-edge bisection of flipped-view fusion plateau
evidence_used: The verified 0.5085625 weight retained 9,331 correct and improved cross-entropy over every higher tested weight; the prior 0.50853125 attempt timed out and supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.46174675016664, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625152359008788, "validation_score": 9331.41797229942}

RECENT RESULT
hypothesis: A 0.508515625 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.196251524 by moving closer to equal fusion.
change: Replace equal evaluation fusion with 0.491484375 original-view and 0.508515625 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of flipped-view fusion plateau
evidence_used: Successive reductions from 0.509 through 0.50853125 retained 9,331 correct while monotonically lowering cross-entropy; bisecting the remaining interval toward the unresolved 0.5085 point is the most conservative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 73.79640879202634, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625151824951173, "validation_score": 9331.417972301286}

RECENT RESULT
hypothesis: A 0.5085078125 flipped-view weight will retain 9,331 correct predictions while reducing cross-entropy below 0.19625151824951173.
change: Shift evaluation fusion to 0.4914921875 original-view and 0.5085078125 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.
mechanism: Left-edge bisection of the flipped-view fusion plateau
evidence_used: Successive reductions from 0.509 through 0.508515625 retained 9,331 correct while monotonically lowering cross-entropy; bisecting the remaining interval toward the unresolved 0.5085 point is the most conservative informative refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.40315091586672, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962515151977539, "validation_score": 9331.417972302352}

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
