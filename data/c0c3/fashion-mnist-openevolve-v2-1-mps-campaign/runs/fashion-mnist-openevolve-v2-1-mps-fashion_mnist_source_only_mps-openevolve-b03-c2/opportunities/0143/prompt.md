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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.56152520910837, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144217742919922, "validation_score": 9256.411718573056}
prior_hypothesis: A center-view weight of 3.3828125 will preserve all 9,256 correct predictions while lowering validation cross-entropy below 0.2144234748840332.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 48.07536750007421, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21445224533081056, "validation_score": 9256.411708242891}
prior_hypothesis: Scaling the ensemble logits by 1.35641 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2144522300720215.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 57.43195170792751, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441860122680664, "validation_score": 9256.411719648806}
prior_hypothesis: A center-view weight of 3.3974609375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 66.21664587501436, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441892204284668, "validation_score": 9257.411719540041}
prior_hypothesis: A center-view weight of 3.39599609375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy to 0.2144522300720215.
change: Change only the evaluation-time ensemble logit multiplier from 1.35641 to 1.35605.
mechanism: Verified evaluation-temperature calibration
evidence_used: Reference Design 2 verified the otherwise-identical implementation at 9,256 correct with lower cross-entropy and a strictly higher validation_score than the current design; unsuccessful attempts reported no contradictory measurement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2147829205 to the verified 0.2144522301.
change: Change only the evaluation-time ensemble logit multiplier from 1.30 to 1.35605.
mechanism: Verified evaluation-temperature calibration
evidence_used: Reference Design 2 is otherwise identical and achieved 9,256 correct with the best verified cross-entropy; unsuccessful attempts supplied no contradictory measurement.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the center-view weight from 3.25 to 3.375 will exceed 9,256 correct predictions by reducing shifted-view influence on borderline examples while preserving most translation-ensemble benefit.
change: Increase only the evaluation ensemble’s center-view weight and corresponding normalization constant; leave training and temperature calibration unchanged.
mechanism: Conservative center-view reweighting
evidence_used: The center-biased 3.25:1 ensemble achieved 9,256 correct, while changing shift-augmentation duration or timing reduced correctness. The unverified 3.5 proposal motivates testing a smaller, more conservative step toward the center prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 69.34729949990287, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144234748840332, "validation_score": 9256.411717996514}

RECENT RESULT
hypothesis: Raising the center-view weight from 3.25 to 3.4375 will exceed 9,256 correct predictions by further reducing harmful shifted-view influence, while retaining enough translation averaging to preserve robustness.
change: Increase the evaluation ensemble’s center-view weight to 3.4375, update its normalization constant, and retain the calibration used by the strongest center-weighted reference.
mechanism: Midpoint center-view ensemble reweighting
evidence_used: Increasing the center weight to 3.375 preserved 9,256 correct while lowering cross-entropy from 0.21445223 to 0.21442347; 3.4375 is a conservative midpoint toward the unmeasured 3.5 proposal.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 66.74498116690665, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2144101219177246, "validation_score": 9254.411722523533}

RECENT RESULT
hypothesis: A center-view weight of 3.40625 will preserve all 9,256 correct predictions while lowering cross-entropy below 0.2144234748840332.
change: Move halfway between the successful 3.375 and accuracy-losing 3.4375 center weights, update normalization, and use the strongest center-weighted reference’s calibration.
mechanism: Midpoint search for the center-view weighting boundary
evidence_used: Weight 3.375 retained 9,256 correct at 0.21442347 cross-entropy, while 3.4375 reduced cross-entropy to 0.21441012 but lost two correct predictions; their midpoint efficiently probes the accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 56.49991483311169, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21441672058105468, "validation_score": 9255.411720286394}

RECENT RESULT
hypothesis: A center-view weight of 3.390625 will preserve 9,256 correct predictions while lowering validation cross-entropy below 0.2144234748840332.
change: Move halfway between the accuracy-preserving 3.375 weight and the 3.40625 weight that lost one prediction, update normalization, and adopt the verified 1.35605 calibration.
mechanism: Binary search of the center-view ensemble weight
evidence_used: Weight 3.375 achieved 9,256 correct at 0.21442347 cross-entropy, while 3.40625 achieved 9,255 correct at the lower 0.21441672; their midpoint is the most informative probe of the correctness boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A center-view weight of 3.3828125 will preserve all 9,256 correct predictions while lowering validation cross-entropy below 0.2144234748840332.
change: Increase the evaluation ensemble’s center-view weight from 3.375 to 3.3828125 and update its normalization constant.
mechanism: Conservative center-weight boundary refinement
evidence_used: Weight 3.375 retained 9,256 correct, while 3.40625 lost one prediction despite lower cross-entropy; 3.3828125 is a conservative quarter-step toward that boundary after the intervening 3.390625 verification timed out without performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.56152520910837, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2144217742919922, "validation_score": 9256.411718573056}

RECENT RESULT
hypothesis: A center-view weight of 3.39453125 will preserve 9,256 correct predictions while lowering validation cross-entropy below 0.2144217742919922.
change: Move halfway between the verified 3.3828125 weight and the accuracy-losing 3.40625 weight, update the normalization constant, and use the best verified calibration multiplier.
mechanism: Binary refinement of center-view ensemble weighting
evidence_used: Weight 3.3828125 retained 9,256 correct at 0.2144217743 cross-entropy, while 3.40625 lost one prediction at 0.2144167206; their midpoint is the most informative remaining probe of the correctness boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.50447379192337, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.2144192337036133, "validation_score": 9257.41171943438}

RECENT RESULT
hypothesis: A center-view weight of 3.400390625 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.
change: Move halfway from the best verified weight, 3.39453125, toward the accuracy-losing 3.40625 weight; update normalization and use the verified 1.35605 calibration.
mechanism: Upper-bound refinement of center-view ensemble weighting
evidence_used: Weight 3.39453125 achieved the best result with 9,257 correct, while 3.40625 achieved 9,255 with slightly lower cross-entropy; their midpoint is the most informative probe of the upper accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 77.57328162482008, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21441797485351563, "validation_score": 9255.411719861162}

RECENT RESULT
hypothesis: A center-view weight of 3.3974609375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.
change: Move halfway from the best verified weight, 3.39453125, toward the accuracy-losing 3.400390625 weight, update normalization, and adopt the verified 1.35605 calibration.
mechanism: Upper-bound refinement of center-view ensemble weighting
evidence_used: Weight 3.39453125 achieved 9,257 correct, while 3.400390625 achieved 9,255 with lower cross-entropy; their midpoint is the most informative probe of the upper 9,257-correct boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 57.43195170792751, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441860122680664, "validation_score": 9256.411719648806}

RECENT RESULT
hypothesis: A center-view weight of 3.39599609375 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.
change: Move halfway from the best verified weight, 3.39453125, toward the 3.3974609375 weight that lost one correct prediction, and update the ensemble normalization constant.
mechanism: Upper-bound refinement of center-view ensemble weighting
evidence_used: Weight 3.39453125 achieved the best result with 9,257 correct, while 3.3974609375 produced 9,256 correct with slightly lower cross-entropy; their midpoint is the most informative probe of the upper 9,257-correct boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 66.21664587501436, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21441892204284668, "validation_score": 9257.411719540041}

RECENT RESULT
hypothesis: A center-view weight of 3.396728515625 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.21441892204284668.
change: Move halfway between the accuracy-preserving 3.39599609375 weight and the accuracy-losing 3.3974609375 weight, updating the ensemble normalization accordingly.
mechanism: Binary refinement of the upper center-weight boundary
evidence_used: Weight 3.39599609375 achieved 9,257 correct, while 3.3974609375 achieved 9,256 with lower cross-entropy; their midpoint is the most informative remaining probe of the upper 9,257-correct boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 54.90483679110184, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21441875915527345, "validation_score": 9256.411719595264}



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
