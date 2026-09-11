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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.43989566690288, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19624986610412598, "validation_score": 9330.41797287855}
prior_hypothesis: Scaling evaluation logits by 1.184 will preserve all 9,330 argmax predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196250.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.07821804098785, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625243644714355, "validation_score": 9331.417971980467}
prior_hypothesis: A 0.51 flipped-view weight will exceed 9,330 correct predictions by exploiting the observed directional advantage of the flipped view without the excessive perturbation of the unsuccessful 0.55 weighting.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.67012754199095, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.196251798248291, "validation_score": 9331.417972203455}
prior_hypothesis: A 0.509 flipped-view weight will preserve 9,331 correct predictions while lowering cross-entropy below 0.196252436 by moving closer to equal fusion.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 69.62122924998403, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19624986610412598, "validation_score": 9330.41797287855}
prior_hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the evaluation-time scale applied to the symmetric flip-logit ensemble.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 2 and 3 independently verified this exact change at 9,330 correct and 0.196249866 cross-entropy, the best reported validation score; later non-completions provide no contrary metric evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 69.62122924998403, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19624986610412598, "validation_score": 9330.41797287855}

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.
change: Increase only the evaluation-time scale applied to the symmetric flip-logit ensemble.
mechanism: Verified flip-ensemble temperature calibration
evidence_used: Reference Designs 1–3 and the latest successful verification independently report 9,330 correct with 0.196249866 cross-entropy at scale 1.184, improving the current design’s tie-breaker without affecting argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Annealing the learning rate fully to zero will exceed 9,330 correct predictions by reducing late-iterate parameter noise while preserving the verified architecture and objective.
change: Remove the 2% terminal learning-rate floor from the cosine schedule; retain all other training and evaluation behavior.
mechanism: Zero-floor cosine annealing
evidence_used: The verified design reaches 9,330 correct with a 2% learning-rate floor, while late-iterate EMA and averaging experiments targeted endpoint variance but timed out; zero-floor annealing tests the same stability mechanism without extra model operations or parameter copies.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting the original validation view 0.55 and its horizontal flip 0.45 will correct at least one borderline flip disagreement and exceed 9,330 correct predictions without materially degrading calibrated cross-entropy.
change: Replace equal evaluation-time logit fusion with a slight original-view bias while retaining the verified 1.184 temperature.
mechanism: Canonical-view-biased flip ensemble
evidence_used: The current equal-weight ensemble achieves 9,330 correct, while temperature scaling is already optimized and cannot alter argmax predictions; the attempted probability-space fusion did not finish, leaving low-cost fusion changes as an untested route to higher correctness.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 77.8688753338065, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19636515579223632, "validation_score": 9320.417932599908}

RECENT RESULT
hypothesis: Weighting the flipped validation view 0.55 and the original view 0.45 will exceed 9,330 correct predictions, because the opposite 0.55 original-view bias reduced correctness by ten.
change: Replace equal evaluation-time fusion with a slight flipped-view bias while retaining the verified 1.184 calibration.
mechanism: Flipped-view-biased logit fusion
evidence_used: Equal fusion verified 9,330 correct, while the 0.55 original/0.45 flipped trial fell to 9,320; this establishes a directional signal favoring the previously downweighted flipped view and motivates the mirrored test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 81.36523770797066, "validation_accuracy": 0.9326, "validation_correct": 9326, "validation_cross_entropy": 0.19634847412109374, "validation_score": 9326.417938427487}

RECENT RESULT
hypothesis: A 0.51 flipped-view weight will exceed 9,330 correct predictions by exploiting the observed directional advantage of the flipped view without the excessive perturbation of the unsuccessful 0.55 weighting.
change: Apply the verified 1.184 calibration while changing evaluation fusion from equal weighting to 0.49 original and 0.51 flipped.
mechanism: Near-equal flipped-view logit fusion
evidence_used: Equal fusion achieved 9,330 correct; a five-point flipped bias retained 9,326 versus only 9,320 for the mirrored original bias, suggesting that any useful asymmetric optimum lies slightly toward the flipped view and close to equal weighting.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.07821804098785, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.19625243644714355, "validation_score": 9331.417971980467}

RECENT RESULT
hypothesis: A 0.516 flipped-view weight will exceed 9,331 correct predictions by moving slightly beyond the successful 0.51 weight while remaining far from the degraded 0.55 setting.
change: Change evaluation fusion from 0.49/0.51 to 0.484/0.516, retaining the verified 1.184 calibration.
mechanism: Interpolated flipped-view logit fusion
evidence_used: Equal fusion produced 9,330 correct, 0.51 flipped weighting improved to 9,331, and 0.55 fell to 9,326; these results place the most promising untested region just beyond 0.51.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting the flipped validation view at 0.51 will reproduce the verified improvement from 9,330 to 9,331 correct predictions while retaining approximately 0.19625 cross-entropy.
change: Change evaluation fusion from equal weighting to 0.49 original and 0.51 flipped, retaining the verified 1.184 calibration.
mechanism: Near-equal flipped-view logit fusion
evidence_used: Reference Design 1 verified this exact fusion at 9,331 correct, outperforming equal fusion’s 9,330; the more aggressive 0.55 flipped weighting fell to 9,326.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting the flipped validation view at 0.51 will reproduce the verified improvement from 9,330 to 9,331 correct predictions while retaining approximately 0.19625 cross-entropy.
change: Change evaluation fusion from equal weighting to 0.49 original and 0.51 flipped, retaining the verified 1.184 calibration.
mechanism: Near-equal flipped-view logit fusion
evidence_used: Reference Design 1 verified this exact fusion at 9,331 correct, outperforming equal fusion’s 9,330; the more aggressive 0.55 flipped weighting fell to 9,326.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the flipped-view weight from 0.510 to 0.512 will exceed 9,331 correct predictions by resolving another borderline disagreement while staying near the verified local improvement and far from the degraded 0.55 weighting.
change: Change evaluation-time fusion to 0.488 original and 0.512 flipped, retaining the verified 1.184 calibration and all training behavior.
mechanism: Fine-grained flipped-view logit fusion
evidence_used: Flipped weight 0.500 produced 9,330 correct, 0.510 improved to 9,331, and 0.550 declined to 9,326; this motivates a small local step beyond the best verified weight.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 79.96637337491848, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19625398559570312, "validation_score": 9330.417971439194}

RECENT RESULT
hypothesis: A 0.511 flipped-view weight will exceed 9,331 correct predictions by capturing a favorable decision transition between the successful 0.510 weight and the degraded 0.512 weight.
change: Change evaluation fusion from 0.490/0.510 to 0.489/0.511 while retaining the verified 1.184 calibration and all training behavior.
mechanism: Midpoint search of the narrow flipped-view fusion optimum
evidence_used: Flipped weight 0.510 achieved 9,331 correct, whereas 0.500 and 0.512 each achieved 9,330; testing their closest unmeasured midpoint brackets the narrow accuracy optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 78.89995537488721, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.1962531707763672, "validation_score": 9331.417971723891}

RECENT RESULT
hypothesis: A 0.509 flipped-view weight will preserve 9,331 correct predictions while lowering cross-entropy below 0.196252436 by moving closer to equal fusion.
change: Change evaluation fusion from 0.500/0.500 to 0.491/0.509 while retaining the verified 1.184 calibration and all training behavior.
mechanism: Leftward refinement of the flipped-view fusion plateau
evidence_used: Flipped weights 0.510 and 0.511 both achieved 9,331 correct, but 0.510 had lower cross-entropy; 0.500 achieved only 9,330. Testing 0.509 probes the unmeasured left edge of the accuracy plateau, where reduced asymmetry should improve the tie-breaker if the extra correct prediction is retained.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.67012754199095, "validation_accuracy": 0.9331, "validation_correct": 9331, "validation_cross_entropy": 0.196251798248291, "validation_score": 9331.417972203455}



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
