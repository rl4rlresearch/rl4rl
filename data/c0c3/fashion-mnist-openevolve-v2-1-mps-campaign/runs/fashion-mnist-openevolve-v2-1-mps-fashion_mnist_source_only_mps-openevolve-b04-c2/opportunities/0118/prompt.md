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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 58.125259374966845, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286717262268067, "validation_score": 9295.415673493617}
prior_hypothesis: BatchNorm1d momentum 0.198125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.19875.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 47.490451332880184, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20286174621582032, "validation_score": 9294.415675368822}
prior_hypothesis: BatchNorm1d momentum 0.1975 will exceed 9,295 correct predictions by preserving the stronger adaptation at 0.2 while slightly reducing running-statistic noise toward 0.19.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 63.12494608317502, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20287260284423828, "validation_score": 9295.41567161711}
prior_hypothesis: BatchNorm1d momentum 0.19875 will retain 9,295 correct predictions while reducing cross-entropy toward the lower value observed at momentum 0.1975, thereby exceeding the current validation_score.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 68.95657641696744, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028644546508789, "validation_score": 9295.415674432865}
prior_hypothesis: BatchNorm1d momentum 0.1978125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.198125.

## Recent verification evidence

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.3 will exceed 9,295 correct predictions by tracking the final classifier representation more closely than momentum 0.2.
change: Increase only the classifier BatchNorm1d momentum from 0.00623359375 to 0.3, preserving all other architecture, optimization, loss, and calibration settings.
mechanism: Faster dense-feature running-statistic adaptation
evidence_used: Validation correctness increased from 9,284 at momentum 0.00623359375 to 9,288 at 0.1 and 9,295 at 0.2, making a further increase the most direct test of the observed adaptation trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 59.471149208024144, "validation_accuracy": 0.9292, "validation_correct": 9292, "validation_cross_entropy": 0.20390634841918945, "validation_score": 9292.415314696742}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.25 will exceed 9,295 correct predictions by balancing the superior adaptation at momentum 0.2 against the slight degradation observed at 0.3.
change: Replace validation-batch statistics with running statistics at momentum 0.25, preserving all other architecture, optimization, loss, and calibration settings.
mechanism: Fine-grained dense BatchNorm momentum tuning
evidence_used: Momentum 0.2 achieved the best result of 9,295 correct, while 0.1 achieved 9,288 and 0.3 achieved 9,292; testing their unmeasured midpoint most directly refines the apparent optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 79.58396562491544, "validation_accuracy": 0.9292, "validation_correct": 9292, "validation_cross_entropy": 0.20335481338500977, "validation_score": 9292.415505048419}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19 will exceed 9,295 correct predictions by retaining the rapid adaptation seen at momentum 0.2 while slightly reducing running-statistic noise.
change: Add classifier BatchNorm1d with momentum 0.19 and use the best-verified evaluation logit scale of 1.20514.
mechanism: Fine-grained dense BatchNorm momentum refinement
evidence_used: Momentum 0.2 achieved 9,295 correct, outperforming 9,288 at 0.1, while momentum 0.25 and 0.3 fell to 9,292; this brackets the optimum near 0.2 and motivates a small downward refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 53.33525920799002, "validation_accuracy": 0.9293, "validation_correct": 9293, "validation_cross_entropy": 0.2027974105834961, "validation_score": 9293.415697602606}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.21 will exceed 9,295 correct predictions by retaining the rapid statistic adaptation at 0.2 while testing the narrow interval before performance declines at 0.25.
change: Increase only the classifier BatchNorm1d momentum from 0.2 to 0.21, preserving the best architecture, optimizer, loss, schedule, and evaluation calibration.
mechanism: Fine-grained dense BatchNorm momentum refinement
evidence_used: Momentum 0.2 achieved the best result with 9,295 correct, while 0.19 achieved 9,293 and both 0.25 and 0.3 achieved 9,292; the evidence brackets a narrow optimum near 0.2 and makes a small upward refinement the most informative next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 56.79437933280133, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20297210845947267, "validation_score": 9294.415637234217}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.205 will exceed 9,295 correct predictions by refining the narrow optimum between the best result at 0.2 and the slightly worse result at 0.21.
change: Increase only the classifier BatchNorm1d momentum from 0.19 to 0.205 while preserving the architecture, optimizer, loss, schedule, and evaluation calibration.
mechanism: Fine-grained dense BatchNorm momentum interpolation
evidence_used: Momentum 0.2 achieved 9,295 correct, compared with 9,293 at 0.19 and 9,294 at 0.21; testing their upper midpoint most directly probes the tightly bracketed optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 53.67643245798536, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20292747535705566, "validation_score": 9295.41565265591}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.1975 will exceed 9,295 correct predictions by preserving the stronger adaptation at 0.2 while slightly reducing running-statistic noise toward 0.19.
change: Change only the classifier BatchNorm1d momentum from 0.25 to 0.1975.
mechanism: Fine-grained dense BatchNorm momentum interpolation
evidence_used: Momentum 0.2 achieved the best score with 9,295 correct, while 0.19 achieved 9,293; testing their midpoint probes the remaining unmeasured interval around the observed optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 47.490451332880184, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20286174621582032, "validation_score": 9294.415675368822}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.2025 will exceed 9,295 correct predictions by interpolating between the two momentum settings that both achieved 9,295 correct.
change: Change only the classifier BatchNorm1d momentum from 0.3 to 0.2025.
mechanism: Fine-grained dense BatchNorm momentum interpolation
evidence_used: Momentum 0.2 and 0.205 each achieved 9,295 correct, outperforming nearby 0.1975 and 0.21; their untested midpoint is the most targeted remaining refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 47.501997291110456, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20290541038513182, "validation_score": 9295.415660280254}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19875 will retain 9,295 correct predictions while reducing cross-entropy toward the lower value observed at momentum 0.1975, thereby exceeding the current validation_score.
change: Decrease only the classifier BatchNorm1d momentum from 0.2 to 0.19875.
mechanism: Lower-bound dense BatchNorm momentum interpolation
evidence_used: Momentum 0.2 achieved 9,295 correct with 0.2028835 cross-entropy, while 0.1975 achieved 9,294 correct with lower 0.2028617 cross-entropy; their midpoint directly probes whether the calibration benefit can be gained without losing a correct prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 63.12494608317502, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20287260284423828, "validation_score": 9295.41567161711}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.198125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.19875.
change: Decrease only the classifier BatchNorm1d momentum from 0.2025 to 0.198125.
mechanism: Lower-bound dense BatchNorm momentum bisection
evidence_used: Momentum 0.19875 achieved 9,295 correct with 0.2028726 cross-entropy, while 0.1975 achieved 9,294 correct with a lower 0.2028617 cross-entropy; bisecting this boundary tests whether more of the calibration benefit is available without losing correctness.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 58.125259374966845, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286717262268067, "validation_score": 9295.415673493617}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.1978125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.198125.
change: Decrease only the classifier BatchNorm1d momentum from 0.205 to 0.1978125.
mechanism: Lower-bound dense BatchNorm momentum bisection
evidence_used: Momentum 0.198125 achieved 9,295 correct with 0.2028672 cross-entropy, while 0.1975 achieved 9,294 correct with the lower 0.2028617 cross-entropy; bisecting this boundary is the most targeted test for further calibration improvement without losing correctness.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 68.95657641696744, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028644546508789, "validation_score": 9295.415674432865}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19765625 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.2028644546508789.
change: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.19765625.
mechanism: Lower-bound dense BatchNorm momentum bisection
evidence_used: Momentum 0.1978125 achieved 9,295 correct at 0.20286445 cross-entropy, while 0.1975 achieved 9,294 correct at the lower 0.20286175; bisecting this boundary is the most targeted test for improved calibration without losing correctness.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 40.32519608410075, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.2028630973815918, "validation_score": 9294.415674901897}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.197734375 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.2028644546508789.
change: Decrease only the classifier BatchNorm1d momentum from 0.1978125 to 0.197734375.
mechanism: Lower-bound dense BatchNorm momentum bisection
evidence_used: Momentum 0.1978125 achieved 9,295 correct with 0.20286445 cross-entropy, while 0.19765625 achieved 9,294 correct with lower 0.20286310 cross-entropy; bisecting this boundary is the most targeted test for improved calibration without losing correctness.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 45.13403887511231, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20286377944946288, "validation_score": 9294.415674666194}



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
