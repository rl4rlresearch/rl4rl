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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 62.81158691714518, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914720371246338, "validation_score": 9328.419648958952}
prior_hypothesis: Temperature 0.800713 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472039.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.06298958393745, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687774658203, "validation_score": 9328.419650107042}
prior_hypothesis: A fusion coefficient of 0.0873046875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.33100033388473, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19147203903198243, "validation_score": 9328.419648958281}
prior_hypothesis: Temperature 0.8007 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.02276470791548, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19146882133483886, "validation_score": 9328.41965009159}
prior_hypothesis: A fusion-margin coefficient of 0.0875 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191472037.

## Recent verification evidence

RECENT RESULT
hypothesis: Temperature 0.800713 will preserve all 9,328 correct predictions while reducing validation cross-entropy below 0.191472182.
change: Sharpen the confidence-adaptive fused evaluation logits using the best verified temperature; training and fusion remain unchanged.
mechanism: Refined evaluation-temperature calibration
evidence_used: Reference Design 3 achieved 9,328 correct with 0.191472037 cross-entropy at temperature 0.800713, strictly improving the current temperature-0.80 result with the same correct count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 68.3671156251803, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914720371246338, "validation_score": 9328.419648958952}

RECENT RESULT
hypothesis: A 0.05 fusion-margin coefficient will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191472037.
change: Reduce the evaluation-only flip-fusion margin coefficient from 0.20 to 0.05 while retaining constant 0.02 label smoothing and temperature 0.800713.
mechanism: Reduced confidence-adaptive flip fusion
evidence_used: Coefficient 0.10 achieved 9,328 correct at 0.191472037 cross-entropy, whereas 0.15 increased cross-entropy and 0.20 lost one correct prediction; testing 0.05 extends the observed favorable direction conservatively.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.39485112507828, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914627384185791, "validation_score": 9327.419652234079}

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.075 will retain 9,328 correct predictions while reducing validation cross-entropy below 0.191472037.
change: Reduce the evaluation-only flip-fusion margin coefficient from 0.10 to 0.075; leave training and temperature calibration unchanged.
mechanism: Boundary-seeking confidence-adaptive flip fusion
evidence_used: Coefficient 0.05 achieved lower cross-entropy (0.191462738) but 9,327 correct, while 0.10 achieved 9,328 correct at 0.191472037; their midpoint tests whether the additional correct prediction can be retained with better calibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.53889549989253, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914662010192871, "validation_score": 9327.4196510145}

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.0875 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191472037.
change: Reduce the evaluation-only flip-fusion margin coefficient from 0.10 to 0.0875, leaving training and temperature calibration unchanged.
mechanism: Boundary-seeking confidence-adaptive flip fusion
evidence_used: Coefficient 0.075 produced 9,327 correct at 0.191466201 cross-entropy, while 0.10 produced 9,328 correct at 0.191472037; their midpoint efficiently searches for the lowest coefficient that preserves the additional correct prediction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.02276470791548, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19146882133483886, "validation_score": 9328.41965009159}

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.08125 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Reduce the evaluation-only fusion coefficient to the midpoint between 0.075 and 0.0875, and use the best verified temperature of 0.800713.
mechanism: Bisection of confidence-adaptive flip-fusion boundary
evidence_used: Coefficient 0.075 produced 9,327 correct, while 0.0875 produced 9,328 correct with lower cross-entropy than 0.10; their midpoint efficiently searches for the smallest coefficient preserving the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.78808333305642, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146743698120117, "validation_score": 9327.419650579177}

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.084375 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between the verified 0.08125 and 0.0875 designs; retain temperature 0.800713 and all training settings.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.08125 produced 9,327 correct with 0.191467437 cross-entropy, while 0.0875 produced 9,328 correct with 0.191468821; bisection is the most informative search for the smallest coefficient preserving the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.02939291717485, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914681121826172, "validation_score": 9327.419650341362}

RECENT RESULT
hypothesis: A fusion-margin coefficient of 0.0859375 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Reduce the evaluation-only flip-fusion coefficient from 0.0875 to the midpoint between 0.084375 and 0.0875; retain all training settings and temperature 0.800713.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.084375 produced 9,327 correct at 0.191468112 cross-entropy, while 0.0875 produced 9,328 correct at 0.191468821; their midpoint is the most informative next test for the smallest coefficient preserving the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 68.05107733304612, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146846160888673, "validation_score": 9327.41965021829}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08671875 will preserve 9,328 correct predictions while lowering cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.0859375 and 0.0875; retain temperature 0.800713 and all training settings.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.0859375 yielded 9,327 correct at 0.191468462 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.77383895893581, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146864013671874, "validation_score": 9327.41965015541}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087109375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.08671875 and 0.0875, and use the best verified temperature of 0.800713.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.08671875 yielded 9,327 correct at 0.191468640 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 62.650059666950256, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19146873474121093, "validation_score": 9327.41965012209}

RECENT RESULT
hypothesis: A fusion coefficient of 0.0873046875 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.
change: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.087109375 and 0.0875, retaining temperature 0.800713 and all training settings.
mechanism: Fine-grained bisection of the confidence-adaptive fusion boundary
evidence_used: Coefficient 0.087109375 yielded 9,327 correct at 0.191468735 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 69.06298958393745, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914687774658203, "validation_score": 9328.419650107042}

RECENT RESULT
hypothesis: A fusion coefficient of 0.08720703125 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.087109375 and 0.0873046875, retaining all training settings and temperature 0.800713.
mechanism: Precision bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.087109375 yielded 9,327 correct at 0.191468735 cross-entropy, while 0.0873046875 yielded 9,328 correct at 0.191468777; their midpoint is the most informative next test for the smallest coefficient retaining the additional correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.25742333289236, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687530517578, "validation_score": 9327.41965011564}

RECENT RESULT
hypothesis: A fusion coefficient of 0.087255859375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.
change: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.08720703125 and 0.0873046875, retaining temperature 0.800713 and all training settings.
mechanism: Precision bisection of the flip-fusion decision boundary
evidence_used: Coefficient 0.08720703125 yielded 9,327 correct, while 0.0873046875 yielded 9,328 correct at 0.191468777 cross-entropy; their midpoint is the most informative next boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.29992450005375, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914687629699707, "validation_score": 9327.419650112148}



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
