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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.33100033388473, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19147203903198243, "validation_score": 9328.419648958281}
prior_hypothesis: Temperature 0.8007 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.39485112507828, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1914627384185791, "validation_score": 9327.419652234079}
prior_hypothesis: A 0.05 fusion-margin coefficient will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191472037.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.02276470791548, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19146882133483886, "validation_score": 9328.41965009159}
prior_hypothesis: A fusion-margin coefficient of 0.0875 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191472037.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 62.81158691714518, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914720371246338, "validation_score": 9328.419648958952}
prior_hypothesis: Temperature 0.800713 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472039.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the verified hard-maximum, constant-0.02-smoothing design will recover 9,328 correct predictions, while temperature 0.80 will preserve its argmax predictions and reduce validation cross-entropy below 0.192145.
change: Restore hard-maximum channel attention, confidence-adaptive flip fusion, constant 0.02 label smoothing, and sharpen fused evaluation logits from the best verified temperature of 0.85 to 0.80.
mechanism: Further evaluation-only temperature sharpening
evidence_used: Temperature 0.85 preserved all 9,328 correct predictions and improved cross-entropy from 0.194148 at temperature 0.90 to 0.192145, motivating one further step in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 66.40186375007033, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914721824645996, "validation_score": 9328.419648907762}

RECENT RESULT
hypothesis: Restoring constant 0.02 label smoothing and temperature-0.80 evaluation will recover 9,328 correct predictions and validation cross-entropy near 0.19147.
change: Train paired flip views with 0.02 label smoothing and divide confidence-adaptive fused log-probabilities by 0.80 during evaluation.
mechanism: Constant label smoothing with calibrated flip-ensemble sharpening
evidence_used: Reference Design 2 achieved the best available score, preserving 9,328 correct predictions while lowering cross-entropy to 0.191472; the current unsmoothed design reached only 9,320 correct.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 83.27676204103045, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914721824645996, "validation_score": 9328.419648907762}

RECENT RESULT
hypothesis: Temperature 0.805 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472, as the improvements from 0.90 to 0.85 to 0.80 place the estimated calibration optimum near 0.805.
change: Slightly relax evaluation-only sharpening from temperature 0.80 to 0.805; training remains unchanged.
mechanism: Interpolated evaluation-temperature calibration
evidence_used: Temperatures 0.90, 0.85, and 0.80 preserved exactly 9,328 correct while cross-entropy decreased from 0.194148 to 0.192145 to 0.191472; quadratic interpolation in inverse temperature estimates a minimum just above 0.80.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 67.01654874999076, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914772232055664, "validation_score": 9328.419647132368}

RECENT RESULT
hypothesis: Temperature 0.8007 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472.
change: Sharpen the confidence-adaptive fused evaluation logits with temperature 0.8007; training remains unchanged.
mechanism: Locally interpolated evaluation-temperature calibration
evidence_used: Temperature 0.80 achieved 0.191472 cross-entropy, while 0.805 was slightly worse at 0.191477; local quadratic interpolation with the 0.85 result places the minimum near 0.8007.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.33100033388473, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19147203903198243, "validation_score": 9328.419648958281}

RECENT RESULT
hypothesis: Temperature 0.800713 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472039.
change: Restore the best-count constant-smoothing design and sharpen its confidence-adaptive fused evaluation logits using the locally estimated optimal temperature.
mechanism: Quadratically refined evaluation-temperature calibration
evidence_used: Temperatures 0.8000, 0.8007, and 0.8050 produced cross-entropies 0.191472182, 0.191472039, and 0.191477223 with identical correct counts; quadratic interpolation places the local minimum near 0.800713.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 62.81158691714518, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.1914720371246338, "validation_score": 9328.419648958952}

RECENT RESULT
hypothesis: Increasing the fusion-margin coefficient from 0.10 to 0.20 will preserve the reliable 9,328-correct constant-smoothing solution and correct additional flip-disagreement cases by giving the more decisive view greater influence.
change: Restore constant 0.02 label smoothing and temperature 0.800713, while strengthening confidence-adaptive validation fusion to coefficient 0.20.
mechanism: Stronger confidence-adaptive flip fusion
evidence_used: Constant 0.02 smoothing produced 9,328 correct, and coefficient-0.10 confidence fusion improved cross-entropy without reducing correct predictions; testing a stronger coefficient directly determines whether view confidence can also improve the primary correct-count objective.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 63.77861508401111, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.1915193328857422, "validation_score": 9327.41963230155}

RECENT RESULT
hypothesis: A 0.15 fusion-margin coefficient will avoid the harmful over-weighting observed at 0.20 while potentially correcting disagreement cases not changed at 0.10, exceeding 9,328 correct predictions.
change: Increase the evaluation-only flip-fusion margin coefficient from 0.10 to the midpoint 0.15; retain constant 0.02 smoothing and temperature 0.800713.
mechanism: Intermediate confidence-adaptive flip fusion
evidence_used: Coefficient 0.10 achieved 9,328 correct with 0.191472037 cross-entropy, while 0.20 changed the decision boundary and fell to 9,327; testing the midpoint isolates whether useful decisive-view corrections occur before the harmful flip.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 65.52950266608968, "validation_accuracy": 0.9328, "validation_correct": 9328, "validation_cross_entropy": 0.19149088897705077, "validation_score": 9328.419642319237}

RECENT RESULT
hypothesis: Temperature 0.800713 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472039.
change: Sharpen the fused evaluation logits with the best verified temperature, leaving training and fusion unchanged.
mechanism: Quadratically refined evaluation-temperature calibration
evidence_used: Reference Design 3 verified 9,328 correct with cross-entropy 0.191472037 at temperature 0.800713, strictly improving the current temperature-0.8007 result of 0.191472039 with the same correct count.
result: the implementation could not be verified

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
