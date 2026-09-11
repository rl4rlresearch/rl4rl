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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.01503616687842, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799136123657227, "validation_score": 9322.417365280067}
prior_hypothesis: A 30.0325% translated-logit contribution will preserve all 9,322 correct predictions and reproduce validation cross-entropy below 0.1979914150238037.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the validation temperature by one float32 ULP will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.
change: Change only the final logit temperature from 0.738156 to its immediately higher float32 value.
mechanism: Opposite-direction one-ULP temperature calibration
evidence_used: Translation-blend refinement plateaued at 9,322 correct, while three lower-temperature probes timed out without validation evidence; probing the previously untested upper direction most directly brackets the argmax-preserving calibration optimum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the argmax-preserving translated-logit contribution from 30% to 30.01% will retain all 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546.
change: Increase only the preserving blend’s translated-logit weight to 0.3001, keeping complementary weights, the optimized correction blend, and temperature unchanged.
mechanism: Argmax-invariant preserving-blend calibration
evidence_used: Correction-blend probes around 37.4884033203125% have plateaued, while temperature probes repeatedly timed out; the preserving blend is a distinct calibration axis whose argmax gate protects the base prediction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the preserving blend’s translated-logit contribution to 29.99% will retain all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.19799208221435546.
change: Change only the preserving blend from 30% translated logits to 29.99%, keeping complementary weights and all correction and temperature calibration unchanged.
mechanism: Opposite-direction argmax-invariant preserving-blend calibration
evidence_used: Correction-blend refinement has plateaued, temperature probes repeatedly timed out, and the 30.01% preserving-blend probe also timed out; testing the unobserved opposite direction most directly brackets this distinct argmax-preserving calibration axis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30.01% translated-logit contribution will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.19799208221435546.
change: Increase only the preserving blend’s translated-logit weight from 30% to 30.01%, keeping complementary weights and all correction and temperature settings unchanged.
mechanism: Argmax-invariant preserving-blend calibration retry
evidence_used: Correction-blend refinement plateaued, while the only 30.01% preserving-blend verification timed out; that timeout provides no validation evidence against this distinct, prediction-preserving calibration change.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.41448049992323, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979918598175049, "validation_score": 9322.417365106367}

RECENT RESULT
hypothesis: Increasing the translated-logit contribution to 30.02% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979918598175049.
change: Increase only the preserving blend’s translated-logit weight from 0.3001 to 0.3002 and keep its weights complementary.
mechanism: Forward preserving-blend calibration
evidence_used: Increasing this contribution from 30% to 30.01% improved cross-entropy from 0.19799208221435546 to 0.1979918598175049 without changing the 9,322 correct predictions, motivating a same-direction probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.70755654200912, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979916358947754, "validation_score": 9322.41736518438}

RECENT RESULT
hypothesis: Increasing the translated-logit contribution to 30.03% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979916358947754.
change: Increase only the preserving blend’s translated-logit weight from 0.3002 to 0.3003 while keeping the weights complementary.
mechanism: Forward preserving-blend calibration
evidence_used: Successive increases from 30.00% to 30.01% and then 30.02% each lowered cross-entropy without changing the 9,322 correct predictions, supporting one further same-direction increment.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.31002045795321, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979914150238037, "validation_score": 9322.417365261328}

RECENT RESULT
hypothesis: Increasing the translated-logit contribution to 30.04% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979914150238037.
change: Increase only the preserving blend’s translated-logit weight from 0.3003 to 0.3004 while keeping the weights complementary.
mechanism: Forward preserving-blend calibration
evidence_used: Successive increases from 30.00% through 30.03% each reduced cross-entropy without changing the 9,322 correct predictions, with 30.03% achieving the current best 0.1979914150238037.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.29041862487793, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19802509956359862, "validation_score": 9322.417353526385}

RECENT RESULT
hypothesis: A 30.035% translated-logit contribution remains below the argmax-gate discontinuity observed at 30.04%, preserves all 9,322 correct predictions, and lowers validation cross-entropy below 0.1979914150238037.
change: Set the preserving blend to the midpoint between the best 30.03% setting and the degraded 30.04% setting, keeping complementary weights.
mechanism: Preserving-blend boundary bisection
evidence_used: Increasing the translated contribution through 30.03% consistently improved cross-entropy, while 30.04% abruptly worsened it despite the argmax-preserving gate; testing 30.035% bisects the narrow boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30.035% translated-logit contribution remains below the discontinuity observed at 30.04%, preserves all 9,322 correct predictions, and lowers validation cross-entropy below 0.1979914150238037.
change: Increase only the preserving blend’s translated-logit weight from 0.3003 to 0.30035 while keeping complementary weights.
mechanism: Preserving-blend boundary bisection retry
evidence_used: Improvements continued through 30.03%, while 30.04% sharply worsened cross-entropy; the sole midpoint verification timed out and supplied no contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30.0325% translated-logit contribution remains below the argmax-gate discontinuity observed at 30.04%, preserves all 9,322 correct predictions, and lowers cross-entropy below 0.1979914150238037.
change: Increase only the preserving blend’s translated-logit weight from 0.3003 to 0.300325 while keeping the weights complementary.
mechanism: Preserving-blend lower-quarter boundary probe
evidence_used: Cross-entropy improved at every tested increment through 30.03% but degraded abruptly at 30.04%; two 30.035% midpoint attempts timed out without validation evidence, motivating a conservative probe halfway between 30.03% and that unresolved midpoint.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 78.51712683402002, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799136123657227, "validation_score": 9322.417365280067}

RECENT RESULT
hypothesis: A 30.03375% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799136123657227.
change: Increase the preserving blend’s translated-logit weight from 0.3003 to 0.3003375 while keeping the weights complementary.
mechanism: Preserving-blend upper-quarter boundary probe
evidence_used: The 30.0325% probe improved cross-entropy to 0.19799136123657227, while 30.04% degraded sharply and 30.035% remains unresolved; 30.03375% probes halfway between the improved setting and unresolved midpoint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30.0325% translated-logit contribution will preserve all 9,322 correct predictions and reproduce validation cross-entropy below 0.1979914150238037.
change: Increase the preserving blend’s translated-logit weight from 0.3003 to 0.300325 while keeping its weights complementary.
mechanism: Preserving-blend lower-quarter retry
evidence_used: The prior 30.0325% probe reported 9,322 correct and lower cross-entropy of 0.19799136123657227, but timed out; retrying it is the strongest evidence-backed opportunity for a strict improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.01503616687842, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799136123657227, "validation_score": 9322.417365280067}



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
