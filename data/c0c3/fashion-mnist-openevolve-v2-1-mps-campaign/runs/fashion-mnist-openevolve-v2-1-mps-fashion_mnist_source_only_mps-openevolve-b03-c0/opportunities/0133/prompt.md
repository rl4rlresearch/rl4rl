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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.70755654200912, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979916358947754, "validation_score": 9322.41736518438}
prior_hypothesis: Increasing the translated-logit contribution to 30.02% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979918598175049.

## Recent verification evidence

RECENT RESULT
hypothesis: A 37.488250732421875% translated-logit contribution will retain 9,322 correct predictions while lowering cross-entropy below 0.19799208221435546.
change: Set the unanimous-correction blend to the midpoint of the two distinct settings that achieved the best measured cross-entropy, keeping complementary weights and the 30% argmax-preserving blend unchanged.
mechanism: Interior translated-logit optimum probe
evidence_used: Both 37.4884033203125% and 37.48809814453125% produced the identical best score, while the nearest tested settings outside that interval were worse; its untested midpoint is the most informative convex-optimum probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 71.9587502090726, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979920837402344, "validation_score": 9322.417365028356}

RECENT RESULT
hypothesis: A 37.4883270263671875% translated-logit contribution will retain 9,322 correct predictions and lower validation cross-entropy below 0.19799208221435546.
change: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest lower setting with worse measured cross-entropy; keep complementary weights and the 30% argmax-preserving blend unchanged.
mechanism: Lower-side translated-logit optimum refinement
evidence_used: The current 37.4884033203125% setting achieved the best verified cross-entropy, while 37.488250732421875% retained the same correct count but was worse; their untested midpoint most directly refines the remaining lower-side interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 73.84275695797987, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979920837402344, "validation_score": 9322.417365028356}

RECENT RESULT
hypothesis: A 37.48836517333984375% translated-logit contribution will retain 9,322 correct predictions and lower validation cross-entropy below 0.19799208221435546.
change: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest lower tested setting, preserving complementary weights and the 30% argmax-preserving blend.
mechanism: Lower-bound translated-logit plateau refinement
evidence_used: The current 37.4884033203125% setting achieved the best verified cross-entropy, while 37.4883270263671875% retained 9,322 correct but was measurably worse; their untested midpoint most directly refines the remaining lower boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 78.78179795807227, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208221435546, "validation_score": 9322.417365028887}

RECENT RESULT
hypothesis: Decreasing the positive validation temperature by one float32 ULP will retain all 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546.
change: Change only the final logit temperature from 0.738156 to its immediately lower float32 value; all training and ensemble behavior remains unchanged.
mechanism: One-ULP temperature calibration
evidence_used: Translation-blend probes around 37.4884033203125% have plateaued or worsened, so the argmax-preserving temperature is the most informative remaining calibration axis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the positive validation temperature by one float32 ULP will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.
change: Change only the final logit temperature from 0.738156 to its immediately lower float32 value.
mechanism: One-ULP temperature calibration retry
evidence_used: Translation-blend refinement has plateaued, while the prior lower-temperature probe timed out and therefore provided no contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the positive validation temperature by two float32 ULPs will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.
change: Change only the final logit temperature from 0.738156 to its second-lower float32 value.
mechanism: Two-ULP temperature calibration
evidence_used: Translation-blend refinement plateaued, while both one-ULP temperature probes timed out without contrary validation evidence; a two-ULP decrease preserves predictions and makes the calibration effect more measurable.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decreasing the positive validation temperature by three float32 ULPs will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.
change: Change only the final logit temperature from 0.738156 to its third-lower float32 value.
mechanism: Three-ULP temperature calibration
evidence_used: Translation-blend refinement plateaued at 9,322 correct, while the one- and two-ULP lower-temperature probes timed out without contrary validation evidence; a third-ULP decrement continues the argmax-preserving calibration search.
result: training did not finish within the verification time limit

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
