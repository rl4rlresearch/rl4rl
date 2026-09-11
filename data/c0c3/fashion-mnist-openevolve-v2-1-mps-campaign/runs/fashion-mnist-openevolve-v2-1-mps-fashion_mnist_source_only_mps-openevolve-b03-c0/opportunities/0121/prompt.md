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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 71.33989529195242, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208221435546, "validation_score": 9322.417365028887}
prior_hypothesis: A 37.4884033203125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979920837402344, or add another correct prediction.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A 37.5% translated-logit contribution will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.1979921051.
change: Raise only the unanimous-correction translation blend from 37.48046875% to 37.5%, keeping the argmax-preserving blend at 30%.
mechanism: Unanimous-correction boundary probe
evidence_used: Successive increases through 37.48046875% retained 9,322 correct while monotonically lowering cross-entropy; the computationally identical 37.5% attempt timed out without producing contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 37.4853515625% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979921051, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.48046875% to 37.4853515625%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: Successive increases through 37.48046875% retained 9,322 correct while monotonically lowering cross-entropy; this is the midpoint between that best verified setting and the timed-out 37.490234375% probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 67.64983158302493, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208984375, "validation_score": 9322.41736502623}

RECENT RESULT
hypothesis: A 37.48779296875% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208984375, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.4853515625% to 37.48779296875%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: Successive increases through 37.4853515625% retained 9,322 correct while monotonically lowering cross-entropy; this is the midpoint between that best verified setting and the timed-out 37.490234375% probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 78.65843549999408, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979920837402344, "validation_score": 9322.417365028356}

RECENT RESULT
hypothesis: A 37.489013671875% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979920837402344, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.48779296875% to 37.489013671875%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: Successive increases through 37.48779296875% retained 9,322 correct while monotonically lowering cross-entropy; this is the midpoint between that best verified setting and the timed-out 37.490234375% probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 37.4884033203125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979920837402344, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.48779296875% to 37.4884033203125%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: The 37.48779296875% setting retained 9,322 correct while achieving the best verified cross-entropy; 37.489013671875% timed out without contrary validation evidence, so their midpoint is the narrowest informative next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 71.33989529195242, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208221435546, "validation_score": 9322.417365028887}

RECENT RESULT
hypothesis: A 37.48870849609375% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.4884033203125% to 37.48870849609375%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: The current 37.4884033203125% setting retained 9,322 correct and achieved the best verified cross-entropy; this patch probes the midpoint toward the timed-out 37.489013671875% setting, for which no contrary validation evidence exists.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 37.488555908203125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.4884033203125% to 37.488555908203125%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: The current 37.4884033203125% setting achieved the best verified cross-entropy while retaining 9,322 correct; this probes the midpoint toward the timed-out 37.48870849609375% setting, for which no contrary validation evidence exists.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 37.4884796142578125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546, or add another correct prediction.
change: Increase only the unanimous-correction translation blend from 37.4884033203125% to 37.4884796142578125%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction boundary search
evidence_used: The current 37.4884033203125% setting has the best verified cross-entropy with 9,322 correct; this probes the midpoint toward the timed-out 37.488555908203125% setting, which supplied no contrary validation evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 67.27067987504415, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208297729493, "validation_score": 9322.417365028621}

RECENT RESULT
hypothesis: A 37.48844146728515625% translated-logit contribution will retain 9,322 correct predictions and achieve cross-entropy below 0.19799208221435546.
change: Increase only the unanimous-correction translation blend from 37.4884033203125% to the midpoint toward the nearest valid worse setting, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained translated-logit optimum bracketing
evidence_used: The current setting achieved the best verified cross-entropy with 9,322 correct, while 37.4884796142578125% retained the same correct count but worsened cross-entropy; their midpoint is the most informative remaining probe of the local optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 72.51990225003101, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208221435546, "validation_score": 9322.417365028887}

RECENT RESULT
hypothesis: A 37.488422393798828125% translated-logit contribution will retain 9,322 correct predictions and lower cross-entropy below 0.19799208221435546.
change: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest higher setting with identical measured cross-entropy, keeping weights complementary.
mechanism: Convex translated-logit plateau refinement
evidence_used: The current 37.4884033203125% and 37.48844146728515625% settings produced identical best scores, while 37.4884796142578125% was worse; the midpoint probes the unresolved interior of this narrow optimum bracket.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 37.48809814453125% translated-logit contribution will retain 9,322 correct predictions and lower validation cross-entropy below 0.19799208221435546.
change: Set the unanimous-correction blend to the midpoint between the current optimum and the nearest lower verified setting, keeping weights complementary.
mechanism: Symmetric translated-logit optimum bracketing
evidence_used: The current 37.4884033203125% setting achieved the best verified cross-entropy, while 37.48779296875% was measurably worse; higher probes were equal or worse, so the untested lower midpoint most directly refines the optimum bracket.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.26564008300193, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799208221435546, "validation_score": 9322.417365028887}



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
