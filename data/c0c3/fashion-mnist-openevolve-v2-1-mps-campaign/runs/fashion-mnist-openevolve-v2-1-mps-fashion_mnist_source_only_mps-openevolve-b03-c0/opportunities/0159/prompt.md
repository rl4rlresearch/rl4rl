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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.0015388338361, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979913360595703, "validation_score": 9322.417365288838}
prior_hypothesis: A 30.0333203125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799134063720703.

## Recent verification evidence

RECENT RESULT
hypothesis: A 30.033330078125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.1979913360595703.
change: Bisect the interval between the current-best blend weight and the higher verified non-improving weight, keeping the blend complementary.
mechanism: Preserving-blend local optimum bisection
evidence_used: The 0.300333203125 setting achieved the best cross-entropy of 0.1979913360595703, while 0.3003333984375 worsened it to 0.1979913375854492 with correctness unchanged; their midpoint is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 70.65351291699335, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799133911132813, "validation_score": 9322.417365287774}

RECENT RESULT
hypothesis: A 30.03330078125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.1979913360595703.
change: Bisect the interval between the current-best blend weight and the lower verified weight, keeping the blend weights complementary.
mechanism: Lower-side local-optimum bisection
evidence_used: Cross-entropy improved from 0.19799134063720703 at 0.3003328125 to 0.1979913360595703 at 0.300333203125, then worsened to 0.19799133911132813 at 0.30033330078125; probing the untested lower midpoint most directly refines the bracketed local optimum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30.03330078125% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.
change: Set the preserving blend to the previously unverified lower midpoint while retaining exactly complementary weights.
mechanism: Lower-side local-optimum retry
evidence_used: The current 0.300333203125 weight is best; 0.3003328125 was worse below it and 0.30033330078125 was worse above it. The intervening 0.3003330078125 probe timed out without validation evidence, making its retry the most informative unresolved test.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 82.8872539170552, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799134063720703, "validation_score": 9322.417365287243}

RECENT RESULT
hypothesis: A 30.0333251953125% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.
change: Bisect the interval between the current-best blend weight and the nearest higher verified non-improving weight, retaining exactly complementary weights.
mechanism: Upper-side local-optimum bisection
evidence_used: Weight 0.300333203125 achieved the best cross-entropy of 0.1979913360595703, while 0.30033330078125 worsened it to 0.19799133911132813 with correctness unchanged; their untested midpoint is the most informative remaining local refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30.0333251953125% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.
change: Set the preserving blend to the unverified midpoint between the current-best weight and the nearest higher verified non-improving weight, retaining exactly complementary coefficients.
mechanism: Upper-side local-optimum retry
evidence_used: Weight 0.300333203125 achieved the best cross-entropy, while 0.30033330078125 worsened it with correctness unchanged; the intervening midpoint previously timed out and remains the most informative unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 71.73235154221766, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799133682250977, "validation_score": 9322.417365288573}

RECENT RESULT
hypothesis: A 30.03332275390625% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.
change: Bisect the interval between the current-best blend weight and the nearest higher verified non-improving weight while retaining exactly complementary coefficients.
mechanism: Upper-side local-optimum bisection
evidence_used: Weight 0.300333203125 produced the best cross-entropy, while 0.300333251953125 preserved correctness but worsened cross-entropy to 0.19799133682250977; their midpoint is the most informative remaining refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 73.02558816689998, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799133834838867, "validation_score": 9322.41736528804}

RECENT RESULT
hypothesis: A 30.033310546875% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.
change: Bisect the unresolved lower interval around the current-best preserving blend while retaining exactly complementary coefficients.
mechanism: Lower-side local-optimum bisection
evidence_used: Weight 0.300333203125 produced the best cross-entropy, while the lower 0.3003330078125 setting worsened it to 0.19799134063720703; their untested midpoint is the most informative remaining lower-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.35278516705148, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979913375854492, "validation_score": 9322.417365288307}

RECENT RESULT
hypothesis: A 30.0333154296875% translated-logit contribution will preserve all 9,322 correct predictions and reduce validation cross-entropy below 0.1979913360595703.
change: Bisect the interval between the current-best blend weight and the nearest lower verified non-improving weight, retaining exactly complementary coefficients.
mechanism: Lower-side local-optimum bisection
evidence_used: Weight 0.300333203125 achieved the best cross-entropy, while the lower 0.30033310546875 setting preserved correctness but worsened cross-entropy to 0.1979913375854492; their midpoint is the most informative remaining lower-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 75.5010265421588, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799133949279785, "validation_score": 9322.417365287642}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
