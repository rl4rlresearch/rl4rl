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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 52.76092537515797, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798034667968, "validation_score": 9267.41138449647}
prior_hypothesis: A 9.83258152008056640625% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.2154079833984375.

## Recent verification evidence

RECENT RESULT
hypothesis: A 9.8325958251953125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.2154079834.
change: Increase the normalized geometric-logit contribution from 9.83258056640625% to 9.8325958251953125%, with the complementary arithmetic-log-probability weight.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258056640625% blend retained 9,267 correct, whereas 9.832611083984375% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 48.70707312505692, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540798110961915, "validation_score": 9266.411384496212}

RECENT RESULT
hypothesis: A 9.83258819580078125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.2154079833984375.
change: Increase the normalized geometric-logit contribution to the midpoint between the highest known 9,267-correct blend and the lowest tested 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258056640625% blend retained 9,267 correct, while 9.8325958251953125% lost one prediction; their midpoint is the most informative refinement of the bracketed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 63.57520320918411, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540798149108886, "validation_score": 9266.411384496083}

RECENT RESULT
hypothesis: A 9.832584381103515625% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.2154079833984375.
change: Increase the geometric-logit contribution to the midpoint between the highest known 9,267-correct blend and the lowest tested 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258056640625% blend retained 9,267 correct, while 9.83258819580078125% lost one prediction; their midpoint is the most informative refinement of the bracketed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 51.52435470884666, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540798149108886, "validation_score": 9266.411384496083}

RECENT RESULT
hypothesis: A 9.8325824737548828125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.2154079833984375.
change: Increase the geometric-logit contribution to the midpoint between the highest known 9,267-correct blend and the lowest tested 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258056640625% blend retained 9,267 correct, while 9.832584381103515625% lost one prediction; their midpoint is the most informative refinement of the bracketed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.28023050003685, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540798301696779, "validation_score": 9266.411384495566}

RECENT RESULT
hypothesis: A 9.83258152008056640625% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.2154079833984375.
change: Increase the geometric-logit contribution to the midpoint between the highest known 9,267-correct blend and the lowest tested 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258056640625% blend retained 9,267 correct, while 9.8325824737548828125% lost one prediction; their midpoint is the most informative refinement of the bracketed decision boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 52.76092537515797, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798034667968, "validation_score": 9267.41138449647}

RECENT RESULT
hypothesis: A 9.832581996917724609375% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Increase the geometric-logit contribution to the midpoint between the highest known 9,267-correct blend and the lowest tested 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258152008056640625% blend retained 9,267 correct, while 9.8325824737548828125% lost one prediction; their midpoint is the most informative refinement of the bracketed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 62.83172995806672, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540798530578614, "validation_score": 9266.411384494791}

RECENT RESULT
hypothesis: A 9.8325817584991455078125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Increase the geometric-logit contribution to the midpoint between the highest known 9,267-correct blend and the lowest tested 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.83258152008056640625% blend retained 9,267 correct, while 9.832581996917724609375% lost one prediction; their midpoint is the most informative refinement of the bracketed decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 48.438070207834244, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798034667968, "validation_score": 9267.41138449647}

RECENT RESULT
hypothesis: A 9.83258187770843505859375% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Increase the geometric-logit contribution to the midpoint between the highest verified 9,267-correct blend and the lowest verified 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.8325817584991455078125% blend retained 9,267 correct, while 9.832581996917724609375% lost one prediction; their midpoint most precisely probes the remaining decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 74.77460891706869, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21540798530578614, "validation_score": 9266.411384494791}

RECENT RESULT
hypothesis: A residual four-head attention layer over the 7×7 feature grid will exceed 9,267 correct predictions by learning content-dependent, garment-wide relationships while preserving the spatial layout required by the existing classifier.
change: Insert a pre-normalized, position-aware self-attention block before the flattened classifier, with a learnable residual scale initialized conservatively; total learned parameters become 249,059.
mechanism: Position-aware global self-attention residual
evidence_used: Static dilated wider-context extraction regressed to 9,238 correct, while position-free pooling produced 9,253/9,249, indicating that spatial layout is load-bearing but fixed receptive-field expansion is insufficient. Dense input-conditioned attention tests a distinct global interaction mechanism without pooling away positions or replacing the successful flattened head.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 9.832581818103790283203125% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.21540798034667968.
change: Increase the geometric-logit contribution to the midpoint between the highest verified 9,267-correct blend and the lowest verified 9,266-correct blend.
mechanism: Binary-refined geometric log-opinion blend
evidence_used: The 9.8325817584991455078125% blend retained 9,267 correct, while 9.83258187770843505859375% lost one prediction; their midpoint is the most informative remaining probe of the decision boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 42.99846224999055, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21540798034667968, "validation_score": 9267.41138449647}

RECENT RESULT
hypothesis: Averaging the final 20% of the optimization trajectory with decay 0.98 will exceed 9,267 correct predictions by reducing late-step parameter noise without changing the successful representation or inference ensemble.
change: Track a bias-corrected EMA of all floating model parameters and normalization buffers during the final training phase, then install the averaged state after the last optimizer step.
mechanism: Late-training exponential weight averaging
evidence_used: Geometric-blend refinements have exhausted the float-level decision boundary without exceeding 9,267 correct, while wider-context architectural changes regressed or timed out; late weight averaging tests a distinct generalization mechanism while preserving the best architecture and TTA.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 70.23605033312924, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.21561995849609375, "validation_score": 9254.411312759803}

RECENT RESULT
hypothesis: A lightweight squeeze-excitation gate will exceed 9,267 correct predictions by adding input-dependent global context while preserving the spatially flattened representation and avoiding the timed-out attention layer’s quadratic cost.
change: Add a 14-unit channel gate after the residual block, initialized to an exact identity mapping and adding 1,638 learned parameters.
mechanism: Identity-initialized global channel recalibration
evidence_used: Full spatial attention timed out, while position-free pooling regressed; channel recalibration uses global context only to modulate intact 7×7 feature maps, testing a cheaper interaction mechanism without discarding load-bearing spatial layout.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 235072, "training_seconds": 59.707203374942765, "validation_accuracy": 0.9224, "validation_correct": 9224, "validation_cross_entropy": 0.21966347122192384, "validation_score": 9224.409949147283}



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
