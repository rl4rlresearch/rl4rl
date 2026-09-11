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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.30421712505631, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1846571502685547, "validation_score": 9359.42206304152}
prior_hypothesis: A 53.80615234375% original / 46.19384765625% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465724296569824.

## Recent verification evidence

RECENT RESULT
hypothesis: A 53.75% original / 46.25% flipped weighting will preserve all 9,359 correct predictions while lowering validation cross-entropy below 0.184717472076416.
change: Move the evaluation ensemble halfway from the successful 52.5% canonical weighting toward the lower-cross-entropy 55% weighting, retaining the established calibration scale.
mechanism: Accuracy-constrained canonical-view weight refinement
evidence_used: The 52.5% weighting preserved 9,359 correct with 0.184717472076416 cross-entropy, while 55% reduced cross-entropy to 0.18461807746887207 but lost one correct prediction; their midpoint efficiently probes the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.3133676671423, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1846593635559082, "validation_score": 9359.422062252983}

RECENT RESULT
hypothesis: A 54.375% original / 45.625% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.
change: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the lower-cross-entropy 55% weighting, retaining the established calibration scale.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The 53.75% weighting preserved 9,359 correct with 0.1846593635559082 cross-entropy, while 55% lowered cross-entropy to 0.18461807746887207 but lost one correct prediction; their midpoint is the most informative probe of the accuracy boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 54.375% original / 45.625% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.
change: Increase the evaluation ensemble’s canonical-image weight from 53.75% to 54.375%, retaining the established calibration scale and training procedure.
mechanism: Accuracy-constrained canonical-view weight bisection retry
evidence_used: The 53.75% weighting preserved 9,359 correct at 0.1846593635559082 cross-entropy, while 55% lowered cross-entropy but lost one correct prediction. The midpoint trial timed out and therefore provided no subject-level evidence against this most informative boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.78834024979733, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.18463662109375, "validation_score": 9357.422070355666}

RECENT RESULT
hypothesis: A 54.0625% original / 45.9375% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.
change: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the unsuccessful 54.375% weighting, retaining the established calibration scale and training procedure.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, whereas 54.375% reduced cross-entropy but fell to 9,357 correct; their midpoint is the most informative remaining probe of the accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 69.76800625002943, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.1846474624633789, "validation_score": 9357.422066493065}

RECENT RESULT
hypothesis: A 53.90625% original / 46.09375% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.
change: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the unsuccessful 54.0625% weighting, retaining the established calibration and training procedure.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, whereas 54.0625% reduced cross-entropy but fell to 9,357 correct; their midpoint is the most informative remaining probe of this accuracy boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 62.54286412498914, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.18465328063964845, "validation_score": 9358.422064420174}

RECENT RESULT
hypothesis: A 53.828125% original / 46.171875% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.
change: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the 53.90625% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, while 53.90625% lowered cross-entropy to 0.18465328063964845 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 75.8761840830557, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.18465629234313966, "validation_score": 9358.422063347176}

RECENT RESULT
hypothesis: A 53.7890625% original / 46.2109375% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.
change: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the 53.828125% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The current 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, while 53.828125% lowered cross-entropy to 0.18465629234313966 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 67.82120974990539, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18465781898498534, "validation_score": 9359.422062803273}

RECENT RESULT
hypothesis: A 53.80859375% original / 46.19140625% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.18465781898498534.
change: Move the evaluation ensemble halfway from the successful 53.7890625% canonical weighting toward the 53.828125% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The current 53.7890625% weighting achieved 9,359 correct at 0.18465781898498534 cross-entropy, while 53.828125% lowered cross-entropy to 0.18465629234313966 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 81.93131116684526, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.18465705108642577, "validation_score": 9358.422063076856}

RECENT RESULT
hypothesis: A 53.798828125% original / 46.201171875% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465781898498534.
change: Move the evaluation ensemble halfway from the successful 53.7890625% canonical weighting toward the 53.80859375% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The current 53.7890625% weighting achieved 9,359 correct at 0.18465781898498534 cross-entropy, while 53.80859375% lowered cross-entropy to 0.18465705108642577 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.044033790938556, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1846574378967285, "validation_score": 9359.422062939046}

RECENT RESULT
hypothesis: A 53.8037109375% original / 46.1962890625% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.1846574378967285.
change: Move the evaluation ensemble halfway from the successful 53.798828125% canonical weighting toward the 53.80859375% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The current 53.798828125% weighting achieved 9,359 correct at 0.1846574378967285 cross-entropy, while 53.80859375% lowered cross-entropy to 0.18465705108642577 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 64.63854629197158, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.18465724296569824, "validation_score": 9359.422063008493}

RECENT RESULT
hypothesis: A 53.80615234375% original / 46.19384765625% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465724296569824.
change: Move the evaluation ensemble halfway from the successful 53.8037109375% canonical weighting toward the 53.80859375% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The current 53.8037109375% weighting achieved 9,359 correct at 0.18465724296569824 cross-entropy, while 53.80859375% lowered cross-entropy to 0.18465705108642577 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.30421712505631, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1846571502685547, "validation_score": 9359.42206304152}

RECENT RESULT
hypothesis: A 53.807373046875% original / 46.192626953125% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.1846571502685547.
change: Move the evaluation ensemble halfway from the successful 53.80615234375% canonical weighting toward the 53.80859375% weighting that lost one correct prediction.
mechanism: Accuracy-constrained canonical-view weight bisection
evidence_used: The current 53.80615234375% weighting preserved 9,359 correct at 0.1846571502685547 cross-entropy, while 53.80859375% reduced cross-entropy to 0.18465705108642577 but produced 9,358 correct; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 60.39869900001213, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.1846571029663086, "validation_score": 9358.422063058371}



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
