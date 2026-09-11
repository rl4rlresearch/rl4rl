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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.00727504189126, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}
prior_hypothesis: A 50.687969970703125% live / 49.312030029296875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969650268556.

## Recent verification evidence

RECENT RESULT
hypothesis: A 50.68797607421875% live / 49.31202392578125% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Move the ensemble weight halfway between the current best mixture and the lower-live mixture that retained accuracy but had slightly worse cross-entropy.
mechanism: Lower-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.5068798828125 live weight is best at 9,290 correct and 0.202489697265625 cross-entropy; 0.506879638671875 retained 9,290 correct but produced 0.2024896987915039, making their untested midpoint the most informative remaining local probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 65.02680920786224, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969650268556, "validation_score": 9290.415803978574}

RECENT RESULT
hypothesis: A 50.687982177734375% live / 49.312017822265625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969650268556.
change: Move the ensemble weight halfway from the current best mixture toward the nearest tested higher-live mixture.
mechanism: Upper-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.5068797607421875 live weight achieved the best cross-entropy; the equidistant tested neighbors were both worse, but the higher-live neighbor was closer in cross-entropy than the lower-live neighbor, making the upper midpoint the most informative remaining probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.92139016697183, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969955444335, "validation_score": 9290.41580397752}

RECENT RESULT
hypothesis: A 50.687969970703125% live / 49.312030029296875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969650268556.
change: Move the ensemble weight halfway from the current best mixture toward the nearest tested lower-live mixture.
mechanism: Lower-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.5068797607421875 mixture achieved the best result; both adjacent tested mixtures were worse, and the upper-side midpoint was also worse, making the corresponding untested lower-side midpoint the most informative remaining probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.00727504189126, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}

RECENT RESULT
hypothesis: A 50.68797302246094% live / 49.31202697753906% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest tested higher-live mixture.
mechanism: Upper-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 live weight achieved the best verified cross-entropy; its equidistant tested neighbors were both worse, but the higher-live neighbor had lower cross-entropy than the lower-live neighbor, making the untested upper midpoint the most informative next probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.68797302246094% live / 49.31202697753906% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Increase the live-model ensemble weight from 0.50687969970703125 to 0.5068797302246094 and reduce the EMA complement accordingly.
mechanism: Upper-side sub-quantization live/EMA mixture refinement retry
evidence_used: The current mixture is best at 9,290 correct and 0.20248969497680663 cross-entropy; the sole verification of the adjacent upper midpoint timed out, so retrying it resolves the closest untested local probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.6879669189453125% live / 49.3120330810546875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest tested lower-live mixture.
mechanism: Lower-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 live weight achieved the best verified cross-entropy; the nearest lower-live mixture was worse, while the corresponding upper midpoint timed out twice, making the untested lower midpoint the most informative remaining local probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.6879669189453125% live / 49.3120330810546875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Decrease the live-model ensemble weight from 0.50687969970703125 to 0.506879669189453125 and increase the EMA complement accordingly.
mechanism: Lower-side sub-quantization live/EMA mixture refinement retry
evidence_used: The current mixture is best at 9,290 correct and 0.20248969497680663 cross-entropy; the corresponding upper midpoint timed out twice, while this closest lower midpoint timed out only once, so retrying it resolves the most informative unverified local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 68.50414575007744, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969955444335, "validation_score": 9290.41580397752}

RECENT RESULT
hypothesis: A 50.68797149658203% live / 49.31202850341797% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the unresolved higher-live midpoint.
mechanism: Upper-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 live weight is the verified best; the nearest lower-live probe was worse, while the nearest higher-live probe timed out twice, making its untested halfway point the most informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.50349141610786, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969955444335, "validation_score": 9290.41580397752}

RECENT RESULT
hypothesis: A 50.68796844482422% live / 49.31203155517578% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest validated lower-live mixture.
mechanism: Lower-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 live weight is the verified best; the adjacent lower-live weight 0.506879669189453125 was worse, while the corresponding upper-side quarter-step was also worse, making the untested lower-side quarter-step the most informative local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 79.01343420892954, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.202489701461792, "validation_score": 9290.415803976859}

RECENT RESULT
hypothesis: A 50.68797073364258% live / 49.31202926635742% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest validated higher-live mixture.
mechanism: Upper-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 weight is best; the nearest tested mixtures on both sides were worse, but the higher-live neighbor produced lower cross-entropy than the lower-live neighbor, motivating refinement on the upper side.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 68.08024295885116, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969955444335, "validation_score": 9290.41580397752}

RECENT RESULT
hypothesis: A 50.68797035217285% live / 49.31202964782715% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest validated higher-live mixture.
mechanism: Upper-side eighth-step live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 live weight is best; the nearest tested mixtures on both sides were worse, but the higher-live neighbor had lower cross-entropy than the lower-live neighbor, making this untested upper midpoint the most informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.97980979201384, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}

RECENT RESULT
hypothesis: A 50.68797016143799% live / 49.31202983856201% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the live-model ensemble weight halfway from the current best toward the nearest tested higher-live mixture that tied its cross-entropy.
mechanism: Upper-side sixteenth-step live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 weight and the higher 0.5068797035217285 weight both achieved 9,290 correct with 0.20248969497680663 cross-entropy, while farther upper and lower probes were worse; their untested midpoint is the most informative remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.90004099998623, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}



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
