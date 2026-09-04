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
hypothesis: A 50.68798828125% live / 49.31201171875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.2024896987915039.
change: Move the ensemble weight halfway from the current best 0.50687890625 live weight toward the accuracy-safe but cross-entropy-worse 0.506880859375 result.
mechanism: Local live/EMA cross-entropy refinement
evidence_used: The current mixture achieved the best score at 9,290 correct and 0.2024896987915039 cross-entropy, while the nearby higher-live mixture retained 9,290 correct but raised cross-entropy; their midpoint is the most informative probe of the apparent local optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.46274729212746, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.202489697265625, "validation_score": 9290.41580397831}

RECENT RESULT
hypothesis: A 50.687939453125% live / 49.312060546875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Move the ensemble weight halfway from the current best toward the nearby lower-live result.
mechanism: Local live/EMA cross-entropy refinement
evidence_used: The current 0.5068798828125 live weight achieved the best cross-entropy, while both adjacent tested weights were worse; the lower-live neighbor was closer in cross-entropy than the upper-live neighbor, motivating a lower-side midpoint probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.09913970786147, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248970184326173, "validation_score": 9290.415803976728}

RECENT RESULT
hypothesis: A 50.688037109375% live / 49.311962890625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Move the ensemble weight halfway from the current best toward the nearby higher-live result.
mechanism: Upper-side local live/EMA cross-entropy refinement
evidence_used: The current 0.5068798828125 live weight achieved the best cross-entropy; the lower-side midpoint was worse, while the next tested higher-live weight also retained 9,290 correct, making the untested upper midpoint the most informative local probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.688037109375% live / 49.311962890625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Increase the live-model ensemble weight from 0.5068798828125 to 0.50688037109375 and reduce the EMA weight complementarily.
mechanism: Upper-side local live/EMA cross-entropy refinement retry
evidence_used: The current 0.5068798828125 mixture is best at 9,290 correct and 0.202489697265625 cross-entropy; the sole verification of the adjacent upper midpoint timed out, so retrying it resolves the most informative untested local probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.6880126953125% live / 49.3119873046875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Increase the live-model ensemble weight from 0.5068798828125 to 0.506880126953125 and reduce the EMA weight complementarily.
mechanism: Conservative upper-side live/EMA cross-entropy refinement
evidence_used: The current weight achieved the best verified result, while the adjacent upper midpoint timed out twice; probing halfway toward that unresolved point is the smallest informative upper-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.73422233294696, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.202489697265625, "validation_score": 9290.41580397831}

RECENT RESULT
hypothesis: A 50.68802490234375% live / 49.31197509765625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Increase the live-model ensemble weight from 0.5068798828125 to 0.5068802490234375 and reduce the EMA weight complementarily.
mechanism: Conservative upper-side live/EMA cross-entropy refinement
evidence_used: The nearby 0.506880126953125 mixture retained 9,290 correct and tied the best cross-entropy, while 0.50688037109375 timed out twice; probing their midpoint is the smallest informative upper-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.112949250033125, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969955444335, "validation_score": 9290.41580397752}

RECENT RESULT
hypothesis: A 50.68800048828125% live / 49.31199951171875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Move the ensemble weight halfway between the current best mixture and the higher-live mixture that tied its cross-entropy.
mechanism: Sub-quantization live/EMA mixture refinement
evidence_used: The 0.5068798828125 mixture is the verified best; 0.506880126953125 retained 9,290 correct and tied its cross-entropy, while the next higher tested weight worsened cross-entropy, making their untested midpoint the most informative local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 66.54584258398972, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024896987915039, "validation_score": 9290.415803977783}

RECENT RESULT
hypothesis: A 50.6879638671875% live / 49.3120361328125% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Move the ensemble weight halfway between the current best mixture and the tested lower-live midpoint.
mechanism: Lower-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.5068798828125 live weight has the best verified cross-entropy; 0.50687939453125 retained 9,290 correct but was worse, while the corresponding upper-side refinement was also worse, making the untested lower half-midpoint the most informative remaining local probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 50.6879638671875% live / 49.3120361328125% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.
change: Retry the unresolved lower-side midpoint by decreasing the live-model ensemble weight from 0.5068798828125 to 0.506879638671875 and increasing the EMA complement.
mechanism: Lower-side sub-quantization live/EMA mixture refinement retry
evidence_used: The current mixture is best at 9,290 correct and 0.202489697265625 cross-entropy; the only verification of this adjacent midpoint timed out, so retrying it resolves the most informative remaining local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.0375761250034, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024896987915039, "validation_score": 9290.415803977783}

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
