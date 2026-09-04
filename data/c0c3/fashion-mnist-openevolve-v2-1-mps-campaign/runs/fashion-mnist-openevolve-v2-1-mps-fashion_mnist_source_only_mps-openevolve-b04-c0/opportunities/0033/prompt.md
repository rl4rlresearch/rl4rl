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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 77.82156541594304, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19386489181518554, "validation_score": 9359.418807859605}
prior_hypothesis: Keeping parameter EMA at 0.96 decay while increasing buffer decay to 0.97 will exceed 9,358 correct predictions—or tie with lower cross-entropy—by smoothing BatchNorm running statistics more strongly without altering the best-tested parameter trajectory.

## Recent verification evidence

RECENT RESULT
hypothesis: An EMA decay of 0.9625 will exceed the 0.96 design’s 9,358 correct predictions—or tie it with lower cross-entropy—by adding modest variance reduction without returning to the less responsive 0.97 horizon.
change: Reduce parameter and floating-buffer EMA update rates from 4% to 3.75%, preserving all other training behavior.
mechanism: Upper-side coherent EMA horizon refinement
evidence_used: Decay 0.96 achieved the best result at 9,358 correct and 0.193622 cross-entropy; 0.97 fell to 9,356 correct, while the lower-side midpoint 0.955 fell to 9,357, making 0.9625 the most informative untested local bracket.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.60182425007224, "validation_accuracy": 0.9357, "validation_correct": 9357, "validation_cross_entropy": 0.19357746963500977, "validation_score": 9357.418908711601}

RECENT RESULT
hypothesis: Keeping parameter EMA at the best-tested 0.96 decay while increasing floating-buffer updates to 0.0625 will exceed 9,358 correct predictions—or tie with lower cross-entropy—by compensating for BatchNorm’s intrinsic running-statistic lag.
change: Preserve the 4% parameter EMA update and increase only the floating-buffer EMA update from 4% to 6.25%.
mechanism: Lag-compensated BatchNorm buffer averaging
evidence_used: Extending parameter EMA to BatchNorm buffers previously improved validation_correct from 9,323 to 9,335, while joint EMA-decay refinement established 0.96 as the best parameter horizon; decoupling the buffer rate now targets BatchNorm’s additional momentum-induced lag without disturbing that optimum.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 81.48273454094306, "validation_accuracy": 0.9355, "validation_correct": 9355, "validation_cross_entropy": 0.1934361976623535, "validation_score": 9355.418958299555}

RECENT RESULT
hypothesis: Keeping parameter EMA at 0.96 decay while increasing buffer decay to 0.97 will exceed 9,358 correct predictions—or tie with lower cross-entropy—by smoothing BatchNorm running statistics more strongly without altering the best-tested parameter trajectory.
change: Preserve the 4% parameter EMA update and reduce only the floating-buffer update from 4% to 3%.
mechanism: Slower BatchNorm-buffer EMA
evidence_used: The coherent 4% EMA achieved 9,358 correct, while increasing only the buffer update to 6.25% regressed to 9,355; testing a modest decrease to 3% directly probes the opposite side of the buffer-rate optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 77.82156541594304, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19386489181518554, "validation_score": 9359.418807859605}

RECENT RESULT
hypothesis: Keeping parameter EMA at 0.96 decay while increasing buffer EMA decay from 0.97 to 0.98 will exceed 9,359 correct predictions—or tie with lower cross-entropy—by further reducing noise in BatchNorm running statistics without changing the best-tested parameter trajectory.
change: Preserve the 4% parameter EMA update and reduce only the floating-buffer EMA update from 3% to 2%.
mechanism: Further-slowed BatchNorm-buffer EMA
evidence_used: Reducing the buffer update from the coherent 4% rate to 3% improved validation_correct from 9,358 to 9,359, whereas increasing it to 6.25% regressed to 9,355; another measured reduction directly tests the successful direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 81.22066495893523, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19450161666870117, "validation_score": 9358.418584615561}

RECENT RESULT
hypothesis: A 3.5% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by retaining most of the 3% rate’s accuracy gain while moving toward the better-calibrated 4% rate.
change: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.5%.
mechanism: Upper-side BatchNorm-buffer EMA midpoint refinement
evidence_used: The 3% buffer update achieved the best result at 9,359 correct; 2% and 4% each achieved 9,358, while 4% had lower cross-entropy than 3%, motivating the untested midpoint between 3% and 4%.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 2.5% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by balancing the 3% rate’s accuracy against the stronger smoothing of the 2% rate while preserving the best-tested 4% parameter EMA.
change: Keep parameter EMA at 4% and reduce only the floating-buffer EMA update from 3% to 2.5%.
mechanism: Lower-side BatchNorm-buffer EMA midpoint refinement
evidence_used: The 3% buffer update achieved the best result at 9,359 correct, while 2% achieved 9,358; testing their untried midpoint is the most informative refinement after the 3.5% verification failed to finish.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 58.70114220818505, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.19409763717651368, "validation_score": 9359.418726228436}

RECENT RESULT
hypothesis: A 3.25% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly improving responsiveness while preserving the smoothing benefit of the best-tested 3% rate.
change: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.25%.
mechanism: Upper-side BatchNorm-buffer EMA refinement
evidence_used: The 3% buffer update achieved the best result at 9,359 correct; 2.5% tied but had worse cross-entropy, while the larger 3.5% test timed out, making 3.25% the closest untested upper-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 66.20031079091132, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19378411865234374, "validation_score": 9358.418836196752}

RECENT RESULT
hypothesis: A 2.75% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by interpolating between the accuracy-preserving 2.5% rate and the best-tested 3% rate while keeping parameter EMA unchanged.
change: Keep parameter EMA at 4% and reduce only the floating-buffer EMA update from 3% to 2.75%.
mechanism: Lower-side BatchNorm-buffer EMA refinement
evidence_used: The 3% buffer update achieved 9,359 correct with 0.193865 cross-entropy, while 2.5% also achieved 9,359 correct but with worse cross-entropy; their untested midpoint is the closest remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 68.64094929187559, "validation_accuracy": 0.9359, "validation_correct": 9359, "validation_cross_entropy": 0.1939667667388916, "validation_score": 9359.418772124927}

RECENT RESULT
hypothesis: A 3.125% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by moving toward the better-calibrated 3.25% rate without crossing its observed accuracy boundary.
change: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.125%.
mechanism: Boundary-refined BatchNorm-buffer EMA
evidence_used: Buffer updates of 2.5%, 2.75%, and 3% all produced 9,359 correct while cross-entropy improved monotonically; 3.25% further improved cross-entropy but lost one correct prediction, making their midpoint the closest untested boundary refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 63.13526958390139, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19382225799560546, "validation_score": 9358.418822816086}

RECENT RESULT
hypothesis: A 3.0625% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by improving responsiveness without crossing the accuracy boundary observed at 3.125%.
change: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.0625%.
mechanism: Fine-grained BatchNorm-buffer EMA boundary search
evidence_used: The 3% update achieved the best 9,359-correct result; 3.125% lowered cross-entropy but lost one correct prediction, so their midpoint is the most informative remaining boundary refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 79.04740833397955, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.1938429786682129, "validation_score": 9358.41881554688}

RECENT RESULT
hypothesis: Reducing the parameter EMA update to 3.75% while retaining the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by combining the stronger parameter smoothing associated with 0.193577 cross-entropy and the buffer horizon responsible for the current accuracy gain.
change: Change only the parameter EMA update rate from 4% to 3.75%; preserve the 3% floating-buffer update and all other behavior.
mechanism: Decoupled weight-and-buffer EMA refinement
evidence_used: The coherent 3.75% EMA produced the lowest observed cross-entropy, 0.193577, but lost one correct prediction; independently slowing buffer updates to 3% raised the best result to 9,359 correct, so decoupling these rates directly tests whether their benefits combine.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 237818, "training_seconds": 59.93516691587865, "validation_accuracy": 0.9358, "validation_correct": 9358, "validation_cross_entropy": 0.19377296028137206, "validation_score": 9358.418840111675}

RECENT RESULT
hypothesis: A 4.25% parameter EMA update with the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly improving weight responsiveness without disturbing the BatchNorm smoothing responsible for the current accuracy gain.
change: Increase only the parameter EMA update rate from 4% to 4.25%, preserving the 3% floating-buffer update and all other behavior.
mechanism: Upper-side decoupled parameter-EMA refinement
evidence_used: Reducing the parameter update to 3.75% with a 3% buffer update regressed from 9,359 to 9,358 correct; the symmetric upper-side refinement is the closest informative untested direction around the best 4% parameter rate.
result: training did not finish within the verification time limit



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
