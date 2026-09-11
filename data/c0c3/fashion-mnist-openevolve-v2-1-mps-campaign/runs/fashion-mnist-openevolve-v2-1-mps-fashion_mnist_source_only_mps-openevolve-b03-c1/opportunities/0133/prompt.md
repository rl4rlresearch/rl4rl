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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 78.9700069159735, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23116647605895996, "validation_score": 9210.406118920328}
prior_hypothesis: Averaging class probabilities across the ten validation views will exceed 9,206 correct predictions by preventing a single unreliable transformed view from disproportionately suppressing the correct class.

## Recent verification evidence

RECENT RESULT
hypothesis: Cycling through center and cardinal translations in the exact 3:2:2:2:2 validation-view ratio will exceed 9,206 correct predictions by reducing per-batch augmentation sampling noise during the fixed two-pass exposure.
change: Replace random triangular shift generation and diagonal cardinalization with a low-overhead deterministic 11-example cycle containing three centered crops and two crops for each one-pixel cardinal translation.
mechanism: Deterministic validation-matched cardinal augmentation cycle
evidence_used: Removing diagonal translations improved correctness from 9,196 to 9,206; the earlier validation-matched 3:2:2:2:2 proposal timed out, while deterministic balanced cardinalization completed successfully, motivating a deterministic formulation of the previously unverified distribution.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Centering one-eleventh of diagonal candidates before cardinalizing the remainder will exceed 9,206 correct predictions by matching the successful validation ensemble’s 3:8 center-to-cardinal exposure ratio while preserving balanced one-pixel shifts.
change: Convert one-eleventh of sampled diagonal shifts into centered crops using a step-rotated batch-index mask; deterministically cardinalize all remaining diagonals as before.
mechanism: Validation-ratio cardinalization
evidence_used: Eliminating diagonal translations improved correctness from 9,196 to 9,206. The current fast implementation yields a 1:3 center-to-cardinal ratio, while the successful validation ensemble uses 3:8; centering one-eleventh of diagonal candidates closes that specific distribution gap without the costly deterministic-cycle implementation that timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.64111574995331, "validation_accuracy": 0.9198, "validation_correct": 9198, "validation_cross_entropy": 0.2278820697784424, "validation_score": 9198.407205229481}

RECENT RESULT
hypothesis: Reducing centered training crops from 25% to approximately 22.7% while distributing the difference across balanced cardinal shifts will exceed 9,206 correct predictions.
change: Deterministically convert one-eleventh of centered candidates into step-rotated one-pixel cardinal crops, preserving the existing diagonal cardinalization and computational profile.
mechanism: Modestly increased cardinal-shift exposure
evidence_used: Increasing center exposure from 25% to 27.3% reduced correctness from 9,206 to 9,198, while replacing diagonal crops with cardinal shifts previously improved correctness from 9,196 to 9,206; a symmetric, modest shift away from center is therefore the most directly supported next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 64.24652804085054, "validation_accuracy": 0.9196, "validation_correct": 9196, "validation_cross_entropy": 0.2282196662902832, "validation_score": 9196.407093302381}

RECENT RESULT
hypothesis: Alternating the recentered axis across optimizer steps will exceed 9,206 correct predictions by removing fixed batch-position coupling while preserving the successful 25% center/75% balanced-cardinal augmentation distribution.
change: Include step parity in the existing low-overhead diagonal cardinalization rule; leave augmentation frequencies, randomness, architecture, and optimization unchanged.
mechanism: Step-rotated balanced cardinalization
evidence_used: Eliminating diagonal shifts with batch-parity cardinalization improved correctness from 9,196 to 9,206, while moving center exposure in either direction reduced it to 9,198 and 9,196; this motivates retaining the winning distribution and isolating its fixed-position artifact.
result: training did not finish within the verification time limit

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Averaging class probabilities across the ten validation views will exceed 9,206 correct predictions by preventing a single unreliable transformed view from disproportionately suppressing the correct class.
change: Replace weighted logit averaging with weighted softmax-probability averaging, returning scaled log-probabilities while preserving all views and weights.
mechanism: Arithmetic-probability test-time augmentation
evidence_used: Adding diagonal validation views previously reduced correctness to 9,155, while removing unsupported diagonal training transforms improved correctness from 9,196 to 9,206; this indicates unequal view reliability and motivates a less veto-sensitive ensemble rule.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 78.9700069159735, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.23116647605895996, "validation_score": 9210.406118920328}

RECENT RESULT
hypothesis: Raising the probability-pooling power from 1 to 1.25 will exceed 9,210 correct predictions by emphasizing confident agreement while retaining the arithmetic ensemble’s resistance to low-probability vetoes.
change: Replace arithmetic probability averaging with a class-wise power mean across the same ten weighted validation views.
mechanism: Mild power-mean test-time pooling
evidence_used: Arithmetic probability pooling improved correctness from 9,206 to 9,210 over logit/geometric pooling, motivating a modest extrapolation beyond the arithmetic endpoint rather than returning toward the inferior geometric rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting centered validation views at exactly 25% will exceed 9,210 correct predictions by matching the augmentation distribution whose higher and lower centered-crop variants both performed worse.
change: Change center-to-cardinal validation weights from 3:2:2:2:2 to 4:3:3:3:3 while retaining arithmetic probability pooling and all ten views.
mechanism: Training-matched test-time view weighting
evidence_used: The 25%-center training distribution achieved 9,206 correct before probability pooling, while increasing center exposure to 27.3% produced 9,198 and decreasing it to 22.7% produced 9,196; the current validation weights assign 27.3% to centered views.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed winner-take-all pooling with learned per-channel max/mean mixtures will exceed 9,210 correct predictions by reducing one-pixel aliasing while preserving the successful position-sensitive classifier.
change: Replace all three max-pooling stages with channel-specific learned mixed pooling initialized at 90% max and 10% average; this adds 144 parameters for a total of 249,933.
mechanism: Channel-adaptive mixed downsampling
evidence_used: Cardinal-only augmentation improved correctness from 9,196 to 9,206, and arithmetic multi-view pooling further improved it to 9,210, indicating persistent alignment sensitivity. Global spatial classification previously regressed to 9,128, so this retains the flattened spatial head while challenging the load-bearing assumption that every downsampling stage should discard all but its local maximum.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A mild convex confidence tilt will exceed 9,210 correct predictions by emphasizing confident agreement across views without restoring the low-probability veto behavior of geometric pooling.
change: Replace arithmetic view-probability averaging with a normalized average of `p + 0.25p²`, preserving all validation views and weights.
mechanism: Quadratically confidence-tilted probability pooling
evidence_used: Arithmetic probability pooling improved correctness from 9,206 to 9,210 over logit/geometric pooling. The fractional power-mean test timed out, so this computationally cheap multiply-add tests a smaller move in the same confidence-emphasizing direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Assigning exactly 25% of ensemble weight to centered views will exceed 9,210 correct predictions by matching the augmentation balance whose higher and lower center rates both performed worse.
change: Change center-to-cardinal validation weights from 3:2:2:2:2 to 4:3:3:3:3 while retaining arithmetic probability pooling.
mechanism: Training-matched test-time view weighting
evidence_used: The 25%-center training distribution reached 9,206 correct, versus 9,198 at 27.3% and 9,196 at 22.7%; the prior verification of this exact low-cost weighting timed out without yielding subject-level evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 79.60080970800482, "validation_accuracy": 0.9205, "validation_correct": 9205, "validation_cross_entropy": 0.23133830718994142, "validation_score": 9205.406062247133}



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
