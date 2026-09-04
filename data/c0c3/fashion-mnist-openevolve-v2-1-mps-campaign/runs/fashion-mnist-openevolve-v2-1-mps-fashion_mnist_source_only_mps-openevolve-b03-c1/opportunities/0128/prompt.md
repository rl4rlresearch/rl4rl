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
hypothesis: Replacing diagonal training shifts with center/cardinal ±1 shifts weighted like the successful validation ensemble will exceed 9,196 correct predictions by retaining beneficial translation invariance while eliminating unsupported diagonal transforms.
change: Sample only the centered crop and four one-pixel cardinal translations, with probabilities proportional to validation weights 3:2:2:2:2; retain flips and all other training settings.
mechanism: Center-and-cardinal translation augmentation
evidence_used: Restricting translations from ±2 to ±1 improved correctness from 9,166 to 9,196, while adding diagonal validation views previously reduced correctness to 9,155; this directly motivates removing the remaining diagonal training transforms.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reassigning diagonal one-pixel training shifts to the centered crop will exceed 9,196 correct predictions by eliminating transforms unsupported by the successful validation ensemble while preserving the baseline frequency of each cardinal shift.
change: Keep the existing triangular ±1 sampling, but convert samples shifted on both axes to centered crops, changing the distribution from 25% center/50% cardinal/25% diagonal to 50% center/50% cardinal/0% diagonal.
mechanism: Diagonal-to-center translation redistribution
evidence_used: Restricting translations from ±2 to ±1 improved correctness from 9,166 to 9,196, while diagonal validation views previously reduced correctness to 9,155. Unlike the timed-out 3:2:2:2:2 proposal, this isolates diagonal removal without increasing total cardinal-shift exposure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reassigning diagonal one-pixel shifts to cardinal shifts will exceed 9,196 correct predictions by eliminating unsupported diagonal transforms while closely matching the successful validation ensemble’s center-to-cardinal exposure ratio.
change: Detect diagonally shifted training samples and randomly center one shifted axis, changing augmentation from 25% center/50% cardinal/25% diagonal to 25% center/75% cardinal/0% diagonal.
mechanism: Cardinalized one-pixel translation augmentation
evidence_used: Restricting translations from ±2 to ±1 improved validation correctness from 9,166 to 9,196, while adding diagonal validation views previously reduced it to 9,155; the prior diagonal-removal attempts timed out and therefore did not provide contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Eliminating diagonal training translations while preserving balanced one-pixel cardinal shifts will exceed 9,196 correct predictions by aligning augmentation support with the successful validation views.
change: For diagonally shifted samples, deterministically alternate which axis is recentered using batch-index parity, producing 25% centered and 75% balanced cardinal crops without additional random sampling.
mechanism: Low-overhead balanced cardinalization
evidence_used: Restricting translations to ±1 improved correctness from 9,166 to 9,196, while diagonal validation views reduced correctness to 9,155. Earlier diagonal-removal implementations timed out, motivating this computationally lean formulation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 67.21863225009292, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22713947525024414, "validation_score": 9206.407451646764}

RECENT RESULT
hypothesis: A zero-initialized class-specific log-mean-exp evidence branch will exceed 9,206 correct predictions by learning translation-tolerant local class cues while retaining the position-sensitive flattened classifier.
change: Challenge the assumption that every class decision must pass through one shared dense spatial bottleneck: add per-class 1×1 evidence maps with smooth maximum pooling, and reduce the dense width from 160 to 158 to remain below the parameter ceiling.
mechanism: Additive class-specific spatial evidence pooling
evidence_used: Replacing spatial classification with global context regressed to 9,128, so this preserves the successful flattened head; cardinalized translation augmentation improved correctness to 9,206, making a complementary translation-tolerant class-evidence pathway plausible.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249009, "training_seconds": 76.16207754216157, "validation_accuracy": 0.9193, "validation_correct": 9193, "validation_cross_entropy": 0.23321033401489258, "validation_score": 9193.405445840186}

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
