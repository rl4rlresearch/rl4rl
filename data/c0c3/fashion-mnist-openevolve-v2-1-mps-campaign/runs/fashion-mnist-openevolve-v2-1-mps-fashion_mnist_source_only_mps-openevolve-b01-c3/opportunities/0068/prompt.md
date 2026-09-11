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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.99758079089224, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21854879875183106, "validation_score": 9281.410324149932}
prior_hypothesis: Linearly averaging 20 evenly spaced iterates across the final 10% will exceed the current 9,281 correct predictions while reducing averaging overhead enough to finish verification.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.10896966606379, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2172163459777832, "validation_score": 9286.410773320333}
prior_hypothesis: Weighting tail iterates by recency^1.5 while retaining uniform ten-view logit pooling will exceed 9,287 correct predictions by reducing mismatch with terminal BatchNorm statistics without discarding the smoothing benefit of the final-10% window.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.95856470800936, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21857940216064453, "validation_score": 9285.41031384505}
prior_hypothesis: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.1797975001391, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.21733987007141115, "validation_score": 9287.410731638955}
prior_hypothesis: Averaging logits across all ten translation/flip views will exceed 9,286 correct predictions by rewarding class evidence that remains consistent across transformations.

## Recent verification evidence

RECENT RESULT
hypothesis: Cardinal augmentation with half-density linear tail averaging and BatchNorm momentum 0.04 will exceed 9,285 correct predictions by aligning evaluation statistics with the averaged parameters while retaining the previously completed averaging workload.
change: Restore the verified five-position augmentation and every-other-step final-10% linear parameter average, while setting all BatchNorm running-stat momentum to 0.04.
mechanism: Tail-average-aligned BatchNorm memory
evidence_used: Half-density averaging completed with 9,283 correct, while dense averaging reached 9,285; momentum 0.04 gives running statistics roughly the same temporal center as the linear tail average. The earlier dense momentum trial timed out, so this tests the idea using the proven lower-cost averaging schedule.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.92729725013487, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.2186310947418213, "validation_score": 9280.410296440126}

RECENT RESULT
hypothesis: Giving the unshifted view twice the weight of each shifted view will exceed 9,285 correct predictions by retaining translation robustness while reducing boundary-induced errors from replicated-padding shifts.
change: Replace uniform probability averaging across the five translation positions with a 2:1 center-to-shifted weighted mixture; training and computation remain unchanged.
mechanism: Center-prioritized translation ensembling
evidence_used: The strongest design already achieves 9,285 correct with dense tail averaging, while averaging-density and BatchNorm variants produced only small regressions or timeouts; its five-position inference mixture still weights the clean center view identically to four padding-altered views, making ensemble weighting an isolated unexplored lever.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fusing all parameter-average updates into multi-tensor operations will make dense final-10% averaging finish within the time limit and recover the reference result of at least 9,285 correct predictions.
change: Average every final-window iterate as in the strongest reference design, replacing per-parameter Python loops with fused foreach lerp and copy operations.
mechanism: Fused dense linear-recency tail averaging
evidence_used: Dense final-10% linear averaging achieved the best verified result of 9,285 correct, while half-density averaging reached 9,283; later dense variants timed out, making averaging overhead the targeted constraint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dense linear-recency averaging over the final 10% will reproduce at least 9,285 correct predictions while completing on time when each iterate is flattened and averaged with one contiguous tensor operation.
change: Replace sparse per-parameter averaging with dense final-window averaging in a single flat parameter buffer, then install that buffer at the final step.
mechanism: Contiguous-buffer dense tail averaging
evidence_used: Dense final-10% averaging achieved the best verified result of 9,285 correct versus 9,281 for every-fourth-step averaging; the later fused multi-tensor attempt timed out, motivating a single contiguous-buffer implementation with fewer averaging dispatches.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Image-conditioned row/column gating will exceed 9,285 correct predictions by preserving garment-part arrangement before global pooling while reducing refinement computation enough to complete verification.
change: Replace the assumption that fixed global mean/max statistics alone capture spatial structure with factorized axial attention that modulates each channel by learned row and column context; slightly narrow the classifier to remain below 250,000 parameters and restore the stronger default BatchNorm behavior.
mechanism: Factorized coordinate-attention refinement
evidence_used: Global-pooling variants plateaued at 9,285 correct while averaging changes moved only 2–4 predictions; spatial refinement previously improved 9,202 to 9,209, but the axial-profile head timed out. Applying compressed axial context before pooling tests the spatial mechanism while replacing the expensive 96×96 pointwise refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Balancing all ten translation/flip transformations within every minibatch will exceed 9,285 correct predictions by reducing augmentation-gradient and BatchNorm-statistic variance without increasing computation.
change: Replace independent per-image transformation sampling with a randomly rotated, near-uniform assignment of the ten training transformations across each batch.
mechanism: Minibatch-stratified transformation sampling
evidence_used: The 9,285-correct current design trains on the same ten transformations used at evaluation, but samples them independently; averaging and BatchNorm variants produced regressions or timeouts, motivating this isolated, zero-overhead alignment of each training batch with the evaluation mixture.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing from 0.04 to 0 while preserving the original average strength of 0.02 will exceed 9,285 correct predictions by regularizing early learning and sharpening class margins during the low-learning-rate tail.
change: Replace constant 0.02 label smoothing with a linear exposure-progress schedule from approximately 0.04 to 0; architecture, augmentation, optimizer, and averaging cost remain unchanged.
mechanism: Mean-preserving label-smoothing annealing
evidence_used: Averaging-density variants moved correctness by only 2–4 images, plateauing at 9,285, while every verified design retained constant 0.02 smoothing. This isolates an untested, computation-neutral training lever suited to the accuracy-first objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging every iterate in the final 10% will reproduce at least 9,285 correct predictions, matching the strongest qualified design.
change: Replace every-fourth-iterate tail averaging with dense per-iterate averaging while preserving the proven averaging window and linear-recency weights.
mechanism: Dense linear-recency tail averaging
evidence_used: Reference Design 2 used this exact dense schedule, completed in 59.96 seconds, and achieved the best qualified result of 9,285 correct versus 9,281 for the current sparse schedule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reconstructing the dense linear-recency average from the first and every second tail iterate will recover at least 9,285 correct predictions while retaining approximately half-density averaging cost.
change: Restore proven default BatchNorm momentum and replace the biased half-density average with online quadrature weights that exactly reproduce dense averaging when intervening parameter trajectories are locally linear.
mechanism: Endpoint-sampled linear-tail quadrature
evidence_used: Dense tail averaging achieved 9,285 correct versus 9,283 for half-density averaging, while momentum 0.04 reduced the latter to 9,280. The prior midpoint reconstruction timed out after adding another parameter copy; sampling the first tail iterate permits the same reconstruction using the existing average buffer.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.66377333295532, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.21857954750061034, "validation_score": 9286.410313796112}

RECENT RESULT
hypothesis: Averaging logits across all ten translation/flip views will exceed 9,286 correct predictions by rewarding class evidence that remains consistent across transformations.
change: Replace probability-space mixing across translations with a single uniform logit average across all ten evaluation views; training and parameter averaging remain unchanged.
mechanism: Geometric ten-view evidence pooling
evidence_used: The current endpoint-reconstructed tail average is strongest at 9,286 correct, while uniform probability pooling over all ten views timed out; consistent logit pooling is the complementary computation-neutral aggregation test and removes the current asymmetric treatment of flips and translations.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.1797975001391, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.21733987007141115, "validation_score": 9287.410731638955}

RECENT RESULT
hypothesis: Weighting the unshifted flip pair twice as strongly as each shifted pair while retaining the 9,287-correct logit-pooling and quadrature design will exceed 9,287 correct predictions by reducing replicated-boundary artifacts without sacrificing translation robustness.
change: Restore the strongest endpoint-reconstructed half-density parameter average and replace uniform ten-view logit averaging with a 2:1 center-to-shifted weighted logit mean.
mechanism: Center-prioritized geometric view pooling with endpoint-sampled tail quadrature
evidence_used: Uniform ten-view logit pooling with endpoint-sampled quadrature achieved the best result, 9,287 correct; the earlier center-prioritized probability-pooling trial timed out, leaving center weighting under the stronger logit-pooling rule unmeasured.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting tail iterates by recency^1.5 while retaining uniform ten-view logit pooling will exceed 9,287 correct predictions by reducing mismatch with terminal BatchNorm statistics without discarding the smoothing benefit of the final-10% window.
change: Restore the strongest ten-view logit pooling and endpoint-sampled quadrature, but reconstruct a moderately later-biased recency^1.5 parameter average instead of the linear-recency average.
mechanism: Three-halves-recency endpoint-sampled tail quadrature
evidence_used: Endpoint-sampled linear quadrature reached 9,286 correct and uniform ten-view logit pooling raised it to 9,287; BatchNorm-statistic experiments indicate parameter/statistic alignment remains unresolved, motivating a computation-neutral shift toward later tail iterates.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.10896966606379, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2172163459777832, "validation_score": 9286.410773320333}



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
