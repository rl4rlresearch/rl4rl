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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.09104441688396, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.21856914138793945, "validation_score": 9283.410317300035}
prior_hypothesis: Linearly averaging every other iterate across the final 10% will exceed 9,281 correct predictions by recovering more of the dense average’s 9,285-correct benefit while retaining substantially lower averaging overhead.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.99758079089224, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21854879875183106, "validation_score": 9281.410324149932}
prior_hypothesis: Linearly averaging 20 evenly spaced iterates across the final 10% will exceed the current 9,281 correct predictions while reducing averaging overhead enough to finish verification.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.95856470800936, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21857940216064453, "validation_score": 9285.41031384505}
prior_hypothesis: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.92729725013487, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.2186310947418213, "validation_score": 9280.410296440126}
prior_hypothesis: Cardinal augmentation with half-density linear tail averaging and BatchNorm momentum 0.04 will exceed 9,285 correct predictions by aligning evaluation statistics with the averaged parameters while retaining the previously completed averaging workload.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing purely translation-invariant global pooling with learned channel projections summarized as row and column activation profiles will exceed 9,285 correct predictions by exposing garment-part arrangement while retaining robust global mean and peak features.
change: Challenge the assumption that global mean/max statistics contain sufficient shape information: project late features into 16 layout channels, preserve their seven-row and seven-column profiles, and classify the combined global-and-layout representation through a parameter-budgeted head.
mechanism: Learned axial-profile classification head
evidence_used: The strongest global-pooling design reached 9,285 correct, while changing its averaging density moved accuracy by only 2–4 images and repeated averaging variants timed out. Local spatial refinement improved correctness from 9,202 to 9,209, indicating spatial organization is useful; axial profiles test that signal directly without costly all-pairs attention and remain under 250,000 parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Densely averaging only the classifier while averaging the backbone every fourth iterate will exceed 9,283 correct predictions by recovering decision-boundary stability from the 9,285-correct dense average without its full runtime cost.
change: Keep the proven sparse final-10% average for all parameters, add a separate dense linear-recency average for the small classifier, and install the dense classifier average after the sparse full-model average.
mechanism: Hybrid-density linear tail averaging
evidence_used: Increasing final-10% averaging density improved correctness monotonically from 9,281 every fourth iterate, to 9,283 every second iterate, to 9,285 every iterate; concentrating dense averaging on the classifier targets prediction-changing parameters with much less overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Quadratically weighting every other iterate across the final 10% will exceed 9,283 correct predictions by retaining the completed half-density runtime while aligning averaged weights more closely with terminal BatchNorm statistics.
change: Restore proven five-way cardinal translation augmentation and half-density tail averaging, but weight sampled iterates quadratically by recency instead of linearly.
mechanism: Quadratic-recency half-density tail averaging
evidence_used: Half-density linear averaging completed with 9,283 correct, while dense linear averaging reached 9,285; stronger recency weighting targets the remaining gap without increasing averaging frequency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging predictive probabilities across all ten equally sampled training transformations will exceed 9,285 correct predictions by avoiding the current flip-pair geometric pooling, which can let one uncertain flipped view suppress an otherwise correct prediction.
change: Replace hierarchical flip-logit/position-probability aggregation with a uniform probability mixture over all five translations and both horizontal orientations.
mechanism: Uniform probability-space ten-view ensembling
evidence_used: The strongest verified design achieves 9,285 correct using uniformly sampled five-position, two-orientation augmentation, but inference combines flips differently from positions; matching the ensemble to that uniform augmentation distribution is an isolated, computation-neutral test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Approximating omitted odd tail iterates by interpolation between adjacent sampled iterates will exceed the current 9,283 correct predictions and approach the dense average’s 9,285, without its per-step averaging cost.
change: Anchor the average immediately before the final-10% window, retain every-other-iterate updates, and use quadrature weights that reproduce the dense linear-recency average when consecutive parameter trajectories are locally linear.
mechanism: Midpoint-reconstructed half-density tail averaging
evidence_used: Dense final-10% linear averaging achieved 9,285 correct, while half-density linear averaging completed with 9,283; reconstructing the missing iterates directly targets that two-image gap with only one additional parameter copy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using evaluation-batch statistics will exceed 9,285 correct predictions by eliminating the mismatch between tail-averaged parameters and terminal BatchNorm running statistics without adding training-time averaging work.
change: Disable running-stat tracking for every BatchNorm2d layer so training remains unchanged while evaluation computes normalization statistics from the full concatenated ten-view batch.
mechanism: Batch-adaptive normalization for averaged weights
evidence_used: Dense tail averaging achieved the best verified 9,285 correct, while explicitly averaging BatchNorm buffers timed out; batch-adaptive normalization targets the same unresolved parameter/statistics mismatch with no additional optimizer-step copies.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 67.18113620812073, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.21914594917297364, "validation_score": 9277.410123168878}

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
