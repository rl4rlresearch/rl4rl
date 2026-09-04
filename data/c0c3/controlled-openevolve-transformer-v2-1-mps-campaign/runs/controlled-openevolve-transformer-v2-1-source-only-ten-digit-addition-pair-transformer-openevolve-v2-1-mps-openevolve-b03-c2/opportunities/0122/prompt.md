# Optimize a transformer for 10-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two 10-digit numbers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% accuracy under the fixed verification process. A
smaller implementation is useful only when it meets that accuracy requirement.
Every submitted implementation is trained from a fresh initialization.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must:

- have nonzero trainable parameters;
- contain and use at least one learned causal self-attention module;
- map token inputs to token logits through the learned model;
- train from a fresh initialization during verification;
- write both `checkpoints/best.pt` and a positive-step `checkpoints/last.pt`;
- keep source code unchanged while training; and
- use the protected generic decoding interface exactly as supplied.

Do not implement or embed decimal arithmetic, carry propagation, place-value
rules, digit lookup tables, finite-state addition transitions, fixed answer
rules, or input-dependent Python logic that directly computes the sum. Do not
hide such a solver in model generation, token processing, training, or saved
weights. Do not add dummy or zero-length parameters to disguise a fixed
algorithm as a learned model.

Do not modify protected files. Do not perform post-training state-dictionary
surgery, substitute a different saved model, truncate weights after training,
or report a parameter count that differs from the submitted model.

## Work boundaries

Minimize parameters. Required result: accuracy >= 0.99.
Editable source files: src/model.py, src/train.py.
Results reported after each verification: accuracy, parameters, training_steps.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or verification yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.998, "parameters": 1484, "training_steps": 4999}
prior_hypothesis: Sharing the first two learned final-LayerNorm gains will reduce the verified 1485-parameter model to 1484 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}
prior_hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9979, "parameters": 1486, "training_steps": 4999}
prior_hypothesis: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Completing Reference Design 3’s first-head normalization and removing only the feature-uniform final-LayerNorm bias direction with an orthonormal Householder chart will reduce the model from 1486 to 1484 parameters while retaining at least 99% accuracy.
change: Normalize the remaining first-head zero-bias query row and transfer its scale into the paired key row, then store the final LayerNorm bias in a seven-dimensional orthonormal centered chart.
mechanism: Orthogonal centered final-bias chart atop the normalized first-head query frame
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters. Final-LayerNorm gain sharing reached 98.91%, suggesting the terminal affine map can nearly tolerate another constraint; centering its bias preserves all seven relative bias directions and leaves the learned gains unrestricted.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7394, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing the second head’s biased-query basis reduction with a conditioned complementary shear will reduce the verified 1485-parameter model to 1484 parameters while retaining at least 99% accuracy.
change: Remove one coordinate from the second biased query after reducing the anchor modulo the two zero-bias queries, and apply the exact inverse transformation to the corresponding key rows.
mechanism: Max-pivot complementary biased-query shear
evidence_used: The current max-pivot query-frame design achieved 99.92% at 1485 parameters, and Reference Design 1 achieved 99.99% with the first shear between these biased queries. Unlike failed normalization, bias tying, and geometric constraints, this retains both independent biases and removes another continuous query/key basis redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6545000000000001, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing Reference Design 3’s normalized orthogonal query frame will reduce the current model from 1486 to 1485 parameters while retaining at least 99% accuracy.
change: Normalize the first-head neighbor query with a fixed-norm max-pivot chart, absorb its initialization scale into the matching key row, and store one fewer learned coordinate.
mechanism: Max-pivot normalization of the first head’s orthogonal zero-bias query row
evidence_used: Reference Design 3 used this exact reduction and achieved 99.92% accuracy with 1485 parameters, while every reported 1484-parameter extension failed the accuracy requirement or verification.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.955, "parameters": 1485, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the first two learned final-LayerNorm gains will reduce the verified 1485-parameter model to 1484 parameters while retaining at least 99% accuracy.
change: Store six final-LayerNorm gains, reuse the first gain for coordinates zero and one, and retain the fixed final gain.
mechanism: Leading-pair terminal LayerNorm gain sharing
evidence_used: Sharing two terminal adjacent gains reached 98.91%, only 0.09 percentage points below the requirement, while the unconstrained design reached 99.92%; testing the distinct leading adjacent pair is the closest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified leading-pair LayerNorm-gain sharing to the first three gains will reduce Reference Design 3 from 1484 to 1483 parameters while retaining at least 99% accuracy.
change: Complete the first-head normalized orthogonal query frame, then represent the first three final-LayerNorm gains with one shared learned scalar.
mechanism: Normalized first-head query frame with leading-triplet final-gain sharing
evidence_used: Reference Design 3 achieved 99.8% accuracy at 1484 parameters with the normalized frame and shared leading gain pair; unlike the failed nonadjacent sharing, extending that successful local group is the closest informative one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6637000000000001, "parameters": 1483, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the first two final-LayerNorm biases will reduce the qualified 1484-parameter model to 1483 parameters while retaining at least 99% accuracy.
change: Store seven final-LayerNorm biases and reuse the first bias for coordinates zero and one, leaving the successful gain sharing unchanged.
mechanism: Leading-pair final-LayerNorm bias sharing
evidence_used: Leading-pair gain sharing achieved 99.8% at 1484 parameters, while extending that gain group to three coordinates collapsed accuracy; sharing the corresponding two zero-initialized biases tests a distinct local constraint without further restricting the gains.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reproducing the qualified 1484-parameter normalized-frame and leading-gain-sharing design, then sharing the corresponding leading final-LayerNorm biases, will yield 1483 parameters while retaining at least 99% accuracy.
change: Normalize the first-head neighboring query and transfer its scale into the paired key row, share the first two final-LayerNorm gains, and share their biases.
mechanism: Normalized first-head query frame with paired terminal affine sharing
evidence_used: Reference Design 3 achieved 99.8% accuracy at 1484 parameters. The direct 1483-parameter leading-bias-sharing trial timed out rather than producing negative accuracy evidence, so completing that unresolved test is the most informative next change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reproducing the qualified 1484-parameter design and sharing its leading final-LayerNorm biases will yield 1483 parameters and at least 99% accuracy; removing intermediate validation will allow the previously timed-out experiment to finish.
change: Add the normalized orthogonal first-head query frame, share the first two final-LayerNorm gains and biases, and evaluate only after the final training step.
mechanism: Normalized first-head query frame with paired terminal affine sharing and final-only validation
evidence_used: Reference Design 3 achieved 99.8% at 1484 parameters. Both corresponding 1483-parameter bias-sharing trials timed out without negative accuracy evidence, so reducing validation overhead directly addresses the unresolved failure mode.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6938, "parameters": 1483, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing coordinates two and three while retaining the verified leading-pair sharing will reduce the model from 1484 to 1483 parameters and preserve at least 99% accuracy.
change: Store five learned final-LayerNorm gains and independently reuse the first two scalars for coordinate pairs zero/one and two/three.
mechanism: Disjoint adjacent-pair final-LayerNorm gain sharing
evidence_used: Leading-pair gain sharing achieved 99.8% at 1484 parameters, while extending that same group to three coordinates collapsed accuracy; a separate adjacent pair tests a less restrictive, previously untested reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reproducing the qualified normalized first-head frame while sharing final-LayerNorm gains for coordinate pairs 0/1 and 2/3 will yield 1483 parameters and at least 99% accuracy; final-only validation will allow the previously timed-out experiment to complete.
change: Normalize the first head’s orthogonal neighboring query and transfer its scale into the paired key row, share two disjoint final-LayerNorm gain pairs, and evaluate only at step 4999.
mechanism: Disjoint adjacent-pair final-gain sharing with final-only validation
evidence_used: The normalized-frame design with leading-pair gain sharing achieved 99.8% at 1484 parameters. The only trial adding disjoint 2/3 sharing timed out without accuracy evidence, while final-only validation previously converted the same timeout pattern into a completed run.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing the first two final-LayerNorm gains will reduce the verified 1489-parameter model to 1488 parameters while retaining at least 99% accuracy.
change: Store six learned final-LayerNorm gains and reuse the first scalar for coordinates zero and one, while retaining the fixed final gain.
mechanism: Leading-pair terminal LayerNorm gain sharing
evidence_used: The current design achieved 99.99% accuracy at 1489 parameters, and the identical leading-pair gain sharing previously achieved 99.8% in Reference Design 3; this tests that qualified reduction on the higher-margin current attention geometry.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Normalizing the second head’s biased anchor query across its relative weights and bias, while transferring the omitted scale into the matching key row, will reduce the qualified 1484-parameter design to 1483 parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 3’s normalized first-head frame and leading final-gain sharing, then remove one exact scale redundancy from the second-head anchor query’s complete affine row.
mechanism: Homogeneous affine query/key scale gauge
evidence_used: Reference Design 3 reached 99.8% at 1484 parameters; prior 1483 failures imposed additional terminal-affine sharing constraints, whereas this change uses an expressivity-preserving query/key gauge analogous to the successful normalized attention rows.
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
blocks have been applied. All blocks must apply. They may edit either or both
editable files, but together they must describe one implementation ready for
verification. The mechanism name is descriptive, not chosen from a fixed list.
Do not paste whole files, lengthy logs, or routine progress reports outside the
patch.
