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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1511, "training_steps": 4999}
prior_hypothesis: Fixing `q_bias[0]` alongside the qualified fixed `q_bias[4]` will reduce the model to 1511 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9992, "parameters": 1509, "training_steps": 4999}
prior_hypothesis: Fixing `q_bias[2]` in the qualified 1510-parameter design will yield 1509 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998, "parameters": 1508, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified 1509-parameter design and replacing its three second-head value gauges with four triangular, initialization-preserving rotations will yield 1508 parameters with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1507, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified 1508-parameter design and triangularizing one additional second-head value coordinate will yield 1507 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Reproducing the qualified 1512-parameter design and fixing the eighth and final `ln1` scale will yield a 1511-parameter model with at least 99% accuracy.
change: Apply the qualified complete embedding, seventh value-output, and query-bias gauges, then make all eight pre-attention LayerNorm scales fixed units.
mechanism: Complete LayerNorm scale gauge
evidence_used: The seven-fixed-scale design achieved 99.96% accuracy at 1512 parameters, after the five- and six-fixed-scale designs achieved 99.84% and 99.91%; fixing the remaining scale is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7413, "parameters": 1511, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[0]` alongside the qualified fixed `q_bias[4]` will reduce the model to 1511 parameters while retaining at least 99% accuracy.
change: Remove one learned query-bias scalar and reconstruct the first coordinate of each attention head as zero.
mechanism: Paired per-head query-bias gauge fixing
evidence_used: Fixing `q_bias[4]` qualified at 99.88% and remained successful in the 1512-parameter model at 99.96%; the failed 1511 attempt instead removed the final learned `ln1` scale and fell to 74.13%, motivating a distinct query-key gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1511, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[0]` as in the qualified 1511-parameter design while sharing its remaining learned `ln1` scale with an existing query-bias scalar will yield a 1510-parameter model with at least 99% accuracy.
change: Reconstruct the first coordinate of both attention-head query biases as zero, fix seven `ln1` scales to one, and derive the remaining scale from an already-counted query-bias parameter.
mechanism: Cross-module query-bias/LayerNorm scale sharing
evidence_used: The distinct query-bias reduction reached 99.96% at 1511 parameters, whereas fixing the final `ln1` scale outright fell to 74.13%; parameter sharing retains an adaptive scale while testing one further deduplicated reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7973, "parameters": 1510, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1511-parameter design and fixing `q_bias[1]` will yield a 1510-parameter model with at least 99% accuracy.
change: Fix seven `ln1` scales while retaining its sole learned scale, and reconstruct query-bias coordinates 0, 1, and 4 as zero.
mechanism: Third zero-initialized query-bias anchor with the qualified adaptive LayerNorm scale
evidence_used: The independent `ln1` scale plus `q_bias[0]`/`q_bias[4]` design achieved 99.96% at 1511 parameters; the failed 1510 design shared that essential scale, so this tests a distinct query-bias reduction while leaving the scale independent.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1510, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1510-parameter design and fixing `q_bias[5]`, the second-head counterpart of the successful `q_bias[1]` anchor, will yield 1509 parameters with at least 99% accuracy.
change: Complete the embedding and seventh value-output gauges, retain the sole learned `ln1` scale, and reconstruct query-bias coordinates 0, 1, 4, and 5 as zero.
mechanism: Symmetric second-head query-bias anchoring
evidence_used: The design fixing query-bias coordinates 0, 1, and 4 achieved 99.89% at 1510 parameters, while sharing or eliminating the remaining LayerNorm scale failed; symmetrically anchoring coordinate 5 tests a distinct zero-initialized query-bias reduction without disturbing that essential scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9406, "parameters": 1509, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `q_bias[2]` in the qualified 1510-parameter design will yield 1509 parameters while retaining at least 99% accuracy.
change: Reconstruct query-bias coordinates 0, 1, 2, and 4 as zero while retaining the sole learned `ln1` scale and all remaining query biases.
mechanism: Asymmetric first-head query-bias anchoring
evidence_used: Fixing first-head `q_bias[1]` achieved 99.89% at 1510 parameters, whereas fixing its second-head counterpart `q_bias[5]` fell to 94.06%; this motivates testing another first-head coordinate rather than extending the harmful second-head constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1509, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the remaining first-head query-bias coordinate `q_bias[3]` in the qualified 1509-parameter design will yield 1508 parameters while retaining at least 99% accuracy.
change: Reconstruct all four first-head query biases and the first second-head query bias as zero, while retaining the three remaining second-head biases and the independent learned `ln1` scale.
mechanism: Complete first-head query-bias anchoring
evidence_used: Fixing first-head coordinates 1 and 2 successively qualified at 99.89% and 99.92%, whereas fixing second-head coordinate 5 fell to 94.06%; this supports extending the successful asymmetric first-head anchoring direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7329000000000001, "parameters": 1508, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1509-parameter design and replacing its three second-head value gauges with four triangular, initialization-preserving rotations will yield 1508 parameters with at least 99% accuracy.
change: Retain the qualified adaptive LayerNorm scale and query-bias coordinates, while imposing one additional second-head value-weight zero and compensating it in the attention output projection.
mechanism: Eighth value-output gauge via triangular second-head rotations
evidence_used: The 1509-parameter query-bias design achieved 99.92%, whereas fixing `q_bias[3]` fell to 73.29% and fixing `q_bias[5]` fell to 94.06%; Reference Design 3’s additional second-head value rotation retained 99.89%, motivating a distinct value-output gauge instead of another harmful bias constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1508, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1508-parameter design and triangularizing one additional second-head value coordinate will yield 1507 parameters while retaining at least 99% accuracy.
change: Complete the token-position embedding gauge, retain one learned pre-attention LayerNorm scale, fix query-bias coordinates 0, 1, 2, and 4, and replace four second-head value rotations with five initialization-preserving triangular rotations compensated in the output projection.
mechanism: Fifth triangular second-head value/output rotation gauge
evidence_used: The eight-value-gauge design achieved 99.98% at 1508 parameters, while further query-bias fixing failed; extending its successful second-head value/output gauge is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1507, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing the sixth and final triangular rotation of the second attention head will reduce the qualified 1507-parameter model to 1506 parameters while retaining at least 99% accuracy.
change: Add a `(second_query, 2)` value rotation, remove the corresponding learned value-weight coordinate, and compensate the initialization through the matching output-projection rotation.
mechanism: Complete triangular second-head value/output rotation gauge
evidence_used: The preceding fifth triangular second-head value rotation reduced the model from 1508 to 1507 parameters while achieving 99.91% accuracy; completing the same initialization-preserving triangularization is the closest supported reduction.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Extending the qualified 1507-parameter model with the sixth and final second-head value rotation will yield 1506 parameters while retaining at least 99% accuracy.
change: Replace the current four-rotation second-head parameterization with all six initialization-preserving triangular rotations, fixing two additional value-weight coordinates and compensating them in the attention output projection.
mechanism: Complete triangular second-head value/output gauge
evidence_used: The nine-value-gauge reference achieved 99.91% accuracy at 1507 parameters; the prior attempt to test the final rotation was malformed because its search matched multiple locations, so the proposed 1506-parameter model remains untested.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7365, "parameters": 1506, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1507-parameter nine-value-gauge model with the missing second-head query-key rotation will yield 1506 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified nine-value-gauge parameterization, then fix second-head query-weight coordinate `(0, 2)` through an initialization-preserving query-key rotation.
mechanism: Complete second-head query-key triangular gauge
evidence_used: The nine-value-gauge design achieved 99.91% accuracy at 1507 parameters; the failed 1506 design removed a final value coordinate, so testing the distinct unfinished query-key triangularization is the closest supported alternative.
result: the implementation could not be verified



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
