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
verified_results: {"accuracy": 0.9991, "parameters": 1513, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified query-bias and five-scale gauges, then fixing a sixth `ln1` scale, will yield a 1513-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9992, "parameters": 1509, "training_steps": 4999}
prior_hypothesis: Fixing `q_bias[2]` in the qualified 1510-parameter design will yield 1509 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1511, "training_steps": 4999}
prior_hypothesis: Fixing `q_bias[0]` alongside the qualified fixed `q_bias[4]` will reduce the model to 1511 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9989, "parameters": 1521, "training_steps": 4999}
prior_hypothesis: Rotating second-head value channels 2–3 on input column 1 will produce a 1521-parameter model with at least 99% accuracy while preserving the initialized function and all five qualified value anchors.

## Recent verification evidence

RECENT RESULT
hypothesis: Completing the proven embedding gauge and fixing one query-bias coordinate through the remaining second-head query-key rotation will produce a 1515-parameter model with at least 99% accuracy.
change: Remove the anchor token’s remaining four coordinates, compensate all seven positional coordinates, and omit `q_bias[4]`, reconstructed as zero within the residual rows-4/5 query-key gauge.
mechanism: Residual second-head query-bias gauge fixing
evidence_used: The complete embedding gauge achieved 99.79% at 1516 parameters. Unlike the failed twelfth weight-based query-key constraint, the proposed residual gauge fixes an initially zero bias coordinate while preserving the initialized function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1515, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1515-parameter design by fixing a fifth pre-attention LayerNorm scale will produce a 1514-parameter model with at least 99% accuracy while preserving the initialized function.
change: Reproduce the complete token-position, seventh value-output, and query-bias gauges from the 1515-parameter reference, then reduce the learned `ln1` scales from four to three.
mechanism: Fifth pre-attention LayerNorm scale gauge
evidence_used: The complete embedding plus residual query-bias design achieved 99.88% at 1515 parameters. Extending the existing four-scale LayerNorm gauge tests a distinct exact redundancy instead of the failed additional query-weight and value-chain constraints.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Fixing a fifth `ln1` scale will reduce the qualified model to 1514 parameters while retaining at least 99% accuracy.
change: Replace the four-parameter `ln1` scale vector with three learned scales and five fixed unit scales.
mechanism: Fifth pre-attention LayerNorm scale gauge
evidence_used: The current four-scale-gauge design achieved 99.88% accuracy at 1515 parameters; the previous attempt at this exact reduction was malformed before verification, so a uniquely matching patch is the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1514, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified query-bias and five-scale gauges, then fixing a sixth `ln1` scale, will yield a 1513-parameter model with at least 99% accuracy.
change: Omit `q_bias[4]` as in the qualified 1514-parameter design and replace the four learned `ln1` scales with two learned scales and six fixed unit scales.
mechanism: Sixth pre-attention LayerNorm scale gauge
evidence_used: The query-bias plus fifth-scale design achieved 99.84% accuracy at 1514 parameters; extending that successful LayerNorm gauge by one coordinate is the closest untested reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1513, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1513-parameter design and fixing a seventh `ln1` scale will yield a 1512-parameter model with at least 99% accuracy.
change: Complete the token-position embedding gauge, omit `q_bias[4]`, and replace four learned `ln1` scales with one learned scale and seven fixed unit scales.
mechanism: Seventh pre-attention LayerNorm scale gauge
evidence_used: The six-scale design achieved 99.91% accuracy at 1513 parameters, while the preceding five-scale design achieved 99.84%; extending the same successful LayerNorm gauge is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1512, "training_steps": 4999}

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
