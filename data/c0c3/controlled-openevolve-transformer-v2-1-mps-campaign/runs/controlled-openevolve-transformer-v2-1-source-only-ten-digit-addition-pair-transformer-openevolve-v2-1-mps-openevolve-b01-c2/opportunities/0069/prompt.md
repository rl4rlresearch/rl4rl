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
verified_results: {"accuracy": 0.9997, "parameters": 1573, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,574-parameter design and representing each attention projection bias in an orthonormal zero-sum basis will yield 1,573 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1574, "training_steps": 4999}
prior_hypothesis: Adding an exact scalar token-embedding translation gauge to the verified 1,575-parameter balanced query-bias design will produce a 1,574-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9991, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,579-parameter design by anchoring positional coordinate 3 will produce a 1,578-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9983, "parameters": 1575, "training_steps": 4999}
prior_hypothesis: Fixing local query-bias coordinate 2 in both attention heads will produce a 1,575-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing one redundant token-position shift coordinate will produce a 1,579-parameter model with at least 99% accuracy.
change: Anchor position 0’s first coordinate, shift its initialized value into that coordinate of every tied token embedding, and retain all other embedding coefficients.
mechanism: Scalar token-position embedding shift gauge
evidence_used: The current 1,580-parameter design achieved 99.83%; the eight-coordinate version collapsed to 70.62%, motivating the smallest one-coordinate ablation of that exact gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified one-coordinate positional gauge to a second coordinate will produce a 1,578-parameter model with at least 99% accuracy.
change: Reproduce the verified query, key, and two value-row gauges, then anchor position 0’s first two coordinates and transfer their initialized shifts into the tied token embeddings.
mechanism: Second scalar token-position embedding shift gauge
evidence_used: The one-coordinate positional gauge achieved 99.79% accuracy at 1,579 parameters, while gauging all eight coordinates collapsed; adding only the adjacent second coordinate is the smallest informative extension, and further QKV-row gauges have already failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7256, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,579-parameter design by anchoring positional coordinate 3 will produce a 1,578-parameter model with at least 99% accuracy.
change: Reproduce the seven verified QKV row gauges and anchor positional coordinates 0 and 3, transferring both initialized shifts into the tied token embeddings.
mechanism: Nonadjacent scalar token-position shift gauge
evidence_used: Coordinate 0 achieved 99.79% at 1,579 parameters, whereas adjacent coordinate 1 collapsed. Coordinate 3 is the first coordinate with an anchored `ln1` scale, making it the most informative nonadjacent test of whether the failure was coordinate-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,578-parameter design by anchoring positional coordinate 4 will produce a 1,577-parameter model with at least 99% accuracy.
change: Reproduce the seven verified QKV row gauges and anchor positional coordinates 0, 3, and 4, transferring their initialized shifts into the tied token embeddings.
mechanism: Third scalar token-position embedding shift gauge
evidence_used: Reference Design 2 achieved 99.91% accuracy at 1,578 parameters with coordinates 0 and 3 anchored; coordinate 4 is also backed by a fixed `ln1` scale, making it the closest informative extension while avoiding the failed unanchored coordinate-1 choice.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7281, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final block MLP output bias’s vocabulary-invariant common-coordinate direction will reduce the model to 1,577 parameters while maintaining at least 99% accuracy.
change: Replace the final block’s `fc2` with a linear layer whose last bias coordinate is fixed to zero, retaining ordinary linear layers in earlier blocks.
mechanism: Final-LayerNorm common-bias gauge
evidence_used: The current 1,578-parameter design achieved 99.91%, while a third positional anchor collapsed to 72.81%; this tests an orthogonal exact null direction because the final LayerNorm removes any common-coordinate shift added by the final MLP bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1232, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,578-parameter coordinates-0-and-3 design by anchoring positional coordinate 7 will produce a 1,577-parameter model with at least 99% accuracy.
change: Reproduce the successful positional anchors at coordinates 0 and 3, then also anchor coordinate 7 and transfer all three initialized shifts into the tied token embeddings.
mechanism: Third scalar token-position shift gauge on the normalized-input pivot coordinate
evidence_used: Coordinates 0 and 3 achieved 99.91% accuracy at 1,578 parameters, while coordinate 4 fell to 72.81%. Coordinate 7 is the strongest untested alternative because it is the fixed-scale input coordinate omitted by every existing normalized-input QKV and MLP row gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.997, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one first-head query-bias coordinate within the verified 1,577-parameter design will produce a 1,576-parameter model with at least 99% accuracy.
change: Reproduce the seven verified QKV row gauges and three positional anchors, then omit query-bias coordinate 3, which can be removed through an invertible within-head Q/K basis change.
mechanism: First-head Q/K basis query-bias gauge
evidence_used: Reference Design 1 achieved 99.7% accuracy at 1,577 parameters; successful first-head value gauges and sharp failures from added second-head constraints motivate a first-head basis gauge that preserves the initialized function because query biases initialize to zero.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9848, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting query-bias coordinate 2 instead of the failed coordinate 3 will produce a 1,576-parameter model with at least 99% accuracy.
change: Reproduce the verified seven QKV-row gauges and positional anchors at coordinates 0, 3, and 7, then fix first-head query-bias coordinate 2 to zero.
mechanism: Alternative first-head Q/K basis query-bias gauge
evidence_used: The 1,577-parameter reference achieved 99.7%, and removing query-bias coordinate 3 reached 98.48%—far closer to the threshold than other failed reductions. Coordinate 2 is the remaining untested, ungauged first-head query coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.991, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing local query-bias coordinate 2 in both attention heads will produce a 1,575-parameter model with at least 99% accuracy.
change: Retain the successful first-head query-bias coordinate-2 constraint and additionally fix global query-bias coordinate 6, the corresponding coordinate in the second head.
mechanism: Balanced per-head Q/K basis query-bias gauge
evidence_used: Fixing first-head coordinate 2 achieved 99.1% accuracy at 1,576 parameters, while balanced query-row gauges across both heads previously achieved 99.93%; applying the same successful basis coordinate symmetrically is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding an exact scalar token-embedding translation gauge to the verified 1,575-parameter balanced query-bias design will produce a 1,574-parameter model with at least 99% accuracy.
change: Reproduce the successful query-bias constraints at coordinates 2 and 6, then anchor the tied token embedding’s first scalar and reconstruct both input embeddings and output logits from the remaining coefficients while preserving initialization modulo LayerNorm and softmax invariances.
mechanism: Global tied-token embedding translation gauge
evidence_used: Fixing query-bias coordinate 2 in both heads achieved 99.83% at 1,575 parameters, whereas constraining query-bias coordinate 3 already fell to 98.48%; this motivates an orthogonal exact translation gauge instead of another query-bias restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing positional coordinate 4 with zero mean, on top of the verified 1,574-parameter design, will yield 1,573 parameters while maintaining at least 99% accuracy.
change: Reproduce the verified balanced query-bias, coordinate-7 positional, and tied-token translation gauges, then remove one additional positional coefficient by centering coordinate 4 across positions and transferring its mean into every token embedding.
mechanism: Mean-centered positional translation gauge
evidence_used: The 1,574-parameter reference reached 99.88%; although anchoring position zero’s coordinate 4 collapsed, the exact translation gauge remains available, and a symmetric zero-mean chart avoids privileging the position-zero coefficient implicated by that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6559999999999999, "parameters": 1573, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,574-parameter design and representing each attention projection bias in an orthonormal zero-sum basis will yield 1,573 parameters while maintaining at least 99% accuracy.
change: Add the verified QKV, balanced query-bias, positional, and tied-token gauges, then remove the attention projection bias’s downstream-LayerNorm-invariant common direction without privileging a hidden coordinate.
mechanism: Orthonormal attention-projection common-bias gauge
evidence_used: The 1,574-parameter reference achieved 99.88% accuracy, while an additional positional gauge failed. The final-MLP anchored-bias experiment motivates testing the distinct attention-projection null direction with an orthonormal centered chart instead of another coordinate anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1573, "training_steps": 4999}



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
