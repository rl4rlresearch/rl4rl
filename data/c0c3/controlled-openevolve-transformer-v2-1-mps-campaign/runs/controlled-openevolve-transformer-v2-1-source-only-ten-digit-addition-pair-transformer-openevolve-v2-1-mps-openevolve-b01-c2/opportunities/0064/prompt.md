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
verified_results: {"accuracy": 0.9994, "parameters": 1586, "training_steps": 4999}
prior_hypothesis: Applying one QKV row gauge to the verified 1,587-parameter design will produce a 1,586-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.997, "parameters": 1577, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,578-parameter coordinates-0-and-3 design by anchoring positional coordinate 7 will produce a 1,577-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9991, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,579-parameter design by anchoring positional coordinate 3 will produce a 1,578-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Adding a gauge for the first key row to the verified four balanced query-row gauges will reduce the model from 1,583 to 1,582 parameters while maintaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified 1,582-parameter design with the first value row will produce a 1,581-parameter model with at least 99% accuracy.
change: Reproduce the five-anchor design with query-row gauges 0, 1, 4, and 5 and key-row gauge 8, then omit the final normalized-input coefficient from value row 16.
mechanism: Value/output-basis normalized-input gauge
evidence_used: The five-gauge reference achieved 99.93% at 1,582 parameters, while additional query and key gauges failed; a value-row gauge is the smallest untested orthogonal extension and can leverage the learned value/output-projection basis freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding value row 17 to the verified gauges will reduce the model from 1,581 to 1,580 parameters while maintaining at least 99% accuracy.
change: Omit the final normalized-input coefficient from the second value coordinate of the first attention head, alongside the verified query, key, and value gauges.
mechanism: Within-head second value-row normalized-input gauge
evidence_used: Gauging value row 16 achieved 99.94% accuracy at 1,581 parameters. Additional gauges in the second head failed sharply, while an adjacent first-head key gauge reached 98.19%, motivating the smallest untested within-first-head value extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,580-parameter design with value row 18 will produce a 1,579-parameter model with at least 99% accuracy.
change: Reproduce the verified query rows 0, 1, 4, and 5, key row 8, and value rows 16 and 17, then omit the final normalized-input coefficient from adjacent value row 18.
mechanism: Within-head third value-row normalized-input gauge
evidence_used: Value rows 16 and 17 successively achieved 99.94% at 1,581 parameters and 99.83% at 1,580 parameters; extending the same successful first-head value-row sequence is the smallest informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.22579999999999997, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding value row 20 to the verified 1,580-parameter gauges will produce a 1,579-parameter model with at least 99% accuracy.
change: Reproduce the verified query rows 0, 1, 4, and 5, key row 8, and value rows 16 and 17, then omit the final normalized-input coefficient from value row 20, the first value coordinate of the second head.
mechanism: Balanced cross-head value-row normalized-input gauge
evidence_used: Value rows 16 and 17 reached 99.94% and 99.83%, while extending within the first head to row 18 collapsed to 22.58%; the successful cross-head balancing of query rows 4 and 5 motivates testing the corresponding value coordinate in the second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7209, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified 1,580-parameter QKV gauges with one additional anchored `ln1` scale will produce a 1,579-parameter model with at least 99% accuracy.
change: Reproduce query rows 0, 1, 4, and 5, key row 8, and value rows 16 and 17, then fix a sixth `ln1` scale to one; its effect remains learnable through the corresponding QKV input column.
mechanism: Sixth attention-input LayerNorm scale gauge
evidence_used: The seven-row QKV design achieved 99.83% accuracy at 1,580 parameters, while further value-row gauges collapsed; an orthogonal LayerNorm-scale reparameterization is therefore the most informative one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.39020000000000005, "parameters": 1579, "training_steps": 4999}

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
