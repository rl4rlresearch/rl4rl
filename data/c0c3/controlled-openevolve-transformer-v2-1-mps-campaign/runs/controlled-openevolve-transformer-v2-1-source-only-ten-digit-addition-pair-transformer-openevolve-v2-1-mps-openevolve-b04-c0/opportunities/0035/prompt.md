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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9995, "parameters": 1613, "training_steps": 4999}
prior_hypothesis: Fixing pre-MLP LayerNorm gain coordinate 5 to one will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` and this tests a distinct redundancy from the failed coordinate-5 bias constraint.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a detached-position seventh anchor between `token[4, 6]` and `position[0, 6]` will reduce the model from 1622 to 1621 parameters while retaining at least 99% accuracy, because the similarly distributed fourth, fifth, and sixth anchors each preserved accuracy above 99.9%.
change: Expand the shared embedding representation to seven anchors, place the new coordinate on token row 4, detach its positional gradient, and preserve gauge-equivalent initialization.
mechanism: Distributed seventh one-sided embedding gauge anchor
evidence_used: The sixth detached anchor on a previously unused token row achieved 99.97% at 1622 parameters, continuing the success of the distributed fourth and fifth anchors; this motivates applying the same mechanism to the next unused token row and coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9972, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a detached-position eighth anchor between `token[5, 7]` and `position[0, 7]` will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because the similarly distributed fourth through seventh anchors all preserved accuracy above 99%.
change: Expand the shared embedding representation to eight anchors, place the new coordinate on unused token row 5, detach its positional gradient, and preserve gauge-equivalent initialization.
mechanism: Distributed eighth one-sided embedding gauge anchor
evidence_used: The distributed seventh anchor achieved 99.72% accuracy at 1621 parameters, following 99.98%, 99.94%, and 99.97% results for the fourth through sixth distributed anchors; this supports applying the same mechanism to the final unused embedding coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the second pre-LayerNorm bias at zero will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because that coordinate’s effect is exactly absorbable by the existing `fc1` bias and the initialization remains unchanged.
change: Add a LayerNorm variant with one fixed-zero bias coordinate and use it only before the MLP.
mechanism: MLP preactivation bias gauge fixing
evidence_used: The current 1620-parameter design achieved 99.97% accuracy after successive exact gauge reductions; this tests another one-parameter functional redundancy while preserving the successful architecture and initialization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second coordinate of the pre-MLP LayerNorm bias at zero will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because both fixed shifts are exactly absorbable by the learned `fc1` bias.
change: Extend `MLPAnchoredLayerNorm` from one fixed-zero bias coordinate to two, preserving its zero initialization and all other architecture and training settings.
mechanism: Two-coordinate MLP preactivation bias gauge fixing
evidence_used: Fixing the first pre-MLP LayerNorm bias coordinate achieved 99.95% accuracy at 1619 parameters; applying one additional instance of the same exact redundancy is the smallest controlled next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third pre-MLP LayerNorm bias coordinate at zero will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because all three fixed shifts are exactly absorbable by the learned `fc1` bias.
change: Extend `MLPAnchoredLayerNorm` from two fixed-zero bias coordinates to three while preserving zero initialization and all other architecture and training settings.
mechanism: Three-coordinate MLP preactivation bias gauge fixing
evidence_used: Fixing the first and then second pre-MLP LayerNorm bias coordinates achieved 99.95% at 1619 parameters and 99.92% at 1618 parameters; extending the same exact redundancy by one coordinate is the smallest controlled next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4874, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final pre-MLP LayerNorm bias coordinate instead of coordinate 2 will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because the shift remains exactly absorbable by `fc1.bias` and avoids the coordinate whose removal caused the 48.74% collapse.
change: Keep bias coordinates 0 and 1 fixed, additionally fix coordinate 7, and retain learned biases for coordinates 2 through 6.
mechanism: Nonconsecutive pre-MLP LayerNorm gauge fixing
evidence_used: Fixing coordinates 0 and 1 achieved 99.92% at 1618 parameters, whereas also fixing coordinate 2 fell to 48.74%. Coordinate 2 is the first detached positional anchor concentrated on token row 0, while coordinate 7 uses the successfully distributed token-row-5 endpoint, motivating a coordinate-specific rather than cumulative test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm bias coordinate 3 in addition to coordinates 0, 1, and 7 will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because its shift is exactly absorbable by `fc1.bias` and coordinate 3 previously supported the strongest distributed-anchor result.
change: Remove LayerNorm bias coordinate 3 from the learned parameter vector and reconstruct it as zero during the forward pass.
mechanism: Distributed-coordinate MLP bias gauge fixing
evidence_used: Fixing coordinates 0, 1, and 7 achieved 99.90% at 1617 parameters, while the distributed embedding anchor associated with coordinate 3 achieved 99.98%; unlike the failing coordinate 2, coordinate 3 uses a separate token-row endpoint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm bias coordinate 5 in addition to coordinates 0, 1, 3, and 7 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its shift is absorbable by `fc1.bias` and its distributed embedding anchor previously achieved 99.97%.
change: Remove LayerNorm bias coordinate 5 from the learned parameter vector and reconstruct it as zero during the forward pass.
mechanism: Fifth pre-MLP bias gauge fixing
evidence_used: Fixing coordinates 0, 1, 3, and 7 achieved 99.93% at 1616 parameters. Among the remaining coordinates, coordinate 5 has the strongest associated distributed-anchor result—99.97%—while the known-sensitive coordinate 2 remains learned.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7323999999999999, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 7 to one will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its zero shift is already fixed and its gain is exactly absorbable by column 7 of `fc1.weight`.
change: Replace gain coordinate 7 with a fixed one while preserving the existing fixed bias coordinates and initialization.
mechanism: Pre-MLP scale-to-weight gauge fixing
evidence_used: The current four-bias-anchor design achieved 99.93% at 1616 parameters, while adding bias coordinate 5 collapsed to 73.24%; fixing a gain already paired with zero bias tests a distinct exact redundancy without adding another shift constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 3 to one will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because its scale is exactly absorbable by column 3 of `fc1.weight`, its bias is already fixed at zero, and coordinate 3 previously tolerated bias gauge fixing.
change: Remove gain coordinate 3 from the learned parameter vector and reconstruct gains 3 and 7 as fixed ones during the forward pass.
mechanism: Second pre-MLP scale-to-weight gauge fixing
evidence_used: Fixing gain coordinate 7 achieved 99.97% accuracy at 1615 parameters, establishing this scale-to-weight reduction; coordinate 3 is the strongest next candidate because fixing its bias retained 99.93% at 1616 parameters and its distributed embedding anchor previously achieved 99.98%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 1 to one will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its scale is absorbable by column 1 of `fc1.weight`, its bias is already fixed at zero, and the initialization is unchanged.
change: Remove gain coordinate 1 from the learned parameter vector and reconstruct gains 1, 3, and 7 as fixed ones during the forward pass.
mechanism: Third pre-MLP scale-to-weight gauge fixing
evidence_used: Fixing gain coordinates 7 and 3 successively retained 99.97% and 99.93% accuracy; coordinate 1 is the strongest controlled next candidate because its LayerNorm bias has also remained successfully fixed throughout every passing bias-anchor design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2721, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing pre-MLP LayerNorm gain coordinate 5 to one will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` and this tests a distinct redundancy from the failed coordinate-5 bias constraint.
change: Remove gain coordinate 5 from the learned vector and reconstruct gains 3, 5, and 7 as fixed ones during the forward pass.
mechanism: Distributed-coordinate pre-MLP scale gauge fixing
evidence_used: Gain coordinates 7 and 3 were fixed successfully at 99.97% and 99.93%, while coordinate 5’s distributed embedding anchor previously achieved 99.97%; its failed bias fixing result motivates testing whether sensitivity is specific to eliminating its shift rather than its scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1613, "training_steps": 4999}



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
