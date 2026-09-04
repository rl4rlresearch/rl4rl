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
verified_results: {"accuracy": 0.9997, "parameters": 1620, "training_steps": 4999}
prior_hypothesis: Adding a detached-position eighth anchor between `token[5, 7]` and `position[0, 7]` will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because the similarly distributed fourth through seventh anchors all preserved accuracy above 99%.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing only the all-ones component of the final positional embedding will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because this component is exactly erased by the pre-LayerNorm/final-LayerNorm architecture and affects only the causally last position’s optimizer coordinates.
change: Store the final positional embedding in a seven-dimensional orthonormal zero-mean basis, preserve full-shaped random-number consumption and baseline-equivalent initialization, and leave all other positional embeddings unchanged.
mechanism: Single-position LayerNorm gauge fixing
evidence_used: The 1628-parameter model achieved 99.76%; quotienting all 23 positional embeddings reached only 50.86%, so isolating one exact gauge direction at the causally least influential position tests whether that failure resulted from perturbing every positional optimizer pathway simultaneously.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9163, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one token-embedding coordinate with the corresponding positional-embedding coordinate will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because a common shift between all token embeddings and all positional embeddings leaves transformer inputs unchanged and changes every output logit equally.
change: Replace the two embedding tables with jointly initialized tables whose first scalar is shared, using a gauge-equivalent transformation of the original initialization and the shared token table for output logits.
mechanism: Shared token/position embedding gauge anchor
evidence_used: Sharing redundant attention bias pathways retained 99.76% at 1628 parameters, while fixing a positional gauge coordinate reached only 91.63%; this tests an exact embedding gauge through pathway-preserving sharing rather than deleting a coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing a second token-embedding coordinate with its positional-embedding counterpart will reduce the model from 1627 to 1626 parameters while retaining at least 99% accuracy, because the corresponding opposing global shifts leave transformer inputs unchanged and add only a common offset to output logits.
change: Expand the shared embedding anchor from one to two coordinates and apply the initialization-preserving gauge transformation independently to both coordinates.
mechanism: Second token/position embedding gauge anchor
evidence_used: The first shared embedding gauge anchor achieved 99.86% accuracy at 1627 parameters; extending the same exact symmetry by one coordinate is the smallest controlled reduction supported by successful evidence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing a third token-embedding coordinate with its positional-embedding counterpart will reduce the model from 1626 to 1625 parameters while retaining at least 99% accuracy, because the corresponding opposing global shifts preserve transformer inputs and add only a common offset to output logits.
change: Expand the shared embedding anchor from two coordinates to three and apply the initialization-preserving gauge transformation to all three coordinates.
mechanism: Third token/position embedding gauge anchor
evidence_used: Sharing the first anchor achieved 99.86% accuracy at 1627 parameters, and extending it to two anchors achieved 100% at 1626 parameters; a third anchor is the smallest controlled continuation of the same successful exact symmetry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9723999999999999, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing a third token coordinate with the causally final positional embedding will reduce the model from 1626 to 1625 parameters while retaining at least 99% accuracy, because it preserves the exact token/position shift symmetry while coupling the new shared parameter to fewer downstream attention computations than position zero.
change: Retain the two successful position-zero anchors and add a third anchor between `token[0, 2]` and the final position’s coordinate 2, with a gauge-equivalent initialization.
mechanism: Causally re-anchored embedding gauge sharing
evidence_used: Two position-zero anchors achieved 100% at 1626 parameters, while a third reached 97.24%; relocating only the failing third constraint tests whether its optimization coupling to the causally most influential position caused the shortfall.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Expanding to three shared embedding coordinates while detaching only the third coordinate’s positional gradient will reduce the model from 1626 to 1625 parameters and achieve at least 99% accuracy, because it preserves the exact token/position gauge constraint but avoids the conflicting joint optimization pathway implicated by the coupled third anchor.
change: Retain the two successful bidirectionally trained anchors, add a third shared coordinate initialized by the same gauge-preserving transformation, and train that new scalar only through the token/input-output embedding pathway.
mechanism: One-sided third embedding gauge anchor
evidence_used: Three fully coupled anchors reached 97.24% at 1625 parameters, close to the threshold, while moving the third coupling to the final position collapsed accuracy to 0.01%; this motivates preserving the exact third gauge reduction while isolating its positional-gradient coupling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the successful one-sided gauge sharing to a fourth embedding coordinate will reduce the model from 1625 to 1624 parameters while maintaining at least 99% accuracy, because the third detached positional pathway achieved 99.95% and the fourth applies the same exact symmetry without adding another coupled positional gradient.
change: Share four token/position embedding coordinates, keeping the first two bidirectionally trained and detaching positional gradients for the third and fourth coordinates.
mechanism: Fourth one-sided embedding gauge anchor
evidence_used: Three fully coupled anchors reached only 97.24%, whereas detaching the third anchor’s positional gradient reached 99.95% at 1625 parameters; this directly motivates extending the successful one-sided mechanism by one coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7422, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the fourth gauge coordinate to `token[1, 3]` instead of concentrating four constraints on token zero will reduce the model to 1624 parameters while retaining at least 99% accuracy, because it preserves the successful position-zero, token-only-gradient mechanism while distributing embedding optimization across token rows.
change: Keep the three successful anchors unchanged and add a detached-position fourth anchor between `token[1, 3]` and `position[0, 3]`, with a gauge-equivalent initialization.
mechanism: Distributed token-endpoint gauge sharing
evidence_used: Three anchors with the third positional gradient detached achieved 99.95% at 1625 parameters, while placing a fourth such anchor on token zero fell to 74.22%; the severe failure after concentrating another constraint on the same token row motivates changing only the fourth token endpoint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a detached-position fifth anchor between `token[2, 4]` and `position[0, 4]` will reduce the model from 1624 to 1623 parameters while retaining at least 99% accuracy, because distributing the fourth anchor to a separate token row restored accuracy to 99.98%, and this applies the same exact gauge symmetry without concentrating another constraint on an existing endpoint.
change: Expand the shared embedding representation to five anchors, placing the new coordinate on token row 2 and detaching its positional gradient while preserving gauge-equivalent initialization.
mechanism: Distributed fifth one-sided embedding gauge anchor
evidence_used: Four anchors concentrated on token row zero achieved only 74.22%, whereas moving the fourth endpoint to token row one achieved 99.98% at 1624 parameters; distributing the next anchor to token row two is the smallest evidence-supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a detached-position sixth anchor between `token[3, 5]` and `position[0, 5]` will reduce the model from 1623 to 1622 parameters while retaining at least 99% accuracy, because the independently distributed fourth and fifth anchors achieved 99.98% and 99.94%, respectively, without concentrating another constraint on an existing token row.
change: Expand the shared embedding representation to six anchors, place the new coordinate on token row 3, detach its positional gradient, and preserve the gauge-equivalent initialization.
mechanism: Distributed sixth one-sided embedding gauge anchor
evidence_used: Distributing the fourth detached anchor to token row 1 recovered accuracy from 74.22% to 99.98%, and distributing the fifth to token row 2 retained 99.94%; this directly supports testing the next unused token row and coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1622, "training_steps": 4999}

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
