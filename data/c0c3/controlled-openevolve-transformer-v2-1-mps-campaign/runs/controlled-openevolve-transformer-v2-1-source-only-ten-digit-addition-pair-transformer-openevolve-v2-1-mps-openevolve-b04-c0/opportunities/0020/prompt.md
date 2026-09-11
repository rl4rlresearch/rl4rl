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
verified_results: {"accuracy": 0.9995, "parameters": 1625, "training_steps": 4999}
prior_hypothesis: Expanding to three shared embedding coordinates while detaching only the third coordinate’s positional gradient will reduce the model from 1626 to 1625 parameters and achieve at least 99% accuracy, because it preserves the exact token/position gauge constraint but avoids the conflicting joint optimization pathway implicated by the coupled third anchor.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing `ln2.bias` will reduce parameters from 1628 to 1620 while retaining at least 99% accuracy, because its contribution is exactly absorbable by the independent `fc1.bias` before GELU.
change: Disable only the second LayerNorm’s bias while preserving its learned scale and all MLP parameters.
mechanism: Pre-MLP bias reparameterization
evidence_used: Tying `ln2.bias` to the post-nonlinearity `fc2.bias` failed at 11.24%; this tests the mathematically valid redundancy with the pre-nonlinearity `fc1.bias` without narrowing the MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1487, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate at zero will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because the affine first LayerNorm and learned attention projections can compensate for this single-coordinate constraint.
change: Learn seven query-bias coordinates and append one fixed zero coordinate during attention.
mechanism: Single-coordinate query-bias gauge fixing
evidence_used: The 1628-parameter bias-shared model reached 99.76%, while eight-parameter bias removals and a 17-parameter feed-forward reduction failed; this motivates the smallest possible one-parameter capacity ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2688, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the unobservable all-ones component from every positional embedding will reduce parameters by `max_seq_len` while retaining at least 99% accuracy, because per-position scalar shifts are erased by every pre-LayerNorm and the final LayerNorm, and the custom optimizer preserves the original eight-coordinate AdamW updates in the seven-dimensional quotient space.
change: Represent each 8-dimensional positional embedding with seven orthonormal zero-mean coordinates, preserve baseline-equivalent initialization, and optimize those coordinates by projecting virtual full-coordinate AdamW updates.
mechanism: Quotient-space positional embeddings with full-coordinate AdamW dynamics
evidence_used: The 1628-parameter model reached 99.76%, while deleting or tying functionally absorbable bias pathways caused severe optimization failures; this patch instead removes a strictly unobservable positional direction and explicitly retains the original optimizer dynamics.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5085999999999999, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln2.bias` to the first eight coordinates of `fc1.bias` will reduce the model from 1628 to 1620 parameters while retaining at least 99% accuracy, because `fc1` can absorb the LayerNorm shift and the shared parameter preserves both optimization pathways.
change: Disable the standalone second LayerNorm bias and reuse the first `d_model` entries of the MLP input bias as its affine shift before `fc1`.
mechanism: Pre-nonlinearity LayerNorm/MLP bias sharing
evidence_used: Removing `ln2.bias` scored only 14.87%, while tying it to the post-nonlinearity `fc2.bias` scored 11.24%; sharing it with the mathematically corresponding pre-nonlinearity `fc1.bias` tests the same redundancy without deleting its gradient pathway or coupling it across GELU.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.15439999999999998, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one `ln2` bias coordinate with one `fc1` bias coordinate will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because both biases affect only the same pre-GELU activations, the remaining seven LayerNorm bias coordinates stay independent, and both gradient pathways remain active.
change: Replace `ln2` with a LayerNorm whose final bias coordinate reuses `mlp.fc1.bias[0]`, eliminating one learned scalar without narrowing the model.
mechanism: Single-coordinate pre-MLP bias sharing
evidence_used: Sharing all eight `ln2` coordinates with `fc1.bias` scored 15.44%, while deleting all eight scored 14.87%; the successful attention-bias tie shows pathway-preserving sharing can work, so a one-coordinate tie tests whether the earlier failure came from imposing eight coupled constraints at once.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2797, "parameters": 1627, "training_steps": 4999}

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
