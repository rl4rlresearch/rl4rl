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
verified_results: {"accuracy": 0.9992, "parameters": 1605, "training_steps": 4999}
prior_hypothesis: Reparameterizing the eight-coordinate MLP output bias as seven orthonormal zero-mean coordinates will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the discarded all-ones component is erased by the final LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying the bias-free second head’s key and value projections will reduce the model from 1609 to 1577 parameters while retaining at least 99% accuracy, because the first head retains fully independent projections and all three load-bearing value-bias coordinates.
change: Challenge the assumption that every attention head needs separate key and value maps by giving the second head a shared learned associative representation, while preserving the fully general bias-carrying first head.
mechanism: Bias-localized key–value sharing
evidence_used: The current model achieved 100% after removing every query- and value-bias coordinate from the second head, whereas reducing the first head’s value bias from three coordinates to two collapsed accuracy to 55.14%; this identifies the first head as the safer place to preserve capacity and the bias-free second head as the informative target for sharing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5229, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning one of eight query-bias coordinates will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because every prior query-bias reduction through two learned coordinates retained at least 99.98% accuracy while preserving the necessary three-coordinate value bias.
change: Store one learned query-bias coordinate and pad the remaining seven coordinates with zeros during attention.
mechanism: Seventh-coordinate query-bias ablation
evidence_used: The two-coordinate query-bias design achieved 100% accuracy with 1609 parameters, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8417, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one attention output-bias coordinate to zero will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because any all-ones component of this residual-branch bias is erased by subsequent LayerNorms.
change: Replace the eight-coordinate attention projection bias with seven learned coordinates padded by one fixed zero, while leaving the verified two-coordinate query bias and three-coordinate value bias intact.
mechanism: Residual-stream output-bias gauge fixing
evidence_used: Reducing the query bias from two coordinates to one collapsed accuracy to 84.17%, so the next reduction should preserve that boundary; unlike the failed second-head key–value sharing, this edit removes only a residual-stream gauge degree of freedom without tying projection maps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one MLP output-bias coordinate to zero will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because the final LayerNorm erases the all-ones component of this residual-branch bias.
change: Replace the eight-coordinate second MLP layer bias with seven learned coordinates padded by one fixed zero.
mechanism: Final-residual MLP output-bias gauge fixing
evidence_used: The analogous attention output-bias gauge fixing reduced the model from 1609 to 1608 parameters while achieving 99.88% accuracy; the MLP output is likewise added to the residual stream immediately before a LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.35359999999999997, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one redundant input weight from a second-head key row will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because bias-free LayerNorm outputs occupy a seven-dimensional subspace and that key row can remain fully expressive using seven coordinates.
change: Replace the joint QKV projection with a full projection for all but one row and a seven-input projection for one second-head key row, while preserving all verified query, value, and output biases.
mechanism: LayerNorm-subspace key-weight elimination
evidence_used: The 1608-parameter model achieved 99.88%, while reducing a load-bearing query-bias coordinate or tying the second head’s key and value maps failed; this patch preserves those capacities and removes only a LayerNorm-subspace-redundant weight.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9653, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one first-LayerNorm scale coordinate to one will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because that scale can be absorbed exactly into the corresponding dense QKV projection column.
change: Replace the first LayerNorm with an equivalent implementation containing seven learned scales and one fixed unit scale, preserving all verified attention biases and dense projection weights.
mechanism: LayerNorm–projection scale gauge fixing
evidence_used: The 1608-parameter model achieved 99.88%, while removing a query-bias coordinate or sparsifying a key row failed; this targets a multiplicative LayerNorm/projection redundancy without removing those capacities or changing the initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one second-LayerNorm scale coordinate to one will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because that scale can be absorbed exactly into the corresponding `fc1` input column.
change: Reuse `ScaleFixedLayerNorm` for the second pre-normalization layer, replacing its eight learned scales with seven learned scales and one fixed unit scale.
mechanism: Second LayerNorm–MLP scale gauge fixing
evidence_used: The structurally equivalent first-LayerNorm gauge fixing achieved 99.92% accuracy with 1607 parameters; the second LayerNorm likewise feeds a dense learned projection, so the same redundancy applies without removing attention or bias capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.20739999999999997, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second first-LayerNorm scale coordinate to one will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because both fixed scales can be absorbed independently into the corresponding dense QKV projection columns.
change: Store six learned scales in `ScaleFixedLayerNorm` and append two fixed unit scales, preserving the initial function and all verified attention capacities.
mechanism: Second first-LayerNorm scale gauge fixing
evidence_used: Fixing one first-LayerNorm scale achieved 99.92% accuracy with 1607 parameters; extending that successful gauge fixing within the same LayerNorm is more directly supported than the failed second-LayerNorm ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8167, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: An orthonormal zero-mean parameterization of the positional embedding will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because adding the same scalar to every positional-embedding coordinate only adds an all-ones residual-stream shift erased by every LayerNorm.
change: Replace the positional embedding with a learned orthonormal coordinate parameterization spanning the zero-mean subspace, removing exactly one redundant scalar while preserving the quotient initialization distribution.
mechanism: Position-embedding global-shift gauge fixing
evidence_used: The first-LayerNorm gauge fixing reached 99.92% with 1607 parameters, while reductions that removed query capacity or projection weights failed; this targets another exact normalization symmetry without reducing attention or MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.49920000000000003, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the single global all-ones degree of freedom from the tied token embedding will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the shift is erased from inputs by LayerNorm and adds only a vocabulary-wide common logit at the output.
change: Represent the tied token embedding in an orthonormal zero-mean basis with one fewer learned coordinate, and use its materialized weight for both input lookup and output logits.
mechanism: Tied-embedding global-shift gauge fixing
evidence_used: The first-LayerNorm gauge fixing retained 99.92% accuracy at 1607 parameters, whereas cuts to query capacity and projection weights failed. Although the positional-shift gauge failed, this tests the distinct tied-embedding symmetry while preserving isotropic quotient initialization and all attention and MLP capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: A seven-channel SwiGLU MLP will retain at least 99% accuracy while reducing the model from 1606 to 1584 parameters, because multiplicative gating can represent conditional feature interactions more efficiently than twelve independent GELU channels.
change: Replace the 12-channel additive GELU MLP with two learned seven-channel projections whose gated product feeds the output projection, while preserving all verified attention, normalization, embedding, and MLP output-bias capacity.
mechanism: Narrow multiplicative feature mixer
evidence_used: Removing an MLP output-bias coordinate collapsed accuracy to 35.36%, showing that the MLP branch is load-bearing, while attention sharing and further query-bias reduction also failed. This motivates preserving those capacities and instead challenging the shared assumption that carry-relevant interactions require a wider additive GELU representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5357, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the eight-coordinate MLP output bias as seven orthonormal zero-mean coordinates will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the discarded all-ones component is erased by the final LayerNorm.
change: Replace the dense `fc2` bias with a seven-parameter Householder basis spanning the zero-mean subspace.
mechanism: Isotropic MLP output-bias gauge fixing
evidence_used: Anchoring one MLP bias coordinate failed at 35.36%, but the same isotropic quotient parameterization successfully removed the tied-embedding shift degree of freedom at 99.84%; this tests whether the prior failure came from the coordinate-asymmetric gauge rather than loss of functional capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1605, "training_steps": 4999}



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
