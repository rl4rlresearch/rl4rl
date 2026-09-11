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
verified_results: {"accuracy": 0.9962000000000001, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Restoring the full value bias while quotienting two final-LayerNorm bias degrees of freedom into otherwise hidden token-row means will produce 1,584 parameters and retain at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the seven remaining value-bias coordinates will reduce parameters from 1,586 to 1,579 while retaining at least 99% accuracy, because normalized attention adds the same value-bias vector at every position and its mean-free projected effect can be represented exactly by the learned attention output bias.
change: Retain only the eight query-bias parameters in QKV and reconstruct both key and value biases as zeros, preserving the original constructor RNG consumption and all projection weights.
mechanism: Value-bias absorption into attention output bias
evidence_used: Exact removal of all key-bias coordinates achieved 99.94% at 1,586 parameters, while the mean-free attention projection retains seven learned output-bias directions; this supports eliminating the remaining algebraically redundant attention bias rather than pruning load-bearing LayerNorm capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.33549999999999996, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Pruning one additional value-bias coordinate will reduce parameters from 1,586 to 1,585 while retaining at least 99% accuracy, because normalized attention contributes that coordinate as a position-independent offset whose projected effect can be learned by the attention output bias.
change: Retain six learned value-bias coordinates instead of seven, reconstructing the final two as zeros while preserving all projection weights, query biases, constructor RNG consumption, and training settings.
mechanism: Incremental value-bias absorption into attention output bias
evidence_used: Removing all seven remaining value-bias coordinates failed at 33.55%, despite their algebraic redundancy, indicating optimization sensitivity; testing a single additional coordinate is the smallest informative extension of the current 99.94%-accurate design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9731000000000001, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Restoring the successful full value bias while removing one exact global-shift degree of freedom from the tied token/output embedding will yield 1,585 parameters and retain at least 99% accuracy.
change: Restore all eight value-bias coordinates, then represent the tied embedding in an orthonormal 111-dimensional mean-free basis while preserving constructor and initialization RNG consumption.
mechanism: Global tied-embedding shift quotient
evidence_used: The 1,586-parameter key-bias-free design achieved 99.94%, whereas pruning one value-bias coordinate fell to 97.31%. Mean-free quotient parameterizations already preserved accuracy for positional embeddings and residual outputs, motivating an exact tied-embedding gauge reduction instead.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the tied token/output embedding with exactly one fewer learned coordinate will reduce the model from 1,586 to 1,585 parameters while retaining at least 99% accuracy, because the previous global-shift quotient achieved 99.99% accuracy and the removed direction is invisible to both LayerNorm and output softmax.
change: Replace the tied full embedding parameter with an orthonormal mean-free flattened parameterization shared by input lookup and output projection, while preserving the original constructor and initialization RNG draws.
mechanism: Correctly deduplicated global tied-embedding shift quotient
evidence_used: The prior tied-embedding quotient reached 99.99% accuracy but remained at 1,586 parameters, proving the constraint is accurate while showing that its previous implementation failed to remove an actual deduplicated parameter; the current key-bias-free design provides the 1,586-parameter baseline.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the tied embedding across vocabulary independently in all eight feature coordinates will reduce the model from 1,585 to 1,578 parameters while retaining at least 99% accuracy, because the removed common token vector can be transferred to every positional embedding and changes output logits only by a softmax-null common offset.
change: Replace the one-dimensional flattened embedding quotient with an eight-dimensional vocabulary-centering quotient, and transfer the removed mean-free initialization component into the positional embeddings.
mechanism: Joint token–position embedding gauge quotient
evidence_used: The existing global tied-embedding quotient achieved 99.97% accuracy at 1,585 parameters, and mean-free positional embeddings have also retained high accuracy; together these support extending the proven quotient to the exact joint token–position gauge rather than pruning optimization-sensitive attention or LayerNorm capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.24710000000000001, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one additional common-token mean-free feature direction and transferring it to every positional embedding will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy.
change: Extend the tied embedding quotient from one to two dimensions, omitting one common-token feature contrast and transferring its initialized value into the positional embeddings without changing the model’s initial function.
mechanism: Incremental joint token–position embedding gauge quotient
evidence_used: The one-dimensional tied-embedding quotient achieved 99.97% at 1,585 parameters, while removing all seven remaining joint-gauge directions at once failed at 24.71%; removing exactly one is the smallest test of whether that failure was caused by the wholesale reparameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0559, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering one shared positional-feature direction and transferring its initialization into the tied token embedding will produce 1,584 parameters while retaining at least 99% accuracy, because it removes an exact token–position gauge without constraining the token embedding’s learned common-feature capacity.
change: Restore the successful full value bias, then remove one scalar from the positional-coordinate table with an orthonormal quotient and transfer the omitted common positional vector into every token embedding at initialization.
mechanism: Position-centered joint token–position embedding gauge
evidence_used: The full-value-bias, tied-embedding quotient reached 99.97% at 1,585 parameters, while pruning one value-bias coordinate reached only 97.31%. Removing additional common-token directions also failed, so this patch keeps those sensitive parameters and places the next exact gauge constraint on the already-successful mean-free positional representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7426, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one learned value channel across the two attention heads, while retaining independent queries, keys, and three private value channels per head, will reduce the verified 1,585-parameter model to 1,577 parameters while maintaining at least 99% accuracy.
change: Replace one pair of independent value-projection rows with a single deduplicated learned row. Preserve full-width initialization draws and reconstruct the shared row inside causal attention.
mechanism: Hybrid shared/private value representation
evidence_used: Fully sharing both key and value heads failed at 36.38%, but that experiment simultaneously removed independent addressing and all private value features. The 99.97%-accurate baseline leaves untested the load-bearing assumption that every value channel must be independent; this patch isolates that assumption while preserving all key capacity and six of eight head-specific value rows.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4065, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at its initial value will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy, because the following `fc1` weight column can represent the removed multiplicative degree of freedom.
change: Add a LayerNorm with two pruned bias coordinates and one fixed scale coordinate, and use it only before the MLP.
mechanism: Downstream-absorbed LayerNorm scale fixing
evidence_used: The current 1,585-parameter model achieved 99.97%, while further additive-bias and embedding-gauge pruning was optimization-sensitive. This edit preserves the initial function exactly and removes a local multiplicative redundancy that `fc1` can absorb.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9798, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Restoring the full value bias while quotienting two final-LayerNorm bias degrees of freedom into otherwise hidden token-row means will produce 1,584 parameters and retain at least 99% accuracy.
change: Restore all eight value-bias coordinates, initialize each token embedding row mean-free, and replace the final LayerNorm bias with six learned mean-free coordinates plus a fixed common component that turns token-row means into a direct output-bias channel.
mechanism: Final-LayerNorm bias absorption through token-row means
evidence_used: The full-value-bias tied-embedding design achieved 99.97% at 1,585 parameters, whereas pruning one value-bias coordinate fell to 97.31%; this keeps the load-bearing value bias and instead removes an exact output-side redundancy while preserving the initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9962000000000001, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one attention output-bias coordinate to a learned value-bias coordinate will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because value bias and projection bias contribute only through their combined position-independent projected offset.
change: Keep all eight learned value-bias coordinates, but reuse the final value-bias scalar as the final mean-free attention projection-bias coordinate instead of learning that coordinate independently.
mechanism: Redundant value/output-bias coordinate tying
evidence_used: Removing even one value-bias coordinate reduced accuracy to 97.31%, while the full-value-bias design reached 99.62% at 1,584 parameters; tying a redundant projection-bias coordinate preserves every value channel and its direct learned gradient path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6277, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one common-direction coordinate from a key-projection row will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because LayerNorm initially produces mean-free features and the removed component changes every key by the same position-independent scalar, which causal-attention softmax cancels.
change: Represent one key-projection row in a seven-dimensional orthonormal mean-free basis, reconstruct the full QKV matrix during attention, and preserve the original full-width initialization draw.
mechanism: LayerNorm-induced key-weight gauge quotient
evidence_used: Eliminating all eight softmax-null key-bias coordinates retained 99.94% accuracy, while tying value and output biases failed at 62.77%; this motivates extending the proven key-side softmax invariance instead of coupling optimization-sensitive value pathways.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4104, "parameters": 1583, "training_steps": 4999}



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
