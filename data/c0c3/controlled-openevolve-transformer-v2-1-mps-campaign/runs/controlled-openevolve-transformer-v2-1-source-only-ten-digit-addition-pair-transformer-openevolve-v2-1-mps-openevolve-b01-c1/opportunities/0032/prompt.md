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
verified_results: {"accuracy": 0.9997, "parameters": 1585, "training_steps": 4999}
prior_hypothesis: Representing the tied token/output embedding with exactly one fewer learned coordinate will reduce the model from 1,586 to 1,585 parameters while retaining at least 99% accuracy, because the previous global-shift quotient achieved 99.99% accuracy and the removed direction is invisible to both LayerNorm and output softmax.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing two independent key/value heads with one shared learned key/value head will reduce parameters from 1,593 to 1,523 while retaining at least 99% accuracy, because the two distinct query heads can retrieve both operand digits from the same content-addressed memory.
change: Keep two learned query heads but broadcast one four-dimensional learned key/value head across them, preserving the original full-width initialization draws so the experiment isolates head sharing.
mechanism: Multi-query attention with a shared learned key/value memory
evidence_used: The current two-head model reaches 99.88%, while recent one-parameter affine pruning has become optimization-sensitive and yielded diminishing reductions. This challenges the load-bearing assumption that each query head requires its own key and value projections, while retaining the two independent attention distributions likely needed for simultaneous operand retrieval.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3638, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln1` bias coordinate to zero will reduce parameters from 1,593 to 1,592 while retaining at least 99% accuracy, because the downstream QKV biases can absorb the omitted coordinate.
change: Add a three-coordinate-pruned LayerNorm and use it only before causal self-attention, leaving the successful two-coordinate `ln2` pruning and all other settings unchanged.
mechanism: Incremental attention LayerNorm bias absorption
evidence_used: Two-coordinate `ln1` pruning achieved 99.88% at 1,593 parameters; although third-coordinate `ln2` pruning reached only 98.12%, testing the same smallest incremental reduction in the still-successful attention path is the most direct next experiment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0641, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all seven remaining learned key-bias coordinates will reduce parameters from 1,593 to 1,586 while retaining at least 99% accuracy, because each head’s position-independent key bias adds the same scalar to every unmasked attention logit and is canceled exactly by softmax.
change: Retain learned query biases and the seven-coordinate value bias, but reconstruct the entire key bias as zeros while preserving the original linear constructor and initialization behavior.
mechanism: Softmax-null key-bias elimination
evidence_used: The current design achieved 99.88% at 1,593 parameters, whereas further LayerNorm-coordinate pruning removed useful capacity and failed; key bias elimination instead removes an exact attention-softmax invariance without narrowing learned queries, keys, values, or attention distributions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1586, "training_steps": 4999}

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
