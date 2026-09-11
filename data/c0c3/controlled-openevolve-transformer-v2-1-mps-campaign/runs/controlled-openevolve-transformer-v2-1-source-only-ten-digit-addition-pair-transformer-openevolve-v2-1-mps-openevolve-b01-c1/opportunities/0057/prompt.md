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
verified_results: {"accuracy": 0.9992, "parameters": 1554, "training_steps": 4999}
prior_hypothesis: Constraining only the sixth positional coordinate to be mean-free across positions will reduce the model from 1,555 to 1,554 parameters while retaining at least 99% accuracy, because it preserves all position-dependent variation in the coordinate whose wholesale removal caused the 52.62% collapse.

## Recent verification evidence

RECENT RESULT
hypothesis: Centering the scaled final-LayerNorm activation before adding its fixed common component will let the explicit token-row means replace the last learned final-bias direction, producing a 1,578-parameter model with at least 99% accuracy.
change: Remove the final LayerNorm’s last learned bias parameter and reserve its common activation direction exclusively for the decay-free token-row-mean output-bias channel.
mechanism: Activation-isolated token-row output-bias quotient
evidence_used: The optimizer-aligned row means reached 100% accuracy with one learned final-bias coordinate at 1,579 parameters, but simply deleting that coordinate fell to 41.71%; activation centering removes the context-dependent common component that otherwise interferes with row means while preserving the original initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6175, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` bias scalar will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy, because `ln2`’s six learned bias coordinates can generate the omitted neuron offset through its learned weight row, while the other `fc1` biases cancel collateral offsets.
change: Replace the MLP’s first linear layer with an otherwise identical layer whose final output-bias coordinate is fixed at zero, preserving all successful attention, embedding, final-LayerNorm, initialization, and optimizer settings.
mechanism: LayerNorm-beta/downstream-bias gauge reorientation
evidence_used: Removing a third `ln2` bias coordinate collapsed accuracy to 40.61%, while the current design with six learned `ln2` bias coordinates reaches 100%; this motivates preserving that optimization pathway and instead removing one downstream bias coordinate from the same exact affine redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second `fc1` bias scalar will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because the preceding `ln2` retains six learned bias coordinates that can supply both omitted neuron offsets through `fc1`’s learned weight rows.
change: Replace the one-pruned MLP input linear layer with a two-pruned version that fixes its final two output-bias coordinates at zero while preserving its full weight matrix and all other model and training settings.
mechanism: Incremental LayerNorm-beta/downstream-bias gauge reorientation
evidence_used: Removing the first `fc1` bias scalar achieved 99.92% accuracy at 1,578 parameters, whereas pruning an additional `ln2` bias coordinate collapsed accuracy; this directly supports extending the successful downstream-bias removal while preserving the LayerNorm optimization pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8448, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` scale coordinate at 1 will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because that coordinate’s bias is already fixed at zero and the following QKV weight column can absorb its learned scale without reducing the attention function family or changing initialization outputs.
change: Add a two-bias-pruned LayerNorm with one fixed unit scale coordinate and use it only for `ln1`, leaving the successful `ln2` and one-pruned `fc1` pathway unchanged.
mechanism: Zero-beta LayerNorm scale gauge
evidence_used: The one-pruned `fc1` design achieved 99.92% at 1,578 parameters, while removing a second `fc1` bias fell to 84.48%; this motivates a single exact gauge removal elsewhere that preserves the six learned `ln2` bias coordinates implicated in the successful result.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8689, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one value-projection bias scalar at zero will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because causal softmax rows sum to one, making every value bias a context-independent offset that the learned attention output-projection bias can exactly absorb.
change: Remove the final value-bias coordinate from `qkv.bias` and restore it as a fixed zero during the forward pass, preserving all weights, query biases, initialization behavior, and successful training settings.
mechanism: Value-bias/output-bias affine quotient
evidence_used: The current 1,578-parameter model achieves 99.92% accuracy. Unlike the unsuccessful second `fc1`-bias and `ln1`-scale removals, this quotient lies entirely within two consecutive affine attention operations: the mean-free output projection maps the omitted value offset into its own representable seven-dimensional bias space.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7924, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing seven learned coordinates per absolute position with two learned causal-distance logits—one per attention head—will reduce the 1,578-parameter model by `5 * INPUT_LEN` parameters while retaining at least 99% accuracy, because fixed-width addition repeatedly addresses operand digits through relative offsets rather than requiring independent absolute-position vectors.
change: Challenge the assumption that absolute position embeddings must represent both place and attention addressing. Remove them from the residual stream and let each attention head directly learn a bias for every causal relative distance, while preserving the successful initialization stream and all other model, optimizer, and decoding behavior.
mechanism: Learned relative-offset attention without absolute position embeddings
evidence_used: The current two-head model reaches 99.92%, showing its attention and MLP have sufficient computational capacity, while three different one-scalar affine quotients fell to 84.48%, 86.89%, and 79.24%. This motivates seeking larger savings through a different representation: directly learned relative addressing, rather than another fragile affine-coordinate removal.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restricting each learned positional vector from seven to six mean-free coordinates will reduce the model by `INPUT_LEN` parameters while retaining at least 99% accuracy, because the current model reaches 99.92% and this preserves learned absolute positioning, all initialization draws, and six independent positional directions.
change: Remove one orthogonal coordinate from `MeanFreePositionEmbedding`, projecting the unchanged full-width initialization draw into the retained six-dimensional subspace.
mechanism: One-coordinate absolute-position bottleneck
evidence_used: The 1,578-parameter design achieves 99.92%, while the attempted wholesale replacement of absolute positions with relative logits could not be verified; this smaller, isolated test targets the same large parameter source without changing attention behavior.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1555, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing each positional vector from six to five learned mean-free coordinates will lower the model from 1,555 to 1,532 parameters while retaining at least 99% accuracy.
change: Remove one additional orthogonal coordinate from `MeanFreePositionEmbedding` while preserving full-width initialization draws and all other model and training behavior.
mechanism: Incremental absolute-position coordinate bottleneck
evidence_used: The immediately preceding reduction from seven to six positional coordinates achieved 100% accuracy at 1,555 parameters, directly motivating one further isolated coordinate reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5262, "parameters": 1532, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining only the sixth positional coordinate to be mean-free across positions will reduce the model from 1,555 to 1,554 parameters while retaining at least 99% accuracy, because it preserves all position-dependent variation in the coordinate whose wholesale removal caused the 52.62% collapse.
change: Reparameterize the sixth positional coordinate with `INPUT_LEN - 1` orthogonal coordinates, removing only its position-independent component while preserving the original full-width initialization draw and all other behavior.
mechanism: Position-axis common-mode quotient
evidence_used: Reducing every positional vector from six to five coordinates failed at 52.62%, whereas six coordinates achieved 100%; retaining the sixth coordinate’s complete relative-position variation isolates whether only its non-positional common mode is dispensable.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1554, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fifth positional coordinate to be mean-free across positions will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because all six position-dependent coordinates remain available and only another position-independent offset is removed.
change: Reparameterize the final two positional coordinates using `INPUT_LEN - 1` orthogonal position-axis coordinates each, preserving the full-width initialization draw and all other model and training behavior.
mechanism: Second position-axis common-mode quotient
evidence_used: Removing one positional common mode achieved 99.92% at 1,554 parameters, whereas deleting an entire positional coordinate collapsed accuracy to 52.62%; this motivates removing another common mode without sacrificing any relative-position variation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3822, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Explicitly isolating the token-common content vector before removing the fifth positional common mode will produce a 1,553-parameter model with at least 99% accuracy, because this single Adam coordinate can absorb the removed position-independent offset without coordinated updates across every token row.
change: Reparameterize token content into mean-free token variation plus an explicit decay-free common vector, then constrain both final positional coordinates to be mean-free across positions.
mechanism: Optimizer-aligned token-common positional gauge
evidence_used: Removing the second positional common mode directly fell to 38.22%, while independently parameterizing the analogous token-row-mean absorption channel previously turned a failing final-bias quotient into a 100%-accurate model; this motivates changing the optimizer coordinates rather than restoring the redundant positional scalar.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7491, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` input-weight scalar will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because `ln2`’s normalized activations lie in a zero-sum hyperplane and the first neuron’s retained bias absorbs the affine offset.
change: Reparameterize the first `fc1` weight row with its final coordinate fixed at zero, while preserving initialization outputs, RNG consumption, all six learned `ln2` bias coordinates, and the successful one-pruned `fc1` output bias.
mechanism: LayerNorm zero-sum input-weight quotient
evidence_used: The 1,554-parameter design achieved 99.92%, while removing a second `fc1` bias fell to 84.48% and fixing an `ln1` scale fell to 86.89%; this motivates removing a distinct exact redundancy that retains both the downstream neuron bias and every LayerNorm scale and bias coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7334, "parameters": 1553, "training_steps": 4999}



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
