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
verified_results: {"accuracy": 1.0, "parameters": 1614, "training_steps": 4999}
prior_hypothesis: Learning seven of eight query-bias coordinates will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because it preserves the three-coordinate value bias that achieved 100% accuracy and makes only a one-coordinate change to the otherwise verified architecture.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower parameters from 1644 to 1627 while retaining at least 99% accuracy because it preserves the verified 8-wide residual stream, all projection biases, and the full two-head attention module.
change: Change the default MLP hidden width from 12 to 11 without altering the architecture or training schedule otherwise.
mechanism: Incremental feed-forward bottleneck narrowing
evidence_used: The 8-wide, bias-enabled baseline reached 99.96% accuracy, while narrowing the residual stream to 6 and removing all linear biases both failed; this motivates a conservative one-unit reduction confined to the MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7395999999999999, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating only the 8 key-bias parameters will reduce the model from 1644 to 1636 parameters while retaining at least 99% accuracy, because adding the same learned key bias to every attended position changes each query’s attention logits by a position-independent constant that softmax cancels exactly.
change: Replace the fused 24-element QKV bias with separate learned query and value biases, preserving the original query/value computations while omitting the functionally redundant key bias.
mechanism: Remove softmax-invariant key-projection bias
evidence_used: The 1644-parameter baseline reached 99.96% accuracy, while removing all 52 linear biases collapsed accuracy to 3.99%; this motivates a targeted removal limited to the key-bias component whose effect on causal attention weights is mathematically invariant.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the two 8-element LayerNorm biases will reduce parameters from 1636 to 1620 while retaining at least 99% accuracy, because the attention LayerNorm bias is absorbable by the query/value biases with its key component softmax-invariant, and the MLP LayerNorm bias is absorbable by `fc1.bias`.
change: Disable only the bias parameters in `ln1` and `ln2`, preserving their learned scales and every other architectural and training setting.
mechanism: Remove downstream-absorbed pre-sublayer normalization biases
evidence_used: The 1636-parameter design reached 99.85% after removing the exactly redundant key bias, while broad bias removal failed; this motivates removing only two further biases with explicit downstream reparameterizations.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the 8-element value bias will reduce parameters from 1620 to 1612 while retaining at least 99% accuracy, because attention weights sum to one, making the value bias a position-independent offset exactly absorbable by `proj.bias`.
change: Delete `v_bias` and its addition in the attention forward pass while preserving the query bias and all other architecture and training settings.
mechanism: Absorb value-projection bias into attention output bias
evidence_used: The 1620-parameter design achieved 99.98% accuracy after targeted removal of mathematically absorbable biases, whereas broad linear-bias removal failed; this motivates another exact, narrowly scoped bias reparameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9753000000000001, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one of eight value-bias coordinates will reduce parameters from 1620 to 1619 while retaining at least 99% accuracy, because the full removal narrowly missed at 97.53% and the retained seven coordinates preserve most of its optimization benefit while `proj.bias` can absorb the missing constant offset.
change: Store seven learned value-bias coordinates and pad the eighth with zero during attention.
mechanism: Single-coordinate value-bias ablation
evidence_used: Removing all eight value-bias parameters produced 97.53% accuracy, whereas the current 1620-parameter model achieved 99.98%; this motivates a minimal one-coordinate ablation rather than another full bias removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second value-bias coordinate will reduce parameters from 1619 to 1618 while retaining at least 99% accuracy, because the first coordinate removal preserved 99.96% accuracy and six learned coordinates still retain most of the optimization benefit lost when all eight were removed.
change: Store six learned value-bias coordinates and pad the remaining two coordinates with zeros during attention.
mechanism: Two-coordinate value-bias ablation
evidence_used: The single-coordinate ablation achieved 99.96% with 1619 parameters, while removing all eight coordinates achieved only 97.53%; this motivates continuing the ablation one coordinate at a time to locate the accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9958, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one additional value-bias coordinate will reduce parameters from 1618 to 1617 while retaining at least 99% accuracy, because the two-coordinate ablation achieved 99.58% and five learned coordinates preserve most of the optimization benefit absent when all eight coordinates were removed.
change: Store five learned value-bias coordinates and pad the remaining three coordinates with zeros during attention.
mechanism: Three-coordinate value-bias ablation
evidence_used: The two-coordinate ablation met the requirement at 99.58% with 1618 parameters, while removing all eight coordinates reached only 97.53%; another single-coordinate ablation is the most direct test of the capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing each learned position with seven coordinates and padding an eighth zero will remove `max_seq_len` parameters while retaining at least 99% accuracy, because position-dependent shifts along the all-ones direction are invisible to every pre-LayerNorm sublayer and the final LayerNorm.
change: Decouple positional-code width from the load-bearing eight-dimensional residual stream by using a seven-dimensional learned positional embedding with a fixed zero gauge coordinate.
mechanism: LayerNorm-gauge-fixed positional codes
evidence_used: Reducing the entire residual width to six collapsed accuracy to 0.01%, while the eight-wide model reaches 99.91%. Prior designs assumed positional and computational widths must match; this patch preserves all attention and MLP capacity while removing a functionally redundant positional degree of freedom.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.005600000000000001, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the learned value bias from five to four coordinates will lower parameters from 1617 to 1616 while retaining at least 99% accuracy, because the three-coordinate ablation achieved 99.91% and remains far above the 97.53% observed only when all eight coordinates were removed.
change: Store four learned value-bias coordinates and pad the remaining four coordinates with zeros during attention.
mechanism: Four-coordinate value-bias ablation
evidence_used: The current five-coordinate value bias reached 99.91% with 1617 parameters, while six coordinates reached 99.58%; continuing the one-coordinate ablation is the most direct test of the accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the learned value bias from four to three coordinates will lower parameters from 1616 to 1615 while retaining at least 99% accuracy, because the four-coordinate version achieved 99.98% and three learned coordinates may preserve enough of the optimization benefit lost only when all eight coordinates were removed.
change: Store three learned value-bias coordinates and pad the remaining five coordinates with zeros during attention.
mechanism: Five-coordinate value-bias ablation
evidence_used: The current four-coordinate value bias reached 99.98% with 1616 parameters, while removing all eight coordinates reached 97.53%; another one-coordinate ablation is the most informative test of the parameter boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the learned value bias from three coordinates to two will lower parameters from 1615 to 1614 while retaining at least 99% accuracy, because the three-coordinate design achieved 100% accuracy and two coordinates may preserve enough of the optimization benefit lost only when all eight were removed.
change: Store two learned value-bias coordinates and pad the remaining six coordinates with zeros during attention.
mechanism: Six-coordinate value-bias ablation
evidence_used: The current three-coordinate value bias reached 100% accuracy with 1615 parameters, while removing all eight coordinates reached 97.53%; another one-coordinate ablation is the most informative test of the parameter boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5514, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning seven of eight query-bias coordinates will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because it preserves the three-coordinate value bias that achieved 100% accuracy and makes only a one-coordinate change to the otherwise verified architecture.
change: Store seven learned query-bias coordinates and pad the remaining coordinate with zero during attention.
mechanism: Single-coordinate query-bias ablation
evidence_used: Reducing the value bias from three coordinates to two collapsed accuracy from 100% to 55.14%, identifying the three-coordinate value bias as a boundary worth preserving; a one-coordinate ablation of the still-intact eight-coordinate query bias is therefore the most informative alternative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1614, "training_steps": 4999}



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
