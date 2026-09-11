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
verified_results: {"accuracy": 0.996, "parameters": 1631, "training_steps": 4999}
prior_hypothesis: Omitting three value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,631-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9943000000000001, "parameters": 1627, "training_steps": 4999}
prior_hypothesis: Omitting one head-1 query-bias coordinate while preserving its critical final value-bias scalar will produce a 1,627-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1625, "training_steps": 4999}
prior_hypothesis: Omitting a second head-0 query-bias coordinate while retaining all three head-1 query-bias coordinates will produce a 1,625-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying one attention output-bias coordinate to the sole retained head-1 value-bias scalar in the qualified 4:3 layout will produce a 1,629-parameter model with at least 99% accuracy.
change: Adopt the qualified 4:3 value-bias omission layout, then replace the eight-scalar attention output bias with seven independent scalars plus the retained value-bias scalar.
mechanism: Attention-local value/output bias gauge tying
evidence_used: The 4:3 layout achieved 99.81% at 1,630 parameters, while 3:4 fell to 74.45%, motivating preservation of head 1’s final value-bias scalar and removal of a redundant degree of freedom from the position-independent output bias instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7056999999999999, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at its initialized value of one will reduce the model to 1,629 parameters while retaining at least 99% accuracy, because the unrestricted corresponding `fc1` weight column can learn the same multiplicative transformation.
change: Replace `ln2` with an equivalent LayerNorm containing seven learned scale coordinates and all eight learned biases, preserving the baseline initialization and leaving the qualified attention parameterization unchanged.
mechanism: Pre-MLP LayerNorm scale gauge fixing
evidence_used: The current attention layout achieved 99.81% at 1,630 parameters, while tying an attention output bias fell to 70.57%; this motivates preserving attention and testing a distinct multiplicative gauge. The failed `ln2` beta omission modified an additive pathway, whereas this patch retains every LayerNorm bias and removes only a scale redundant with `fc1`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.35090000000000005, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Preserving the qualified 4:3 value-bias layout while fixing the attention output bias’s LayerNorm-invisible common-mode degree of freedom will produce a 1,629-parameter model with at least 99% accuracy.
change: Adopt the verified 4:3 per-head value-bias omission and replace the eight-parameter attention projection bias with seven learned relative coordinates, reconstructing the final coordinate as zero.
mechanism: Common-mode attention residual bias quotient
evidence_used: The 4:3 value layout achieved 99.81% at 1,630 parameters, while removing head 1’s final value bias failed. Unlike the failed value/output tie, this patch preserves that scalar and removes only a uniform attention-residual shift annihilated by the following LayerNorms.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7223, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying an exact token/position embedding translation quotient to the qualified 4:3 value-bias design will produce a 1,629-parameter model with at least 99% accuracy.
change: Adopt the verified 4:3 value-bias omission, transfer one positional scalar into the tied token embeddings, and reconstruct that positional scalar as zero.
mechanism: Token-position translation gauge fixing
evidence_used: The 4:3 value-bias layout achieved 99.81% at 1,630 parameters. Prior 1,629 changes to attention-output and `ln2` parameters failed, motivating a distinct embedding gauge that preserves initialized hidden inputs and changes output logits only by a class-independent shift.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified embedding translation quotient to a second hidden coordinate will reduce the model to 1,628 parameters while maintaining at least 99% accuracy.
change: Transfer the first two positional coordinates into the tied token embeddings and reconstruct both corresponding positional scalars as zero.
mechanism: Two-coordinate token-position translation gauge fixing
evidence_used: The one-coordinate quotient achieved 99.95% accuracy at 1,629 parameters; the second coordinate has the same exact invariance, preserving initialized hidden inputs while changing logits only by a class-independent shift.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7126, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified one-coordinate token/position translation quotient with omission of the final key-bias coordinate will produce a 1,628-parameter model with at least 99% accuracy.
change: Compact one positional-embedding scalar and all eight attention key biases while preserving initialized hidden inputs, output probabilities, all QKV weights, and the qualified 4:3 value-bias layout.
mechanism: Orthogonal exact embedding and key-softmax gauge fixing
evidence_used: The one-coordinate embedding quotient achieved 99.95% at 1,629 parameters, while extending it to a second coordinate fell to 71.26%; using the distinct key-bias softmax common-mode invariance is therefore the most informative next one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the qualified 1,628-parameter design with removal of the final MLP bias’s LayerNorm-invisible common mode will produce a 1,627-parameter model with at least 99% accuracy.
change: Apply the verified one-coordinate embedding quotient, omit all key biases and value biases in the qualified 4:3 layout, and represent the eight-coordinate final MLP bias using seven orthonormal zero-mean coordinates.
mechanism: Orthonormal final-residual bias quotient
evidence_used: The qualified embedding-plus-key-bias design achieved 99.95% at 1,628 parameters. The failed asymmetric attention-output quotient motivates testing the distinct final-MLP residual bias immediately before `ln_f`, using an orthonormal basis to preserve balanced optimization geometry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6019, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting one head-1 query-bias coordinate while preserving its critical final value-bias scalar will produce a 1,627-parameter model with at least 99% accuracy.
change: Reconstruct the final head-1 query bias as zero, retaining all QKV weights, all key-bias omissions, the verified positional quotient, and the qualified 4:3 value-bias layout.
mechanism: Cross-projection per-head bias-capacity balancing
evidence_used: The current 1,628-parameter design achieved 99.95%; the 4:3 value layout achieved 99.81% while the mirrored 3:4 layout collapsed to 74.45%, motivating preservation of head 1’s sole value bias and testing the smallest reduction in the untouched query-bias family. Assigning it to head 1 balances total per-head QKV bias omissions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9943000000000001, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting the final query-bias coordinate from head 0 as well as head 1 will produce a 1,626-parameter model with at least 99% accuracy.
change: Extend the qualified 1,627-parameter layout to reconstruct one query-bias coordinate per head as zero, while retaining all QKV weights and head 1’s critical value-bias scalar.
mechanism: Alternating per-head query-bias compaction
evidence_used: The 1,627-parameter design achieved 99.43% after omitting one head-1 query bias; distributing the next query-bias omission to head 0 preserves three learned query-bias coordinates in each head and tests the smallest remaining QKV-bias reduction without removing the value scalar whose absence previously caused collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,626-parameter design by omitting a second head-1 query-bias coordinate will produce a 1,625-parameter model with at least 99% accuracy.
change: Apply the verified one-coordinate positional quotient and 4:3 value/all-key bias layout, while retaining three head-0 and two head-1 query-bias coordinates.
mechanism: Alternating per-head query-bias compaction
evidence_used: Omitting one head-1 query bias achieved 99.43% at 1,627 parameters, and then omitting one head-0 query bias improved accuracy to 99.86% at 1,626; continuing the successful alternating reduction on head 1 is the smallest untested QKV-bias change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7290000000000001, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting a second head-0 query-bias coordinate while retaining all three head-1 query-bias coordinates will produce a 1,625-parameter model with at least 99% accuracy.
change: Retain two head-0 and three head-1 query-bias coordinates, preserving all QKV weights, the one-coordinate positional quotient, all key-bias omissions, and the qualified 4:3 value-bias layout.
mechanism: Head-asymmetric query-bias compaction
evidence_used: The 1,626-parameter design achieved 99.86%, but removing a second head-1 query bias collapsed accuracy to 72.9%; independently, the successful 4:3 versus failed 3:4 value-bias layouts show that head 0 tolerates stronger bias compaction than head 1.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining one head-0 and three head-1 query-bias coordinates will produce a 1,624-parameter model with at least 99% accuracy.
change: Extend the qualified 1,625-parameter layout by omitting a third head-0 query bias while preserving all QKV weights, three head-1 query biases, the critical head-1 value bias, all key-bias omissions, and the positional quotient.
mechanism: Head-asymmetric query-bias compaction
evidence_used: The 1,625-parameter 2:3 query-bias layout achieved 99.99%, whereas reducing head 1 to two query biases collapsed accuracy to 72.9%; the successful head asymmetry motivates placing the next omission in head 0.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9868000000000001, "parameters": 1624, "training_steps": 4999}



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
