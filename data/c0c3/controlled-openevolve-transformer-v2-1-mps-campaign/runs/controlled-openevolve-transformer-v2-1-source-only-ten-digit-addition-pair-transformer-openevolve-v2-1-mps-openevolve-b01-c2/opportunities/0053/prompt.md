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
verified_results: {"accuracy": 0.9994, "parameters": 1581, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,582-parameter design with the first value row will produce a 1,581-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9981, "parameters": 1585, "training_steps": 4999}
prior_hypothesis: Gauge-fixing QKV row 1 alongside the verified row 0 gauge will reduce the model from 1,586 to 1,585 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9994, "parameters": 1586, "training_steps": 4999}
prior_hypothesis: Applying one QKV row gauge to the verified 1,587-parameter design will produce a 1,586-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Adding a gauge for the first key row to the verified four balanced query-row gauges will reduce the model from 1,583 to 1,582 parameters while maintaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-fixing QKV row 1 alongside the verified row 0 gauge will reduce the model to 1,585 parameters while maintaining at least 99% accuracy.
change: Omit and reconstruct the final input coefficient of QKV rows 0 and 1, preserving their initialized functions while leaving the remaining QKV rows unconstrained.
mechanism: Second-row QKV LayerNorm-nullspace gauge
evidence_used: The current single-row QKV gauge achieved 99.94% accuracy at 1,586 parameters, substantially above threshold; adding one adjacent query-row gauge is the smallest incremental reduction along this successful path.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-fixing QKV row 1 alongside the verified row 0 gauge will reduce the model from 1,586 to 1,585 parameters while maintaining at least 99% accuracy.
change: Replace the fused QKV linear layer with a learned projection that omits the final input coefficient from query rows 0 and 1, reconstructs those coefficients as zero, retains query bias, and preserves the initialized row functions.
mechanism: Second-row QKV normalized-input gauge
evidence_used: The single-row QKV gauge achieved 99.94% accuracy at 1,586 parameters. The prior two-row attempt could not be verified and supplied no accuracy result, so a correctly generalized second-row gauge is the smallest informative continuation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-row QKV gauge to query row 2 will reduce the model from 1,585 to 1,584 parameters while maintaining at least 99% accuracy.
change: Reproduce the verified five-anchor, two-row-gauged design and omit the final input coefficient from QKV rows 0, 1, and 2.
mechanism: Third-row QKV normalized-input gauge
evidence_used: The two-row QKV gauge achieved 99.81% accuracy at 1,585 parameters; adding one adjacent query-row gauge is the smallest incremental reduction along that successful path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9590000000000001, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing query row 4, the first row of the second attention head, alongside verified rows 0 and 1 will produce a 1,584-parameter model with at least 99% accuracy.
change: Reproduce the verified five-anchor LayerNorm and two-row QKV design, then omit the final input coefficient from query row 4 to distribute the three QKV gauges across both attention heads.
mechanism: Cross-head QKV normalized-input gauge
evidence_used: Rows 0 and 1 achieved 99.81% accuracy at 1,585 parameters, while adding row 2 fell to 95.90%; gauging row 4 tests whether the failure arose from concentrating three constrained query rows in the first head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing query row 5 alongside verified rows 0, 1, and 4 will produce a 1,583-parameter model with at least 99% accuracy.
change: Add the second query coordinate of the second attention head to the normalized-input QKV gauge, balancing two gauged query rows per head.
mechanism: Balanced cross-head query-row gauge fixing
evidence_used: The cross-head rows 0, 1, and 4 design achieved 99.93% at 1,584 parameters, while concentrating the third gauge on row 2 fell to 95.90%; extending the successful balanced placement to row 5 is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified four-row balanced QKV gauge with exact common-shift gauges on both residual-branch output biases will reduce the model from 1,583 to 1,581 parameters while maintaining at least 99% accuracy.
change: Reproduce query-row gauges 0, 1, 4, and 5, then replace the attention projection and MLP output linears with learned linears that omit one redundant bias coordinate each.
mechanism: Residual-stream common-shift bias gauge
evidence_used: Reference Design 3 achieved 99.98% accuracy with 1,583 parameters using four balanced query-row gauges. A common shift in either residual-branch output is preserved through pre-norm residual blocks and canceled by the final LayerNorm, making one bias direction per branch functionally redundant.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6844, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified four-row balanced QKV gauge with an exact 8-parameter token-position embedding shift gauge will produce a 1,575-parameter model with at least 99% accuracy.
change: Reproduce query-row gauges 0, 1, 4, and 5, then anchor positional row 0 to zero while shifting its initialized vector into every tied token embedding.
mechanism: Token-position embedding shift gauge
evidence_used: Reference Design 3 reached 99.98% accuracy at 1,583 parameters. Anchoring one positional row removes eight redundant parameters without reducing the represented distributions: shifting that row into all token embeddings preserves every input embedding sum, while the tied output logits change only by a vocabulary-wide common offset.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7062, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified four-row QKV gauge and additionally gauge-fixing query row 6 will produce a 1,582-parameter model with at least 99% accuracy.
change: Anchor five `ln1` scales and omit the final normalized-input coefficient from query rows 0, 1, 4, 5, and 6 while preserving their initialized functions.
mechanism: Second-head query-row LayerNorm-nullspace gauge
evidence_used: The balanced rows 0, 1, 4, and 5 design achieved 99.98% accuracy at 1,583 parameters; adding one row in the second head is the smallest reduction from that high-margin result and avoids the previously unsuccessful row-2 placement.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6568, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a gauge for the first key row to the verified four balanced query-row gauges will reduce the model from 1,583 to 1,582 parameters while maintaining at least 99% accuracy.
change: Retain query-row gauges 0, 1, 4, and 5, and omit the final normalized-input coefficient from fused QKV row 8, the first key coordinate of the first attention head.
mechanism: Cross-projection normalized-input gauge fixing
evidence_used: The four balanced query gauges achieved 99.98% accuracy at 1,583 parameters, while adding query row 6 fell to 65.68%; testing one key-row gauge is the smallest orthogonal reduction and avoids further constraining either head’s query representation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,582-parameter design with the corresponding first key row of the second attention head will produce a 1,581-parameter model with at least 99% accuracy.
change: Gauge query rows 0, 1, 4, and 5 and key rows 8 and 12, balancing both query and key gauges across the two attention heads.
mechanism: Balanced cross-head key-row LayerNorm-nullspace gauge
evidence_used: Reference Design 3 achieved 99.93% accuracy at 1,582 parameters after adding key row 8 to four balanced query gauges; adding key row 12 is the smallest cross-head-balanced extension and avoids the failed strategy of constraining another query coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.66, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,582-parameter design with key row 9 will produce a 1,581-parameter model with at least 99% accuracy.
change: Reproduce the four balanced query-row gauges and key-row-8 gauge, then omit the final normalized-input coefficient from adjacent key row 9.
mechanism: Within-head second key-row normalized-input gauge
evidence_used: Key row 8 achieved 99.93% accuracy at 1,582 parameters, while adding second-head key row 12 fell to 66.0%; row 9 tests whether the first head can tolerate another key gauge without constraining the sensitive second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9819, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,582-parameter design with the first value row will produce a 1,581-parameter model with at least 99% accuracy.
change: Reproduce the five-anchor design with query-row gauges 0, 1, 4, and 5 and key-row gauge 8, then omit the final normalized-input coefficient from value row 16.
mechanism: Value/output-basis normalized-input gauge
evidence_used: The five-gauge reference achieved 99.93% at 1,582 parameters, while additional query and key gauges failed; a value-row gauge is the smallest untested orthogonal extension and can leverage the learned value/output-projection basis freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1581, "training_steps": 4999}



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
