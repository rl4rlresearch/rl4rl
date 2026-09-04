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
verified_results: {"accuracy": 0.9995, "parameters": 1513, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,535-parameter four-anchor design by removing the common scalar component of every nonzero position row will reduce parameters by `INPUT_LEN - 1` while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9982, "parameters": 1535, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,536-parameter query-row-7 design by anchoring position-zero coordinate 1 and transferring its shift into the tied token embedding will produce a 1,535-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1544, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,545-parameter design to second-head key row 14 will produce a 1,544-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1541, "training_steps": 4999}
prior_hypothesis: Gauging value row 21 while training it with ordinary reduced-coordinate AdamW will produce a 1,541-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauging second-head query row 7 with ordinary reduced-coordinate AdamW will produce a 1,536-parameter model with at least 99% accuracy.
change: Add query row 7 to the normalized-input gauge while retaining dense-coordinate updates only for sensitive key row 15 and value rows 20 and 23.
mechanism: Reduced-chart final second-head query-row quotient
evidence_used: First-head query rows 2–3 succeeded with reduced-coordinate AdamW at 1,537 parameters; unlike failed row 6, row 7 retains its learned query bias, isolating whether row 6’s fixed-bias coordinate caused its narrow 98.89% miss.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1536, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,536-parameter query-row-7 design by anchoring position-zero coordinate 1 and transferring its shift into the tied token embedding will produce a 1,535-parameter model with at least 99% accuracy.
change: Reproduce the verified query-row-3 and query-row-7 gauges, then remove one additional positional parameter through the existing token–position translation gauge.
mechanism: Fourth token–position translation quotient
evidence_used: Query row 7 achieved 99.96% at 1,536 parameters, while the current model’s three transferred positional anchors coexist with 99.93% accuracy; extending that exact positional quotient is more promising than query row 6, which missed 99% under both reduced- and dense-coordinate AdamW.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,535-parameter design by anchoring position-zero coordinate 2 and transferring its shift into the tied token embedding will produce a 1,534-parameter model with at least 99% accuracy.
change: Reproduce all verified query, key, and value-row gauges, retain dense-coordinate AdamW for sensitive rows 15, 20, and 23, and extend the positional quotient from coordinates 0, 1, 3, and 7 to coordinates 0, 1, 2, 3, and 7.
mechanism: Fifth token–position translation quotient
evidence_used: The fourth positional anchor produced 99.82% accuracy at 1,535 parameters, while the underlying query-row-7 design achieved 99.96% at 1,536 parameters; coordinate 2 applies the same exact token–position translation gauge for one additional reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9804999999999999, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing attention LayerNorm coordinate 2 and absorbing its scale into QKV weights will reduce the verified 1,535-parameter design to 1,534 parameters while retaining at least 99% accuracy.
change: Reproduce the verified query-row-7 and four-position-anchor design, preserve dense-coordinate AdamW for sensitive QKV rows 15, 20, and 23, and fix one additional attention LayerNorm scale.
mechanism: Sixth attention-LayerNorm scale quotient
evidence_used: The 1,535-parameter design achieved 99.82%, while its attention LayerNorm already fixes five scales through the same QKV column-scaling symmetry; this tests that distinct quotient after a fifth positional anchor at coordinate 2 fell to 98.05%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9420000000000001, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring position-zero coordinate 2 while recovering dense AdamW moments for its coupled token/position column will produce a 1,534-parameter model with at least 99% accuracy.
change: Extend the verified four-coordinate positional gauge to coordinate 2 and optimize that quotient in its original dense coordinates while leaving all other embedding coordinates on ordinary reduced-coordinate AdamW.
mechanism: Dense-AdamW fifth token–position translation quotient
evidence_used: The same fifth anchor reached only 98.05% with reduced-coordinate AdamW, while dense-coordinate optimization rescued other sensitive exact quotients, including key row 15 from 90.92% to 99.87%; the verified four-anchor model reached 99.82%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9887, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring position-zero coordinate 4 in addition to the verified coordinates 0, 1, 3, and 7 will produce a 1,534-parameter model with at least 99% accuracy.
change: Reproduce the verified four-coordinate positional gauge, then transfer and anchor coordinate 4 using ordinary reduced-coordinate AdamW while leaving the verified QKV optimizer unchanged.
mechanism: Alternative fifth token–position translation quotient
evidence_used: The coordinate-1 anchor reached 99.82% at 1,535 parameters, whereas coordinate 2 failed with both reduced and dense AdamW; coordinate 4 tests the same exact quotient on a feature that, unlike coordinate 2, retains its learned query bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8589, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,535-parameter design by fixing first-head query-bias coordinate 3 will produce a 1,534-parameter model with at least 99% accuracy.
change: Reproduce the verified query-row-7 and four-position-anchor design, retain dense-coordinate AdamW for QKV rows 15, 20, and 23, and remove one additional first-head query-bias parameter.
mechanism: Additional first-head query-bias basis quotient
evidence_used: The 1,535-parameter design achieved 99.82% while already fixing one query-bias coordinate per head; testing another coordinate of that successful attention-basis quotient is more informative than positional coordinates 2 and 4 or a sixth LayerNorm scale, which all failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2889, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,535-parameter design by anchoring position-zero coordinate 5 will produce a 1,534-parameter model with at least 99% accuracy.
change: Reproduce the verified query-row-7 and four-position-anchor design, retain dense-coordinate AdamW for sensitive QKV rows 15, 20, and 23, and add coordinate 5 to the positional translation gauge.
mechanism: Alternative fifth token–position translation quotient at coordinate 5
evidence_used: The four-anchor design achieved 99.82% at 1,535 parameters. Coordinates 2 and 4 were optimization-sensitive failures, so testing the remaining coordinate 5—while preserving its learned query bias—is the most direct informative one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9594, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing attention LayerNorm coordinate 1 and absorbing its scale into QKV weights will produce a 1,534-parameter model with at least 99% accuracy.
change: Fix coordinate 1 alongside the five verified trailing LayerNorm scales, and expose the reconstructed dense scale vector to the sensitive-row QKV optimizer.
mechanism: Alternative sixth attention-LayerNorm scale quotient at coordinate 1
evidence_used: The 1,535-parameter model reached 99.82% with five fixed LayerNorm scales, while coordinate 1 also tolerated the latest successful positional quotient; testing it distinguishes coordinate-specific optimization sensitivity from coordinate 2’s 94.20% sixth-scale failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9309000000000001, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,535-parameter four-anchor design by removing the common scalar component of every nonzero position row will reduce parameters by `INPUT_LEN - 1` while retaining at least 99% accuracy.
change: Reproduce the verified coordinate-1 positional anchor, represent positions 1 onward in the zero-mean feature subspace, and update those rows with recovered eight-coordinate AdamW moments.
mechanism: Dense-AdamW per-position residual-scalar quotient
evidence_used: The four-anchor design achieved 99.82% at 1,535 parameters, and dense-coordinate optimization preserved other sensitive exact quotients. A position-specific scalar shift survives the residual stream but is erased by every LayerNorm, so this removes exact null directions without reducing learned function capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1513, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauging the remaining second-head query row 6 with reduced-coordinate AdamW in the verified 1,513-parameter position-scalar design will produce a 1,512-parameter model with at least 99% accuracy.
change: Add query row 6 to the normalized-input QKV gauge while retaining dense-coordinate updates for sensitive key row 15 and value rows 20 and 23.
mechanism: Complete second-head query-row quotient
evidence_used: Query row 6 previously reached 98.89%, only 0.11 percentage points below the requirement, while the newer dense-position quotient changed the optimization geometry and achieved 99.95% at 1,513 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.022400000000000003, "parameters": 1512, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding second-head query row 6 through a Helmert orthonormal chart to the verified 1,513-parameter position-scalar design will produce a 1,512-parameter model with at least 99% accuracy.
change: Reproduce the qualified dense-AdamW position-row scalar quotient, then remove query row 6’s exact normalized-input null direction using an orthonormal seven-coordinate basis while retaining the verified dense updates for rows 15, 20, and 23.
mechanism: Orthonormal final-query-row LayerNorm quotient
evidence_used: The position-scalar quotient achieved 99.95% at 1,513 parameters, while query row 6 failed in a last-coordinate reduced chart at 1,512 parameters. Query row 7 succeeded under the same capacity reduction, so testing a symmetric orthonormal chart directly targets optimizer geometry without removing additional function capacity.
result: the patch search text matched more than once



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
