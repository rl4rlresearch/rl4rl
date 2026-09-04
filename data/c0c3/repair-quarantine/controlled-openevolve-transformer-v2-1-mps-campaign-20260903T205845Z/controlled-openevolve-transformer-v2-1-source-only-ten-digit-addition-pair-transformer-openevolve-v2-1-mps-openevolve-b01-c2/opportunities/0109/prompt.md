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
verified_results: {"accuracy": 0.996, "parameters": 1533, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1,534-parameter design by gauging its sole remaining QKV row—second-head query row 6—and training that row in dense coordinates will produce a 1,533-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1534, "training_steps": 4999}
prior_hypothesis: Anchoring position-zero coordinate 6 alongside verified coordinates 0, 1, 3, and 7 will reduce the model to 1,534 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9982, "parameters": 1535, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,536-parameter query-row-7 design by anchoring position-zero coordinate 1 and transferring its shift into the tied token embedding will produce a 1,535-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1544, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,545-parameter design to second-head key row 14 will produce a 1,544-parameter model with at least 99% accuracy.

## Recent verification evidence

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
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Anchoring position-zero coordinate 6 alongside verified coordinates 0, 1, 3, and 7 will reduce the model to 1,534 parameters while retaining at least 99% accuracy.
change: Reproduce the verified four-coordinate positional gauge, then transfer and anchor coordinate 6 using ordinary reduced-coordinate AdamW without changing the verified QKV optimization.
mechanism: Fifth token–position translation quotient at coordinate 6
evidence_used: The four-anchor design achieved 99.82% at 1,535 parameters; fifth anchors at coordinates 2, 4, and 5 failed, leaving coordinate 6 as the only untested positional translation quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,534-parameter design by gauging its sole remaining QKV row—second-head query row 6—and training that row in dense coordinates will produce a 1,533-parameter model with at least 99% accuracy.
change: Reproduce the qualified five-coordinate positional gauge and its QKV gauges, add query row 6, and retain dense-coordinate AdamW for rows 6, 15, 20, and 23.
mechanism: Dense-AdamW completion of the normalized-input QKV row quotient
evidence_used: Reference Design 1 reached 99.90% accuracy with 1,534 parameters and 23 of 24 QKV rows gauged; prior evidence that dense-coordinate optimization rescued key row 15 from 90.92% to 99.87% motivates using the same optimizer geometry for the final row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.996, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,533-parameter design, fixing attention LayerNorm input scale coordinate 1 and absorbing that scale into the learned QKV column will yield 1,532 parameters while retaining at least 99% accuracy.
change: Reproduce the verified five-position-anchor, all-row QKV-gauged design with dense-coordinate updates for sensitive QKV rows, then fix LayerNorm scale coordinate 1 in addition to coordinates 3–7.
mechanism: Alternative sixth attention-LayerNorm column-scale quotient
evidence_used: Reference Design 3 reached 99.6% with 1,533 parameters and all 24 QKV rows gauged. Five LayerNorm scale quotients already train successfully; coordinate 1 is an untested sixth quotient, while the failed coordinate-2 result argues against repeating that coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8398, "parameters": 1532, "training_steps": 4999}



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
