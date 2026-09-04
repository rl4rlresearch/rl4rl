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
verified_results: {"accuracy": 0.9982, "parameters": 1535, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,536-parameter query-row-7 design by anchoring position-zero coordinate 1 and transferring its shift into the tied token embedding will produce a 1,535-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1536, "training_steps": 4999}
prior_hypothesis: Gauging second-head query row 7 with ordinary reduced-coordinate AdamW will produce a 1,536-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1544, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,545-parameter design to second-head key row 14 will produce a 1,544-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1541, "training_steps": 4999}
prior_hypothesis: Gauging value row 21 while training it with ordinary reduced-coordinate AdamW will produce a 1,541-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauging value row 21 while training it with ordinary reduced-coordinate AdamW will produce a 1,541-parameter model with at least 99% accuracy.
change: Add second-head value row 21 to the normalized-input gauge, while retaining dense-coordinate updates only for key row 15 and value row 20.
mechanism: Reduced-chart second-head value-row quotient
evidence_used: Dense-coordinate training of row 21 reached 98.28%, while first-head value rows 16–19 succeeded with ordinary reduced-coordinate updates and the current 1,542-parameter design reached 100%; this isolates optimizer geometry as the next variable.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1541, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,541-parameter design to second-head value row 22 with ordinary reduced-coordinate AdamW will produce a 1,540-parameter model with at least 99% accuracy.
change: Gauge second-head value rows 20–22, retaining dense-coordinate AdamW only for optimization-sensitive key row 15 and value row 20 while rows 21–22 use ordinary reduced-coordinate updates.
mechanism: Reduced-chart third second-head value-row quotient
evidence_used: Value row 21 reached 99.93% with reduced-coordinate AdamW after dense-coordinate training reached only 98.28%; this favors the same reduced geometry for adjacent row 22, while preserving the 100%-accurate dense treatment of row 20.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9962000000000001, "parameters": 1540, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,540-parameter design to second-head value row 23 with ordinary reduced-coordinate AdamW will produce a 1,539-parameter model with at least 99% accuracy.
change: Gauge every key and value row, retaining dense-coordinate AdamW only for sensitive key row 15 and value row 20 while value rows 21–23 use ordinary reduced-coordinate updates.
mechanism: Reduced-chart final second-head value-row quotient
evidence_used: Value row 22 achieved 99.62% at 1,540 parameters with reduced-coordinate AdamW, and row 21 similarly improved from 98.28% with dense-coordinate training to 99.93% with reduced-coordinate training; this motivates applying the same geometry to adjacent final value row 23.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7021999999999999, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauging value row 23 while training it with recovered eight-coordinate AdamW moments will produce a 1,539-parameter model with at least 99% accuracy.
change: Reproduce the verified 1,540-parameter design, add the final second-head value-row gauge, and use dense-coordinate updates for key row 15 and value rows 20 and 23 while retaining reduced-coordinate AdamW for value rows 21–22.
mechanism: Dense-AdamW final value-row quotient
evidence_used: Reduced-coordinate training of final value row 23 collapsed to 70.22%, whereas dense-coordinate training rescued the analogous final key row 15 from 90.92% to 99.87%; dense treatment of value row 20 also achieved 100%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauging first-head query row 2 with ordinary reduced-coordinate AdamW will reduce the model to 1,538 parameters while retaining at least 99% accuracy.
change: Add query row 2 to the normalized-input gauge; existing custom QKV optimization leaves it on the reduced-coordinate path while preserving dense updates for sensitive rows 15, 20, and 23.
mechanism: Reduced-chart first-head query-row quotient
evidence_used: The 1,539-parameter design reached 99.95%, and ordinary reduced-coordinate updates successfully trained adjacent normalized-input gauges including all first-head value rows and second-head value rows 21–22; extending the already verified query-row 0–1 pattern is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,538-parameter design by gauging first-head query row 3 with ordinary reduced-coordinate AdamW will produce a 1,537-parameter model with at least 99% accuracy.
change: Reproduce the verified query-row-2 and final-value-row gauges, then gauge adjacent query row 3 while retaining dense-coordinate updates for sensitive key row 15 and value rows 20 and 23.
mechanism: Reduced-chart final first-head query-row quotient
evidence_used: Query row 2 achieved 99.93% at 1,538 parameters with reduced-coordinate AdamW, while all first-head key and value rows were also successfully gauged; this supports completing the adjacent first-head query row in the same reduced chart.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1537, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,537-parameter design by gauging second-head query row 6 with ordinary reduced-coordinate AdamW will produce a 1,536-parameter model with at least 99% accuracy.
change: Reproduce every verified key, value, and first-head query gauge, add second-head query row 6, and retain dense-coordinate updates for sensitive key row 15 and value rows 20 and 23.
mechanism: Reduced-chart second-head query-row quotient
evidence_used: First-head query rows 2 and 3 successively achieved 99.93% and 99.95% with reduced-coordinate AdamW; row 6 is the adjacent untested query row and shares the already-fixed local query-bias coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9889, "parameters": 1536, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauging second-head query row 6 while training it with recovered eight-coordinate AdamW moments will produce a 1,536-parameter model with at least 99% accuracy.
change: Reproduce all gauges of the verified 1,537-parameter design, add query row 6, and use dense-coordinate updates for that row plus the previously sensitive key row 15 and value rows 20 and 23.
mechanism: Dense-AdamW second-head query-row quotient
evidence_used: Reduced-coordinate training of query row 6 narrowly missed at 98.89%, while dense-coordinate AdamW previously rescued key row 15 from 90.92% to 99.87%; this directly tests whether optimizer geometry, rather than capacity, caused the query-row miss.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9852, "parameters": 1536, "training_steps": 4999}

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
