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
verified_results: {"accuracy": 0.9995, "parameters": 1539, "training_steps": 4999}
prior_hypothesis: Gauging value row 23 while training it with recovered eight-coordinate AdamW moments will produce a 1,539-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9962000000000001, "parameters": 1540, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,541-parameter design to second-head value row 22 with ordinary reduced-coordinate AdamW will produce a 1,540-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1544, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,545-parameter design to second-head key row 14 will produce a 1,544-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1541, "training_steps": 4999}
prior_hypothesis: Gauging value row 21 while training it with ordinary reduced-coordinate AdamW will produce a 1,541-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified 1,548-parameter design to first-head key row 11 will produce a 1,547-parameter model with at least 99% accuracy.
change: Gauge the remaining first-head key row while retaining all previously verified query, value, output, embedding, and positional quotients.
mechanism: Fourth first-head key-row normalized-input gauge
evidence_used: Successive first-head key-row extensions reached 99.91% at 1,549 parameters and 99.96% at 1,548 parameters; row 11 completes the same adjacent first-head pattern without imposing a fragile second-head constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1547, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,547-parameter design to second-head key row 12 will produce a 1,546-parameter model with at least 99% accuracy.
change: Retain every verified gauge, complete the four first-head key-row gauges, and gauge the adjacent first key row of the second attention head.
mechanism: First second-head key-row normalized-input gauge
evidence_used: Successive first-head key extensions achieved 99.91% at 1,549 parameters, 99.96% at 1,548, and 99.90% at 1,547; continuing to adjacent row 12 is the smallest reduction and directly tests the previously fragile head boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1546, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,546-parameter design to second-head key row 13 will produce a 1,545-parameter model with at least 99% accuracy.
change: Reproduce all verified first-head key and value gauges plus second-head key row 12, then gauge the adjacent second-head key row 13.
mechanism: Second second-head key-row normalized-input gauge
evidence_used: The verified progression from key rows 9–12 maintained 99.90%–99.99% accuracy, with row 12 achieving 99.99% at 1,546 parameters; extending the same adjacent-row pattern is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1545, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,545-parameter design to second-head key row 14 will produce a 1,544-parameter model with at least 99% accuracy.
change: Reproduce all verified first-head key and value gauges plus second-head key rows 12–13, then gauge the adjacent second-head key row 14.
mechanism: Third second-head key-row normalized-input gauge
evidence_used: The verified progression through key rows 9–13 maintained 99.90%–99.99% accuracy, and row 13 achieved 99.95% at 1,545 parameters; extending the same adjacent-row pattern is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1544, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,544-parameter design to second-head key row 15 will produce a 1,543-parameter model with at least 99% accuracy.
change: Retain every verified gauge and gauge the remaining second-head key row by fixing its final normalized-input coefficient.
mechanism: Fourth second-head key-row normalized-input gauge
evidence_used: Successive second-head key-row extensions through rows 12–14 maintained 99.85%–99.99% accuracy, with row 14 reaching 99.85% at 1,544 parameters; row 15 completes the same adjacent within-head pattern.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9092, "parameters": 1543, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauging both remaining second-head key rows while training row 15 with recovered eight-coordinate AdamW moments will produce a 1,543-parameter model with at least 99% accuracy.
change: Add key rows 14–15 to the normalized-input gauge and project a dense-coordinate AdamW update for row 15 back into its anchored seven-parameter chart.
mechanism: Dense-AdamW final key-row quotient
evidence_used: Row 14 achieved 99.85% at 1,544 parameters, whereas row 15 collapsed to 90.92%; dense-coordinate AdamW previously rescued an exact quotient from 98.89% to 99.97%, implicating optimizer geometry rather than capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1543, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,543-parameter design to second-head value row 20, while recovering dense-coordinate AdamW moments for both optimization-sensitive row 15 and the new row 20 quotient, will produce a 1,542-parameter model with at least 99% accuracy.
change: Complete all second-head key-row gauges, add the first second-head value-row gauge, and train rows 15 and 20 through projected eight-coordinate AdamW updates while retaining the verified dense output-quotient optimizer.
mechanism: Dense-AdamW second-head value-row quotient
evidence_used: Completing key row 15 with ordinary reduced-coordinate AdamW collapsed to 90.92%, whereas recovered dense moments reached 99.87% at 1,543 parameters; all four first-head value-row gauges also retained at least 99.78%, motivating the adjacent row-20 reduction with the more robust optimizer geometry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1542, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,542-parameter design to second-head value row 21, while retaining dense-coordinate AdamW moments for key row 15 and value rows 20–21, will produce a 1,541-parameter model with at least 99% accuracy.
change: Reproduce every gauge and optimizer from the 1,542-parameter reference, add the adjacent second-head value-row gauge at row 21, and train all optimization-sensitive QKV quotients through projected eight-coordinate AdamW updates.
mechanism: Dense-AdamW second-head value-row quotient extension
evidence_used: The first second-head value-row extension reached 100% accuracy at 1,542 parameters when trained with recovered dense-coordinate moments; row 21 is the nearest untested continuation of that successful pattern.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9828, "parameters": 1541, "training_steps": 4999}

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
