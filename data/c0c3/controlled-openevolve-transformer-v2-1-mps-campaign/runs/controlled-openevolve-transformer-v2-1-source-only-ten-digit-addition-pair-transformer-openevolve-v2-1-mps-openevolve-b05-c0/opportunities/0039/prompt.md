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
verified_results: {"accuracy": 0.9992, "parameters": 1622, "training_steps": 4999}
prior_hypothesis: Fixing value-bias index 3 in addition to indices 0, 1, 2, 4, 6, and 7 will reduce the model from 1623 to 1622 parameters while maintaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the softmax-invisible coordinate from the fourth key row of the first attention head will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, testing whether optimization sensitivity is confined to the previously failed second and third rows.
change: Add the fourth key row of the first head to `GaugeFixedQKV.fixed_rows`; the generalized virtual-coordinate AdamW and gradient-clipping logic will handle the sixth omitted coordinate.
mechanism: Remaining first-head LayerNorm key gauge
evidence_used: Removing all four key coordinates from the second head succeeded at 99.40%, while additions from the second and third rows of the first head failed; its fourth row is the only untested exact key gauge and therefore the smallest informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5202, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second value-bias coordinate to zero will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because value bias is functionally absorbable into the attention output bias and the successful current model already fixes one such coordinate.
change: Store six rather than seven value-bias coordinates and reconstruct the final two coordinates as zero.
mechanism: Incremental value/output-bias quotient
evidence_used: The 1628-parameter model achieved 99.40% while fixing one value-bias coordinate, whereas every attempted additional first-head key gauge failed; titrating the distinct value/output-bias redundancy is therefore the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third value-bias coordinate to zero will reduce the model from 1627 to 1626 parameters while retaining at least 99% accuracy, because a position-independent value bias is absorbable into the attention output bias and the first two removals both trained successfully.
change: Store five rather than six value-bias coordinates and reconstruct the final three coordinates as zero.
mechanism: Third incremental value/output-bias quotient
evidence_used: Fixing the second value-bias coordinate achieved 99.92% accuracy at 1627 parameters, while further first-head key gauges repeatedly failed; continuing the successful value-bias mechanism by one coordinate is the smallest informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0406, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final value-bias coordinate of the first attention head, while retaining the two successful fixed coordinates in the second head, will reduce the model from 1627 to 1626 parameters while maintaining at least 99% accuracy.
change: Store five value-bias coordinates and reconstruct an eight-dimensional bias with zeros at index 3 of the first head and indices 6–7 of the second head.
mechanism: Cross-head value/output-bias quotient
evidence_used: Fixing two second-head value-bias coordinates reached 99.92% at 1627 parameters, while fixing a third adjacent coordinate in that head collapsed to 4.06%; distributing exact gauges across heads previously succeeded for key projections where a same-head extension failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3065, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 4 while retaining the successful fixed indices 6–7 will reduce the model from 1627 to 1626 parameters and maintain at least 99% accuracy, indicating that the failed index-5 removal was coordinate-specific rather than a three-gauge limit.
change: Store five value-bias coordinates and reconstruct an eight-dimensional bias with zeros at indices 4, 6, and 7.
mechanism: Nonadjacent second-head value/output-bias quotient
evidence_used: Fixing indices 6–7 achieved 99.92% at 1627 parameters, while additionally fixing adjacent index 5 collapsed to 4.06%; testing nonadjacent index 4 is the smallest remaining titration within the otherwise successful second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9420000000000001, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 0 while retaining the successful fixed indices 6–7 will reduce the model from 1627 to 1626 parameters and maintain at least 99% accuracy, testing whether prior failures were coordinate-specific.
change: Store five value-bias coordinates and reconstruct the eight-dimensional bias with zeros at indices 0, 6, and 7.
mechanism: First-coordinate value/output-bias quotient
evidence_used: Fixing indices 6–7 achieved 99.92% at 1627 parameters, while additional removals at indices 3, 4, and 5 produced substantially different accuracies; this coordinate sensitivity makes the untested index 0 the smallest informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9692000000000001, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 1 while retaining fixed indices 6–7 will reduce the model from 1627 to 1626 parameters and maintain at least 99% accuracy, testing the remaining coordinate nearest index 0, whose 96.92% result was the strongest unsuccessful third-bias removal.
change: Store five value-bias coordinates and reconstruct the eight-dimensional bias with zeros at indices 1, 6, and 7.
mechanism: Untested near-edge value/output-bias quotient
evidence_used: Fixing indices 6–7 achieved 99.92% at 1627 parameters; additional removals at indices 0, 3, 4, and 5 achieved 96.92%, 30.65%, 94.20%, and 4.06%, respectively, making untested index 1 the most informative neighboring coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 2 in addition to the successful fixed indices 1, 6, and 7 will reduce the model from 1626 to 1625 parameters while maintaining at least 99% accuracy.
change: Store four value-bias coordinates and reconstruct the eight-dimensional bias with zeros at indices 1, 2, 6, and 7.
mechanism: Adjacent value/output-bias quotient extension
evidence_used: Fixing index 1 alongside indices 6–7 achieved 99.98% accuracy at 1626 parameters; index 2 is the only coordinate not previously tested as an additional value-bias gauge, making it the most informative one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 0 in addition to indices 1, 2, 6, and 7 will reduce the model from 1625 to 1624 parameters while maintaining at least 99% accuracy.
change: Store three value-bias coordinates and reconstruct the eight-dimensional bias with zeros at indices 0, 1, 2, 6, and 7.
mechanism: Fifth value/output-bias quotient coordinate
evidence_used: The current indices 1, 2, 6, and 7 reached 99.94% at 1625 parameters; among remaining coordinates, index 0 had the strongest prior unsuccessful result at 96.92%, making it the most informative fifth-gauge candidate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 4 in addition to indices 0, 1, 2, 6, and 7 will reduce the model from 1624 to 1623 parameters while maintaining at least 99% accuracy.
change: Store only value-bias coordinates 3 and 5, reconstructing the eight-dimensional bias with zeros at indices 0, 1, 2, 4, 6, and 7.
mechanism: Sixth value/output-bias quotient coordinate
evidence_used: Among the remaining coordinates, index 4 produced the strongest prior unsuccessful removal at 94.20%, compared with 30.65% for index 3 and 4.06% for index 5; later successful removals of indices 1, 2, and 0 may provide the optimization geometry needed for this exact gauge to train successfully.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 3 in addition to indices 0, 1, 2, 4, 6, and 7 will reduce the model from 1623 to 1622 parameters while maintaining at least 99% accuracy.
change: Store only value-bias coordinate 5 and reconstruct the eight-dimensional bias with zeros at every other index.
mechanism: Seventh value/output-bias quotient coordinate
evidence_used: Index 3 is the strongest remaining candidate: its prior removal reached 30.65%, versus 4.06% for index 5, and the successful index-4 removal showed that later quotient reductions can improve optimization enough for a previously unsuccessful coordinate to train.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the MLP output bias will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because an all-coordinate bias shift survives only in the residual stream and is removed by subsequent LayerNorms.
change: Store seven MLP output-bias coordinates, reconstruct the eighth as zero during the learned projection, and train the reduced bias with the existing virtual-coordinate AdamW and gradient-clipping logic.
mechanism: Post-MLP residual common-shift quotient
evidence_used: The 1622-parameter design reached 99.92%, while additional first-head key gauges repeatedly failed and the remaining value-bias coordinate was historically the weakest removal candidate; the current successful attention projection-bias quotient already uses the same LayerNorm-null common-shift mechanism and optimizer treatment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.078, "parameters": 1621, "training_steps": 4999}



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
