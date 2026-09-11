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
verified_results: {"accuracy": 0.9994, "parameters": 1604, "training_steps": 4999}
prior_hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, and 6 will reduce the model from 1,605 to 1,604 learned parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9975, "parameters": 1601, "training_steps": 4999}
prior_hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, 8, and 9 will produce a 1,601-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1591, "training_steps": 4999}
prior_hypothesis: Removing the redundant learned `ln2` scale and gauge-fixing every `fc1` row will produce a 1,591-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-fixing the final input-weight coordinate in the first four `fc1` rows will produce a 1,606-parameter model with at least 99% accuracy, because the verified three-row implementation achieved 99.97% and each additional row has the same normalized-input null direction.
change: Anchor one scale coordinate in both pre-sublayer LayerNorms and replace `fc1` with a linear layer that omits one coefficient from each of its first four rows while preserving equivalent initialization.
mechanism: Fourth-row LayerNorm-nullspace weight gauge fixing
evidence_used: The 1,607-parameter three-row gauge-fixed design achieved 99.97% accuracy; extending the same row-local reduction by one row is the smallest directly supported next probe.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9777, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one common-mode coordinate from the first positional embedding will reduce the model to 1,606 parameters while retaining at least 99% accuracy, because per-token pre-LayerNorms and the final LayerNorm make uniform hidden-coordinate shifts functionally invisible.
change: Replace the positional embedding with an otherwise equivalent learned embedding that omits the final coordinate of its first row, reconstructs it as zero, and gauge-adjusts initialization to preserve the initial model function and RNG stream.
mechanism: Positional-embedding common-mode gauge fixing
evidence_used: The three-row `fc1` gauge reached 99.97% at 1,607 parameters, while extending it to a fourth row fell to 97.77%; this tests an orthogonal exact null direction instead of further constraining the sensitive MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8989, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing rows 0, 1, 2, and 4 of `fc1` will produce a 1,606-parameter model with at least 99% accuracy, testing whether the failed contiguous fourth-row reduction was specific to row 3 rather than a four-row capacity boundary.
change: Generalize `LayerNormGaugedLinear` to omit the final coefficient from four selected rows, reconstruct those coefficients as zero, and gauge-adjust the same rows during initialization.
mechanism: Noncontiguous LayerNorm-nullspace weight gauge fixing
evidence_used: Gauging rows 0–2 achieved 99.97% at 1,607 parameters, while adding contiguous row 3 fell to 97.77%; substituting a different fourth row is the smallest direct test of row-specific optimization sensitivity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, and 5 will produce a 1,605-parameter model with at least 99% accuracy, because rows 0, 1, 2, and 4 already achieved 99.95% while the failed contiguous experiment specifically implicated row 3.
change: Anchor one scale coordinate in both pre-sublayer LayerNorms and replace `fc1` with a linear layer that omits the final coefficient from five selected rows, reconstructing those coefficients as zero and gauge-adjusting initialization.
mechanism: Fifth noncontiguous LayerNorm-nullspace weight gauge
evidence_used: The noncontiguous four-row design reached 99.95% at 1,606 parameters, whereas adding row 3 to rows 0–2 reached only 97.77%; extending the successful set with row 5 is the smallest probe that avoids the identified sensitive row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, and 6 will reduce the model from 1,605 to 1,604 learned parameters while retaining at least 99% accuracy.
change: Extend the successful noncontiguous gauge set with row 6, omitting and reconstructing its final input coefficient through the existing learned linear reparameterization.
mechanism: Sixth noncontiguous LayerNorm-nullspace weight gauge
evidence_used: The current five-row set achieved 99.83% accuracy at 1,605 parameters, while the principal failed extension involved row 3; adding adjacent row 6 is the smallest informative reduction that preserves the successful exclusion of row 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, and 7 will produce a 1,603-parameter model with at least 99% accuracy.
change: Extend the successful six-row gauge set with row 7, omitting its final input coefficient and reconstructing it as zero through the existing reparameterization.
mechanism: Seventh noncontiguous LayerNorm-nullspace weight gauge
evidence_used: Gauging rows 0, 1, 2, 4, 5, and 6 achieved 99.94% accuracy with 1,604 parameters; adding row 7 is the smallest informative reduction while continuing to exclude sensitive row 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, and 8 will produce a 1,602-parameter model with at least 99% accuracy.
change: Extend the verified noncontiguous gauge set with row 8, omitting its final input coefficient and reconstructing it as zero while preserving gauge-equivalent initialization.
mechanism: Eighth noncontiguous LayerNorm-nullspace weight gauge
evidence_used: Gauging rows 0, 1, 2, 4, 5, 6, and 7 achieved 99.94% accuracy at 1,603 parameters; adding row 8 is the smallest incremental reduction that continues to avoid sensitive row 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, 8, and 9 will produce a 1,601-parameter model with at least 99% accuracy.
change: Extend the verified eight-row gauge set with row 9, omitting its final input coefficient and reconstructing it as zero through the existing learned linear reparameterization.
mechanism: Ninth noncontiguous LayerNorm-nullspace weight gauge
evidence_used: The current eight-row design achieved 99.99% accuracy with 1,602 parameters; adding row 9 is the smallest incremental reduction and continues to avoid row 3, whose inclusion previously reduced accuracy to 97.77%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, 8, 9, and 10 will produce a 1,600-parameter model with at least 99% accuracy.
change: Extend the current gauge set through the verified rows 8 and 9, then omit and reconstruct the final input coefficient of row 10 using the existing gauge-equivalent reparameterization.
mechanism: Tenth noncontiguous LayerNorm-nullspace weight gauge
evidence_used: The nine-row design achieved 99.75% accuracy with 1,601 parameters; adding row 10 is the smallest incremental reduction that continues to avoid row 3, whose inclusion previously reduced accuracy to 97.77%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1` row 11 in addition to rows 0, 1, 2, and 4–10 will produce a 1,599-parameter model with at least 99% accuracy.
change: Extend the verified noncontiguous gauge set through rows 7–11, omitting each selected row’s final input coefficient while continuing to leave sensitive row 3 ungauged.
mechanism: Eleventh noncontiguous LayerNorm-nullspace weight gauge
evidence_used: The 1,600-parameter design gauging rows 0, 1, 2, and 4–10 achieved 99.91% accuracy; adding row 11 is the smallest untested reduction that preserves the empirically important exclusion of row 3.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9840000000000001, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the redundant learned `ln2` scale and gauge-fixing every `fc1` row will produce a 1,591-parameter model with at least 99% accuracy.
change: Make the MLP’s pre-normalization parameter-free and extend `fc1` gauge fixing to all 12 rows, preserving the initial function while making the common-coefficient null direction exact throughout training.
mechanism: Affine-free MLP normalization with exact all-row nullspace gauge
evidence_used: The 1,600-parameter model achieved 99.91%, but separately gauging row 11 fell to 98.40% and earlier gauging row 3 fell to 97.77%; these failures suggest interaction with the learned `ln2` scale, which prevents the current row-wise gauge from remaining an exact null direction after initialization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing `ln1`’s redundant learned scale and gauge-fixing every QKV row will produce a 1,560-parameter model with at least 99% accuracy.
change: Make both pre-sublayer normalizations affine-free, extend `fc1` gauge fixing to all rows, and omit one LayerNorm-null coefficient from every learned QKV projection row.
mechanism: Affine-free attention normalization with exact QKV row gauges
evidence_used: The analogous affine-free `ln2` plus all-row `fc1` gauge achieved 99.96% at 1,591 parameters; applying the same exact normalized-input nullspace argument to `ln1` and QKV is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7304999999999999, "parameters": 1560, "training_steps": 4999}



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
