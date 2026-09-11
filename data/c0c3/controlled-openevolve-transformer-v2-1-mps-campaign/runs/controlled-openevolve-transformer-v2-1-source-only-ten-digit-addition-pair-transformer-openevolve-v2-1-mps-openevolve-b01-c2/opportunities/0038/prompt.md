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
verified_results: {"accuracy": 0.9995, "parameters": 1590, "training_steps": 4999}
prior_hypothesis: Starting from the verified 1,591-parameter design, fixing one additional `ln1` scale coordinate will produce a 1,590-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1588, "training_steps": 4999}
prior_hypothesis: Fixing a fourth `ln1` scale coordinate in the verified 1,589-parameter design will produce a 1,588-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9975, "parameters": 1601, "training_steps": 4999}
prior_hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, 8, and 9 will produce a 1,601-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1591, "training_steps": 4999}
prior_hypothesis: Removing the redundant learned `ln2` scale and gauge-fixing every `fc1` row will produce a 1,591-parameter model with at least 99% accuracy.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Combining the verified affine-free `ln2` and all-row `fc1` gauge with removal of `fc2`’s exact common-mode bias direction will produce a 1,590-parameter model with at least 99% accuracy.
change: Reproduce the verified 1,591-parameter design, then replace `fc2` with an equivalent learned linear layer whose final bias coordinate is fixed at zero.
mechanism: Residual-stream common-bias gauge fixing
evidence_used: The affine-free `ln2` plus all-row `fc1` design achieved 99.96% at 1,591 parameters, while the broad 1,560-parameter `ln1`/QKV reduction failed; this tests one orthogonal null direction whose zero-bias initialization leaves the initial function unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5282, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only `ln1`’s seven learned scales while retaining the full QKV projections will produce a 1,584-parameter model with at least 99% accuracy.
change: Replace the anchored affine `ln1` with parameter-free LayerNorm, leaving QKV weights unconstrained so they can absorb the removed feature scales.
mechanism: Affine-free attention pre-normalization
evidence_used: The 1,591-parameter design reached 99.96% after the analogous `ln2` scale removal, while the 1,560-parameter `ln1` experiment also gauged all 24 QKV rows and fell to 73.05%; isolating `ln1` scale removal tests which part caused that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8393999999999999, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,591-parameter design by gauge-fixing the all-ones feature direction in every positional embedding row will remove `max_seq_len` additional parameters while retaining at least 99% accuracy.
change: Make `ln2` affine-free, gauge every `fc1` row as in Reference Design 3, and replace the positional embedding with a learned embedding that omits one exact common-shift coordinate per position.
mechanism: Per-position residual-stream shift gauge fixing
evidence_used: Reference Design 3 reached 99.96% with 1,591 parameters after making all `fc1` gauges exact; unlike the failed learned-`ln1` QKV gauges, a positional row’s common feature shift is canceled throughout the zero-dropout pre-norm residual network and by the final LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3186, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,591-parameter design, fixing one additional `ln1` scale coordinate will produce a 1,590-parameter model with at least 99% accuracy.
change: Make `ln2` affine-free, gauge-fix every `fc1` row, and retain six learned `ln1` scales while fixing its final two scale coordinates to one.
mechanism: Single-coordinate attention LayerNorm scale ablation
evidence_used: The affine-free `ln2` with all-row `fc1` gauge reached 99.96% at 1,591 parameters, whereas removing all seven remaining `ln1` scales fell to 83.94%; removing one scale is the smallest incremental probe of that optimization-sensitive redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln1` scale coordinate in the verified 1,590-parameter design will produce a 1,589-parameter model with at least 99% accuracy.
change: Reproduce the affine-free `ln2`, all-row `fc1` gauge, reduced QKV bias, and anchor three of eight `ln1` scales to one.
mechanism: Incremental attention LayerNorm scale anchoring
evidence_used: Fixing two `ln1` scale coordinates achieved 99.95% accuracy at 1,590 parameters, while removing all remaining scales failed; a single additional anchored coordinate is the smallest informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fourth `ln1` scale coordinate in the verified 1,589-parameter design will produce a 1,588-parameter model with at least 99% accuracy.
change: Retain four learned `ln1` scales and anchor the remaining four coordinates to one.
mechanism: Incremental attention LayerNorm scale anchoring
evidence_used: Anchoring three `ln1` scales achieved 99.73% accuracy with 1,589 parameters; anchoring one additional coordinate is the smallest incremental reduction supported by the successful two- and three-coordinate results.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1588, "training_steps": 4999}



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
