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
verified_results: {"accuracy": 0.9962000000000001, "parameters": 1540, "training_steps": 4999}
prior_hypothesis: Extending the qualified two-coordinate `ln1` scale anchor to three coordinates will produce a 1540-parameter model with at least 99% accuracy, because the third fixed scale can be absorbed into the corresponding unrestricted `qkv` input column while five learned scales remain.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9956999999999999, "parameters": 1536, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1537-parameter design with an independent query-key rotation in the second attention head will produce a 1536-parameter model with at least 99% accuracy, because the rotation preserves attention scores while fixing one additional query coefficient.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9976, "parameters": 1542, "training_steps": 4999}
prior_hypothesis: Fixing one `ln1` scale coordinate at its initialized value will produce a 1542-parameter model with at least 99% accuracy, because that scale can be absorbed into the corresponding unrestricted `qkv` input column while retaining seven learned normalization scales.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1537, "training_steps": 4999}
prior_hypothesis: Fixing one query-weight coefficient through a shared orthogonal rotation of two query/key channels will produce a 1537-parameter model with at least 99% accuracy because it preserves initialized attention scores and removes only an exact attention-basis symmetry.

## Recent verification evidence

RECENT RESULT
hypothesis: Applying the qualified 1555-parameter design and removing `ln1`’s eight scale parameters will produce a 1547-parameter model with at least 99% accuracy, because the scale can be absorbed exactly into the unconstrained input columns of `qkv`.
change: Fold the value bias into a full mean-zero attention projection bias, make both block LayerNorms affine-free, and retain the learned query bias.
mechanism: Pre-attention LayerNorm scale folding
evidence_used: The affine-free `ln2` design achieved 99.89% accuracy with 1555 parameters; `ln1` has the same learned-linear successor structure, while the qualified bias-free `ln1` result shows its additive shift is already unnecessary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7125, "parameters": 1547, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified 1555-parameter design plus a mean-zero input parameterization for `fc1` will achieve at least 99% accuracy with 1543 parameters, because affine-free `ln2` always supplies zero-mean features, making one common-mode weight per `fc1` output exactly unobservable.
change: Reproduce the qualified three-gauge, folded-attention-bias, bias-free `ln1`, affine-free `ln2` design, then represent each `fc1` weight row in the seven-dimensional mean-zero basis while preserving full-size initialization draws.
mechanism: Null-space reduction of the post-normalization MLP input map
evidence_used: The affine-free-`ln2` design achieved 99.89% at 1555 parameters. Unlike the failed removal of active `ln1` scales at 1547 parameters, this removes only the 12 `fc1` directions annihilated by affine-free LayerNorm outputs.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1543, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` scale coordinate at its initialized value will produce a 1542-parameter model with at least 99% accuracy, because that scale can be absorbed into the corresponding unrestricted `qkv` input column while retaining seven learned normalization scales.
change: Replace `ln1` with a LayerNorm whose final scale is fixed to one and whose other seven scales remain trainable, preserving the freshly initialized function.
mechanism: Single-coordinate pre-attention scale-gauge anchoring
evidence_used: The current 1543-parameter model achieved 99.83%; removing all eight `ln1` scales collapsed to 71.25%, motivating an incremental one-coordinate gauge reduction instead of repeating the all-at-once restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976, "parameters": 1542, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the qualified mean-zero `fc1` parameterization with two fixed `ln1` scale coordinates will produce a 1541-parameter model with at least 99% accuracy, because the second fixed scale can be absorbed into its unrestricted `qkv` input column while six learned scales remain.
change: Reproduce the qualified `fc1` null-space reduction and extend the successful single-coordinate `ln1` scale anchor to the final two coordinates.
mechanism: Second-coordinate pre-attention scale-gauge anchoring
evidence_used: Mean-zero `fc1` achieved 99.83% at 1543 parameters, and fixing one `ln1` scale retained 99.76% at 1542; this supports testing one incremental scale anchor, while the 71.25% result from removing all eight scales cautions against a larger jump.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1541, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-coordinate `ln1` scale anchor to three coordinates will produce a 1540-parameter model with at least 99% accuracy, because the third fixed scale can be absorbed into the corresponding unrestricted `qkv` input column while five learned scales remain.
change: Reproduce the qualified mean-zero `fc1` and affine-free `ln2` reductions, then fix the final three `ln1` scale coordinates at one.
mechanism: Third-coordinate pre-attention scale-gauge anchoring
evidence_used: The two-coordinate anchor achieved 99.82% accuracy with 1541 parameters, supporting one incremental anchor; removing all eight `ln1` scales collapsed to 71.25%, so this retains most learned scales.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9962000000000001, "parameters": 1540, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified three-coordinate `ln1` scale anchor to four coordinates will produce a 1539-parameter model with at least 99% accuracy, because the additional fixed scale can be absorbed into the corresponding unrestricted `qkv` input column while four learned scales remain.
change: Reproduce the qualified 1540-parameter design’s folded attention bias, mean-zero `fc1`, affine-free `ln2`, and three embedding gauges, then fix a fourth `ln1` scale coordinate at one.
mechanism: Fourth-coordinate pre-attention scale-gauge anchoring
evidence_used: Three anchored `ln1` scales achieved 99.62% at 1540 parameters after one and two anchors also qualified; the collapse from removing all eight scales motivates testing one additional coordinate incrementally.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fifth `ln1` scale coordinate at one will produce a 1538-parameter model with at least 99% accuracy, because the scale is absorbable into the corresponding unrestricted `qkv` input column while three learned scales remain.
change: Extend the qualified four-coordinate `ln1` scale anchor by one coordinate without changing initialization or any other model component.
mechanism: Fifth-coordinate pre-attention scale-gauge anchoring
evidence_used: Four anchored scales achieved 99.95% accuracy at 1539 parameters after each of the preceding one-, two-, and three-coordinate anchors also qualified; the failure from removing all eight scales motivates continuing one coordinate at a time.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7304, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the qualified four-coordinate `ln1` anchor with a mean-zero input-row parameterization for `qkv` will produce a 1515-parameter model with at least 99% accuracy, because each QKV output row retains a complete gauge representative on centered LayerNorm features while the learned scales remain available for optimization.
change: Use the qualified four-scale anchor and remove one redundant input-direction parameter from each of the 24 QKV output rows, preserving full-sized initialization draws and the initialized function.
mechanism: LayerNorm-input gauge fixing for QKV weights
evidence_used: Four anchored `ln1` scales achieved 99.95% at 1539 parameters, while the analogous mean-zero `fc1` input reduction removed 12 LayerNorm-null directions and achieved 99.83%; the fifth scale anchor’s 73.04% failure motivates retaining four learned scales and reducing linear-map gauge directions instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4648, "parameters": 1515, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying one mean-zero QKV-row gauge to the qualified 1539-parameter four-anchor design will produce a 1538-parameter model with at least 99% accuracy, because it removes one LayerNorm-null direction while avoiding the optimization disruption caused by constraining all 24 rows.
change: Extend `ln1` from two to four fixed scales and represent only the final QKV weight row in the seven-dimensional mean-zero basis, preserving full-sized initialization draws.
mechanism: Incremental LayerNorm-input gauge fixing for one QKV row
evidence_used: Four anchored scales achieved 99.95% at 1539 parameters, whereas reducing all 24 QKV rows collapsed to 46.48%; this motivates testing the same gauge one row at a time.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining a second QKV weight row to the mean-zero basis will reduce the qualified 1538-parameter design to 1537 parameters while retaining at least 99% accuracy, because it removes one additional LayerNorm-null direction without the optimization disruption observed when all 24 rows were constrained.
change: Reproduce the qualified four-scale `ln1` anchor and extend the successful single-row mean-zero QKV parameterization to exactly two rows, preserving full-sized fresh initialization draws.
mechanism: Incremental second-row LayerNorm-input gauge fixing
evidence_used: The one-row QKV gauge achieved 99.77% accuracy at 1538 parameters, while constraining all 24 rows collapsed to 46.48%; this motivates another one-row incremental reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9645, "parameters": 1537, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-weight coefficient through a shared orthogonal rotation of two query/key channels will produce a 1537-parameter model with at least 99% accuracy because it preserves initialized attention scores and removes only an exact attention-basis symmetry.
change: Retain the qualified single mean-zero QKV row, rotate the first two query channels and their paired key channels so the first query coefficient is zero, and omit that fixed coefficient from the trainable parameterization.
mechanism: Query-key orthogonal gauge fixing
evidence_used: The 1538-parameter single-row design achieved 99.77%, while extending its LayerNorm-input constraint to a second row fell to 96.45%; this tests a distinct exact query-key rotational gauge instead of repeating the disruptive row constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1537, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1537-parameter design with an independent query-key rotation in the second attention head will produce a 1536-parameter model with at least 99% accuracy, because the rotation preserves attention scores while fixing one additional query coefficient.
change: Reproduce the qualified single-row QKV input gauge and first-head rotation, then rotate the second head’s first two query/key channels and omit its fixed query coefficient.
mechanism: Independent per-head query-key orthogonal gauge fixing
evidence_used: The single-row plus first-head rotation design achieved 99.96% accuracy at 1537 parameters; unlike the failed second mean-zero QKV-row constraint, this additional reduction uses a disjoint exact query-key symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956999999999999, "parameters": 1536, "training_steps": 4999}



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
