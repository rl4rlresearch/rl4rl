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
verified_results: {"accuracy": 0.9983, "parameters": 1605, "training_steps": 4999}
prior_hypothesis: Gauge-fixing `fc1` rows 0, 1, 2, 4, and 5 will produce a 1,605-parameter model with at least 99% accuracy, because rows 0, 1, 2, and 4 already achieved 99.95% while the failed contiguous experiment specifically implicated row 3.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 1607, "training_steps": 4999}
prior_hypothesis: Gauge-fixing the final input weight in each of the first three `fc1` rows will reduce the model from 1,610 to 1,607 learned parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1606, "training_steps": 4999}
prior_hypothesis: Gauge-fixing rows 0, 1, 2, and 4 of `fc1` will produce a 1,606-parameter model with at least 99% accuracy, testing whether the failed contiguous fourth-row reduction was specific to row 3 rather than a four-row capacity boundary.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a second `ln2` scale coordinate at one will reduce the model from 1,611 to 1,610 learned parameters while retaining at least 99% accuracy, because the successful first anchor achieved 100% accuracy and the following MLP input weights can absorb both fixed coordinate scales.
change: Retain six learned `ln2` scales and synthesize two fixed unit scales.
mechanism: Second-coordinate LayerNorm scale gauge fixing
evidence_used: The 1,611-parameter model with one fixed `ln2` scale coordinate achieved 100% accuracy, while removing all normalization scales failed; extending the successful reduction by exactly one coordinate is the most conservative capacity probe.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7340000000000001, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified single-coordinate `ln2` scale anchor with a fixed zero coordinate in the attention output bias will produce a 1,610-parameter model with at least 99% accuracy after 5,000 steps.
change: Retain seven learned `ln2` scales, retain seven learned attention projection-bias coordinates, and synthesize fixed unit-scale and zero-bias coordinates during forward passes.
mechanism: Residual common-mode bias gauge fixing
evidence_used: The one-coordinate `ln2` anchor achieved 100% accuracy with 1,611 parameters, while a second scale anchor fell to 73.4%; removing an orthogonal additive gauge is therefore better motivated. A uniform shift in the attention projection bias is invisible to `ln2` and can be absorbed exactly by the MLP output bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9777, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one scale coordinate in each pre-sublayer LayerNorm will produce a 1,610-parameter model with at least 99% accuracy, because each normalization retains seven adaptive scales and its downstream learned projection can absorb the fixed coordinate.
change: Replace both biased LayerNorms with bias-free LayerNorms that learn seven scale coordinates and synthesize one fixed unit scale.
mechanism: Distributed single-coordinate LayerNorm scale anchoring
evidence_used: A single anchored `ln2` coordinate achieved 100% accuracy with 1,611 parameters, while anchoring a second `ln2` coordinate failed at 73.4%; distributing the second anchor to `ln1` preserves seven adaptive coordinates in both pathways. Removing all `ln1` scales failed, motivating this one-coordinate reduction instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the MLP output bias at zero will reduce the verified 1,610-parameter model to 1,609 parameters while retaining at least 99% accuracy, because any uniform component of that bias is erased by the final LayerNorm and the remaining seven coordinates still span all functionally relevant bias differences.
change: Retain seven learned `fc2` bias coordinates, synthesize one fixed zero coordinate, and preserve the existing linear weight shape and initialization path.
mechanism: Final-residual common-mode bias gauge fixing
evidence_used: The distributed LayerNorm anchoring design achieved 99.97% accuracy with 1,610 parameters, while removing a second normalization scale caused a large failure; this motivates removing one exact additive gauge from the final residual branch instead of further reducing normalization capacity. The earlier attention-output bias anchor reached 97.77%, so targeting the later MLP output bias avoids perturbing activations consumed by another learned sublayer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6988, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified distributed LayerNorm anchors with one fixed-zero query-bias coordinate will yield a 1,609-parameter model with at least 99% accuracy, because an invertible query/key basis change within the affected attention head can eliminate that bias coordinate without changing attention logits.
change: Anchor one scale coordinate in `ln1` as in the qualified 1,610-parameter design, then retain seven learned query-bias coordinates and synthesize the eighth as zero.
mechanism: Single-coordinate query-bias head-basis gauge fixing
evidence_used: Distributed LayerNorm anchoring achieved 99.97% with 1,610 parameters. The failed 1,609-parameter `fc2` bias anchor targeted the residual pathway; this instead removes an independently redundant attention-head basis coordinate while preserving its zero-initialized value.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6092, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` input-weight coordinate from the verified distributed-anchor design will yield 1,609 parameters and at least 99% accuracy, because bias-free LayerNorm outputs have zero mean and the omitted coefficient is exactly absorbable by the other coefficients in that output row; gauge-adjusted initialization preserves the original model’s initial function and RNG stream.
change: Anchor one scale coordinate in each pre-sublayer LayerNorm, then replace `fc1` with a linear layer that learns all but one weight, synthesizes the omitted weight as zero, and initializes the retained row coefficients to an equivalent gauge.
mechanism: LayerNorm-nullspace weight gauge fixing
evidence_used: The distributed LayerNorm-anchor design achieved 99.97% with 1,610 parameters. Both prior 1,609-parameter attempts altered sensitive additive pathways and failed, motivating a different one-parameter reduction based on the exact zero-mean nullspace immediately downstream of `ln2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the redundant final `fc1` input-weight coordinate from a second output row will reduce the model to 1,608 parameters while retaining at least 99% accuracy, because every row independently acts on the same zero-mean normalized vector and the first-row gauge achieved 99.97%.
change: Gauge-fix the final input weight in each of the first two `fc1` rows, synthesizing both as zero and gauge-adjusting both rows during initialization.
mechanism: Second-row LayerNorm-nullspace weight gauge fixing
evidence_used: The verified 1,609-parameter design successfully removed one `fc1` coordinate with 99.97% accuracy; extending that exact row-local nullspace reduction by one row is the smallest and most directly supported next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948999999999999, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the final input weight in each of the first three `fc1` rows will reduce the model from 1,610 to 1,607 learned parameters while retaining at least 99% accuracy.
change: Replace `fc1` with a linear layer that omits one coefficient from each of its first three rows, reconstructs those coefficients as zero, and gauge-adjusts initialization to preserve the initial function.
mechanism: Third-row LayerNorm-nullspace weight gauge fixing
evidence_used: Removing one `fc1` coordinate achieved 99.97% at 1,609 parameters, and extending the same row-local reduction to two rows achieved 99.49% at 1,608; a third row is the most direct incremental capacity-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1607, "training_steps": 4999}

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
