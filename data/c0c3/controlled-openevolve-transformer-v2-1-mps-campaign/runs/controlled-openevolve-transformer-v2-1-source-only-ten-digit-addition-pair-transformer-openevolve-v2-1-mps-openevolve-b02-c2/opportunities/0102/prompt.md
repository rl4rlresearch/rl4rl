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
verified_results: {"accuracy": 0.9942, "parameters": 1594, "training_steps": 4999}
prior_hypothesis: Helmert-parameterizing one `fc2` weight column will reduce the verified 1,595-parameter model to 1,594 parameters while retaining at least 99% accuracy, because the removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1602, "training_steps": 4999}
prior_hypothesis: Helmert-parameterizing a second query row in the second attention head will reduce the verified model from 1,603 to 1,602 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component is absorbable by that row’s independent learned query bias.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9954999999999999, "parameters": 1598, "training_steps": 4999}
prior_hypothesis: Helmert-parameterizing the penultimate value row will reduce the verified model from 1,599 to 1,598 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component produces only a position-independent attention output that the learned projection-offset subspace can absorb.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1593, "training_steps": 4999}
prior_hypothesis: Reproducing the verified 1,594-parameter design and Helmert-parameterizing a second `fc2` weight column will yield 1,593 parameters with at least 99% accuracy, because each removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Helmert-parameterizing a third value row will reduce the qualified 1,598-parameter design to 1,597 parameters while retaining at least 99% accuracy.
change: Extend `CompactQKV` to reconstruct the final three value rows from independent seven-dimensional zero-mean coordinates.
mechanism: Third downstream-projection-absorbed value-row LayerNorm gauge
evidence_used: Compacting one value row achieved 99.97% accuracy at 1,599 parameters, and compacting the adjacent second row achieved 99.55% at 1,598; extending the same successful quotient to the next adjacent row is the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9892, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,598-parameter design by Helmert-parameterizing positional row 3 will produce 1,597 parameters with at least 99% accuracy, because a position-local uniform residual shift is removed by every downstream LayerNorm without changing learned attention or logits.
change: Reproduce the qualified four-query/two-value QKV compaction, then center positional rows 1–3 instead of rows 1–2.
mechanism: Residual-uniform positional-row gauge quotient
evidence_used: The two-value-row reference achieved 99.55% at 1,598 parameters, while a third value-row quotient fell to 98.92%; the current verified designs already compact positional rows 1 and 2, motivating extension along that independent exact gauge instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,597-parameter design and Helmert-parameterizing positional row 4 will yield 1,596 parameters with at least 99% accuracy, because its position-local uniform residual shift is removed by downstream LayerNorms.
change: Apply the qualified four-query/two-value QKV compaction and center positional rows 1–4 instead of rows 1–2.
mechanism: Residual-uniform positional-row gauge quotient
evidence_used: Centering positional row 3 produced 99.82% accuracy at 1,597 parameters, while compacting a third value row fell to 98.92%; extending the successful exact positional gauge is the strongest next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9725, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a sixth token/position translation coordinate will reduce the verified 1,597-parameter model to 1,596 parameters while retaining at least 99% accuracy, because the input embeddings remain unchanged and the tied output embedding changes logits only by a vocabulary-uniform offset.
change: Extend `CompactPositionEmbedding` and its initialization from five to six fixed translation coordinates while preserving the centered positional-origin tail and positional rows 1–3.
mechanism: Tied-embedding token/position translation gauge quotient
evidence_used: The current design achieves 99.82% accuracy with five coordinatewise translation gauges at 1,597 parameters, whereas extending positional-row centering to row 4 fell to 97.25%; testing another instance of the already-used exact translation gauge is therefore the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.991, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,596-parameter design and fixing a seventh token/position translation coordinate will yield 1,595 parameters with at least 99% accuracy, because inputs remain unchanged while tied-output logits change only by a vocabulary-uniform offset.
change: Extend the compact position embedding to seven fixed translation coordinates and reproduce centering of positional rows 1–3.
mechanism: Seventh tied-embedding token/position translation gauge quotient
evidence_used: Six fixed translations with positional rows 1–3 achieved 99.10% accuracy at 1,596 parameters, while centering positional row 4 failed; extending the already-qualified translation gauge is the most directly supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing the qualified 1,596-parameter design and fixing the seventh token/position translation without creating a zero-length parameter will yield 1,595 parameters with at least 99% accuracy.
change: Add the qualified two-value-row and third positional-row compactions, fix seven translation coordinates, and reconstruct the remaining centered one-coordinate positional origin as a constant zero tensor.
mechanism: Parameter-free singleton positional-origin gauge
evidence_used: Six fixed translations with three centered positional rows and two compact value rows achieved 99.10% accuracy at 1,596 parameters; the seventh translation is the closest exact gauge extension, while its prior implementation produced no accuracy result.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,595-parameter design and Helmert-parameterizing a second `fc1` row will yield 1,594 parameters with at least 99% accuracy, because each row’s independent bias can absorb the removed LayerNorm-uniform weight component.
change: Apply the qualified seven-translation, three-position-row, four-query-row, and two-value-row compactions, then reconstruct the first two MLP input rows from independent seven-dimensional zero-mean coordinates.
mechanism: Second independently biased MLP LayerNorm-row gauge quotient
evidence_used: The 1,595-parameter reference achieved 99.97% accuracy while already applying this quotient to one independently biased `fc1` row; extending the same local parameterization to the adjacent independently biased row is more directly supported than the failed third-value-row or fourth-position-row reductions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3728, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Helmert-parameterizing one `fc2` weight column will reduce the verified 1,595-parameter model to 1,594 parameters while retaining at least 99% accuracy, because the removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.
change: Extend `CompactResidualLinear` to reconstruct its first weight column from seven zero-mean coordinates while retaining the other columns and the qualified compact bias.
mechanism: Activation-scaled residual-uniform MLP output gauge quotient
evidence_used: The current 1,595-parameter design achieved 99.97% accuracy and already removes the residual-uniform direction from `fc2` bias; applying the same exact output-space quotient to one activation-scaled weight column is more directly supported than repeating the second `fc1` row compaction that fell to 37.28%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9942, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,594-parameter design and Helmert-parameterizing a second `fc2` weight column will yield 1,593 parameters with at least 99% accuracy, because each removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.
change: Fix the qualified seventh token/position translation and reconstruct the first two `fc2` weight columns from independent seven-dimensional zero-mean coordinates.
mechanism: Second activation-scaled residual-uniform MLP output gauge quotient
evidence_used: Seven translations plus one compact `fc2` column achieved 99.42% accuracy at 1,594 parameters; applying the same exact output-space quotient to the adjacent column is more directly supported than the failed second-`fc1`, third-value-row, or fourth-position-row reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,593-parameter design and Helmert-parameterizing a third `fc2` weight column will yield 1,592 parameters with at least 99% accuracy, because each removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.
change: Apply the qualified seven-translation and three-position-row embedding compactions, then reconstruct the first three `fc2` weight columns from independent seven-dimensional zero-mean coordinates.
mechanism: Third activation-scaled residual-uniform MLP output gauge quotient
evidence_used: The verified two-column quotient achieved 99.96% accuracy at 1,593 parameters; extending the same exact output-space gauge to the adjacent third column is more directly supported than the failed third-value-row, fourth-position-row, or second-`fc1` reductions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.508, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,593-parameter design and Helmert-parameterizing the first attention output-projection column will yield 1,592 parameters with at least 99% accuracy, because its removed activation-scaled uniform residual component is eliminated by downstream LayerNorm.
change: Apply the qualified seven-translation, three-position-row, four-query-row, two-value-row, and two-`fc2`-column compactions, then reconstruct the first attention projection column from seven zero-mean coordinates.
mechanism: Attention-projection residual-uniform column quotient
evidence_used: The two-`fc2`-column residual-uniform quotient achieved 99.96% accuracy at 1,593 parameters. Since extending it to an adjacent third MLP column fell to 50.8%, applying the same exact output-space quotient in the attention projection tests an independent branch without further constraining the MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5284, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing one coordinate of the first head’s zero-bias query-row pair, with the matching orthogonal key-row rotation, will reduce the verified 1,593-parameter model to 1,592 parameters while retaining at least 99% accuracy because it preserves the initialized attention score matrix exactly.
change: Givens-rotate query rows 2–3 and matching key rows 2–3, omit the resulting zero query coordinate, and reconstruct that coordinate as zero during forward passes.
mechanism: Zero-bias attention Q/K rotation gauge
evidence_used: The current two-column `fc2` quotient achieved 99.96% at 1,593 parameters, while a third `fc2` column and an attention-projection column both failed; an independent internal Q/K factorization symmetry is therefore the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9470000000000001, "parameters": 1592, "training_steps": 4999}



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
