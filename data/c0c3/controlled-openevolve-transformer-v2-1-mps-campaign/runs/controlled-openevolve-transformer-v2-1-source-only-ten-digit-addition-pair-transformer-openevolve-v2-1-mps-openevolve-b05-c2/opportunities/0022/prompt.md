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
verified_results: {"accuracy": 0.9995, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Fixing a second mean-zero token/position shift coordinate will reduce the qualified 1579-parameter model to 1578 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.998, "parameters": 1577, "training_steps": 4999}
prior_hypothesis: Extending the qualified two-coordinate gauge constraint by one coordinate will produce a 1577-parameter model with at least 99% accuracy while preserving the initialized input function and output probabilities.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9979, "parameters": 1581, "training_steps": 4999}
prior_hypothesis: Restoring the qualified 1582-parameter design’s full value bias while fixing one attention output-projection bias coordinate will retain at least 99% accuracy with 1581 parameters, because the missing mean-zero output bias component can be represented through the learned value bias.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate will reduce the qualified 1583-parameter design to 1582 parameters while retaining at least 99% accuracy, because that scale is exactly absorbable into the corresponding `fc1` input column without constraining the MLP function.
change: Restore all eight learned value-bias coordinates, remove `ln2` bias as in Reference Design 3, and represent its scale with seven learned coordinates plus one fixed unit coordinate.
mechanism: One-coordinate pre-MLP LayerNorm scale anchoring
evidence_used: The full-value-bias, bias-free-`ln2` design achieved 99.96% accuracy with 1583 parameters, whereas the prior 1582-parameter combination failed after removing a sensitive value-bias coordinate; this instead removes an algebraically redundant LayerNorm scale coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7484000000000001, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the tied token/output embeddings across vocabulary will reduce the qualified 1583-parameter design to 1575 parameters while retaining at least 99% accuracy, because the removed vector produces only a common logit shift and its input-side effect is absorbable by positional embeddings up to LayerNorm-invisible common mode.
change: Apply the qualified mean-zero residual parameterization and bias-free `ln2`, then represent tied token/output embeddings in an orthonormal vocabulary-centered basis while preserving the qualified model’s initial function.
mechanism: Vocabulary-common-mode gauge compression
evidence_used: The 1583-parameter mean-zero, bias-free-`ln2` design achieved 99.96%; prior smaller failures altered sensitive value-bias or LayerNorm-scale paths, whereas this patch removes a distinct exact tied-embedding/softmax gauge without reducing attention, MLP, or normalization capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7145999999999999, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the single global all-ones component of the tied token/output embedding will reduce the qualified model from 1583 to 1582 parameters while retaining at least 99% accuracy, because it only shifts residual streams by a LayerNorm-invisible constant and shifts every output logit equally.
change: Replace the tied embedding with a gauge-fixed parameterization in which the final token vector is mean-zero, using the exact global shift symmetry to remove one scalar while preserving initialization and weight tying.
mechanism: Global tied-embedding common-mode anchoring
evidence_used: The current 1583-parameter model achieved 99.96%. Full eight-dimensional vocabulary centering failed at 71.46%, motivating this narrower one-dimensional gauge that requires no positional-embedding compensation and leaves all other token coordinates directly parameterized.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Restoring the qualified 1582-parameter design’s full value bias while fixing one attention output-projection bias coordinate will retain at least 99% accuracy with 1581 parameters, because the missing mean-zero output bias component can be represented through the learned value bias.
change: Adopt the qualified bias-free-ln2 and globally gauge-fixed tied embedding, restore all eight value-bias coordinates, and remove one coordinate only from the attention projection bias.
mechanism: Attention value/output-bias redundancy anchoring
evidence_used: Reference Design 3 reached 99.99% with 1582 parameters and full value bias; the seven-value-bias combination fell to 98.56%, so this patch preserves that sensitive pathway and instead removes one redundant projection-bias coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second mean-zero attention projection-bias coordinate while retaining all eight value-bias coordinates will produce a 1580-parameter model with at least 99% accuracy.
change: Apply the qualified mean-zero residual parameterization, bias-free `ln2`, and globally gauge-fixed tied embedding, then reconstruct the final two attention projection-bias coordinates as zero.
mechanism: Incremental attention value/output-bias redundancy anchoring
evidence_used: The first attention projection-bias anchor achieved 99.79% with 1581 parameters, while removing a second value-bias coordinate collapsed accuracy; this motivates preserving the full value pathway and testing the smallest adjacent reduction in its redundant output bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third mean-zero attention projection-bias coordinate while retaining all eight value-bias coordinates will produce a 1579-parameter model with at least 99% accuracy.
change: Reduce the attention output projection’s learned mean-zero bias from five coordinates to four, reconstructing the final three coordinates as zero.
mechanism: Incremental attention value/output-bias redundancy anchoring
evidence_used: Two projection-bias anchors achieved 99.97% accuracy with 1580 parameters, improving on the one-anchor model’s 99.79%; this supports testing one additional coordinate without disturbing the sensitive value-bias pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7433, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing the third anchored projection-bias coordinate from an existing value-bias scalar will yield a 1579-parameter model with at least 99% accuracy, because it preserves a trainable direct output offset while removing its redundant independent parameter.
change: Retain four independent mean-zero attention projection-bias coordinates, derive the fifth from the aligned learned value-bias coordinate, and keep the two already-qualified coordinates fixed at zero.
mechanism: Value-tied attention output-bias anchoring
evidence_used: The 1580-parameter two-anchor design achieved 99.97%, while fixing the third coordinate to zero collapsed accuracy to 74.33%; tying that sensitive coordinate to the full learned value-bias pathway tests whether its trainability—not an independent degree of freedom—is required.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7177, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding one exactly compensated mean-zero token/position gauge constraint to the qualified 1580-parameter two-anchor model will retain at least 99% accuracy with 1579 parameters.
change: Use the qualified twice-anchored attention projection, then remove one coordinate from the final token embedding row while transferring its initialized contribution into every positional embedding.
mechanism: Single-coordinate token/position shift-gauge anchoring
evidence_used: The two-projection-bias-anchor design achieved 99.97% at 1580 parameters; the one-dimensional global embedding gauge also retained 99.99%, whereas removing all vocabulary-common modes collapsed accuracy, motivating a single compensated gauge reduction rather than a third projection-bias anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second mean-zero token/position shift coordinate will reduce the qualified 1579-parameter model to 1578 parameters while retaining at least 99% accuracy.
change: Remove another coordinate from the final token embedding row and transfer its initialized two-coordinate contribution into every positional embedding, leaving the initialized input function and output probabilities unchanged.
mechanism: Second compensated token/position shift-gauge anchor
evidence_used: The first compensated token/position gauge constraint achieved 99.97% accuracy at 1579 parameters, while removing all vocabulary-common modes failed; this motivates one incremental extension of the qualified gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-coordinate gauge constraint by one coordinate will produce a 1577-parameter model with at least 99% accuracy while preserving the initialized input function and output probabilities.
change: Fix the final three mean-zero coordinates of the last token embedding row and transfer their initialized contribution into every positional embedding.
mechanism: Third compensated token/position shift-gauge anchor
evidence_used: The first and second compensated gauge anchors achieved 99.97% and 99.95% accuracy at 1579 and 1578 parameters respectively; this supports one incremental extension, while the failed full vocabulary-centering experiment cautions against removing all coordinates at once.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified three-coordinate embedding gauge to four coordinates will produce a 1576-parameter model with at least 99% accuracy while preserving its initialized input function and output probabilities.
change: Adopt the qualified twice-anchored attention projection and constrain four coordinates of the final token row, transferring their initialized common shift into every positional embedding.
mechanism: Fourth compensated token/position shift-gauge anchor
evidence_used: Three compensated gauge anchors achieved 99.8% accuracy at 1577 parameters, while a third attention projection-bias anchor collapsed to 74.33%; this supports extending the successful embedding gauge instead of further restricting attention.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9590000000000001, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the localized first mean-zero basis coordinate alongside the qualified final three coordinates will produce a 1576-parameter model with at least 99% accuracy, showing that the prior collapse was specific to the adjacent fourth coordinate rather than the gauge count.
change: Replace the failed fourth-adjacent gauge with a non-adjacent four-coordinate gauge over coordinates 0, 4, 5, and 6, with exact positional compensation.
mechanism: Non-adjacent token/position shift-gauge anchoring
evidence_used: Three final-coordinate anchors achieved 99.8% at 1577 parameters, while adding adjacent coordinate 3 fell to 95.9%; testing coordinate 0 preserves the qualified anchors and isolates coordinate-selection effects.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8918, "parameters": 1576, "training_steps": 4999}



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
