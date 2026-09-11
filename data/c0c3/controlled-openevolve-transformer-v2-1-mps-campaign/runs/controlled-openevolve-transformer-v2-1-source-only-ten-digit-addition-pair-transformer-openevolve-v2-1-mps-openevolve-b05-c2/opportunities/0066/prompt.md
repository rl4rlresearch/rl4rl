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
verified_results: {"accuracy": 0.9979, "parameters": 1516, "training_steps": 4999}
prior_hypothesis: Fixing the seventh and final mean-zero coordinate of the anchor token will produce a 1516-parameter model with at least 99% accuracy while preserving initialized inputs and output softmax probabilities.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1514, "training_steps": 4999}
prior_hypothesis: Fixing a fifth `ln1` scale will reduce the qualified model to 1514 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1517, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1518-parameter design with a sixth exact token-position common-shift gauge will produce a 1517-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9989, "parameters": 1521, "training_steps": 4999}
prior_hypothesis: Rotating second-head value channels 2–3 on input column 1 will produce a 1521-parameter model with at least 99% accuracy while preserving the initialized function and all five qualified value anchors.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a column-one stabilizer rotation within the already column-zero-anchored first-head value subspace to the qualified 1524-parameter mirrored-head design will produce a 1523-parameter model with at least 99% accuracy.
change: Retain the two qualified first-head value rotations, add the qualified first second-head rotation, then rotate first-head value channels 0–1 on input column 1 and omit the resulting fixed coefficient while inversely rotating the matching projection columns.
mechanism: Residual first-head value-output stabilizer
evidence_used: The mirrored first second-head rotation achieved 99.72% at 1524 parameters, whereas extending either head sequentially into another channel failed; the proposed residual rotation acts only within the two first-head rows whose column-zero coefficients are already fixed, testing a distinct stabilizer gauge analogous to the successful query-key stabilizers.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956999999999999, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating first-head value channels 2–3 on input column 1 will produce a 1522-parameter model with at least 99% accuracy while preserving the initialized model function and all four qualified value anchors.
change: Reproduce the qualified 1523-parameter four-value-gauge design, add an orthogonal rotation confined to first-head value rows 2–3, omit the resulting fixed `value_weight[2, 1]` coefficient, and inversely rotate projection columns 2–3.
mechanism: Complement-subspace value-output orthogonal gauge fixing
evidence_used: The four-value-gauge design achieved 99.57% accuracy at 1523 parameters. Additional column-zero chain rotations failed, whereas the qualified column-one stabilizer showed that changing the gauge direction can retain trainability; this complementary rotation preserves every existing zero without extending either failed column-zero chain.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1522, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating second-head value channels 2–3 on input column 1 will produce a 1521-parameter model with at least 99% accuracy while preserving the initialized function and all five qualified value anchors.
change: Reproduce the qualified eleven query-key and five value-output gauges, then add an inverse-compensated rotation of second-head value rows 6–7 that fixes `value_weight[6, 1]`.
mechanism: Mirrored second-head complement-subspace value-output gauge fixing
evidence_used: The analogous first-head complement rotation improved the qualified 1523-parameter design to 99.95% at 1522 parameters; mirroring it avoids the failed second-head column-zero chain, which achieved only 80.13%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1521, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating second-head value rows 5 and 7 on input column 1 will produce a 1520-parameter model with at least 99% accuracy while preserving the initialized function and all six qualified value anchors.
change: Add a seventh, nonadjacent value-output Givens rotation, omit the resulting fixed `value_weight[5, 1]` coefficient, and inversely rotate attention-output columns 5 and 7.
mechanism: Residual second-head column-one value-output gauge
evidence_used: The mirrored complement rotation qualified at 99.89% with 1521 parameters, while the row-5–6 column-zero extension fell to 80.13%; using the residual rows 5 and 7 on the successful column-one direction avoids disturbing either qualified second-head zero.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9154000000000001, "parameters": 1520, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the residual second-head rows 5 and 7 on input column 2 will produce a 1520-parameter model with at least 99% accuracy while preserving the initialized function and all six qualified value anchors.
change: Extend the current model with the qualified second-head complement rotation, then add a nonadjacent rows-5/7 column-two rotation and inverse output-projection compensation.
mechanism: Alternate-axis residual second-head value-output gauge
evidence_used: The six-value-gauge design achieved 99.89% at 1521 parameters, while the same residual rows anchored on column 1 achieved only 91.54%; changing the gauge direction tests whether that failure was anchor-direction-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1520, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the 1520-parameter qualified design with one additional exact token/position common-shift gauge will produce a 1519-parameter model with at least 99% accuracy.
change: Reproduce the qualified seven-value QKV parameterization, then fix a fourth mean-zero coordinate of the anchor token embedding and compensate it in the positional embedding.
mechanism: Fourth token-position shift gauge atop the qualified seven-value attention gauge
evidence_used: The seven-value design achieved 99.96% accuracy at 1520 parameters. Using a distinct exact embedding symmetry avoids extending the value-rotation chains whose less-supported anchor directions previously caused large accuracy losses.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 1519, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1519-parameter design with a fifth exact token-position common-shift gauge will produce a 1518-parameter model with at least 99% accuracy.
change: Reproduce the qualified seven-value QKV parameterization, then remove a fifth anchor-token embedding coordinate and compensate it in the positional embedding.
mechanism: Fifth token-position common-shift gauge
evidence_used: The seven-value design reached 99.96% at 1520 parameters, and its fourth token-position shift reduction reached 99.74% at 1519; the proposed reduction extends that same exact embedding symmetry by one coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1518, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1518-parameter design with a sixth exact token-position common-shift gauge will produce a 1517-parameter model with at least 99% accuracy.
change: Remove a sixth anchor-token embedding coordinate, compensate it in the positional embedding, and preserve the initialized input function and softmax probabilities.
mechanism: Sixth token-position common-shift gauge
evidence_used: The fifth token-position gauge achieved 100% accuracy at 1518 parameters, while the preceding fourth-gauge design achieved 99.74%; this applies the same exact embedding symmetry for one further reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1517, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the seventh and final mean-zero coordinate of the anchor token will produce a 1516-parameter model with at least 99% accuracy while preserving initialized inputs and output softmax probabilities.
change: Extend the qualified six-coordinate embedding gauge to all seven mean-zero coordinates, remove the now-empty anchor parameter entirely, reconstruct the anchor row as zero, and compensate the full shift in positional embeddings.
mechanism: Complete anchor-token/position common-shift gauge
evidence_used: The fifth gauge achieved 100% accuracy at 1518 parameters and the sixth achieved 99.99% at 1517; completing the same exact symmetry is the closest supported reduction and avoids the value-rotation directions that previously harmed accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1516, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing the proven embedding gauge and fixing one query-bias coordinate through the remaining second-head query-key rotation will produce a 1515-parameter model with at least 99% accuracy.
change: Remove the anchor token’s remaining four coordinates, compensate all seven positional coordinates, and omit `q_bias[4]`, reconstructed as zero within the residual rows-4/5 query-key gauge.
mechanism: Residual second-head query-bias gauge fixing
evidence_used: The complete embedding gauge achieved 99.79% at 1516 parameters. Unlike the failed twelfth weight-based query-key constraint, the proposed residual gauge fixes an initially zero bias coordinate while preserving the initialized function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1515, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1515-parameter design by fixing a fifth pre-attention LayerNorm scale will produce a 1514-parameter model with at least 99% accuracy while preserving the initialized function.
change: Reproduce the complete token-position, seventh value-output, and query-bias gauges from the 1515-parameter reference, then reduce the learned `ln1` scales from four to three.
mechanism: Fifth pre-attention LayerNorm scale gauge
evidence_used: The complete embedding plus residual query-bias design achieved 99.88% at 1515 parameters. Extending the existing four-scale LayerNorm gauge tests a distinct exact redundancy instead of the failed additional query-weight and value-chain constraints.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Fixing a fifth `ln1` scale will reduce the qualified model to 1514 parameters while retaining at least 99% accuracy.
change: Replace the four-parameter `ln1` scale vector with three learned scales and five fixed unit scales.
mechanism: Fifth pre-attention LayerNorm scale gauge
evidence_used: The current four-scale-gauge design achieved 99.88% accuracy at 1515 parameters; the previous attempt at this exact reduction was malformed before verification, so a uniquely matching patch is the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1514, "training_steps": 4999}



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
