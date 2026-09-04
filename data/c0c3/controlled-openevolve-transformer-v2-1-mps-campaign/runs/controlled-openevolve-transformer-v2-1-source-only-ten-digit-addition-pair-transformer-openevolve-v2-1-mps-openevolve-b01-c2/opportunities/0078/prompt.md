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
verified_results: {"accuracy": 0.9991, "parameters": 1549, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,550-parameter design to first-head key row 9 will produce a 1,549-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9994, "parameters": 1550, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,551-parameter design to first-head value row 19 will produce a 1,550-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1547, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,548-parameter design to first-head key row 11 will produce a 1,547-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9978, "parameters": 1551, "training_steps": 4999}
prior_hypothesis: Gauge-fixing value row 18’s final normalized-input coefficient will reduce the model to 1,551 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding an exact scalar token-embedding translation gauge to the verified 1,575-parameter balanced query-bias design will produce a 1,574-parameter model with at least 99% accuracy.
change: Reproduce the successful query-bias constraints at coordinates 2 and 6, then anchor the tied token embedding’s first scalar and reconstruct both input embeddings and output logits from the remaining coefficients while preserving initialization modulo LayerNorm and softmax invariances.
mechanism: Global tied-token embedding translation gauge
evidence_used: Fixing query-bias coordinate 2 in both heads achieved 99.83% at 1,575 parameters, whereas constraining query-bias coordinate 3 already fell to 98.48%; this motivates an orthogonal exact translation gauge instead of another query-bias restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1574, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing positional coordinate 4 with zero mean, on top of the verified 1,574-parameter design, will yield 1,573 parameters while maintaining at least 99% accuracy.
change: Reproduce the verified balanced query-bias, coordinate-7 positional, and tied-token translation gauges, then remove one additional positional coefficient by centering coordinate 4 across positions and transferring its mean into every token embedding.
mechanism: Mean-centered positional translation gauge
evidence_used: The 1,574-parameter reference reached 99.88%; although anchoring position zero’s coordinate 4 collapsed, the exact translation gauge remains available, and a symmetric zero-mean chart avoids privileging the position-zero coefficient implicated by that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6559999999999999, "parameters": 1573, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1,574-parameter design and representing each attention projection bias in an orthonormal zero-sum basis will yield 1,573 parameters while maintaining at least 99% accuracy.
change: Add the verified QKV, balanced query-bias, positional, and tied-token gauges, then remove the attention projection bias’s downstream-LayerNorm-invariant common direction without privileging a hidden coordinate.
mechanism: Orthonormal attention-projection common-bias gauge
evidence_used: The 1,574-parameter reference achieved 99.88% accuracy, while an additional positional gauge failed. The final-MLP anchored-bias experiment motivates testing the distinct attention-projection null direction with an orthonormal centered chart instead of another coordinate anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1573, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the final MLP output bias in an orthonormal zero-sum basis will reduce the verified model from 1,573 to 1,572 parameters while maintaining at least 99% accuracy.
change: Replace the MLP’s `fc2` with the existing orthonormal common-bias-gauged linear layer, removing the bias direction erased by the final LayerNorm.
mechanism: Orthonormal final-MLP common-bias gauge
evidence_used: The orthonormal attention-projection bias gauge achieved 99.97% at 1,573 parameters, while the coordinate-anchored final-MLP bias gauge collapsed to 12.32%; this tests the same final-LayerNorm null direction using the successful symmetric, isometric chart.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9889, "parameters": 1572, "training_steps": 4999}

RECENT RESULT
hypothesis: Projecting ordinary eight-coordinate AdamW updates onto an orthonormal seven-parameter final-MLP bias will produce a 1,572-parameter model with at least 99% accuracy.
change: Apply the successful orthonormal common-bias gauge to the attention projection and final MLP output, while training the latter with full-coordinate AdamW moments projected into its seven-dimensional learned parameterization.
mechanism: Dense-coordinate AdamW on a zero-mean final-MLP bias quotient
evidence_used: The orthonormal attention-projection gauge reached 99.97% at 1,573 parameters; the orthonormal final-MLP gauge narrowly missed at 98.89%, suggesting its exact null-direction removal preserves capacity but its rotated elementwise-Adam geometry needs correction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1572, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing every common-output weight direction from the attention projection and final MLP, while projecting ordinary dense-coordinate AdamW updates into the orthonormal quotient, will produce a 1,552-parameter model with at least 99% accuracy.
change: Extend the verified common-bias gauges to all affine columns of both residual-branch output layers, retain dense-coordinate AdamW geometry for the newly gauged weights and final-MLP bias, and reproduce the verified tied-token translation gauge.
mechanism: Dense-AdamW common-output affine quotient
evidence_used: The 1,573-parameter orthonormal attention-bias quotient achieved 99.97%, and dense-coordinate AdamW raised the final-MLP bias quotient from 98.89% to 99.97% at 1,572 parameters; each removed weight direction likewise contributes only an input-dependent common residual translation erased by downstream LayerNorm.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1552, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,552-parameter common-output affine quotient with a fourth positional translation gauge at coordinate 4, while projecting ordinary dense-coordinate AdamW updates back into the anchored chart, will yield 1,551 parameters and at least 99% accuracy.
change: Reproduce the verified balanced query-bias, tied-token, three-position, and common-output affine gauges; additionally anchor positional coordinate 4 and train its coupled token-position quotient with a recovered dense AdamW moment for the omitted coefficient.
mechanism: Dense-AdamW token-position translation quotient
evidence_used: The dense-coordinate optimizer rescued the exact final-MLP quotient from 98.89% to 99.97%, and the resulting common-output extension achieved 99.8% at 1,552 parameters; this directly motivates retesting the exact coordinate-4 token-position gauge whose ordinary and centered-coordinate optimizers previously failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5152, "parameters": 1551, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing value row 18’s final normalized-input coefficient will reduce the model to 1,551 parameters while retaining at least 99% accuracy.
change: Extend the seven verified QKV row gauges to local value coordinate 2 in the first attention head.
mechanism: Third first-head value-row LayerNorm gauge
evidence_used: The 1,552-parameter design reached 99.8% with adjacent first-head value rows 16 and 17 already gauged; extending that successful local pattern avoids the previously reported failures from additional second-head constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1551, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,551-parameter design to first-head value row 19 will produce a 1,550-parameter model with at least 99% accuracy.
change: Reproduce the verified common-output affine quotients and dense-coordinate AdamW updates, then gauge the remaining first-head value row alongside rows 16–18.
mechanism: Fourth first-head value-row LayerNorm gauge
evidence_used: Gauging value row 18 reduced the verified common-output design to 1,551 parameters with 99.78% accuracy; row 19 continues the successful adjacent first-head pattern without adding a previously fragile second-head constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1550, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,550-parameter design to first-head key row 9 will produce a 1,549-parameter model with at least 99% accuracy.
change: Reproduce the verified common-output affine quotients and dense-coordinate AdamW updates, retain all four first-head value-row gauges, and gauge the adjacent second key row in the first attention head.
mechanism: Second first-head key-row LayerNorm gauge
evidence_used: The 1,550-parameter reference achieved 99.94% accuracy after adjacent first-head value-row extensions; key row 8 is already verified, so extending to adjacent first-head key row 9 is the smallest untested reduction that avoids the previously fragile second-head constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1549, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,549-parameter design to first-head key row 10 will produce a 1,548-parameter model with at least 99% accuracy.
change: Reproduce the verified common-output quotients, dense-coordinate AdamW updates, tied-token and positional gauges, balanced query-bias constraints, four first-head value-row gauges, and two first-head key-row gauges; then gauge adjacent key row 10.
mechanism: Third first-head key-row normalized-input gauge
evidence_used: Gauging adjacent first-head key row 9 achieved 99.91% accuracy at 1,549 parameters, while the preceding 1,550-parameter design achieved 99.94%; extending the same successful first-head pattern is the smallest untested reduction and avoids fragile second-head constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1548, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,548-parameter design to first-head key row 11 will produce a 1,547-parameter model with at least 99% accuracy.
change: Gauge the remaining first-head key row while retaining all previously verified query, value, output, embedding, and positional quotients.
mechanism: Fourth first-head key-row normalized-input gauge
evidence_used: Successive first-head key-row extensions reached 99.91% at 1,549 parameters and 99.96% at 1,548 parameters; row 11 completes the same adjacent first-head pattern without imposing a fragile second-head constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1547, "training_steps": 4999}



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
