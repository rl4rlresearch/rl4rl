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
verified_results: {"accuracy": 0.9972, "parameters": 1524, "training_steps": 4999}
prior_hypothesis: Adding the first value-output rotation to the less-constrained second head will produce a 1524-parameter model with at least 99% accuracy while preserving the initialized model function.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1526, "training_steps": 4999}
prior_hypothesis: Fixing one first-head value coefficient through a shared rotation of the value rows and output-projection columns will produce a 1526-parameter model with at least 99% accuracy while preserving the initialized model function.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9971, "parameters": 1525, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1526-parameter value-output design with a second first-head value rotation will produce a 1525-parameter model with at least 99% accuracy while preserving the initialized model function.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1529, "training_steps": 4999}
prior_hypothesis: Adding a ninth Givens rotation on first-head query/key channels 1–2 using input column 1 will produce a 1529-parameter model with at least 99% accuracy, because it preserves attention scores and all eight existing anchors while eliminating one additional query coefficient.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a second sequential rotation within the second attention head will reduce the model to 1533 parameters while retaining at least 99% accuracy, because it removes another exact query-key rotational gauge without changing initialized attention scores.
change: Rotate second-head query/key channels 1–2 after its existing rotation, fix the resulting query coefficient to zero, and omit that coefficient from the trainable QKV parameterization.
mechanism: Incremental second-head query-key rotational gauge fixing
evidence_used: The current four-rotation design achieved 99.26% at 1534 parameters, and each preceding query-key rotation qualified; applying the same established symmetry to the less-constrained second head is better supported than another mean-zero QKV-row constraint, whose second application achieved only 96.45%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9952, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified five-rotation 1533-parameter design with a third sequential rotation in the second attention head will produce a 1532-parameter model with at least 99% accuracy, because it removes one additional exact query-key rotational gauge while preserving initialized attention scores.
change: Reproduce the qualified five-rotation parameterization, then rotate second-head query/key channels 2–3 and omit the resulting fixed query coefficient.
mechanism: Complete second-head query-key Givens gauge fixing
evidence_used: Five rotations achieved 99.52% accuracy at 1533 parameters, and every preceding incremental query-key rotation qualified; this supports completing the analogous three-rotation chain in the second head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1532, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a seventh Givens rotation on first-head query/key channels 0–1 using input column 1 will produce a 1531-parameter model with at least 99% accuracy, because it preserves attention scores and the six existing column-zero anchors while eliminating one additional query coefficient.
change: Reproduce the qualified six-rotation QKV parameterization, apply one stabilizer rotation within the first head, and omit the resulting fixed `q_weight[0, 1]` coefficient while preserving full-sized initialization draws.
mechanism: Residual query-key orthogonal gauge fixing
evidence_used: The six-rotation 1532-parameter design achieved 99.97% accuracy, and every preceding incremental query-key rotation qualified; this supports extending the same exact symmetry into the remaining first-head subspace.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1531, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding the qualified column-one stabilizer rotation to the second attention head will produce a 1530-parameter model with at least 99% accuracy because it preserves attention scores and all seven existing anchors while removing one additional query coefficient.
change: Replace the single-rotation QKV parameterization with eight sequential Givens rotations: three column-zero anchors per head and one column-one anchor per head, preserving full-sized initialization draws.
mechanism: Mirrored per-head residual query-key orthogonal gauge fixing
evidence_used: Seven rotations achieved 99.94% accuracy at 1531 parameters, and every preceding incremental query-key rotation qualified; mirroring the seventh rotation in the previously untouched second-head stabilizer subspace is the closest supported one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1530, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a ninth Givens rotation on first-head query/key channels 1–2 using input column 1 will produce a 1529-parameter model with at least 99% accuracy, because it preserves attention scores and all eight existing anchors while eliminating one additional query coefficient.
change: Extend the eight-rotation QKV parameterization with a first-head `(1, 1)` stabilizer rotation and omit the resulting fixed `q_weight[1, 1]` coefficient.
mechanism: Incremental first-head column-one stabilizer rotation
evidence_used: Eight rotations achieved 99.84% accuracy at 1530 parameters, while every preceding incremental query-key rotation qualified; continuing the same exact symmetry is better supported than adding another LayerNorm-input row constraint, whose second application fell to 96.45%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1529, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified nine-rotation 1529-parameter design with a second-head channels 1–2 rotation on input column 1 will produce a 1528-parameter model with at least 99% accuracy, because it preserves attention scores and all nine existing anchors while removing one exact query coefficient.
change: Upgrade the current QKV parameterization to ten sequential Givens rotations, incorporating the qualified eighth and ninth rotations and adding the mirrored second-head tenth rotation while preserving full-sized initialization draws.
mechanism: Mirrored second-head column-one query-key stabilizer rotation
evidence_used: Nine rotations achieved 99.93% accuracy at 1529 parameters, and the earlier mirrored second-head stabilizer achieved 99.84%; alternating the established column-one rotation chain into the second head is the closest supported one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1528, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a first-head channels 0–1 rotation on input column 2 will produce a 1527-parameter model with at least 99% accuracy, because it preserves attention scores and all ten qualified anchors while removing one exact query coefficient.
change: Upgrade the current six-rotation QKV parameterization to eleven sequential rotations, incorporating the qualified column-one rotations and adding a first-head column-two stabilizer rotation while preserving full-sized initialization draws.
mechanism: Eleventh residual query-key Givens gauge fixing
evidence_used: The ten-rotation 1528-parameter design achieved 99.92% accuracy, and every preceding incremental query-key rotation qualified; the remaining first-head channels 0–1 subspace preserves the established column-zero and column-one anchors.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1527, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing the second head’s six-dimensional orthogonal gauge will produce a 1526-parameter model with at least 99% accuracy because the added rotation preserves attention scores while removing one exact query coefficient.
change: Upgrade the five-rotation QKV parameterization to twelve rotations, including all eleven qualified anchors plus the mirrored second-head column-two rotation, while preserving full-sized initialization draws.
mechanism: Complete per-head triangular query-key gauge fixing
evidence_used: The eleven-rotation design achieved 99.92% accuracy at 1527 parameters, and every preceding incremental query-key rotation qualified; the remaining second-head channels 0–1 rotation is the exact mirror of its qualified first-head counterpart.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6361, "parameters": 1526, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one first-head value coefficient through a shared rotation of the value rows and output-projection columns will produce a 1526-parameter model with at least 99% accuracy while preserving the initialized model function.
change: Retain the qualified eleven query-key rotations, rotate first-head value channels 0–1 to eliminate `value_weight[0, 0]`, omit that coefficient, and apply the inverse rotation to the corresponding attention-output projection columns.
mechanism: Coupled value-output orthogonal gauge fixing
evidence_used: The 1527-parameter eleven-rotation design achieved 99.92%, whereas extending the second head to twelve query-key anchors fell to 63.61%; this tests a distinct exact value-output factorization symmetry without adding the failed query constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1526, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1526-parameter value-output design with a second first-head value rotation will produce a 1525-parameter model with at least 99% accuracy while preserving the initialized model function.
change: Incorporate the qualified eleventh query-key anchor and first value-output gauge, then rotate first-head value channels 1–2, omit the resulting second fixed value coefficient, and inversely rotate the matching projection columns.
mechanism: Sequential coupled value-output orthogonal gauge fixing
evidence_used: The first coupled value-output rotation achieved 99.96% accuracy at 1526 parameters, while a twelfth query-key constraint failed at 63.61%; extending the successful value-output symmetry is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9971, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing the first-head value-column rotation chain will produce a 1524-parameter model with at least 99% accuracy while preserving the initialized model function.
change: Reproduce the qualified eleven query-key and two value-output gauges, then rotate first-head value channels 2–3, omit the resulting fixed coefficient, and inversely rotate the matching projection columns.
mechanism: Third coupled value-output orthogonal gauge fixing
evidence_used: The two-value-rotation design achieved 99.71% accuracy at 1525 parameters; extending the same exact symmetry is the closest supported reduction, while the alternative twelfth query-key constraint failed at 63.61%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5328, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding the first value-output rotation to the less-constrained second head will produce a 1524-parameter model with at least 99% accuracy while preserving the initialized model function.
change: Reproduce the qualified eleven query-key and two first-head value-output gauges, then rotate second-head value channels 0–1, omit the resulting fixed coefficient, and inversely rotate projection columns 4–5.
mechanism: Mirrored second-head coupled value-output gauge fixing
evidence_used: The two-value-rotation design achieved 99.71% at 1525 parameters, while adding a third sequential rotation within the first head fell to 53.28%; mirroring the successful first value rotation in the independent second head tests whether that failure came from overconstraining one head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9972, "parameters": 1524, "training_steps": 4999}



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
