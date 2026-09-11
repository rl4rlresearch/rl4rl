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
verified_results: {"accuracy": 1.0, "parameters": 1501, "training_steps": 4999}
prior_hypothesis: Fixing a second relative attention-output coordinate in the second head will reduce the qualified 1502-parameter design to 1501 parameters while maintaining at least 99% accuracy, because each head admits an independent invertible value/output basis change and a well-conditioned two-pivot initialization preserves the represented function.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1501, "training_steps": 4999}
prior_hypothesis: Extending the second attention head from one to two function-preserving projection-coordinate gauges will reduce the model to 1501 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1503, "training_steps": 4999}
prior_hypothesis: Fixing the same relative attention-output coordinate independently in the first head will reduce the verified 1504-parameter model to 1503 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9964, "parameters": 1499, "training_steps": 4999}
prior_hypothesis: Applying the verified normalized scalar value/output gauge independently to the second attention head will reduce the model from 1500 to 1499 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one query-bias coordinate to zero will reduce the current model from 1506 to 1505 parameters while maintaining at least 99% accuracy.
change: Store seven learned query-bias coordinates and reconstruct the eighth as zero, leaving the protected decoding interface and training procedure unchanged.
mechanism: Single-coordinate query-key basis gauge fixing
evidence_used: Reference Design 1 verified this exact 1505-parameter gauge at 100% accuracy; the 1504-parameter per-head extension reached only 98.61%, identifying the single-coordinate version as the qualified boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1505, "training_steps": 4999}

RECENT RESULT
hypothesis: Concentrating both fixed query-bias coordinates in the already-gauged second head will produce a 1504-parameter model with at least 99% accuracy, because it preserves a fully unconstrained first head while using the second head’s within-head query/key basis symmetry.
change: Apply the verified final-LayerNorm scale gauge and store six query-bias coordinates, reconstructing the final two coordinates of the second head as zero.
mechanism: Single-head concentrated query-basis gauge fixing
evidence_used: The single-coordinate query gauge reached 99.97% at 1505 parameters, while distributing two fixed coordinates across both heads reached 98.61% at 1504; concentrating both constraints in one head tests whether leaving one head fully unconstrained restores the missing margin.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6047, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified 1505-parameter quotient design and fixing one relative attention-output coordinate through the exact within-head value/output basis symmetry will yield 1504 parameters with at least 99% accuracy.
change: Reproduce the qualified weight, embedding, residual-output, LayerNorm-scale, and single query-bias quotients, then store one attention-projection row with six coordinates and reconstruct its seventh coordinate as zero after a function-preserving initialization basis change.
mechanism: Independent attention value/output basis gauge fixing
evidence_used: The single query-bias gauge achieved 99.97–100% accuracy at 1505 parameters, while both 1504 query-only extensions failed; an independent value/output basis symmetry removes one parameter without imposing another constraint on attention logits.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the same relative attention-output coordinate independently in the first head will reduce the verified 1504-parameter model to 1503 parameters while maintaining at least 99% accuracy.
change: Store six relative coordinates for one projection row in each attention head, reconstruct both fixed coordinates as zero, and apply function-preserving within-head value/output basis changes at initialization.
mechanism: Per-head attention value/output basis gauge fixing
evidence_used: The current 1504-parameter design achieved 99.79% after one value/output gauge; using the independent symmetry of the other head extends that successful mechanism without adding another query-bias restriction, whose tested 1504-parameter variants failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1503, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second relative attention-output coordinate in the first head will reduce the verified 1503-parameter design to 1502 parameters while maintaining at least 99% accuracy, because two conditioned pivot directions can select this representative through an invertible value/output basis change without changing the initialized function.
change: Reproduce the verified per-head projection gauges, then store five coordinates for the first head’s gauged projection row and six for the second head’s row, reconstructing the omitted coordinates as zero and applying matching value-weight basis changes at initialization.
mechanism: Additional within-head value/output basis gauge fixing
evidence_used: Reference Design 1 achieved 99.90% accuracy at 1503 parameters after independently fixing one attention-output coordinate in each head; extending that successful value/output symmetry by one coordinate is the smallest symmetry-grounded reduction, while prior additional query-bias restrictions failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1502, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second relative attention-output coordinate in the second head will reduce the qualified 1502-parameter design to 1501 parameters while maintaining at least 99% accuracy, because each head admits an independent invertible value/output basis change and a well-conditioned two-pivot initialization preserves the represented function.
change: Reproduce the qualified projection gauges, but store five coordinates for the final projection row of each head, reconstruct two omitted coordinates per row as zero, and apply matching within-head value-weight basis changes at initialization.
mechanism: Balanced per-head value/output basis gauge fixing
evidence_used: The 1502-parameter design achieved 99.99% accuracy with two fixed projection coordinates in the first head and one in the second; extending the same successful two-coordinate construction to the second head is the smallest symmetry-grounded reduction and avoids the query-bias restrictions that previously failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified balanced two-coordinate gauges by fixing a third relative projection coordinate in one attention head will reduce the model from 1501 to 1500 parameters while retaining at least 99% accuracy, because a best-conditioned three-pivot within-head basis change preserves the freshly initialized function.
change: Reproduce the qualified query-bias, final-LayerNorm scale, embedding, and balanced projection quotients; then store four rather than five coordinates for the first head’s gauged projection row, with initialization selecting the best-conditioned omitted coordinate triple.
mechanism: Conditioned third-coordinate value/output basis gauge
evidence_used: Reference Design 3 achieved 100% accuracy at 1501 parameters after fixing two projection coordinates in each head. The proposed one-parameter reduction uses the same successful value/output symmetry, while adaptive pivot-coordinate selection addresses the conditioning risk of extending the gauge to three coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9882, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one disjoint relative projection coordinate in another row of the first attention head will reduce the model from 1501 to 1500 parameters while retaining at least 99% accuracy, because a best-pivot scalar shear preserves the initialized function without the three-coordinate same-row conditioning that reached only 98.82%.
change: Add a six-coordinate auxiliary projection-row parameter, reconstruct its omitted third-from-last relative coordinate as zero, and initialize it through a function-preserving one-pivot value/output basis change.
mechanism: Secondary-row scalar value/output basis gauge
evidence_used: The balanced two-coordinate-per-head design achieved 100% at 1501 parameters, while extending one row to three simultaneously fixed coordinates scored 98.82%; this tests an unused, independently conditioned scalar gauge on a different row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6437, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the second attention head from one to two function-preserving projection-coordinate gauges will reduce the model to 1501 parameters while retaining at least 99% accuracy.
change: Store five relative coordinates for each head’s final projection row, reconstruct two omitted coordinates per row, and initialize the second head with the same conditioned two-pivot basis change already used for the first head.
mechanism: Balanced per-head value/output basis gauge fixing
evidence_used: Reference Design 3 verified this balanced construction at 100% accuracy with 1501 parameters; both tested 1500-parameter extensions fell below 99%, making the qualified 1501 design the strongest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the failed three-pivot shear with a norm-preserving SVD basis rotation will reduce the balanced 1501-parameter design to 1500 parameters while retaining at least 99% accuracy.
change: Fix three relative projection coordinates in the first attention head and two in the second; use an orthogonal first-head initialization transform and the verified conditioned two-pivot transform for the second head.
mechanism: Orthogonal nullspace value/output basis gauge
evidence_used: The balanced two-coordinate-per-head design reached 100% at 1501 parameters, while the three-coordinate pivot construction narrowly missed at 98.82%; an orthogonal transform tests whether conditioning, rather than the additional gauge itself, caused that loss.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8768, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: A scalar value/output gauge on the balanced 1501-parameter model will reduce it to 1500 parameters while retaining at least 99% accuracy, because normalizing one already-gauged projection row and absorbing its scale into the matching value row preserves the initialized function without imposing a third zero-coordinate constraint.
change: Reproduce the verified final-scale, query-bias, embedding, and balanced two-coordinate projection quotients, then represent the first head’s gauged projection row by four max-pivot chart coordinates with fixed norm and transfer its omitted scale into the corresponding value weight.
mechanism: Max-pivot normalized value/output scale gauge
evidence_used: The balanced two-coordinate-per-head design achieved 100% at 1501 parameters, while adding a third zero coordinate reached only 98.82% and its SVD variant reached 87.68%; this tests an unused diagonal value/output symmetry with a bounded max-pivot chart instead of another projection-direction restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified normalized scalar value/output gauge independently to the second attention head will reduce the model from 1500 to 1499 parameters while retaining at least 99% accuracy.
change: Store both gauged projection rows with four chart coordinates and fixed norm, absorb each omitted scale into its matching value-weight row at initialization, and propagate both pivot indices through optimizer-side reconstruction.
mechanism: Independent second-head max-pivot value/output scale gauge
evidence_used: The current first-head scalar gauge achieved 99.94% accuracy at 1500 parameters, while prior third-coordinate restrictions failed; the second head has the same independent diagonal value/output symmetry and therefore offers the closest evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1499, "training_steps": 4999}



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
