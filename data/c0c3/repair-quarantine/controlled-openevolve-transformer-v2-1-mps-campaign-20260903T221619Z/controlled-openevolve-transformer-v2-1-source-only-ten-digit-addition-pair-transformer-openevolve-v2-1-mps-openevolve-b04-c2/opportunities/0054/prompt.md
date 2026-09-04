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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1608, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1,609-parameter value triangularization by removing the function-null uniform output component of one attention projection column will yield 1,608 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1607, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1,608-parameter design by quotienting the second attention output-projection column will yield 1,607 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9973000000000001, "parameters": 1610, "training_steps": 4999}
prior_hypothesis: Rotating two already first-column-zero value channels in the first attention head to eliminate one second-column coefficient will reduce the model from 1,611 to 1,610 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9987, "parameters": 1609, "training_steps": 4999}
prior_hypothesis: Completing first-column value alignment in both heads, then eliminating two second-column coefficients within the first head’s preserved zero subspace, will produce a 1,609-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Reproducing the qualified 1,615-parameter design and gauge-fixing a second disjoint value-channel pair in the first head will yield 1,614 parameters while retaining at least 99% accuracy.
change: Add the qualified projection-bias and seventh positional quotients, compact one value coefficient per head, and remove one additional coefficient through an independently compensated rotation of the first head’s remaining value-channel pair.
mechanism: Additional disjoint intra-head value-basis gauge fixing
evidence_used: Reference Design 3 achieved 99.89% at 1,615 parameters after independent value-basis gauge fixes in both heads; this motivates extending the same exact symmetry to a disjoint channel pair instead of repeating positional or MLP quotients that failed sharply.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the remaining disjoint value-channel pair in the second attention head will reduce the verified 1,614-parameter model to 1,613 parameters while retaining at least 99% accuracy.
change: Apply the compensated value-channel rotation to both disjoint channel pairs in every four-dimensional attention head, removing one additional learned QKV coefficient.
mechanism: Complete per-head disjoint value-basis gauge fixing
evidence_used: The current design achieved 100% accuracy after adding the second disjoint pair in the first head; applying the same exact intra-head symmetry to the untouched pair in the second head is the smallest direct extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,613-parameter design by rotating the two remaining nonzero first-column value coordinates in the first head will yield 1,612 parameters while retaining at least 99% accuracy.
change: Reproduce both disjoint value-pair fixes in every head, then apply one compensated cross-pair rotation in the first head and omit the additional coefficient made exactly zero.
mechanism: Cross-pair intra-head value-basis gauge fixing
evidence_used: Complete disjoint value-basis gauge fixing achieved 99.85% at 1,613 parameters; the added rotation uses the same successful exact symmetry while preserving the previously fixed zeros.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,612-parameter design and applying its successful cross-pair rotation to the second attention head will yield 1,611 parameters while retaining at least 99% accuracy.
change: Compact six value-projection coefficients by applying two disjoint rotations and one cross-pair rotation independently in each four-dimensional attention head, with exactly compensated output-projection rotations.
mechanism: Symmetric cross-pair intra-head value-basis gauge fixing
evidence_used: The first cross-pair extension reduced the fully disjoint 1,613-parameter design to 1,612 parameters with 99.98% accuracy; applying the identical remaining symmetry to the second head is the closest incremental test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating two already first-column-zero value channels in the first attention head to eliminate one second-column coefficient will reduce the model from 1,611 to 1,610 parameters while retaining at least 99% accuracy.
change: Generalize value-basis rotations to select their pivot input column, then add one compensated rotation within the first head’s three-dimensional zero subspace and omit the newly zero second-column coefficient.
mechanism: Residual value-subspace gauge fixing
evidence_used: The successive compensated value-basis rotations from 1,617 through 1,611 parameters all met the accuracy requirement, with the current symmetric cross-pair design reaching 99.75%; this tests the next unused exact orthogonal gauge direction without adding another sensitive positional constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the qualified 1,610-parameter design’s second-column value rotation to the corresponding zero subspace in the second attention head will yield 1,609 parameters while retaining at least 99% accuracy.
change: Generalize rotations to select an input-column pivot, complete the verified first-column gauge fixes in both heads, and eliminate one second-column coefficient per head with compensated output-projection rotations.
mechanism: Symmetric residual value-subspace gauge fixing
evidence_used: The first residual-subspace rotation achieved 99.73% accuracy at 1,610 parameters; the identical unused gauge direction remains in the second head, and every value-basis reduction from 1,617 through 1,610 met the requirement.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9899, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing first-column value alignment in both heads, then eliminating two second-column coefficients within the first head’s preserved zero subspace, will produce a 1,609-parameter model with at least 99% accuracy.
change: Generalize compensated value-basis rotations to arbitrary channel pairs and input columns; remove three first-column coefficients per head and two second-column coefficients in the first head.
mechanism: First-head residual value-subspace triangularization
evidence_used: The first residual-subspace reduction in head 0 achieved 99.73% at 1,610 parameters, while applying the next reduction to head 1 narrowly failed at 98.99%; using the remaining independent stabilizer in the already-qualified first-head zero subspace tests a distinct 1,609-parameter allocation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating a third-column coefficient within head 0’s two-dimensional subspace that already vanishes on the first two input columns will reduce the model from 1,609 to 1,608 parameters while retaining at least 99% accuracy.
change: Add one compensated value-basis rotation between head 0 channels 0 and 1 at input column 2, then omit the coefficient made exactly zero.
mechanism: Nested residual value-subspace triangularization
evidence_used: The current design reached 99.87% at 1,609 parameters by placing both residual second-column reductions in head 0, while distributing the second reduction to head 1 scored 98.99%; this extends the successful head-0 triangularization through its remaining exact stabilizer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7415, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1,609-parameter value triangularization and replacing the failed third-column value constraint with one exact query–key co-rotation will yield 1,608 parameters while maintaining at least 99% accuracy.
change: Add the qualified second residual value rotation in head 0, then co-rotate one query/key channel pair and omit the resulting zero key coefficient.
mechanism: Compensated query–key basis gauge fixing
evidence_used: Reference Design 3 achieved 99.87% at 1,609 parameters, while extending its increasingly constrained value subspace to 1,608 collapsed to 74.15%; an independent query–key orthogonal gauge avoids that failed value-path allocation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.142, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,609-parameter value triangularization by removing the function-null uniform output component of one attention projection column will yield 1,608 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified three-first-column/two-second-column value alignment, then parameterize the first attention output-projection column in the same seven-dimensional zero-sum basis already used for its bias and train it with full-coordinate quotient moments.
mechanism: Residual-output common-mode quotient
evidence_used: Reference Design 3 achieved 99.87% at 1,609 parameters, whereas further value triangularization collapsed to 74.15% and query–key co-rotation to 14.2%; the already-qualified projection-bias common-mode quotient shows that removing an attention-output uniform direction is a distinct, stable mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,608-parameter design by quotienting the second attention output-projection column will yield 1,607 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified value, bias, and positional compactions, then parameterize the first two attention output-projection columns in a seven-dimensional zero-sum basis and optimize them with full-coordinate AdamW moments.
mechanism: Multi-column residual-output common-mode quotient
evidence_used: The 1,608-parameter reference achieved 99.90% after removing the function-null uniform component of its first projection column; the identical LayerNorm-erased direction exists independently in the second column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the third attention output-projection column will reduce the qualified 1,607-parameter model to 1,606 parameters while maintaining at least 99% accuracy.
change: Parameterize the first three attention output-projection columns in the seven-dimensional zero-sum basis and retain full-coordinate AdamW moments through the existing matrix-aware quotient optimizer.
mechanism: Third-column residual-output common-mode quotient
evidence_used: Quotienting the first projection column achieved 99.90% at 1,608 parameters, and extending the same exact LayerNorm-erased quotient to the second achieved 99.96% at 1,607; the third column has the identical independent function-null uniform direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9871, "parameters": 1606, "training_steps": 4999}



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
