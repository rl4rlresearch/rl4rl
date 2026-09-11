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
verified_results: {"accuracy": 0.9945999999999999, "parameters": 1599, "training_steps": 4999}
prior_hypothesis: Quotienting `fc2` columns 0, 1, 2, and 4 will produce a 1,599-parameter model with at least 99% accuracy, avoiding the optimization collapse observed when column 3 was the fourth quotient.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9997, "parameters": 1595, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1,596-parameter design by quotienting `fc2` column 8 will produce a 1,595-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1600, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,601-parameter design by quotienting the third `fc2` output column will produce a 1,600-parameter model with at least 99% accuracy, because its uniform residual-channel component is independently erased by the final LayerNorm.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9981, "parameters": 1596, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,597-parameter design by quotienting `fc2` column 7 will produce a 1,596-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified 1,603-parameter design by quotienting the next late positional row will produce a 1,602-parameter model with at least 99% accuracy.
change: Compact the first two and final ten positional rows, updating initialization, reconstruction, and full-coordinate AdamW projection consistently.
mechanism: Twelfth positional common-mode quotient
evidence_used: Reference Design 3 achieved 99.91% accuracy at 1,603 parameters after compacting the first two and final nine positional rows; the closest informative reduction extends the same successful LayerNorm-invariant quotient by one row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9462999999999999, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,603-parameter design by quotienting positional row 2 instead of the failed tenth late row will produce a 1,602-parameter model with at least 99% accuracy.
change: Reproduce the qualified two-column attention-output quotient and first-two/final-nine positional compaction, then additionally compact positional row 2 with consistent reconstruction and full-coordinate AdamW moments.
mechanism: Alternate early-row positional common-mode quotient
evidence_used: The first two and final nine positional quotients achieved 99.91% at 1,603 parameters, while adding the next late row collapsed to 94.63%; reallocating the same exact LayerNorm-invariant quotient to the adjacent early row tests whether the failure was specific to that late-row optimization path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9871, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Restricting `ln2.bias` to six orthonormal directions will reduce the model to 1,602 parameters while retaining at least 99% accuracy, because the omitted bias direction is exactly absorbable by the unrestricted `fc1.bias`.
change: Remove one additional coordinate from `ln2.bias` and its reconstruction basis; the existing full-coordinate quotient optimizer already supports the reduced basis.
mechanism: Additional post-attention LayerNorm-bias absorption
evidence_used: The 1,603-parameter design reached 99.91%, while adding either the next late positional quotient or row-2 positional quotient scored 94.63% and 98.71%. This tests a distinct redundancy whose effect can be absorbed directly before GELU by `fc1.bias`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6498999999999999, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Compacting one `fc2` output column on top of the qualified 1,603-parameter design will yield 1,602 parameters while retaining at least 99% accuracy, because its uniform residual-channel component is exactly erased by the final LayerNorm.
change: Reproduce the qualified first-two/final-nine positional compaction, then parameterize the first MLP output column in the existing seven-dimensional zero-sum basis and optimize it with full-coordinate AdamW moments.
mechanism: MLP residual-output common-mode quotient
evidence_used: Reference Design 3 achieved 99.91% at 1,603 parameters, while additional positional and `ln2.bias` restrictions failed; the successful attention-output common-mode quotients motivate testing the analogous, previously untested exact quotient in the MLP output path.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.996, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Compacting the first two `fc2` output columns on the qualified 1,602-parameter design will yield 1,601 parameters while retaining at least 99% accuracy, because each omitted uniform output component is independently erased by the final LayerNorm.
change: Reproduce the qualified first-two/final-nine positional compaction, then parameterize the first two MLP output columns in the seven-dimensional zero-sum basis and train them with full-coordinate AdamW moments.
mechanism: Two-column MLP residual-output common-mode quotient
evidence_used: Compacting the first `fc2` column produced 99.60% accuracy at 1,602 parameters; the second column has the same exact LayerNorm-null uniform direction, making it the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,601-parameter design by quotienting the third `fc2` output column will produce a 1,600-parameter model with at least 99% accuracy, because its uniform residual-channel component is independently erased by the final LayerNorm.
change: Reproduce the qualified two-column attention quotient and first-two/final-nine positional compaction, then parameterize the first three MLP output columns in the seven-dimensional zero-sum basis with full-coordinate AdamW moments.
mechanism: Three-column MLP residual-output common-mode quotient
evidence_used: Compacting the first two `fc2` columns achieved 99.80% accuracy at 1,601 parameters; extending the same exact LayerNorm-null quotient to the adjacent third MLP column is the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,600-parameter design by quotienting the fourth `fc2` output column will produce a 1,599-parameter model with at least 99% accuracy, because its uniform residual-channel component is independently erased by the final LayerNorm.
change: Parameterize the first four `fc2` output columns in the existing seven-dimensional zero-sum basis and leave the remaining eight columns unrestricted; the matrix-aware quotient optimizer already supports the new shape.
mechanism: Four-column MLP residual-output common-mode quotient
evidence_used: Successively quotienting one, two, and three `fc2` columns achieved 99.60% at 1,602 parameters, 99.80% at 1,601 parameters, and 99.90% at 1,600 parameters, so extending the same exact function-null quotient to the adjacent fourth column is the closest informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9554, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting `fc2` columns 0, 1, 2, and 4 will produce a 1,599-parameter model with at least 99% accuracy, avoiding the optimization collapse observed when column 3 was the fourth quotient.
change: Extend the qualified three-column quotient with original `fc2` column 4, preserve column ordering during reconstruction, and leave column 3 unrestricted.
mechanism: Alternate-column MLP residual-output quotient
evidence_used: Quotienting columns 0–2 achieved 99.90% at 1,600 parameters, while adding adjacent column 3 fell to 95.54%; reallocating the same exact LayerNorm-null quotient to column 4 directly tests whether that failure was specific to column 3’s optimization path.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9945999999999999, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,599-parameter design by quotienting `fc2` column 5 will yield 1,598 parameters with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while the optimization-sensitive column 3 remains unrestricted.
change: Compact `fc2` columns 0, 1, 2, 4, and 5, reconstruct their original ordering in the forward pass, and train the compact matrix with full-coordinate AdamW moments.
mechanism: Five-column MLP residual-output common-mode quotient
evidence_used: Reference Design 1 achieved 99.46% accuracy at 1,599 parameters with columns 0, 1, 2, and 4 compacted, whereas using column 3 as the fourth quotient collapsed to 95.54%; adding adjacent column 5 is the closest incremental reduction that preserves the successful unrestricted treatment of column 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,598-parameter design by quotienting `fc2` column 6 will produce a 1,597-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while optimization-sensitive column 3 remains unrestricted.
change: Compact `fc2` columns 0, 1, 2, 4, 5, and 6, reconstruct their original ordering during the forward pass, and optimize the compact matrix with full-coordinate AdamW moments.
mechanism: Six-column MLP residual-output common-mode quotient
evidence_used: Quotienting columns 0, 1, 2, 4, and 5 achieved 99.93% accuracy at 1,598 parameters, while the earlier failure involved column 3; extending the successful contiguous run after column 4 to adjacent column 6 is the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,597-parameter design by quotienting `fc2` column 7 will produce a 1,596-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.
change: Compact `fc2` columns 0, 1, 2, 4, 5, 6, and 7, preserve their original ordering during reconstruction, and retain full-coordinate AdamW optimization through the existing quotient machinery.
mechanism: Seven-column MLP residual-output common-mode quotient
evidence_used: Quotienting columns 0, 1, 2, 4, 5, and 6 achieved 99.93% accuracy at 1,597 parameters; extending the successful run after column 4 to adjacent column 7 is the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,596-parameter design by quotienting `fc2` column 8 will produce a 1,595-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.
change: Parameterize `fc2` columns 0, 1, 2, and 4 through 8 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave column 3 and columns 9 through 11 unrestricted.
mechanism: Eighth-column MLP residual-output common-mode quotient
evidence_used: Reference Design 3 achieved 99.81% accuracy at 1,596 parameters after successfully extending the compacted run through column 7; quotienting adjacent column 8 is the closest incremental reduction using the same exact LayerNorm-null direction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1595, "training_steps": 4999}



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
