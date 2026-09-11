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
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1619, "training_steps": 4999}
prior_hypothesis: Combining the verified one-direction `ln2.bias` quotient with removal of the attention projection bias’s uniform direction will yield 1,619 parameters and at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1617, "training_steps": 4999}
prior_hypothesis: Quotienting the uniform residual-channel direction of one additional late positional row will reduce the verified model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because the change is function-null and applies only one more optimization constraint.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9989, "parameters": 1619, "training_steps": 4999}
prior_hypothesis: Removing only the uniform direction of `ln1.bias` will reduce the current model from 1,620 to 1,619 parameters while retaining at least 99% accuracy, because the retained query and unrestricted projection biases can absorb its effects without the optimization interaction introduced by simultaneously quotienting `attn.proj.bias`.

## Recent verification evidence

RECENT RESULT
hypothesis: Combining the verified one-direction `ln2.bias` quotient with removal of the attention projection bias’s uniform direction will yield 1,619 parameters and at least 99% accuracy.
change: Represent `ln2.bias` and `attn.proj.bias` in seven-dimensional zero-sum bases, reconstruct them during inference, and preserve projected full-coordinate AdamW updates.
mechanism: Function-null attention output-bias quotient
evidence_used: The one-direction `ln2.bias` quotient achieved 99.95% at 1,620 parameters, while removing a second `ln2` direction failed at 12.27%; the next reduction instead targets an exactly function-null uniform attention-output shift erased by the subsequent `ln2` and final LayerNorms.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,619-parameter design with a zero-sum quotient of `ln1.bias` will yield 1,618 parameters and at least 99% accuracy, because its removed uniform shift is absorbable by the retained query and projection biases.
change: Apply the verified query-only attention, `ln2.bias`, and attention-projection quotients, then represent `ln1.bias` in a seven-dimensional zero-sum basis with reconstructed full-coordinate AdamW updates.
mechanism: One-direction pre-attention LayerNorm bias quotient
evidence_used: The independent `ln2.bias` and attention-projection quotients achieved 99.85% at 1,619 parameters, while removing a second `ln2` direction failed; this motivates testing one direction from the distinct, structurally redundant pre-attention bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining every `fc2.weight` column to the seven-dimensional zero-sum output subspace will reduce the model from 1,618 to 1,606 parameters while retaining at least 99% accuracy, because each removed component produces only a token-local uniform residual shift erased by subsequent LayerNorms.
change: Store `fc2.weight` as a 7-by-12 compact matrix, reconstruct its eight output coordinates during inference, preserve baseline initialization, and train it with virtual full-coordinate AdamW moments.
mechanism: Columnwise MLP residual common-mode quotient
evidence_used: The 1,618-parameter design achieved 99.94% while already quotienting the uniform component of `fc2.bias`; applying the same verified function-null output direction independently to every `fc2.weight` column is the closest structural extension.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding the verified zero-sum quotient of `ln1.bias` to the current 1,619-parameter model will reduce it to 1,618 parameters while retaining at least 99% accuracy.
change: Represent `ln1.bias` in a seven-dimensional orthonormal zero-sum basis, reconstruct it in the forward pass, and preserve virtual full-coordinate AdamW updates during training.
mechanism: One-direction pre-attention LayerNorm bias quotient
evidence_used: Reference Design 2 used this exact extension and achieved 99.94% accuracy with 1,618 parameters; it is the smallest qualified design and directly improves the current verified model.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7314, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining every `fc2.weight` column to the seven-dimensional zero-sum output subspace will reduce the verified 1,620-parameter model to 1,608 parameters while retaining at least 99% accuracy, because the removed component contributes only a token-local uniform residual shift erased by the final LayerNorm.
change: Preserve full baseline initialization, project `fc2.weight` into compact zero-sum coordinates, reconstruct its eight output rows in the forward pass, and extend virtual full-coordinate AdamW updates to the compact matrix.
mechanism: Columnwise MLP residual common-mode quotient
evidence_used: The current 1,620-parameter design achieved 99.95%, and its successful `fc2.bias` quotient removes the same function-null output direction. The previous columnwise attempt could not be verified rather than failing accuracy, motivating an isolated implementation on the robust current design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.38539999999999996, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the single grand-common-mode direction from the tied token embedding will yield 1,617 parameters and at least 99% accuracy, because it changes every input embedding by the same uniform residual-channel shift and every output logit by the same scalar, leaving the modeled distribution unchanged.
change: Represent the tied token embedding in a one-dimension-smaller orthonormal zero-sum basis, reconstruct it for input lookup and output logits, and train it with the existing full-coordinate AdamW quotient updates.
mechanism: Tied-embedding grand-common-mode quotient
evidence_used: The current 1,618-parameter design achieved 99.94%, while larger `fc2.weight` and additional `ln2.bias` quotients failed; this tests one independent, exactly function-null direction without further constraining those sensitive components.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5459, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining only the first `fc2.weight` column to the seven-dimensional zero-sum output subspace will reduce the model to 1,618 parameters while retaining at least 99% accuracy, because the removed activation-dependent uniform residual shift is erased by the final LayerNorm.
change: Compact one `fc2.weight` column after baseline initialization, reconstruct the full matrix during inference, and train it with projected full-coordinate AdamW moments.
mechanism: Single-column MLP residual common-mode quotient
evidence_used: The 1,619-parameter current design achieved 99.85%, while quotienting all twelve `fc2.weight` columns at once fell to 38.54%; applying the same function-null quotient to exactly one column tests whether the failure came from the simultaneous twelve-direction optimization change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.21350000000000002, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the uniform direction of `ln1.bias` will reduce the current model from 1,620 to 1,619 parameters while retaining at least 99% accuracy, because the retained query and unrestricted projection biases can absorb its effects without the optimization interaction introduced by simultaneously quotienting `attn.proj.bias`.
change: Represent `ln1.bias` in a seven-dimensional zero-sum basis, reconstruct it in the forward pass, and train it using the existing full-coordinate AdamW quotient updates.
mechanism: Isolated pre-attention LayerNorm common-mode quotient
evidence_used: The current 1,620-parameter design achieved 99.95%, and the qualified design containing this quotient achieved 99.94% at 1,618 parameters. The later combined 1,618-parameter rerun fell to 73.14%, motivating an isolated test that preserves the full attention-projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the uniform residual-channel direction of one additional late positional row will reduce the verified model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because the change is function-null and applies only one more optimization constraint.
change: Extend positional compaction from the first two and final four rows to the first two and final five rows, preserving initialization and full-coordinate AdamW moments.
mechanism: Incremental positional common-mode quotient
evidence_used: The current six-row positional quotient achieved 99.94% at 1,618 parameters; the reported 96.93% failure from further positional compression motivates testing the smallest possible one-row extension rather than compressing many additional rows simultaneously.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,617-parameter design by quotienting one additional late positional row will produce 1,616 parameters and retain at least 99% accuracy, because the seven-row design achieved 100% and this adds only one function-null optimization constraint.
change: Add the qualified `ln1.bias` quotient and extend positional compaction from the first two plus final five rows to the first two plus final six rows, preserving full-coordinate AdamW moments.
mechanism: Incremental late-position common-mode quotient
evidence_used: Reference Design 2 achieved 100% accuracy at 1,617 parameters with the `ln1.bias` quotient and seven compact positional rows; the immediately preceding one-row positional extension improved accuracy from 99.94% to 100%, motivating another single-row extension rather than a larger compression jump.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8894, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding the exact two quotients from qualified Reference Design 2 will reduce the current model from 1,619 to 1,617 parameters while retaining at least 99% accuracy.
change: Store the attention projection bias in a seven-dimensional zero-sum basis and extend positional compaction from the first two plus final four rows to the first two plus final five rows, preserving full-coordinate AdamW moments.
mechanism: Dual common-mode quotient for attention output bias and one late positional row
evidence_used: Reference Design 2 used this exact 1,617-parameter structure and achieved 100% accuracy; the subsequent 1,616-parameter failure indicates this qualified boundary is safer than compacting another positional row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7411, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Compacting the sixth-from-last positional row with native compact-coordinate AdamW while preserving the verified seven rows’ full-coordinate updates will produce 1,616 parameters and at least 99% accuracy.
change: Store the additional late positional row as a separate seven-parameter zero-sum vector, optimize it normally, and retain the qualified optimizer behavior for all existing compact rows.
mechanism: Split-coordinate positional common-mode quotient
evidence_used: The seven-row quotient achieved 100% at 1,617 parameters, while adding this same eighth row with reconstructed full-coordinate AdamW fell to 88.94%; isolating the new row under compact-coordinate AdamW directly tests whether optimizer geometry caused that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7289, "parameters": 1616, "training_steps": 4999}



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
