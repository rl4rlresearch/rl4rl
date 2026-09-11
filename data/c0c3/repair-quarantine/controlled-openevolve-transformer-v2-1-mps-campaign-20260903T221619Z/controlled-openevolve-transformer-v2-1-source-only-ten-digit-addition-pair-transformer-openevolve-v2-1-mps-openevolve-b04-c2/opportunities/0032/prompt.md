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
verified_results: {"accuracy": 0.9994, "parameters": 1618, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,619-parameter design with a zero-sum quotient of `ln1.bias` will yield 1,618 parameters and at least 99% accuracy, because its removed uniform shift is absorbable by the retained query and projection biases.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1620, "training_steps": 4999}
prior_hypothesis: Combining the verified 1,621-parameter query-only attention and four-tail positional design with a single zero-sum quotient of `ln2.bias` will yield 1,620 parameters and at least 99% accuracy; removing one redundant direction should disturb optimization substantially less than the failed removal of all eight bias coordinates.

## Recent verification evidence

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final four rows will reproduce at least 99% accuracy with 1,629 parameters; the adjacent fifth-tail-row reduction is excluded because it fell to 96.93%.
change: Add the fourth-from-last positional row to the orthonormal zero-sum quotient and its reconstructed full-coordinate AdamW update.
mechanism: Incremental tail-localized positional gauge fixing with full-coordinate AdamW
evidence_used: The qualified final-four-row design achieved 99.81% accuracy at 1,629 parameters, while extending the quotient to five final rows failed at 96.93%.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final four rows will yield 1,629 parameters and at least 99% accuracy, reproducing the qualified 99.81% reference design.
change: Extend the current positional quotient from the final two positions to the final four and update reconstructed full-coordinate AdamW projection accordingly.
mechanism: Qualified four-row tail positional quotient
evidence_used: The final-four-row design achieved 99.81% accuracy with 1,629 parameters, while extending to five final rows fell to 96.93%; this patch stops at the verified boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight-dimensional attention value bias will reduce the model from 1,629 to 1,621 parameters while retaining at least 99% accuracy, because a position-independent value bias passes unchanged through normalized attention and is exactly absorbable by the retained output-projection bias.
change: Store only the learned query bias in `qkv.bias`; reconstruct zero key and value biases in the attention forward pass while leaving the full output-projection bias trainable.
mechanism: Value-bias/output-bias reparameterization
evidence_used: The current 1,629-parameter design achieved 99.81%, whereas further positional compression failed at 96.93%; the prior output-bias quotient also failed, motivating a structurally redundant reduction that preserves every coordinate of the sensitive output bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified value-bias removal with elimination of the eight-dimensional pre-MLP LayerNorm bias will yield 1,613 parameters and at least 99% accuracy, because that LayerNorm shift is exactly absorbable by the retained `fc1` bias.
change: Retain only the learned attention query bias, then remove `ln2.bias` while preserving module initialization and all downstream learned weights and biases.
mechanism: Pre-MLP LayerNorm bias absorption
evidence_used: The 1,621-parameter query-only attention-bias design achieved 99.98% accuracy, demonstrating substantial margin after an exact bias absorption; `ln2.bias` has the same direct redundancy since `fc1(W)` maps its position-independent shift into the unrestricted learned `fc1.bias`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9052, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified 1,621-parameter query-only attention and four-tail positional design with a single zero-sum quotient of `ln2.bias` will yield 1,620 parameters and at least 99% accuracy; removing one redundant direction should disturb optimization substantially less than the failed removal of all eight bias coordinates.
change: Apply the verified value-bias removal and fourth-tail positional quotient, then represent `ln2.bias` in a seven-dimensional orthonormal zero-sum basis and train it with reconstructed full-coordinate AdamW moments.
mechanism: One-dimensional pre-MLP LayerNorm bias quotient
evidence_used: The query-only attention design achieved 99.98% at 1,621 parameters, while removing all eight `ln2.bias` parameters fell to 90.52%; this tests the smallest incremental version of that exact bias-absorption reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one additional direction from `ln2.bias` will reduce the model to 1,619 parameters while retaining at least 99% accuracy, because the one-direction quotient achieved 99.95% and is the smallest incremental step toward the eight-direction removal that failed.
change: Represent `ln2.bias` in a six-dimensional orthonormal subspace and retain its reconstructed full-coordinate AdamW updates.
mechanism: Incremental two-direction pre-MLP bias quotient
evidence_used: The verified 1,620-parameter model removed one `ln2.bias` direction at 99.95% accuracy, while removing all eight bias coordinates fell to 90.52%; testing exactly one additional direction most directly locates the compression boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.12269999999999999, "parameters": 1619, "training_steps": 4999}

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
