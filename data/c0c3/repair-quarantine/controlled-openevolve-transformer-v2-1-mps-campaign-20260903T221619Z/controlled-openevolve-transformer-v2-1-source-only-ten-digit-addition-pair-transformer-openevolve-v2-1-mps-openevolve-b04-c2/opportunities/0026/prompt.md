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
verified_results: {"accuracy": 0.9998, "parameters": 1621, "training_steps": 4999}
prior_hypothesis: Removing the eight-dimensional attention value bias will reduce the model from 1,629 to 1,621 parameters while retaining at least 99% accuracy, because a position-independent value bias passes unchanged through normalized attention and is exactly absorbable by the retained output-projection bias.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9948, "parameters": 1629, "training_steps": 4999}
prior_hypothesis: Quotienting positional rows zero, one, and the final four rows will yield 1,629 parameters and at least 99% accuracy, reproducing the qualified 99.81% reference design.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1620, "training_steps": 4999}
prior_hypothesis: Combining the verified 1,621-parameter query-only attention and four-tail positional design with a single zero-sum quotient of `ln2.bias` will yield 1,620 parameters and at least 99% accuracy; removing one redundant direction should disturb optimization substantially less than the failed removal of all eight bias coordinates.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified two-row positional quotient to row two will reduce the model from 1,633 to 1,632 parameters while retaining at least 99% accuracy, because two rows achieved 99.86% and quotienting all 23 rows only narrowly missed at 98.16%.
change: Parameterize positional rows zero through two in the orthonormal zero-sum basis while preserving baseline initialization and reconstructed full-coordinate AdamW dynamics.
mechanism: Incremental positional gauge fixing with full-coordinate AdamW
evidence_used: The verified two-row design reached 99.86% accuracy at 1,633 parameters; this supports the smallest incremental extension before attempting broader positional compression.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9897, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final row will produce a 1,632-parameter model with at least 99% accuracy; unlike the failed row-two quotient, the final row has no long causal downstream influence while sharing the same exact LayerNorm-invariant common-mode redundancy.
change: Apply the verified terminal-MLP quotient and two-row positional quotient, then remove the final positional row’s common-mode coordinate while preserving baseline initialization and reconstructed full-coordinate AdamW dynamics.
mechanism: Tail-localized positional gauge fixing with full-coordinate AdamW
evidence_used: The two-row positional quotient achieved 99.86% at 1,633 parameters, while adding row two narrowly missed at 98.97%; selecting the final row tests the same one-parameter reduction with substantially less causal propagation of numerical optimization differences.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final two rows will produce a 1,631-parameter model with at least 99% accuracy because the verified final-row quotient reached 99.99%, while the penultimate row has similarly limited causal influence compared with the failed row-two quotient.
change: Add the penultimate positional row to the orthonormal zero-sum quotient and extend its reconstructed full-coordinate AdamW update accordingly.
mechanism: Tail-localized positional gauge fixing with full-coordinate AdamW
evidence_used: The current 1,632-parameter design achieved 99.99% after quotienting the final row, whereas quotienting row two achieved only 98.97%; this motivates testing the adjacent penultimate row as the smallest tail-localized reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final three rows will reduce the model from 1,633 to 1,630 parameters while retaining at least 99% accuracy, because the verified final-two-row design achieved 99.96% and tail rows have substantially shorter causal influence than the failed row-two quotient.
change: Store positional rows zero, one, and the final three rows in the seven-dimensional orthonormal zero-sum basis, reconstruct them during inference, and preserve virtual eight-coordinate AdamW moments during training.
mechanism: Incremental tail-localized positional gauge fixing with full-coordinate AdamW
evidence_used: The 1,631-parameter design quotienting rows zero, one, and the final two rows achieved 99.96%, whereas adding early row two produced only 98.97%; this supports testing the adjacent third-from-last row as the smallest tail-localized reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976, "parameters": 1630, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final four rows will reduce the verified 1,630-parameter design to 1,629 parameters while retaining at least 99% accuracy, because the final-three-row design achieved 99.76% and the next tail row still has much shorter causal influence than the failed early row-two quotient.
change: Preserve baseline initialization, then store positional rows zero, one, and the final four rows in a seven-dimensional orthonormal zero-sum basis; reconstruct them in the forward pass and preserve virtual eight-coordinate AdamW moments during training.
mechanism: Incremental tail-localized positional gauge fixing with full-coordinate AdamW
evidence_used: Quotienting rows zero, one, and the final three rows achieved 99.76% at 1,630 parameters, while quotienting early row two missed at 98.97%; this motivates the smallest additional tail-localized reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting positional rows zero, one, and the final five rows will reduce the verified model from 1,629 to 1,628 parameters while retaining at least 99% accuracy, because the final-four-row design achieved 99.81% and the next tail row has substantially less causal influence than the failed early row-two quotient.
change: Add the fifth-from-last positional row to the orthonormal zero-sum quotient and extend its reconstructed full-coordinate AdamW update.
mechanism: Incremental tail-localized positional gauge fixing with full-coordinate AdamW
evidence_used: Quotienting the final four rows achieved 99.81% at 1,629 parameters, while quotienting early row two achieved 98.97%; this supports the smallest additional tail-localized reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9693, "parameters": 1628, "training_steps": 4999}

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
