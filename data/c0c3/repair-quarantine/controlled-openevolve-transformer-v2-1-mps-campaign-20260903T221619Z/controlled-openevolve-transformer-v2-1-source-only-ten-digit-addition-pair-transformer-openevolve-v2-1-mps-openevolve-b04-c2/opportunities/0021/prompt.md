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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1631, "training_steps": 4999}
prior_hypothesis: Quotienting positional rows zero, one, and the final two rows will produce a 1,631-parameter model with at least 99% accuracy because the verified final-row quotient reached 99.99%, while the penultimate row has similarly limited causal influence compared with the failed row-two quotient.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9981, "parameters": 1629, "training_steps": 4999}
prior_hypothesis: Quotienting positional rows zero, one, and the final four rows will reduce the verified 1,630-parameter design to 1,629 parameters while retaining at least 99% accuracy, because the final-three-row design achieved 99.76% and the next tail row still has much shorter causal influence than the failed early row-two quotient.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9976, "parameters": 1630, "training_steps": 4999}
prior_hypothesis: Quotienting positional rows zero, one, and the final three rows will reduce the model from 1,633 to 1,630 parameters while retaining at least 99% accuracy, because the verified final-two-row design achieved 99.96% and tail rows have substantially shorter causal influence than the failed row-two quotient.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the redundant attention key bias and the terminal MLP common-mode bias will reduce the model from 1,644 to 1,635 parameters while retaining at least 99% accuracy if the terminal bias uses a centered orthonormal basis with full-coordinate AdamW moments.
change: Apply the verified fused key-bias elimination, represent the terminal MLP bias in the seven-dimensional zero-sum subspace, and optimize it with reconstructed eight-coordinate AdamW dynamics.
mechanism: Orthonormal quotient-space bias optimization
evidence_used: Key-bias elimination achieved 99.71% at 1,636 parameters, while full-coordinate optimization of a last-coordinate-fixed 1,635-parameter bias reached 98.68%; the narrow miss motivates preserving the same virtual dynamics in a centered orthonormal gauge that also preserves gradient norms.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,635-parameter design to remove the attention output bias’s common-mode coordinate will produce a 1,634-parameter model with at least 99% accuracy, because downstream LayerNorms cancel uniform residual-channel shifts and the orthonormal parameterization preserves gradient norms.
change: Apply the verified key-bias elimination and terminal MLP bias quotient, additionally represent the attention output bias in the same seven-dimensional zero-sum basis, and optimize both quotient biases using reconstructed eight-coordinate AdamW moments.
mechanism: Dual residual-bias gauge fixing with full-coordinate AdamW dynamics
evidence_used: The centered orthonormal terminal-bias design achieved 99.78% accuracy at 1,635 parameters, showing that one-dimensional LayerNorm gauge removal remains trainable when full-coordinate optimizer dynamics are preserved; the attention output bias has an analogous common-mode redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4961, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,635-parameter design with row-wise zero-sum positional embeddings will remove one redundant parameter per sequence position while retaining at least 99% accuracy, provided initialization RNG and virtual eight-coordinate AdamW dynamics are preserved.
change: Apply the verified key-bias and terminal-MLP quotient reductions, then store each positional embedding in a seven-dimensional orthonormal zero-sum basis and optimize both quotient parameter types through reconstructed full-coordinate AdamW moments.
mechanism: Full-coordinate AdamW on quotient positional embeddings
evidence_used: The centered terminal-bias quotient reached 99.78% at 1,635 parameters, showing that orthonormal gauge removal succeeds when full-coordinate optimizer dynamics are retained; positional common-mode shifts are likewise canceled by every pre-norm path and the final LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9815999999999999, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,635-parameter design by removing only one positional embedding common-mode coordinate will yield a 1,634-parameter model with at least 99% accuracy; the 1,612-parameter all-row quotient reached 98.16%, so limiting the same exact symmetry reduction to one row should greatly reduce its optimization disturbance.
change: Apply the verified key-bias and terminal-MLP quotient reductions, then parameterize only positional row zero in a seven-dimensional orthonormal zero-sum basis while preserving baseline initialization RNG and reconstructed full-coordinate AdamW dynamics.
mechanism: Single-row positional gauge fixing with full-coordinate AdamW
evidence_used: The terminal-bias quotient achieved 99.78% at 1,635 parameters, while quotienting all 23 positional rows produced a close 98.16% at 1,612; this tests the smallest conservative positional reduction using the already successful optimization treatment.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,634-parameter design by quotienting positional row one will produce a 1,633-parameter model with at least 99% accuracy, because the single-row design reached 99.83% while quotienting all 23 rows narrowly missed at 98.16%.
change: Parameterize positional rows zero and one in the orthonormal zero-sum basis, preserving baseline initialization and reconstructed full-coordinate AdamW dynamics.
mechanism: Dual-row positional gauge fixing with full-coordinate AdamW
evidence_used: The current single-row quotient achieved 99.83% at 1,634 parameters; the 1,612-parameter all-row quotient’s 98.16% result supports testing the smallest incremental extension rather than quotienting many additional rows simultaneously.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1633, "training_steps": 4999}

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
