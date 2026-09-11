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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1632, "training_steps": 4999}
prior_hypothesis: Quotienting positional rows zero, one, and the final row will produce a 1,632-parameter model with at least 99% accuracy; unlike the failed row-two quotient, the final row has no long causal downstream influence while sharing the same exact LayerNorm-invariant common-mode redundancy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9978, "parameters": 1635, "training_steps": 4999}
prior_hypothesis: Removing the redundant attention key bias and the terminal MLP common-mode bias will reduce the model from 1,644 to 1,635 parameters while retaining at least 99% accuracy if the terminal bias uses a centered orthonormal basis with full-coordinate AdamW moments.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9986, "parameters": 1633, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,634-parameter design by quotienting positional row one will produce a 1,633-parameter model with at least 99% accuracy, because the single-row design reached 99.83% while quotienting all 23 rows narrowly missed at 98.16%.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing only the eight-parameter `ln2` bias will reduce the model from 1,644 to 1,636 parameters while retaining at least 99% accuracy, because its effect can be represented by the immediately following learned `fc1` bias.
change: Disable the bias in the LayerNorm feeding the MLP while preserving all projection biases, widths, training settings, and decoding behavior.
mechanism: Absorbable pre-MLP normalization offset removal
evidence_used: Projection-bias removal and feed-forward width reduction failed at 72.22% and 74.39%; this instead preserves those parameters and removes an algebraically redundant normalization offset.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7236, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight-dimensional attention key bias while preserving the baseline RNG consumption and initial weights will reduce parameters from 1,644 to 1,636 and retain at least 99% accuracy.
change: Construct the original biasful QKV layer to preserve initialization randomness, unregister its bias, and replace it with learned query/value biases only.
mechanism: Initialization-preserving softmax-invariant key-bias elimination
evidence_used: The previous key-bias removal reached only 64.55% despite the key bias being mathematically canceled by attention softmax, indicating that altered initialization—not lost representational capacity—likely caused the regression.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9458, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the mathematically redundant eight-element key bias while preserving baseline RNG consumption, parameter registration order, and fused QKV bias addition will reduce parameters from 1,644 to 1,636 and recover at least 99% accuracy.
change: Resize the existing QKV bias parameter to query/value biases only, reconstruct a zero key-bias slice during the forward pass, and retain the original fused linear operation.
mechanism: Optimizer-order-preserving key-bias elimination
evidence_used: Initialization-preserving key-bias removal improved accuracy from 64.55% to 94.58%; preserving the original parameter slot and fused QKV computation targets the remaining optimizer-order and numerical differences without restoring redundant parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9971, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Building on the verified 1,636-parameter key-bias design, fixing one common-mode degree of freedom in the terminal MLP bias will reduce the model to 1,635 parameters while retaining at least 99% accuracy, because the immediately following final LayerNorm exactly cancels uniform shifts across the eight residual channels.
change: Apply the verified initialization- and optimizer-order-preserving key-bias removal, then store seven terminal MLP output-bias coordinates and reconstruct an eight-dimensional bias with the final coordinate fixed to zero.
mechanism: Gauge-fixed terminal MLP bias plus softmax-invariant key-bias elimination
evidence_used: The optimizer-order-preserving key-bias design achieved 99.71% accuracy with 1,636 parameters, whereas larger or optimization-disrupting reductions failed; this motivates removing only one additional algebraically redundant parameter while preserving widths and all weight matrices.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7337, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified key-bias elimination with a seven-parameter terminal MLP bias will reach at least 99% accuracy at 1,635 parameters if training reconstructs the omitted coordinate’s gradient and preserves the original eight-coordinate AdamW and gradient-clipping dynamics.
change: Remove the redundant attention key bias and one common-mode terminal MLP bias parameter, while maintaining virtual optimizer state for the eliminated MLP coordinate.
mechanism: Quotient-space terminal bias with full-coordinate AdamW dynamics
evidence_used: The optimizer-order-preserving key-bias design achieved 99.71% at 1,636 parameters, while the direct seven-coordinate terminal-bias parameterization fell to 73.37%; this isolates altered optimization geometry as the most informative issue to address.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9868000000000001, "parameters": 1635, "training_steps": 4999}

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
