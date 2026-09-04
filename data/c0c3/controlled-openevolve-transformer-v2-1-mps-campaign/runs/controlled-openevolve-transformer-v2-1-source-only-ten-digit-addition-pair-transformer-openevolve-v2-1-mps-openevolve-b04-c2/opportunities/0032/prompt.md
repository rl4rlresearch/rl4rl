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
verified_results: {"accuracy": 1.0, "parameters": 1612, "training_steps": 4999}
prior_hypothesis: Removing the verified redundant QKV biases and quotienting one key-projection row into a seven-dimensional zero-sum basis will produce a 1,612-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1613, "training_steps": 4999}
prior_hypothesis: Removing the remaining zero-initialized query bias from the verified 1,621-parameter design will produce a 1,613-parameter model with at least 99% accuracy, because a learned pre-attention LayerNorm bias and query projection can jointly represent a constant query offset without changing the initial forward pass.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1610, "training_steps": 4999}
prior_hypothesis: Extending the verified two-row key quotient to the adjacent third-from-last key row will produce a 1,610-parameter model with at least 99% accuracy, because each compacted row independently removes only a position-constant key component that cancels inside attention softmax.

## Recent verification evidence

RECENT RESULT
hypothesis: Combining the verified final-four positional quotient with removal of one functionally redundant `ln2` bias coordinate will produce a 1,628-parameter model with at least 99% accuracy, because `fc1.bias` can absorb the fixed LayerNorm offset.
change: Extend positional compaction from the final three to the final four rows, then store seven coordinates of the second LayerNorm bias and reconstruct its eighth coordinate as zero.
mechanism: Downstream-bias absorption with tail positional gauge fixing
evidence_used: The final-four positional design achieved 99.81% at 1,629 parameters, while extending compaction to the fifth-from-last row fell to 96.93%; this motivates seeking the next reduction in the affine redundancy between `ln2.bias` and the following `fc1.bias`.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9329000000000001, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the single global common mode of the tied token-embedding matrix on top of the verified final-four positional design will produce a 1,628-parameter model with accuracy at least 99%, because this mode only adds LayerNorm-canceled input shifts and a softmax-invariant common logit shift.
change: Extend positional compaction to the verified final four rows, represent the tied token matrix in an implicit orthonormal zero-sum basis, and preserve its virtual full-coordinate AdamW dynamics.
mechanism: Global tied-embedding gauge quotient with verified tail positional compaction
evidence_used: The final-four positional design achieved 99.81% at 1,629 parameters, while extending to a fifth tail row fell to 96.93% and removing an `ln2` coordinate fell to 93.29%; this tests a different exact one-dimensional symmetry without further compressing sensitive positional or LayerNorm parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7387999999999999, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all eight value-bias parameters will reduce the verified 1,629-parameter model to 1,621 parameters while retaining at least 99% accuracy, because softmax attention rows sum to one and therefore any constant value bias is exactly absorbable by the learned output-projection bias.
change: Store only the learned query bias in `qkv.bias`; reconstruct both key and value biases as zero while retaining the full learned attention weights and output-projection bias.
mechanism: Value-bias absorption into the attention output projection
evidence_used: The current design reaches 99.81% at 1,629 parameters, while further positional, LayerNorm, output-bias, and embedding quotients failed. The already successful removal of the equally functionally redundant key bias motivates testing the distinct exact redundancy between value bias and output-projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Building on the verified 1,621-parameter design, fixing the mean final-LayerNorm gain to one will yield a 1,620-parameter model with at least 99% accuracy because positive global scaling of its gain and bias cannot change greedy argmax decoding.
change: Apply the verified value-bias removal and final-four positional quotient, then represent the final LayerNorm gain as seven learned zero-mean deviations around a fixed unit mean.
mechanism: Final-LayerNorm output-scale gauge fixing
evidence_used: The value-bias-free, final-four-position design achieved 99.93% accuracy at 1,621 parameters; unlike the failed fifth positional-row and hidden LayerNorm-bias reductions, this removes only a global output scale that is irrelevant to the protected decoder’s argmax.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0319, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the remaining zero-initialized query bias from the verified 1,621-parameter design will produce a 1,613-parameter model with at least 99% accuracy, because a learned pre-attention LayerNorm bias and query projection can jointly represent a constant query offset without changing the initial forward pass.
change: Apply the verified value-bias removal and final-four positional quotient, then remove the query bias so the QKV projection is entirely bias-free.
mechanism: Bias-free attention with verified tail positional quotient
evidence_used: The 1,621-parameter design removed all value-bias parameters and retained 99.93% accuracy. Unlike the failed positional and final-LayerNorm reductions, eliminating query bias preserves initialization exactly and tests a distinct affine redundancy through `ln1.bias` and the query weight.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the attention output-projection bias to the seven-dimensional zero-sum subspace will reduce the verified 1,613-parameter model to 1,612 parameters while retaining at least 99% accuracy, because its omitted uniform component passes unchanged through both residual additions and is canceled exactly by the downstream LayerNorms.
change: Store the attention output-projection bias in the same orthonormal zero-sum basis used for the verified terminal MLP bias quotient, reconstruct it in the forward pass, and preserve virtual eight-coordinate AdamW moments during training.
mechanism: Attention-output common-mode gauge fixing
evidence_used: The current bias-free-attention design achieved 99.84% accuracy at 1,613 parameters while already using a successful common-mode quotient for `fc2.bias`; applying that same exact residual-stream symmetry to `attn.proj.bias` is a distinct one-parameter reduction that avoids the positional, LayerNorm, and embedding reductions that previously failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9887, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the verified redundant QKV biases and quotienting one key-projection row into a seven-dimensional zero-sum basis will produce a 1,612-parameter model with at least 99% accuracy.
change: Make attention bias-free, store one key row without its LayerNorm-induced constant-offset coordinate, reconstruct its computation from the un-affined normalized residual stream, and preserve virtual eight-coordinate AdamW dynamics.
mechanism: Single-row key-projection gauge quotient
evidence_used: The bias-free 1,613-parameter design achieved 99.84%; the failed 1,612-parameter output-bias quotient reached 98.87%, motivating a distinct exact attention symmetry where a key component constant across positions cancels from every softmax row.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified single-row key quotient to the adjacent penultimate key row will produce a 1,611-parameter model with at least 99% accuracy, because each removed coordinate independently contributes only a position-constant key offset that cancels inside attention softmax.
change: Adopt the verified bias-free attention and final-four positional quotient, then store the final two key-projection rows in independent seven-dimensional zero-sum bases with reconstructed full-coordinate AdamW updates.
mechanism: Two-row key-projection gauge quotient
evidence_used: The 1,612-parameter single-key-row quotient achieved 100% accuracy, while the alternative 1,612-parameter attention-output-bias quotient missed at 98.87%; incrementally quotienting a second key row therefore tests the strongest demonstrated reduction mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-row key quotient to the adjacent third-from-last key row will produce a 1,610-parameter model with at least 99% accuracy, because each compacted row independently removes only a position-constant key component that cancels inside attention softmax.
change: Store the final three key-projection rows in independent seven-dimensional zero-sum bases while retaining their reconstructed full-coordinate AdamW updates.
mechanism: Three-row key-projection gauge quotient
evidence_used: The current two-row key quotient achieved 100% accuracy at 1,611 parameters, and the preceding one-row quotient also achieved 100% at 1,612 parameters; this supports the smallest incremental extension of the strongest verified reduction mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-row key quotient to the adjacent fourth-from-last key row will produce a 1,609-parameter model with at least 99% accuracy, because each removed common-mode coordinate contributes only a position-constant key offset that cancels in attention softmax.
change: Store the final four key-projection rows in independent seven-dimensional zero-sum bases and preserve their reconstructed eight-coordinate AdamW updates.
mechanism: Four-row key-projection gauge quotient
evidence_used: One-, two-, and three-row key quotients achieved 100%, 100%, and 99.96% accuracy at 1,612, 1,611, and 1,610 parameters respectively, making a one-row extension the strongest supported next reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Quotienting four key rows in one batched parameter will produce a 1,609-parameter model that completes training and achieves at least 99% accuracy; the three-row quotient achieved 99.96%, while the previous four-row attempt reported only a timeout, and batching removes its per-row execution overhead.
change: Compact the final four key-projection rows into one seven-coordinate matrix, fuse the remaining query/key/value rows into one projection, and update the compact matrix with virtual eight-coordinate AdamW moments.
mechanism: Batched four-row key-projection gauge quotient
evidence_used: One-, two-, and three-row key quotients achieved 100%, 100%, and 99.96% accuracy respectively; the four-row extension had no reported accuracy failure because verification timed out, motivating an execution-efficient retry of the same incremental reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1,609-parameter four-row key quotient will complete within the time limit and retain at least 99% accuracy when its compact rows are batched, the remaining Q/K/V projections are fused, redundant LayerNorm computation is eliminated, and validation frequency is reduced.
change: Extend key compaction from three to four rows, store all compact rows in one matrix with batched virtual AdamW updates, fuse the other attention projections, compute normalization once per block, and evaluate every 1,000 steps.
mechanism: Fused four-row key gauge quotient with single-pass normalization
evidence_used: Three compacted key rows achieved 99.96% at 1,610 parameters; both four-row attempts timed out without an accuracy failure, so the strongest supported next test is the same one-parameter reduction with additional execution savings.
result: training did not finish within the verification time limit



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
