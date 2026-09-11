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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9998, "parameters": 1618, "training_steps": 4999}
prior_hypothesis: Fixing one coordinate of the second pre-LayerNorm bias to zero will reduce parameters from 1,619 to 1,618 while retaining at least 99% accuracy, because its effect can be absorbed exactly into the retained `fc1` bias.

## Recent verification evidence

RECENT RESULT
hypothesis: Compressing each learned positional vector from 8 dimensions to the 4-dimensional attention-head width and broadcasting it across heads will retain at least 99% accuracy while removing `max_seq_len * 4` parameters.
change: Store one head-width positional embedding per sequence position and repeat it across the model dimension before adding token embeddings.
mechanism: Head-width learned positional bottleneck
evidence_used: The current 1-layer, 8-dimensional model reaches 99.96% accuracy; its two attention heads each operate on only four dimensions, motivating a rank-4 positional representation while preserving the successful attention and MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5039, "parameters": 1552, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing linear-layer biases while preserving the full 8-dimensional positional embeddings will retain at least 99% accuracy because pre-LayerNorm affine terms and residual paths provide learned offsets without reducing attention or MLP rank.
change: Disable biases in both attention projections and both MLP projections, reducing learned parameters while leaving model width, positional capacity, and training unchanged.
mechanism: Bias-free transformer projections
evidence_used: The rank-4 positional bottleneck reduced accuracy from 99.96% to 50.39%, showing that positional capacity should be preserved; this patch instead removes projection offsets that are typically redundant in a pre-LayerNorm transformer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.14880000000000002, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will preserve at least 99% accuracy while removing 17 learned parameters, because it retains the full positional embeddings and all projection biases shown to be important by the two failed compressions.
change: Change the default feed-forward width from 12 to 11 without altering the successful attention, embedding, bias, or training configuration.
mechanism: Single-neuron MLP width reduction
evidence_used: The 1,644-parameter baseline reached 99.96% accuracy, while positional compression and wholesale bias removal collapsed accuracy; this motivates a smaller, orthogonal reduction in MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7501000000000001, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the value-projection bias will retain at least 99% accuracy while reducing parameters from 1,644 to 1,636, because softmax attention weights sum to one and the value bias can therefore be represented exactly by the retained attention output bias.
change: Replace the combined QKV bias with learned query and key biases only; retain every other model dimension, bias, positional embedding, and training setting.
mechanism: Algebraically redundant value-bias removal
evidence_used: Removing all projection biases collapsed accuracy to 14.88%, so most biases should remain; unlike those broader removals, the value bias is mathematically redundant with the attention output bias and does not reduce the model’s representable function class.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7171, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one algebraically redundant value-projection bias coordinate to zero while preserving the baseline constructor RNG sequence will reduce parameters from 1,644 to 1,643 and retain at least 99% accuracy.
change: Retain all query and key biases and seven of eight value biases; reconstruct the final value-bias coordinate as a fixed zero during the QKV projection.
mechanism: Single-coordinate value-bias pruning with initialization preservation
evidence_used: The baseline reached 99.96%, whereas removing all eight value biases reached only 71.71%; pruning just one redundant coordinate is the smallest informative compression and avoids changing subsequent weight initialization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1643, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second algebraically redundant value-projection bias coordinate to zero will reduce parameters from 1,643 to 1,642 while retaining at least 99% accuracy.
change: Preserve constructor RNG consumption, retain six of eight learned value biases, and reconstruct the final two value-bias coordinates as fixed zeros during projection.
mechanism: Two-coordinate value-bias pruning with initialization preservation
evidence_used: Pruning one value-bias coordinate with unchanged initialization achieved 99.96% accuracy at 1,643 parameters; extending that successful compression by one coordinate is the smallest informative next change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5146000000000001, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all eight key-projection bias parameters will reduce the model from 1,643 to 1,635 parameters while retaining at least 99% accuracy, because adding the same learned key bias to every attended position changes each query’s attention logits by a row-wise constant that softmax cancels exactly.
change: Preserve constructor RNG consumption, store only the eight query biases and seven retained value biases, and reconstruct the key bias and final value-bias coordinate as fixed zeros during projection.
mechanism: Softmax-invariant key-bias elimination
evidence_used: Single-coordinate value-bias pruning retained 99.96% accuracy, while removing a second value coordinate failed; key bias is a more informative target because, unlike value bias, it has no effect on the attention output even before considering downstream parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7177, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one softmax-invariant key-bias coordinate to zero will reduce parameters from 1,643 to 1,642 while retaining at least 99% accuracy.
change: Preserve constructor RNG consumption, retain seven of eight key biases and seven of eight value biases, and reconstruct their final coordinates as fixed zeros during QKV projection.
mechanism: Single-coordinate key-bias pruning with initialization preservation
evidence_used: The one-coordinate value-bias reduction achieved 99.96%, while removing all eight key biases achieved only 71.77%; this tests the smallest incremental key-bias reduction without repeating the failed wholesale removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.996, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second key-projection bias coordinate to zero will reduce parameters from 1,642 to 1,641 while retaining at least 99% accuracy, because key bias contributes only a row-wise constant to attention logits.
change: Preserve constructor RNG consumption, retain six of eight key biases and seven of eight value biases, and reconstruct the omitted coordinates as fixed zeros during QKV projection.
mechanism: Incremental softmax-invariant key-bias pruning
evidence_used: Pruning one key-bias coordinate achieved 99.6% accuracy at 1,642 parameters; unlike the failed second value-bias pruning, another key coordinate is softmax-invariant, making this the smallest informative extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7483, "parameters": 1641, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing each learned 8-dimensional positional vector in the 7-dimensional mean-free subspace will reduce parameters from 1,642 to 1,619 while retaining at least 99% accuracy, because per-position all-ones offsets are removed by every pre-LayerNorm and the final LayerNorm.
change: Replace unconstrained positional embeddings with seven learned orthonormal coordinates reconstructed into a mean-zero 8-dimensional vector, while preserving the baseline initialization function and RNG consumption.
mechanism: LayerNorm-quotient positional embeddings
evidence_used: The rank-4 broadcast positional bottleneck failed at 50.39%, showing positional capacity is load-bearing; unlike that compression, this removes only the exact LayerNorm-invariant direction and retains all seven observable positional degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the second pre-LayerNorm bias to zero will reduce parameters from 1,619 to 1,618 while retaining at least 99% accuracy, because its effect can be absorbed exactly into the retained `fc1` bias.
change: Add a LayerNorm variant with seven learned bias coordinates and use it only before the MLP, reconstructing the eighth coordinate as zero without changing initialization RNG consumption.
mechanism: Single-coordinate MLP LayerNorm bias quotient
evidence_used: The current mean-free positional design achieved 99.64% at 1,619 parameters, while broader removals caused optimization failures; this tests the smallest possible reduction in a directly redundant affine parameter.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1618, "training_steps": 4999}



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
