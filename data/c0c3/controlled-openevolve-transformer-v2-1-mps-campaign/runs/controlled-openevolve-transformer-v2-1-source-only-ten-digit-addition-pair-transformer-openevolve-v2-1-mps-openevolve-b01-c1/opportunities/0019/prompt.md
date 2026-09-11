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
verified_results: {"accuracy": 0.9962000000000001, "parameters": 1594, "training_steps": 4999}
prior_hypothesis: Pruning one `ln1` bias coordinate will reduce parameters from 1,595 to 1,594 while retaining at least 99% accuracy, because its effect can be represented by the downstream QKV and attention-output biases.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Representing the `fc2` bias with seven orthonormal mean-free coordinates will reduce parameters from 1,618 to 1,617 while retaining at least 99% accuracy, because its omitted all-ones component is canceled by the downstream final LayerNorm.
change: Replace the eight-parameter MLP output bias with a seven-parameter basis representation, preserving constructor RNG consumption, MLP width, and all observable bias directions.
mechanism: Final-LayerNorm quotient for MLP output bias
evidence_used: Mean-free positional embeddings removed the same LayerNorm-invariant direction at every position and achieved 99.64% accuracy; the current design subsequently achieved 99.98%, supporting another exact one-parameter LayerNorm quotient instead of reducing load-bearing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the attention output-projection bias with seven mean-free coordinates will reduce parameters from 1,617 to 1,616 while retaining at least 99% accuracy, because its omitted all-ones component propagates unchanged through the residual stream and is canceled by downstream LayerNorms.
change: Reuse `MeanFreeResidualLinear` for the attention output projection, preserving its full weight matrix, all seven observable bias directions, and constructor RNG consumption.
mechanism: Final-LayerNorm quotient for attention output bias
evidence_used: The identical mean-free quotient applied to the MLP output bias reduced the model from 1,618 to 1,617 parameters while retaining 99.98% accuracy; the attention output bias has the same downstream constant-shift invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln2` bias coordinate to zero will reduce parameters from 1,616 to 1,615 while retaining at least 99% accuracy, because both omitted coordinates can be absorbed into the retained `fc1` bias.
change: Retain six of eight second pre-LayerNorm bias coordinates and reconstruct the final two as fixed zeros, leaving all other model capacity and training settings unchanged.
mechanism: Incremental MLP LayerNorm bias absorption
evidence_used: Pruning one `ln2` bias coordinate achieved 99.98% accuracy, and the following affine `fc1` bias provides an exact representational replacement; this is the smallest extension of that successful compression.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln2` bias coordinate to zero will reduce parameters from 1,615 to 1,614 while retaining at least 99% accuracy, because the omitted coordinate can be absorbed into the retained `fc1` bias.
change: Retain five of eight second pre-LayerNorm bias coordinates and reconstruct the final three as fixed zeros, without changing other model capacity or training settings.
mechanism: Incremental MLP LayerNorm bias absorption
evidence_used: Pruning one and then two `ln2` bias coordinates achieved 99.98% and 99.97% accuracy respectively; a third coordinate is the smallest extension of this successful exact bias-absorption mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9812000000000001, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at its initial value of one will reduce parameters from 1,615 to 1,614 while retaining at least 99% accuracy, because the following `fc1` weight column can learn the omitted scale exactly.
change: Retain seven learned `ln2` scale coordinates and six learned bias coordinates, reconstructing the final scale as one and the final two biases as zero.
mechanism: MLP LayerNorm scale–linear weight quotient
evidence_used: Removing two `ln2` bias coordinates retained 99.97% accuracy, while removing a third bias fell to 98.12%; testing the independent scale–weight redundancy is therefore more informative than repeating the failed bias-pruning direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3968, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining both residual-output weight matrices to the seven-dimensional mean-free output subspace will reduce parameters from 1,615 to 1,595 while retaining at least 99% accuracy, because the removed input-dependent all-ones components pass unchanged through residual connections and are canceled by downstream LayerNorms.
change: Represent the attention projection and MLP `fc2` weights with orthonormal mean-free coordinates, reconstruct full weights during forward passes, and preserve full-width initialization RNG consumption.
mechanism: LayerNorm-quotient residual output weights
evidence_used: Mean-free quotients already reduced the positional embeddings and both residual-output biases while retaining 99.64%–99.98% accuracy; applying the same exact LayerNorm-invariant quotient independently to every weight column removes 20 redundant parameters without narrowing observable output capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9961, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Pruning one `ln1` bias coordinate will reduce parameters from 1,595 to 1,594 while retaining at least 99% accuracy, because its effect can be represented by the downstream QKV and attention-output biases.
change: Add a one-coordinate-pruned LayerNorm and use it before causal self-attention, leaving the successful two-coordinate `ln2` pruning unchanged.
mechanism: Single-coordinate attention LayerNorm bias absorption
evidence_used: Two `ln2` bias coordinates were pruned while achieving 99.97% accuracy; because pruning a third fell to 98.12%, the most informative next step is the smallest equivalent bias-absorption test in the previously unpruned `ln1`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9962000000000001, "parameters": 1594, "training_steps": 4999}



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
