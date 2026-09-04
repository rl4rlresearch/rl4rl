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
verified_results: {"accuracy": 0.9994, "parameters": 1586, "training_steps": 4999}
prior_hypothesis: Removing all seven remaining learned key-bias coordinates will reduce parameters from 1,593 to 1,586 while retaining at least 99% accuracy, because each head’s position-independent key bias adds the same scalar to every unmasked attention logit and is canceled exactly by softmax.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing a second `ln1` bias coordinate to zero will reduce parameters from 1,594 to 1,593 while retaining at least 99% accuracy, because its effect can be absorbed by the learned QKV and attention-output biases.
change: Use the existing two-coordinate-pruned LayerNorm before causal self-attention, leaving all other model capacity, initialization, and training settings unchanged.
mechanism: Incremental attention LayerNorm bias absorption
evidence_used: Pruning one `ln1` bias coordinate achieved 99.62% accuracy at 1,594 parameters, while two-coordinate `ln2` bias pruning achieved 99.97%; this is the smallest incremental test of the same successful bias-absorption mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing two independent key/value heads with one shared learned key/value head will reduce parameters from 1,593 to 1,523 while retaining at least 99% accuracy, because the two distinct query heads can retrieve both operand digits from the same content-addressed memory.
change: Keep two learned query heads but broadcast one four-dimensional learned key/value head across them, preserving the original full-width initialization draws so the experiment isolates head sharing.
mechanism: Multi-query attention with a shared learned key/value memory
evidence_used: The current two-head model reaches 99.88%, while recent one-parameter affine pruning has become optimization-sensitive and yielded diminishing reductions. This challenges the load-bearing assumption that each query head requires its own key and value projections, while retaining the two independent attention distributions likely needed for simultaneous operand retrieval.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3638, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln1` bias coordinate to zero will reduce parameters from 1,593 to 1,592 while retaining at least 99% accuracy, because the downstream QKV biases can absorb the omitted coordinate.
change: Add a three-coordinate-pruned LayerNorm and use it only before causal self-attention, leaving the successful two-coordinate `ln2` pruning and all other settings unchanged.
mechanism: Incremental attention LayerNorm bias absorption
evidence_used: Two-coordinate `ln1` pruning achieved 99.88% at 1,593 parameters; although third-coordinate `ln2` pruning reached only 98.12%, testing the same smallest incremental reduction in the still-successful attention path is the most direct next experiment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0641, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all seven remaining learned key-bias coordinates will reduce parameters from 1,593 to 1,586 while retaining at least 99% accuracy, because each head’s position-independent key bias adds the same scalar to every unmasked attention logit and is canceled exactly by softmax.
change: Retain learned query biases and the seven-coordinate value bias, but reconstruct the entire key bias as zeros while preserving the original linear constructor and initialization behavior.
mechanism: Softmax-null key-bias elimination
evidence_used: The current design achieved 99.88% at 1,593 parameters, whereas further LayerNorm-coordinate pruning removed useful capacity and failed; key bias elimination instead removes an exact attention-softmax invariance without narrowing learned queries, keys, values, or attention distributions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the seven remaining value-bias coordinates will reduce parameters from 1,586 to 1,579 while retaining at least 99% accuracy, because normalized attention adds the same value-bias vector at every position and its mean-free projected effect can be represented exactly by the learned attention output bias.
change: Retain only the eight query-bias parameters in QKV and reconstruct both key and value biases as zeros, preserving the original constructor RNG consumption and all projection weights.
mechanism: Value-bias absorption into attention output bias
evidence_used: Exact removal of all key-bias coordinates achieved 99.94% at 1,586 parameters, while the mean-free attention projection retains seven learned output-bias directions; this supports eliminating the remaining algebraically redundant attention bias rather than pruning load-bearing LayerNorm capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.33549999999999996, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Pruning one additional value-bias coordinate will reduce parameters from 1,586 to 1,585 while retaining at least 99% accuracy, because normalized attention contributes that coordinate as a position-independent offset whose projected effect can be learned by the attention output bias.
change: Retain six learned value-bias coordinates instead of seven, reconstructing the final two as zeros while preserving all projection weights, query biases, constructor RNG consumption, and training settings.
mechanism: Incremental value-bias absorption into attention output bias
evidence_used: Removing all seven remaining value-bias coordinates failed at 33.55%, despite their algebraic redundancy, indicating optimization sensitivity; testing a single additional coordinate is the smallest informative extension of the current 99.94%-accurate design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9731000000000001, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Restoring the successful full value bias while removing one exact global-shift degree of freedom from the tied token/output embedding will yield 1,585 parameters and retain at least 99% accuracy.
change: Restore all eight value-bias coordinates, then represent the tied embedding in an orthonormal 111-dimensional mean-free basis while preserving constructor and initialization RNG consumption.
mechanism: Global tied-embedding shift quotient
evidence_used: The 1,586-parameter key-bias-free design achieved 99.94%, whereas pruning one value-bias coordinate fell to 97.31%. Mean-free quotient parameterizations already preserved accuracy for positional embeddings and residual outputs, motivating an exact tied-embedding gauge reduction instead.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1586, "training_steps": 4999}



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
