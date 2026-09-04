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
verified_results: {"accuracy": 0.9995, "parameters": 1643, "training_steps": 4999}
prior_hypothesis: Fixing one coordinate of the first LayerNorm bias at zero will reduce the model from 1,644 to 1,643 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offset.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower parameters from 1,644 to 1,576 while retaining at least 99% accuracy, because the current model’s 99.96% accuracy provides margin for a modest MLP-width reduction.
change: Change the default feed-forward width from 12 to 8 while preserving the attention architecture and training procedure.
mechanism: Narrower feed-forward residual bottleneck
evidence_used: The current 1,644-parameter design achieved 99.96% accuracy after 4,999 steps, motivating a conservative 68-parameter reduction in its MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6399, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower the parameter count from 1,644 to 1,627 while retaining at least 99% accuracy.
change: Remove one feed-forward hidden unit while preserving the successful attention architecture and training procedure.
mechanism: Single-neuron feed-forward bottleneck ablation
evidence_used: `d_ff=12` achieved 99.96% accuracy, whereas `d_ff=8` fell to 63.99%; this suggests a sharp capacity threshold and motivates the smallest possible width reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7406, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the 8-dimensional key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because adding the same learned key bias to every position shifts all unmasked attention logits by an equal amount and therefore cannot change the attention probabilities.
change: Replace the combined QKV bias with separate learned query and value biases, leaving the key projection unbiased while preserving `d_ff=12`.
mechanism: Eliminate softmax-invariant key bias
evidence_used: The 1,644-parameter `d_ff=12` model reached 99.96%, while reducing `d_ff` to 11 collapsed accuracy to 74.06%; this motivates preserving demonstrated feed-forward capacity and removing a mathematically redundant attention parameter instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8187000000000001, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the first LayerNorm bias at zero will reduce the model from 1,644 to 1,643 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offset.
change: Add a LayerNorm equivalent with seven learned bias coordinates and one fixed-zero coordinate, and use it only before self-attention.
mechanism: One-coordinate pre-attention bias anchoring
evidence_used: The 1,644-parameter model reached 99.96%, while larger structural reductions collapsed accuracy; this motivates the smallest possible reduction in a bias that is representationally redundant with the following affine projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1643, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second coordinate of the first LayerNorm bias at zero will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offsets.
change: Store six learned bias coordinates in the pre-attention LayerNorm and append two fixed-zero coordinates.
mechanism: Incremental pre-attention bias anchoring
evidence_used: Fixing one LayerNorm bias coordinate reduced the model to 1,643 parameters and achieved 99.95% accuracy, while larger structural reductions failed; this supports testing the smallest incremental extension of the successful redundancy removal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.47759999999999997, "parameters": 1642, "training_steps": 4999}



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
