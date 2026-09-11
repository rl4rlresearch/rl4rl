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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9971, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Removing the mathematically redundant eight-element key bias while preserving baseline RNG consumption, parameter registration order, and fused QKV bias addition will reduce parameters from 1,644 to 1,636 and recover at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing redundant projection biases will reduce parameters from 1,644 to 1,592 while preserving at least 99% accuracy.
change: Disable biases in all attention and MLP linear layers without changing model width, depth, training, or decoding.
mechanism: Bias-free transformer projections
evidence_used: The current 1,644-parameter model reaches 99.96% accuracy, providing enough margin for a conservative 52-parameter reduction that preserves every learned weight matrix.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7222, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the attention key bias will reduce parameters from 1,644 to 1,636 while preserving at least 99% accuracy because a shared key bias adds the same constant to every visible attention logit and cancels under softmax.
change: Replace the combined QKV bias with learned query and value biases, leaving the key projection unbiased while retaining every other projection bias.
mechanism: Softmax-invariant key-bias elimination
evidence_used: Removing all 52 attention and MLP biases reduced accuracy to 72.22%, so those biases should not be removed wholesale; this patch targets only the eight-dimensional key bias whose contribution is mathematically invariant under attention softmax.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6455, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower the model from 1,644 to 1,627 deduplicated learned parameters while retaining at least 99% accuracy.
change: Remove one MLP hidden unit while preserving every attention, MLP, and normalization bias.
mechanism: Single-neuron feed-forward width ablation
evidence_used: The 1,644-parameter design achieved 99.96% accuracy, while broader and attention-specific bias removal fell below the threshold; this motivates a conservative capacity reduction that leaves all apparently important biases intact.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7439, "parameters": 1627, "training_steps": 4999}

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
