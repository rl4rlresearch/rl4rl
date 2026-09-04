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
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Removing only the attention key bias will reduce parameters from 1644 to 1636 while preserving at least 99% accuracy because adding the same bias to every key changes each query’s attention logits by a row-wise constant, which softmax cancels exactly.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the sole MLP hidden width from 12 to 11 will lower parameters from 1644 to 1627 while retaining at least 99% accuracy, given the current model’s 99.96% accuracy margin.
change: Change the default feed-forward width from 12 to 11 while preserving architecture, attention, training, and decoding behavior.
mechanism: One-neuron feed-forward bottleneck reduction
evidence_used: The current 1644-parameter design achieved 0.9996 accuracy at step 4999, motivating a conservative 17-parameter capacity reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7221, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the attention key bias will reduce parameters from 1644 to 1636 while preserving at least 99% accuracy because adding the same bias to every key changes each query’s attention logits by a row-wise constant, which softmax cancels exactly.
change: Replace the fused 24-element QKV bias with learned query and value biases while fixing the functionally redundant key bias to zero; preserve the feed-forward width of 12.
mechanism: Softmax-invariant key-bias elimination
evidence_used: The 1644-parameter width-12 model reached 99.96%, whereas reducing d_ff to 11 collapsed accuracy to 72.21%; this motivates retaining nonlinear capacity and removing an attention parameter that cannot affect the model’s output.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining only the QKV query bias will reduce the model to 1628 parameters while maintaining at least 99% accuracy, because key bias cancels in softmax and value bias can be represented exactly by the existing attention output-projection bias.
change: Replace the fused QKV bias with an 8-element learned query bias and fixed-zero key/value biases, preserving all weight matrices, MLP capacity, training, and decoding behavior.
mechanism: Softmax-invariant key-bias and output-bias-absorbed value-bias elimination
evidence_used: Removing key bias alone achieved 99.84% accuracy with 1636 parameters, while reducing MLP width failed at 72.21%; this motivates eliminating another functionally redundant attention bias without reducing nonlinear capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.47909999999999997, "parameters": 1628, "training_steps": 4999}



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
