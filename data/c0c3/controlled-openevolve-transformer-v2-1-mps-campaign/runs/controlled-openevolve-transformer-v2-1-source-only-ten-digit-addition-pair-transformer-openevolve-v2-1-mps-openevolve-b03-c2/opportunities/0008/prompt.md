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
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1643, "training_steps": 4999}
prior_hypothesis: Fixing only one key-bias coordinate at zero will reduce the model to 1643 parameters while retaining at least 99% accuracy, because any constant key bias is mathematically canceled by attention softmax and this minimal parameterization change should perturb optimization less than removing all eight key-bias coordinates.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower parameters from 1644 to 1576 while preserving at least 99% accuracy because the verified model has substantial accuracy headroom at 99.96% and retains its full attention and embedding capacity.
change: Change the default feed-forward width from 12 to 8, removing 68 learned parameters.
mechanism: Narrower nonlinear feed-forward bottleneck
evidence_used: The current `d_model=8, d_ff=12` design achieved 0.9996 accuracy with 1644 parameters, motivating a focused reduction in the feed-forward sublayer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7129000000000001, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: A `d_ff` width of 10 will retain at least 99% accuracy while reducing the model from 1644 to 1610 parameters, locating whether the capacity threshold lies between the failed width 8 and verified width 12.
change: Change the default feed-forward width from 12 to 10.
mechanism: Intermediate feed-forward bottleneck search
evidence_used: The width-12 model reached 99.96% accuracy, while width 8 fell to 71.29%; testing the midpoint is the most informative next capacity reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1277, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will retain at least 99% accuracy while reducing parameters from 1644 to 1627.
change: Change the default feed-forward width from 12 to 11.
mechanism: Single-neuron feed-forward boundary search
evidence_used: Width 12 achieved 99.96% accuracy, while widths 10 and 8 failed; width 11 is the remaining untested point at the observed capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7462000000000001, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the 8-dimensional key-projection bias will reduce parameters from 1644 to 1636 while preserving at least 99% accuracy, because adding the same learned key bias to every position shifts all unmasked attention logits for a query by the same constant and therefore cannot change the softmax attention weights.
change: Replace the fused QKV bias with separate learned query and value biases, leaving the key projection bias-free.
mechanism: Eliminate softmax-invariant key bias
evidence_used: The verified 1644-parameter design achieved 99.96%, while every reduction of the feed-forward width failed; this motivates removing a functionally redundant attention parameter instead of further reducing demonstrated nonlinear capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9460999999999999, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the softmax-invariant key bias while preserving the fused QKV operation and seeded initialization trajectory will achieve at least 99% accuracy with 1636 parameters.
change: Instantiate the original biased fused projection to preserve RNG consumption, discard its bias, add learned query/value biases, and reconstruct the fused bias with a fixed zero key segment.
mechanism: RNG-preserving fused key-bias elimination
evidence_used: The 1644-parameter model achieved 99.96%, while the prior 1636-parameter key-bias trial reached only 94.61% despite the key bias being functionally redundant; preserving the original fused projection and initialization trajectory isolates that redundancy more precisely.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.47090000000000004, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing only one key-bias coordinate at zero will reduce the model to 1643 parameters while retaining at least 99% accuracy, because any constant key bias is mathematically canceled by attention softmax and this minimal parameterization change should perturb optimization less than removing all eight key-bias coordinates.
change: Preserve the original fused QKV layer and constructor RNG consumption, replace its 24-element bias with 23 learned elements, and reconstruct one fixed-zero key-bias coordinate during the fused projection.
mechanism: Single-coordinate softmax-invariant key-bias removal
evidence_used: The 1644-parameter baseline reached 99.96%, whereas removing all eight theoretically redundant key-bias parameters failed; a one-coordinate reduction is the smallest informative probe of that optimization sensitivity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1643, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing two key-bias coordinates at zero will reduce the model from 1644 to 1642 parameters while retaining at least 99% accuracy, because the qualified one-coordinate design achieved 99.88% and every constant key-bias coordinate is canceled by attention softmax.
change: Preserve the fused QKV layer and constructor RNG consumption, replace its 24-element bias with 22 learned elements, and reconstruct two fixed-zero key-bias coordinates during projection.
mechanism: Two-coordinate softmax-invariant key-bias removal
evidence_used: The 1643-parameter one-coordinate design achieved 99.88%, while removing all eight key-bias coordinates failed; removing one additional invariant coordinate is the smallest informative continuation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9852, "parameters": 1642, "training_steps": 4999}



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
