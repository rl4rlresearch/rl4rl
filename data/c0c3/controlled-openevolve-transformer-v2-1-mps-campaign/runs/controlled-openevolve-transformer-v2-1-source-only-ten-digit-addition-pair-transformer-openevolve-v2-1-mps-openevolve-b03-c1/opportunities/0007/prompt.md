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
verified_results: {"accuracy": 0.9976999999999999, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Disabling the second LayerNorm bias will reduce parameters from 1,636 to 1,628 while retaining at least 99% accuracy, because the following `fc1` bias can represent the same pre-GELU offset.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower deduplicated parameters from 1,644 to 1,576 while retaining at least 99% accuracy after 5,000 training steps.
change: Change the default feed-forward width to 8 without altering the learned attention architecture or training procedure.
mechanism: Narrower feed-forward bottleneck
evidence_used: The current 1,644-parameter model achieved 99.96% accuracy at step 4,999, providing substantial accuracy margin for a moderate 68-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.71, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the key-projection bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy because a position-independent key bias adds the same constant to every unmasked attention score and therefore cannot change softmax attention weights.
change: Replace the combined QKV bias with learned query and value biases, preserving the full d_ff=12 architecture while eliminating the redundant eight-parameter key bias.
mechanism: Remove softmax-invariant key bias
evidence_used: The 1,644-parameter d_ff=12 model achieved 99.96% accuracy, whereas narrowing d_ff to 8 reduced accuracy to 71%; this motivates preserving feed-forward capacity and removing a mathematically redundant attention parameter instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9593, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower parameters from 1,644 to 1,627 while retaining at least 99% accuracy.
change: Remove one feed-forward hidden unit while preserving the attention architecture and training procedure.
mechanism: Single-neuron feed-forward narrowing
evidence_used: The `d_ff=12` baseline reached 99.96%, while the larger reduction to `d_ff=8` fell to 71%; testing the smallest possible width reduction is the most informative next capacity-boundary experiment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7413, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight key-bias parameters while preserving the baseline constructor RNG stream and parameter ordering will retain at least 99% accuracy with 1,636 parameters.
change: Retain the combined QKV weight, replace its 24-element bias with 16 learned query/value biases, and apply those biases explicitly.
mechanism: RNG-aligned removal of softmax-invariant key bias
evidence_used: The 1,644-parameter baseline reached 99.96%; the prior key-bias removal reached 95.93% despite key bias being mathematically irrelevant, motivating an initialization-aligned implementation that isolates the parameter removal from an RNG-trajectory change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9909, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Disabling the second LayerNorm bias will reduce parameters from 1,636 to 1,628 while retaining at least 99% accuracy, because the following `fc1` bias can represent the same pre-GELU offset.
change: Construct `ln2` without a learned bias while preserving its learned scale, the full `d_ff=12` capacity, initialization RNG stream, and training procedure.
mechanism: Remove affine redundancy before the MLP
evidence_used: The 1,636-parameter RNG-aligned design achieved 99.09%, while reducing feed-forward width to 11 or 8 failed badly; this motivates removing an exactly absorbable affine parameter instead of reducing MLP capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight value-projection bias parameters will reduce the model from 1,628 to 1,620 parameters while retaining at least 99% accuracy, because attention weights sum to one and therefore the value bias contributes a position-independent vector that the following output-projection bias can represent exactly.
change: Retain only the learned query bias in the RNG-aligned combined QKV module and stop adding a separate value bias.
mechanism: Remove output-bias-redundant value bias
evidence_used: The current RNG-aligned 1,628-parameter design achieved 99.77% accuracy, and the successful LayerNorm-bias removal shows that eliminating exactly absorbable affine parameters can preserve accuracy without narrowing the capacity-critical feed-forward network.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3085, "parameters": 1620, "training_steps": 4999}



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
