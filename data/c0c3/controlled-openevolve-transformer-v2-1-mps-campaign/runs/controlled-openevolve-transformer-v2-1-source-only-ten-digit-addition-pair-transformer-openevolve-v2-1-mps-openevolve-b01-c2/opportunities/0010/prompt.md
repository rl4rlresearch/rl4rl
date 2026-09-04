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
verified_results: {"accuracy": 0.9997, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Removing both the softmax-invariant key bias and the value bias will reduce the model to 1,628 parameters while retaining at least 99% accuracy, because attention weights sum to one, making the value bias a position-independent offset that the learned output-projection bias can represent.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9989, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Removing the eight softmax-invariant key-bias parameters while retaining the baseline’s constructor RNG stream and single fused linear bias addition will achieve at least 99% accuracy in 5,000 steps with 1,636 parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1612, "training_steps": 4999}
prior_hypothesis: Retaining only the effective query bias and removing both pre-sublayer LayerNorm biases will reduce the model to 1,612 parameters while maintaining at least 99% accuracy, because those normalization offsets are representable by the downstream query, attention-output, and MLP biases.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the MLP width from 12 to 8 will remove 68 learned parameters while retaining at least 99% accuracy after 5,000 training steps.
change: Change the default transformer feed-forward width to 8.
mechanism: Narrow feed-forward bottleneck
evidence_used: The current 1,644-parameter model achieves 99.96% accuracy, providing enough margin to test a moderate reduction in MLP capacity while preserving the attention architecture.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4579, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will remove 17 learned parameters while retaining at least 99% accuracy after 5,000 training steps.
change: Change the default transformer feed-forward width to 11.
mechanism: One-unit feed-forward bottleneck reduction
evidence_used: The 12-wide model achieved 99.96% accuracy, while width 8 failed at 45.79%; testing the nearest smaller width conservatively probes the capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7289, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating the 8 key-bias parameters will reduce the model from 1,644 to 1,636 learned parameters while retaining at least 99% accuracy, because adding the same learned bias to every key shifts all attention logits for a query equally and therefore does not change softmax attention weights.
change: Replace the combined QKV bias with separate learned query and value biases, leaving keys unbiased.
mechanism: Remove softmax-invariant key bias
evidence_used: The 1,644-parameter design achieved 99.96% accuracy, whereas narrowing `d_ff` caused severe accuracy loss even at width 11; removing a mathematically redundant attention parameter is therefore better motivated than another capacity reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.862, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Keeping the fused QKV weight layout and original initialization stream while removing only the 8 ineffective key-bias parameters will retain at least 99% accuracy with 1,636 parameters.
change: Resize the fused projection’s learned bias to query and value components only, then apply those components explicitly around the unchanged fused weight projection.
mechanism: Initialization-preserving removal of softmax-invariant key bias
evidence_used: The 1,644-parameter baseline reached 99.96%, while capacity reductions failed; the prior key-bias removal reached 86.2%, motivating a more tightly controlled implementation that preserves the successful model’s fused projection layout and initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9779000000000001, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,636-parameter fused-QKV model will exceed 99% accuracy when trained for 7,500 steps, because its prior 5,000-step result reached 97.79% and suggests insufficient convergence rather than a severe capacity loss.
change: Remove the eight softmax-invariant key-bias parameters while preserving the fused QKV weight layout, and extend the default cosine-scheduled training run from 5,000 to 7,500 steps.
mechanism: Longer optimization for initialization-preserving key-bias removal
evidence_used: Initialization-preserving key-bias removal achieved 97.79% with 1,636 parameters, substantially outperforming the separate-bias version at 86.2% and the narrower 1,627-parameter MLP at 72.89%; additional optimization is therefore the most targeted next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.441, "parameters": 1636, "training_steps": 7499}

RECENT RESULT
hypothesis: Removing the eight softmax-invariant key-bias parameters while retaining the baseline’s constructor RNG stream and single fused linear bias addition will achieve at least 99% accuracy in 5,000 steps with 1,636 parameters.
change: Preserve fused QKV construction, replace its bias with learned query/value components, synthesize the zero key component, and pass the complete bias through one fused `F.linear` call.
mechanism: Fused zero-key-bias reparameterization
evidence_used: The 1,644-parameter baseline achieved 99.96%, while the prior 1,636-parameter implementation reached 97.79% after applying biases separately; preserving the baseline fused bias-add computation targets that remaining numerical difference.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both the softmax-invariant key bias and the value bias will reduce the model to 1,628 parameters while retaining at least 99% accuracy, because attention weights sum to one, making the value bias a position-independent offset that the learned output-projection bias can represent.
change: Preserve the baseline fused QKV construction and single fused linear call, but retain only the query-bias parameters and synthesize zero key and value bias components.
mechanism: Output-bias absorption of constant value bias
evidence_used: The initialization-preserving fused zero-key-bias design achieved 99.89% accuracy with 1,636 parameters; extending the same successful fused reparameterization to the value bias targets another mathematically redundant eight-parameter component without narrowing the failed MLP bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining only the effective query bias and removing both pre-sublayer LayerNorm biases will reduce the model to 1,612 parameters while maintaining at least 99% accuracy, because those normalization offsets are representable by the downstream query, attention-output, and MLP biases.
change: Apply the verified query-only fused-QKV reparameterization and disable the biases in `ln1` and `ln2`.
mechanism: Pre-normalization bias absorption
evidence_used: The 1,628-parameter query-only-bias design achieved 99.97% accuracy; its margin motivates removing 16 additional parameters that are initialized to zero and functionally absorbable by existing downstream biases.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both pre-sublayer LayerNorm scale vectors will reduce the model from 1,612 to 1,596 parameters while maintaining at least 99% accuracy, because each scale can be represented exactly by the following QKV or MLP input-projection weights.
change: Make `ln1` and `ln2` entirely parameter-free while retaining their normalization behavior.
mechanism: Downstream absorption of pre-normalization scales
evidence_used: The 1,612-parameter design reached 99.95% accuracy after removing both pre-sublayer normalization biases; the remaining scales are more directly redundant because they multiply normalized features immediately before learned linear projections.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7963, "parameters": 1596, "training_steps": 4999}



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
