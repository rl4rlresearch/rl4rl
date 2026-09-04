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
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Removing only the 8 key-projection bias parameters will reduce the model from 1644 to 1636 parameters while retaining at least 99% accuracy, because a position-independent key bias adds the same attention-logit offset for every attended key and cancels exactly under softmax.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the sole MLP width from 12 to 8 will lower parameters from 1644 to 1576 while retaining at least 99% accuracy.
change: Change the default feed-forward dimension to match the 8-dimensional residual stream.
mechanism: Narrower feed-forward bottleneck
evidence_used: The current 1644-parameter model achieves 99.96% accuracy, providing substantial accuracy margin for a 68-parameter reduction in its MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4484, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the query bias and full 12-unit MLP while eliminating key and value biases will reduce parameters from 1644 to 1628 and preserve at least 99% accuracy.
change: Make the packed QKV projection bias-free, add back only a learned query bias, and apply it after splitting Q, K, and V.
mechanism: Remove functionally redundant key/value attention biases
evidence_used: Reducing the MLP width to 8 cut accuracy to 44.84%, so this patch preserves the successful nonlinear capacity. Key bias cancels inside softmax, while the value-bias contribution can be represented by the existing output-projection bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5273, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the 8 key-projection bias parameters will reduce the model from 1644 to 1636 parameters while retaining at least 99% accuracy, because a position-independent key bias adds the same attention-logit offset for every attended key and cancels exactly under softmax.
change: Replace the packed QKV bias with separate learned query and value biases, leaving the key projection bias-free while preserving all other architecture and training settings.
mechanism: Exact softmax-invariant key-bias removal
evidence_used: Removing both key and value biases reduced accuracy to 52.73%, so value-bias removal may have disrupted optimization; isolating the mathematically redundant key bias tests a smaller, function-preserving reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Disabling both LayerNorm bias vectors will reduce the model from 1636 to 1620 parameters while retaining at least 99% accuracy, because the MLP LayerNorm offset is absorbable by `fc1.bias`, while the attention LayerNorm offset is absorbable by the retained query/value biases and produces only a softmax-invariant constant key shift.
change: Keep both learned LayerNorm scale vectors but construct the two block LayerNorms without learned biases.
mechanism: Remove affine offsets duplicated by downstream biases
evidence_used: The 1636-parameter design reaches 99.84% accuracy while preserving query and value biases; removing value bias caused optimization failure, so this patch retains those successful parameters and instead removes upstream offsets that duplicate them and the existing MLP input bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9112, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the 8-parameter attention LayerNorm bias will reduce the model from 1636 to 1628 parameters while retaining at least 99% accuracy, because its query and value effects are absorbable by the retained biases and its key effect cancels under softmax.
change: Disable the learned bias only in `ln1`, preserving `ln2` and every other successful architectural and training setting.
mechanism: Attention-LayerNorm offset absorption
evidence_used: Removing both block LayerNorm biases reached only 91.12%, but that experiment confounded the attention and fragile MLP paths; the 1636-parameter model reached 99.84%, so isolating the theoretically redundant attention-side offset is the most informative smaller change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.848, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing each head’s query bias with one learned scalar will reduce parameters from 1636 to 1630 while retaining at least 99% accuracy, because within-head Q/K rotations can orient any query-bias vector along one coordinate without changing attention logits.
change: Replace the eight-element query bias with two headwise scalars expanded onto the first coordinate of each four-dimensional head.
mechanism: Headwise canonical query-bias gauge
evidence_used: Exact removal of the softmax-redundant key bias retained 99.84% accuracy, whereas removing the attention LayerNorm offset failed; this motivates preserving a learned query offset while removing only its rotationally redundant coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5251, "parameters": 1630, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 12 to 11 will lower parameters from 1636 to 1619 while retaining at least 99% accuracy.
change: Remove one hidden unit from the sole feed-forward layer while preserving the successful attention and normalization design.
mechanism: One-neuron MLP capacity titration
evidence_used: The 12-unit model achieved 99.84%, while the larger reduction to 8 units collapsed to 44.84%; testing 11 isolates the smallest capacity reduction and avoids altering optimization-sensitive attention biases.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9544, "parameters": 1619, "training_steps": 4999}



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
