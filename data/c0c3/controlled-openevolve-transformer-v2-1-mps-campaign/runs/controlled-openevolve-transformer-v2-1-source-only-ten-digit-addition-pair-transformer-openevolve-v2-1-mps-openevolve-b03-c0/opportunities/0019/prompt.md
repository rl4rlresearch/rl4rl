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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1618, "training_steps": 4999}
prior_hypothesis: Removing one value-projection bias coordinate alongside the validated missing key-bias coordinate will reduce parameters from 1,619 to 1,618 while retaining at least 99% accuracy, because the attention output-projection bias can learn the equivalent constant offset.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying `ln_f.bias` to the already shared block LayerNorm bias will reduce deduplicated parameters from 1,628 to 1,620 while retaining at least 99% accuracy.
change: Share the final LayerNorm bias with `blocks[0].ln1.bias`, preserving separate normalization scales and all attention and MLP capacity.
mechanism: Shared normalization offset across block and output norms
evidence_used: Sharing both block LayerNorm affine parameters achieved 99.59% accuracy at 1,628 parameters; this motivates an isolated eight-parameter bias-sharing extension rather than reducing the capacity components that previously failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9940000000000001, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln_f.weight` to the shared block LayerNorm weight will reduce deduplicated parameters from 1,620 to 1,612 while retaining at least 99% accuracy.
change: Share both the final LayerNorm scale and bias with `blocks[0].ln1`, preserving all attention and MLP capacity.
mechanism: Shared normalization scale across all LayerNorm sites
evidence_used: Sharing the block LayerNorm scale and bias achieved 99.59% accuracy, and extending bias sharing to the final LayerNorm still achieved 99.40%; this motivates testing the remaining isolated eight-parameter normalization sharing constraint.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2707, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one of eight `ln_f` scale coordinates with the block LayerNorm will reduce deduplicated parameters from 1,620 to 1,619 while retaining at least 99% accuracy.
change: Replace the final LayerNorm with an equivalent learned normalization whose first scale coordinate and bias are shared with `blocks[0].ln1`, while its other seven scale coordinates remain independent.
mechanism: Single-coordinate final-normalization scale sharing
evidence_used: Sharing only the final bias achieved 99.40% at 1,620 parameters, whereas sharing all eight final scale coordinates collapsed accuracy to 27.07%; a one-coordinate constraint is the smallest informative step toward locating that capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.731, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the key-projection bias will reduce parameters from 1,620 to 1,612 while retaining at least 99% accuracy, because a position-independent key bias adds the same constant to every attention logit in a query row and is canceled by softmax.
change: Replace the combined QKV bias with learned query and value biases, leaving the key projection bias-free while preserving all projection widths and other training settings.
mechanism: Eliminate softmax-invariant key bias
evidence_used: Removing all 24 QKV biases failed at 49.86%, showing that indiscriminate bias removal is harmful; the key-bias component is uniquely removable without changing the attention function, making this a more targeted eight-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2639, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating all eight key-bias parameters while preserving the original `nn.Linear` construction RNG trajectory will reduce parameters from 1,620 to 1,612 and retain at least 99% accuracy.
change: Retain the combined QKV layer’s original construction, replace its stored bias with query/value biases only, and reconstruct a zero key bias during the learned attention computation.
mechanism: Initialization-preserving removal of softmax-invariant key bias
evidence_used: The current model reaches 99.40% accuracy, while the earlier key-bias removal unexpectedly failed despite key bias being canceled row-wise by softmax. Preserving the original biased-layer construction isolates that mathematical reduction from initialization changes.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9599, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one key-projection bias coordinate while preserving the original layer-construction RNG trajectory will reduce the model from 1,620 to 1,619 parameters and retain at least 99% accuracy.
change: Store 23 learned QKV bias values and reconstruct one fixed-zero key-bias coordinate during attention.
mechanism: Single-coordinate softmax-invariant key-bias removal
evidence_used: Initialization-preserving removal of all eight mathematically redundant key biases reached 95.99%; removing only one coordinate is the smallest informative constraint and minimizes the optimizer perturbation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 1619, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second key-projection bias coordinate while preserving the original layer-construction RNG trajectory will reduce the model from 1,619 to 1,618 parameters and retain at least 99% accuracy.
change: Store 22 learned QKV bias values and reconstruct two fixed-zero key-bias coordinates during attention.
mechanism: Two-coordinate softmax-invariant key-bias removal
evidence_used: Removing one key-bias coordinate achieved 99.74% accuracy at 1,619 parameters, while removing all eight reached only 95.99%; removing one additional coordinate is the smallest informative step toward locating the constraint boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7084999999999999, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one value-projection bias coordinate alongside the validated missing key-bias coordinate will reduce parameters from 1,619 to 1,618 while retaining at least 99% accuracy, because the attention output-projection bias can learn the equivalent constant offset.
change: Store 22 learned QKV bias values and reconstruct fixed-zero first key and first value bias coordinates, preserving the original linear-layer construction and all other model and training settings.
mechanism: Single-coordinate value/output bias redundancy
evidence_used: Removing one key-bias coordinate achieved 99.74% accuracy at 1,619 parameters, but removing a second key coordinate fell to 70.85%; testing a distinct one-coordinate affine redundancy is therefore more informative than further restricting the key bias, while retaining the query biases whose wholesale removal previously failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second value-projection bias coordinate will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because attention weights sum to one and the output-projection bias can learn the resulting constant offset.
change: Store 21 learned QKV bias values and reconstruct fixed-zero first-key and first-two-value bias coordinates, preserving the original layer-construction RNG trajectory and all training settings.
mechanism: Incremental value/output bias redundancy
evidence_used: Removing one key and one value bias coordinate achieved 99.96% accuracy at 1,618 parameters, while removing a second key coordinate failed; extending the successful value-bias reduction by one coordinate is the smallest informative next constraint.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9611, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one redundant `fc1` weight coordinate while preserving the initialized function will reduce the model from 1,618 to 1,617 parameters and retain at least 99% accuracy, because LayerNorm outputs lie on an affine hyperplane and therefore one weight per downstream affine row is functionally redundant.
change: Store 95 rather than 96 `fc1` weights, reconstruct the first coordinate as zero during inference, and shift the remaining first-row weights after initialization so the initial MLP preactivation is unchanged.
mechanism: LayerNorm-hyperplane canonicalization of one MLP input weight
evidence_used: The current one-key/one-value-bias design achieved 99.96% at 1,618 parameters, while removing another value-bias coordinate fell to 96.11%; this motivates testing a distinct single-coordinate redundancy while preserving the successful initialization trajectory and initial model function.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7134999999999999, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one positional-embedding coordinate via an initialization-preserving per-position constant shift will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because pre-LayerNorm attention and MLP inputs are invariant to that shift and the final LayerNorm removes it exactly.
change: Canonicalize position zero’s embedding so its first coordinate is fixed at zero, store only the remaining positional parameters, and reconstruct the full embedding tensor during the learned forward pass.
mechanism: Positional-embedding shift-gauge elimination
evidence_used: The current one-key/one-value-bias design achieved 99.96% at 1,618 parameters, while other one-parameter reductions failed despite preserving capacity; this motivates testing a distinct exact model symmetry with both the initialized function and all attention/MLP capacity preserved.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6875, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one attention output-projection bias coordinate at zero will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because a uniform shift of that bias can be absorbed by the MLP output bias without changing the block’s function.
change: Preserve the original projection-layer construction, store seven learned output biases, and reconstruct a fixed-zero first coordinate during attention.
mechanism: Residual-stream bias gauge elimination
evidence_used: The current one-key/one-value-bias design reached 99.96% at 1,618 parameters. Since additional QKV, positional-embedding, and MLP-input reductions failed, this tests a distinct exact one-parameter redundancy: LayerNorm makes uniform residual shifts invisible between the attention and MLP sublayers.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4841, "parameters": 1617, "training_steps": 4999}



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
