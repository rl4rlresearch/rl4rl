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
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1525, "training_steps": 4999}
prior_hypothesis: Removing the position-common component of the fifth retained positional coordinate will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because its input contribution can be transferred exactly to the corresponding token-embedding coordinate and remains softmax-null under the tied output projection.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate six at one will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.
change: Retain learned `ln2` scales only for coordinates zero and three, reconstructing coordinates one, two, and four through seven as ones.
mechanism: Fully connected final LayerNorm-scale absorption
evidence_used: Successively fixing fully connected coordinates four and five preserved 99.97% and 99.82% accuracy; coordinate six has the same full downstream connectivity, unlike failed coordinates zero and three, which are each omitted by an `fc1` row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.665, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` scale coordinate seven at one will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because all query, key, and value rows have trainable coefficients that can absorb its scale.
change: Retain seven learned `ln1` scales and reconstruct the final scale as one during the forward pass.
mechanism: Downstream-absorbed attention LayerNorm scale
evidence_used: Fixing `ln2` coordinate seven preserved 99.96% accuracy despite sparse downstream use; `ln1` coordinate seven is used by every dense QKV row, and its bias is already fixed at zero, making absorption less constrained.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.408, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln2` scale coordinate six to the mean of the load-bearing scales at coordinates zero and three will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because coordinate six remains trainable instead of being fixed at one.
change: Retain independent `ln2` scales for coordinates zero and three, and reconstruct coordinate six as their arithmetic mean.
mechanism: Adaptive shared LayerNorm scale
evidence_used: Fixing coordinate six at one fell to 66.5%, whereas fixing coordinates four and five retained 99.97% and 99.82%; this suggests the final scale needs adaptive training dynamics, which parameter sharing preserves while testing whether it needs an independent degree of freedom.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9773000000000001, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the position-common component of the fifth retained positional coordinate will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because its input contribution can be transferred exactly to the corresponding token-embedding coordinate and remains softmax-null under the tied output projection.
change: Represent positional coordinate four with a mean-free basis, transfer its initialized common component to token embeddings, and retain coordinate zero as the sole dense positional coordinate.
mechanism: Fifth positional common-mode quotient
evidence_used: The verified 1,526-parameter design reached 99.82% accuracy while already quotienting four positional common modes by this mechanism; unlike the failed LayerNorm-scale reductions, this removes another instance of the same input-sum redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the position-common component of the sole remaining dense positional coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its constant input contribution can be transferred exactly to the matching token-embedding coordinate.
change: Represent positional coordinate zero in the mean-free position basis, retain its removed common component in a buffer, and add that component to token embeddings at initialization.
mechanism: Complete positional common-mode quotient
evidence_used: Removing the common component of positional coordinate four preserved 99.88% accuracy at 1,525 parameters; this applies the same exact input-sum reparameterization to the only positional coordinate still storing a common mode.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9388, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one LayerNorm-normal-direction coefficient from each of the 24 independent QKV rows will reduce the model from 1,525 to 1,501 parameters while retaining at least 99% accuracy, because each projection row has only seven observable linear degrees of freedom on centered LayerNorm states.
change: Replace the assumption that attention requires dense QKV rows with learned row-specific tangent-space projections, distributing omitted coordinates evenly while preserving every head-specific query, key, and value row.
mechanism: Distributed LayerNorm-tangent QKV projections
evidence_used: The verified 1,525-parameter model reaches 99.88% while applying the same distributed one-coordinate quotient to all twelve `fc1` rows. Unlike key-weight sharing, which collapsed accuracy to 21.68%, this removes no head-specific addressing functions: query/value constants remain representable by their biases, and key constants are softmax-null.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4014, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the vocabulary-common component of the first token-content coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because the combined input offset remains trainable through the load-bearing positional common mode and the removed output component is softmax-null.
change: Represent the first token-content coordinate in a vocabulary-mean-free basis, transfer its initialized common component to positional coordinate zero, and preserve no-weight-decay treatment for all token-embedding parameters.
mechanism: Reverse token–position common-mode quotient
evidence_used: Transferring positional coordinate zero’s common component into token embeddings reduced accuracy to 93.88%, indicating that the positional parameterization is optimization-important. This patch reverses that quotient: it preserves the positional common degree of freedom from the 99.88%-accurate design while removing the mathematically redundant token-side common component.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7453, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate three at zero will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its position-independent QKV contribution can be absorbed by the retained query/value biases and is softmax-null for keys.
change: Extend the verified four-coordinate `ln1` bias quotient to coordinate three while leaving all attention weights and query/value biases unchanged.
mechanism: Fifth attention-LayerNorm bias absorption
evidence_used: The 1,525-parameter design achieves 99.88% accuracy with coordinates four through seven already removed from `ln1` bias. Unlike the failed value-bias removal, this preserves the full learned value bias and applies the same established `ln1` reparameterization to one additional coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9813, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one centered-input coefficient from the first key-projection row will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because the row’s omitted coefficient is representationally redundant on centered LayerNorm states and its constant key contribution is softmax-null.
change: Store the QKV matrix with the final input coefficient of the first key row omitted, reconstruct it as zero during inference, and canonicalize that row at initialization while preserving the original RNG sequence.
mechanism: Single-key-row LayerNorm-tangent quotient
evidence_used: Simultaneously pruning one coefficient from all 24 QKV rows collapsed accuracy to 40.14%, while the current unpruned-QKV design reaches 99.88%; isolating the quotient to one key row tests whether the failure came from applying 24 optimization-changing reparameterizations at once.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6202000000000001, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Deriving `ln2` coordinate-six scale from the corresponding `fc1` column norm will reduce the model to 1,524 parameters while retaining at least 99% accuracy, because it preserves the verified initialization and supplies adaptive radial training dynamics without an independent scale parameter.
change: Remove the learned coordinate-six `ln2` scale, reconstruct it from the square root of the normalized `fc1` coordinate-six column norm, and retain the column’s initialized norm as a non-trainable buffer.
mechanism: Parameter-free radial LayerNorm scale
evidence_used: Fixing coordinate six at one collapsed accuracy to 66.5%, while tying it to the other learned scales reached 97.73%; this shows adaptive scale dynamics help, but coupling them to load-bearing LayerNorm coordinates remains too restrictive.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.141, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln1` bias coordinate three to the mean of coordinates zero through two will reduce the model to 1,524 parameters while retaining at least 99% accuracy by restoring adaptive dynamics absent when that coordinate was fixed at zero.
change: Store three independent `ln1` bias values, reconstruct coordinate three as their mean, and keep coordinates four through seven fixed at zero.
mechanism: Adaptive mean-tied attention LayerNorm bias
evidence_used: Fixing `ln1` bias coordinate three scored 98.13%, while adaptively mean-tying the essential `ln2` coordinate six improved its fixed-value result from 66.5% to 97.73%; the same adaptive sharing may recover the smaller remaining gap here.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5446, "parameters": 1524, "training_steps": 4999}



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
