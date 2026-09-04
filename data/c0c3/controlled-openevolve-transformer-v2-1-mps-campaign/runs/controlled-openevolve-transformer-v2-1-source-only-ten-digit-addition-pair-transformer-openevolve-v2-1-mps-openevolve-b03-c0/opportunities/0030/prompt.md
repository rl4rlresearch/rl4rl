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
verified_results: {"accuracy": 0.9991, "parameters": 1614, "training_steps": 4999}
prior_hypothesis: Tying key-bias coordinates 1 and `head_dim + 1` will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and a learned symmetric tie preserves more optimization freedom than fixing a second key coordinate at zero.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one attention output-projection bias coordinate at zero will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because a uniform shift of that bias can be absorbed by the MLP output bias without changing the block’s function.
change: Preserve the original projection-layer construction, store seven learned output biases, and reconstruct a fixed-zero first coordinate during attention.
mechanism: Residual-stream bias gauge elimination
evidence_used: The current one-key/one-value-bias design reached 99.96% at 1,618 parameters. Since additional QKV, positional-embedding, and MLP-input reductions failed, this tests a distinct exact one-parameter redundancy: LayerNorm makes uniform residual shifts invisible between the attention and MLP sublayers.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4841, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the first value-bias coordinate of the second attention head, instead of a second coordinate from the first head, will reduce the model to 1,617 parameters while retaining at least 99% accuracy by distributing the exact value/output-bias redundancy evenly across heads.
change: Store 21 QKV bias values and reconstruct fixed-zero key coordinate 0 plus value coordinates 0 and `head_dim`, preserving the original layer-construction RNG trajectory and all training settings.
mechanism: Balanced per-head value-bias elimination
evidence_used: Removing one key and one value bias reached 99.96% at 1,618 parameters, while removing the adjacent second value coordinate reached 96.11%; testing the corresponding coordinate in the other head isolates whether the failure was caused by concentrating both value constraints within one head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing key-bias coordinate 0 from each attention head will reduce the model from 1,617 to 1,616 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and distributing constraints across heads previously succeeded for value biases where adjacent same-head removal failed.
change: Store 20 QKV bias values and reconstruct fixed-zero key coordinates 0 and `head_dim`, plus the already validated fixed-zero value coordinates 0 and `head_dim`, while preserving layer-construction RNG trajectory.
mechanism: Balanced per-head key-bias elimination
evidence_used: Concentrating two removed value-bias coordinates in the first head reached only 96.11%, whereas distributing them across the two heads reached 99.97% at 1,617 parameters; this motivates applying the same balanced-head strategy to the next key-bias reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5328, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying value-bias coordinates 1 and `head_dim + 1` will reduce the model to 1,616 parameters while retaining at least 99% accuracy, because both coordinates represent constant per-head offsets absorbable by the output-projection bias, while a learned symmetric tie preserves more optimization freedom than fixing another coordinate to zero.
change: Store 20 QKV bias parameters and reconstruct the first learned value-bias coordinate of the second head from the corresponding coordinate of the first head, retaining the validated zero key coordinate and balanced zero value coordinates.
mechanism: Cross-head value-bias coordinate tying
evidence_used: Removing value coordinates 0 and `head_dim` symmetrically across heads achieved 99.97% at 1,617 parameters, whereas concentrating two removed value coordinates in the first head achieved only 96.11%; this motivates a cross-head tie that preserves the successful balanced structure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying value-bias coordinates 2 and `head_dim + 2` will reduce the model from 1,616 to 1,615 parameters while retaining at least 99% accuracy, because the successful cross-head tie at coordinate 1 shows that symmetric per-head value offsets can share a learned degree of freedom without imposing a fixed-zero constraint.
change: Store 19 QKV bias parameters and reconstruct the second head’s value-bias coordinates 1 and 2 from the corresponding learned coordinates of the first head.
mechanism: Incremental cross-head value-bias tying
evidence_used: The current symmetric cross-head value-bias tie achieved 99.86% accuracy at 1,616 parameters, whereas fixing an additional value coordinate at zero achieved only 96.11%; extending the successful learned tie by one coordinate is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying value-bias coordinates 3 and `head_dim + 3` will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because the successful ties at coordinates 1 and 2 show that symmetric per-head value offsets can share learned degrees of freedom without the optimization failure caused by fixing additional offsets to zero.
change: Store 18 QKV bias parameters and reconstruct all three nonzero value-bias coordinates of the second head from the corresponding learned coordinates of the first head, while retaining the validated fixed-zero key coordinate and balanced fixed-zero value coordinates.
mechanism: Incremental cross-head value-bias tying
evidence_used: Cross-head tying at value coordinate 1 achieved 99.86% accuracy at 1,616 parameters, and extending the tie to coordinate 2 achieved 99.75% at 1,615; tying coordinate 3 is the smallest direct continuation of this successful reduction path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9601999999999999, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying key-bias coordinates 1 and `head_dim + 1` will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and a learned symmetric tie preserves more optimization freedom than fixing a second key coordinate at zero.
change: Store 18 QKV bias parameters and reconstruct the second head’s key-bias coordinate 1 from the corresponding learned coordinate of the first head, retaining the validated value-bias zeros and ties.
mechanism: Cross-head key-bias coordinate tying
evidence_used: Fixing a second key-bias coordinate at zero failed at 53.28%, while learned cross-head value-bias ties succeeded at 1,616 and 1,615 parameters; this motivates testing a learned cross-head tie on the mathematically redundant key bias rather than another fixed-zero constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying key-bias coordinates 2 and `head_dim + 2` will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because the successful coordinate-1 key tie demonstrates that a learned symmetric constraint can preserve optimization despite fixed-zero key constraints failing.
change: Store 17 QKV bias parameters and reconstruct the second head’s key-bias coordinates 1 and 2 from the corresponding learned coordinates of the first head, retaining the validated value-bias zeros and ties.
mechanism: Incremental cross-head key-bias coordinate tying
evidence_used: Tying key-bias coordinate 1 across heads achieved 99.91% accuracy at 1,614 parameters, whereas fixing an additional key coordinate at zero failed at 53.28%; extending the successful learned key tie by one coordinate is the smallest informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5392, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying key-bias coordinate 3 across heads will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because key biases are softmax-invariant and the successful coordinate-1 tie shows this redundancy can be removed without fixing another coordinate at zero.
change: Store 17 learned QKV bias values and reconstruct the second head’s key-bias coordinate 3 from the first head’s coordinate 3, while retaining the validated coordinate-1 key tie and value-bias constraints.
mechanism: Alternative cross-head key-bias coordinate tying
evidence_used: The coordinate-1 key tie achieved 99.91% accuracy at 1,614 parameters, whereas tying coordinate 2 fell to 53.92%; testing coordinate 3 isolates whether that failure was coordinate-specific while continuing the mathematically redundant key-bias reduction path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9651000000000001, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the attention output bias in the zero-sum subspace will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because uniform output-bias shifts are removed by subsequent LayerNorms and symmetric centering avoids the failed coordinate-anchored constraint.
change: Store seven attention projection-bias parameters and reconstruct an eight-dimensional zero-mean bias during the learned attention forward pass.
mechanism: Zero-mean attention output-bias gauge
evidence_used: Fixing one projection-bias coordinate at zero failed at 48.41%, despite the exact residual-shift redundancy; the current 1,614-parameter design reaches 99.91%, motivating a symmetry-preserving parameterization of that distinct redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4664, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying coordinate 1 of the MLP output bias to coordinate 1 of the attention projection bias will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because uniform opposite shifts of these biases leave the block function unchanged and a learned tie avoids the failed fixed-zero gauge constraints.
change: Store seven `fc2` bias values, reconstruct its missing coordinate from the learned attention projection bias, and pass that bias into the MLP forward path.
mechanism: Learned cross-sublayer residual-bias gauge tie
evidence_used: The 1,614-parameter design achieved 99.91%, and learned coordinate-1 tying succeeded where fixed-zero key constraints failed; prior projection-bias gauge reductions used fixed or centered constraints and failed, motivating a learned cross-sublayer tie that retains the complete function class.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.49090000000000006, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the second head’s coordinate-3 key bias with its coordinate-3 value bias will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because key bias is softmax-invariant and the value coordinate remains independently learnable.
change: Store 17 QKV bias parameters and reconstruct the second head’s final key coordinate from its learned final value-bias coordinate, preserving every value-bias degree of freedom in the successful design.
mechanism: Same-coordinate key/value bias reuse
evidence_used: Directly tying the remaining value coordinates reduced accuracy to 96.02%, while the 1,614-parameter design reached 99.91%; reusing an output-relevant value scalar for an invariant key coordinate removes a parameter without imposing the value constraint associated with that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6848000000000001, "parameters": 1613, "training_steps": 4999}



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
