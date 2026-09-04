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
verified_results: {"accuracy": 0.9997, "parameters": 1637, "training_steps": 4999}
prior_hypothesis: Sharing all eight softmax-invariant key-bias coordinates as one learned scalar will produce a 1637-parameter model with at least 99% accuracy; unlike the failed 1637 design, it removes the asymmetric fixed-zero anchor.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1638, "training_steps": 4999}
prior_hypothesis: Extending the verified shared key-bias parameterization by one coordinate will produce a 1638-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1635, "training_steps": 4999}
prior_hypothesis: Representing each MLP output bias with seven relative coordinates and applying the difference of the corresponding eight-coordinate AdamW updates will produce a 1635-parameter model with at least 99% accuracy.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing one proven softmax-invariant key-bias coordinate and one uniform MLP-output bias degree will produce a 1642-parameter model with at least 99% accuracy, since the final LayerNorm cancels any feature-uniform shift from `fc2.bias`.
change: Preserve constructor RNG consumption while shortening the fused QKV bias by one element and the MLP output bias by one element, reconstructing fixed-zero gauge coordinates during each projection.
mechanism: Orthogonal attention-key and residual-shift gauge removal
evidence_used: The one-coordinate key-bias design achieved 99.88% with 1643 parameters, while removing a second key coordinate fell to 98.52%; this tests a distinct exact invariance instead of further perturbing the sensitive key-bias parameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7182, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying a second key-bias coordinate to a surviving learned key bias will produce a 1642-parameter model with at least 99% accuracy, because the one-coordinate design achieved 99.88% and sharing avoids the additional fixed-zero coordinate used by the near-threshold 98.52% two-coordinate design.
change: Preserve fused-QKV construction and initialization while replacing its 24-element bias with 22 learned values; reconstruct one zero key-bias coordinate and one tied key-bias coordinate during projection.
mechanism: Shared softmax-invariant key-bias coordinate
evidence_used: The qualified one-coordinate key-bias reduction reached 99.88% at 1643 parameters, while fixing two coordinates reached 98.52% at 1642; tying the second invariant coordinate is the smallest change that tests whether retaining a learned shared value improves the sensitive optimization trajectory.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one additional key-bias coordinate to the surviving shared value will yield a 1641-parameter model with at least 99% accuracy, because the analogous 1642-parameter shared-bias design achieved 99.91%.
change: Replace the 24-element fused QKV bias with 21 learned values, reconstructing one zero key-bias coordinate and three key coordinates from one shared parameter.
mechanism: Incremental sharing of softmax-invariant key bias
evidence_used: Sharing a second redundant key-bias coordinate succeeded at 1642 parameters, whereas fixing two coordinates independently at zero scored only 98.52%; this motivates extending the successful sharing mechanism by one parameter.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 1641, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one additional key-bias coordinate to the existing shared value will produce a 1640-parameter model with at least 99% accuracy.
change: Reduce the fused QKV bias from 21 to 20 learned values, reconstructing one zero key-bias coordinate and four key coordinates from one shared parameter.
mechanism: Incremental sharing of a softmax-invariant key-bias coordinate
evidence_used: The same incremental sharing mechanism reduced the qualified model from 1642 parameters at 99.91% accuracy to 1641 parameters at 99.74%, leaving sufficient accuracy headroom for the next one-parameter probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1640, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified shared key-bias parameterization by one coordinate will produce a 1639-parameter model with at least 99% accuracy.
change: Reduce the fused QKV bias from 22 to 19 learned values, reconstructing one zero key-bias coordinate and five key coordinates from one shared parameter.
mechanism: Incremental sharing of a softmax-invariant key bias
evidence_used: The 1640-parameter design achieved 99.98% accuracy after the same sharing mechanism succeeded at 1642 and 1641 parameters, motivating the next one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9954999999999999, "parameters": 1639, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified shared key-bias parameterization by one coordinate will produce a 1638-parameter model with at least 99% accuracy.
change: Reduce the fused QKV bias from 23 to 18 learned values, reconstructing one zero key-bias coordinate and six key coordinates from one shared parameter.
mechanism: Incremental sharing of a softmax-invariant key bias
evidence_used: The 1639-parameter design achieved 99.55% accuracy after the same incremental sharing mechanism succeeded at 1642, 1641, and 1640 parameters, making one further shared coordinate the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1638, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the final independent key-bias coordinate to the existing shared value will produce a 1637-parameter model with at least 99% accuracy.
change: Reduce the fused QKV bias from 18 to 17 learned values, reconstructing one fixed-zero key coordinate and all seven remaining key coordinates from one shared parameter.
mechanism: Complete sharing of softmax-invariant key bias
evidence_used: The current 1638-parameter design achieved 99.96% accuracy after incremental key-bias sharing also qualified at 1642, 1641, 1640, and 1639 parameters, providing strong evidence for the final one-parameter continuation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8371999999999999, "parameters": 1637, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing all eight softmax-invariant key-bias coordinates as one learned scalar will produce a 1637-parameter model with at least 99% accuracy; unlike the failed 1637 design, it removes the asymmetric fixed-zero anchor.
change: Replace the current one-zero/five-shared key-bias parameterization with a single learned value shared across every key-bias coordinate.
mechanism: Symmetric complete sharing of key-projection bias
evidence_used: Six-coordinate sharing achieved 99.96% accuracy at 1638 parameters, while the asymmetric 1637 design with one fixed-zero coordinate and seven shared coordinates collapsed to 83.72%; fully symmetric sharing isolates whether that anchoring caused the optimization discontinuity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1637, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the mean learned query bias across all key-bias coordinates will produce a 1636-parameter model with at least 99% accuracy.
change: Replace the independent shared key-bias scalar with the mean of the learned query bias, while retaining one fused QKV parameter and a symmetric key bias.
mechanism: Cross-projection sharing of the softmax-invariant key bias
evidence_used: Symmetric complete key-bias sharing achieved 99.97% at 1637 parameters, whereas fixed-zero key-bias elimination failed; reusing a learned query-bias statistic removes the last independent key parameter without reintroducing a fixed-zero anchor.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing each MLP output bias with seven relative coordinates and applying the difference of the corresponding eight-coordinate AdamW updates will produce a 1635-parameter model with at least 99% accuracy.
change: Remove the final-LayerNorm-invariant uniform degree of each `fc2` bias, reconstruct a mean-anchored eight-coordinate bias, and preserve full-bias gradient clipping and AdamW quotient dynamics.
mechanism: Mean-anchored quotient bias with AdamW trajectory preservation
evidence_used: The current 1636-parameter model reached 99.92%; the earlier fixed-zero MLP-output gauge reduction collapsed to 71.82%, showing that this exact redundancy is optimization-sensitive and motivating trajectory-preserving removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1635, "training_steps": 4999}



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
