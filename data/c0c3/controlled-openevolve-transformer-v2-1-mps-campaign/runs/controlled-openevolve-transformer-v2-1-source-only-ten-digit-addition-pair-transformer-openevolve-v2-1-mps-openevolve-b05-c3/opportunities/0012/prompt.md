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
verified_results: {"accuracy": 0.9986, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Removing all eight key-bias parameters while preserving the baseline constructor RNG stream will produce 1636 parameters and retain at least 99% accuracy, because zero key bias is functionally invisible to causal softmax and all remaining weights receive the proven baseline initialization.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9992, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Removing the eight value-projection bias parameters in addition to the already qualified key-bias removal will produce 1628 parameters and retain at least 99% accuracy, because softmax-normalized attention makes value bias an input-independent vector that the existing output-projection bias can represent.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.998, "parameters": 1604, "training_steps": 4999}
prior_hypothesis: Representing each positional vector with seven coordinates modulo its LayerNorm-invisible common-channel shift will remove `max_seq_len` parameters from the qualified 1627-parameter model while retaining at least 99% accuracy, provided initialization, gradient clipping, and AdamW updates preserve the original eight-coordinate quotient dynamics.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the single block’s feed-forward width from 12 to 10 will lower deduplicated parameters from 1644 to 1610 while retaining at least 99% accuracy, given the current model’s 99.96% accuracy margin.
change: Set the default feed-forward width to 10 and update run labels accordingly; all training and decoding behavior remains unchanged.
mechanism: Narrower nonlinear feature bottleneck
evidence_used: The d_model=8, d_ff=12 design achieved 0.9996 accuracy at 1644 parameters after 4999 steps, indicating enough margin to test a modest two-unit reduction in the MLP bottleneck.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1793, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: An intermediate d_ff=11 model will reduce parameters from 1644 to 1627 while retaining at least 99% accuracy, testing whether the failure at d_ff=10 marks a sharp capacity threshold.
change: Reduce the default feed-forward width from 12 to 11 and update run labels accordingly.
mechanism: Single-unit MLP bottleneck reduction
evidence_used: The d_ff=12 design reached 99.96% accuracy, while d_ff=10 reached only 17.93%; d_ff=11 is the most informative untested point between these outcomes.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7341, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Keeping d_ff=12 while removing the key-projection bias will reduce parameters from 1644 to 1636 and retain at least 99% accuracy because a shared key bias adds the same constant to every unmasked attention logit and therefore cancels under softmax.
change: Replace the fused QKV bias with learned query and value biases only; leave architecture, training, and decoding otherwise unchanged.
mechanism: Remove softmax-invariant key bias
evidence_used: The d_ff=12 model achieved 99.96% accuracy, while d_ff=11 fell to 73.41%, motivating retention of the proven MLP capacity and removal of a functionally redundant attention parameter instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8615, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one common-mode degree of freedom from the final positional embedding will reduce parameters from 1644 to 1643 while retaining at least 99% accuracy, because every LayerNorm is invariant to per-token shifts shared across all eight residual channels.
change: Replace the positional embedding with an otherwise identical learned embedding whose final row fixes its last coordinate to zero, while initializing that row to a functionally equivalent gauge transformation of the baseline initialization.
mechanism: Gauge-fix one functionally redundant positional-embedding scalar
evidence_used: The d_ff=12 baseline reached 99.96% accuracy, whereas reducing nonlinear capacity or changing attention biases failed; this motivates preserving the proven architecture and function class while removing only a LayerNorm-invisible positional degree of freedom.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.045, "parameters": 1643, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all eight key-bias parameters while preserving the baseline constructor RNG stream will produce 1636 parameters and retain at least 99% accuracy, because zero key bias is functionally invisible to causal softmax and all remaining weights receive the proven baseline initialization.
change: Keep the fused QKV weight and baseline initialization sequence, replace its 24-element bias with 16 learned query/value biases, and insert a fixed zero key bias only during the forward pass.
mechanism: Initialization-preserving removal of softmax-invariant key bias
evidence_used: The 1644-parameter baseline reached 99.96%, while the earlier 1636-parameter key-bias removal reached 86.15%; since key bias mathematically adds only a row-wise softmax constant, the failed result motivates preserving the baseline RNG stream and initialization exactly while removing the same redundant degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight value-projection bias parameters in addition to the already qualified key-bias removal will produce 1628 parameters and retain at least 99% accuracy, because softmax-normalized attention makes value bias an input-independent vector that the existing output-projection bias can represent.
change: Preserve the baseline constructor RNG stream, learn only the eight query-bias parameters in fused QKV, and insert fixed zero key and value biases during the forward pass.
mechanism: Absorb constant value bias into the attention output bias
evidence_used: The initialization-preserving key-bias design achieved 99.86% accuracy with 1636 parameters; this retains its proven architecture and initialization while eliminating another analytically redundant eight-parameter bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1628-parameter qualified design can remove all eight `ln2` bias parameters and retain at least 99% accuracy, because `fc1.bias` can represent the LayerNorm bias’s entire downstream effect without reducing the function class, producing 1620 parameters.
change: Apply the qualified query-only QKV bias design, retain `d_ff=12`, and remove the second LayerNorm bias while preserving the proven initialization RNG stream.
mechanism: Absorb the second LayerNorm bias into the MLP input bias
evidence_used: The query-only QKV design achieved 99.92% accuracy at 1628 parameters; unlike the failed `d_ff` reductions, this removes an analytically redundant affine parameter while preserving nonlinear capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1727, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the qualified query-only QKV bias with a seven-dimensional mean-zero parameterization of `fc2.bias` will reduce the model to 1627 parameters while retaining at least 99% accuracy, because the omitted all-ones bias direction is exactly canceled by the immediately following final LayerNorm.
change: Apply the proven key/value-bias removal and represent the eight-coordinate MLP output bias using seven learned coefficients in a fixed orthonormal basis perpendicular to the all-ones vector, while preserving the baseline initialization RNG stream.
mechanism: Orthonormal gauge removal of the final pre-LayerNorm bias
evidence_used: The query-only QKV design achieved 99.92% accuracy at 1628 parameters. Unlike the failed positional gauge edit, this removes a zero-initialized null direction immediately before `ln_f`, requiring no transformation of randomly initialized embeddings.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3967, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining seven learned query-bias coordinates while fixing one to zero will reduce the qualified 1628-parameter design to 1627 parameters and preserve at least 99% accuracy, because LayerNorm bias and the remaining attention biases provide compensating flexibility without altering the proven weight initialization stream.
change: Remove key and value biases as in the qualified design, then learn only seven of the eight query-bias coordinates while fixing the final coordinate to zero.
mechanism: Single-coordinate query-bias ablation
evidence_used: The query-only QKV-bias design achieved 99.92% accuracy at 1628 parameters; unlike the failed broad LayerNorm and gauge reparameterizations, this tests the smallest possible localized reduction while leaving every weight tensor and seven query-bias coordinates unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing each positional vector with seven coordinates modulo its LayerNorm-invisible common-channel shift will remove `max_seq_len` parameters from the qualified 1627-parameter model while retaining at least 99% accuracy, provided initialization, gradient clipping, and AdamW updates preserve the original eight-coordinate quotient dynamics.
change: Replace the full positional table with a last-coordinate-zero gauge representation, preserve the proven constructor and initialization RNG streams, and train it with a gauge-aware AdamW optimizer that reconstructs the omitted gradient and optimizer state.
mechanism: Gauge-quotient positional embeddings with AdamW-equivalent updates
evidence_used: The 1627-parameter query-bias design reached 99.88%, while an earlier positional gauge edit failed at 4.5%; the key-bias experiments showed that an analytically redundant change recovered from 86.15% to 99.86% when the original RNG stream was preserved. This patch additionally preserves AdamW and clipping dynamics, directly addressing a second load-bearing assumption of the failed gauge parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1604-parameter gauge-aware design to represent `fc2.bias` modulo its LayerNorm-invisible common-channel direction will produce 1603 parameters while retaining at least 99% accuracy.
change: Apply the qualified seven-coordinate positional and query-bias parameterizations, gauge-fix the MLP output bias to seven coordinates, and preserve full-space AdamW and gradient-clipping dynamics for both gauge-fixed parameters.
mechanism: Gauge-quotient positional embeddings and MLP output bias with AdamW-equivalent updates
evidence_used: The gauge-aware positional design achieved 99.8% accuracy at 1604 parameters after a naïve positional gauge failed, while the naïve `fc2.bias` gauge failed at 39.67%; this directly motivates applying the successful optimizer-state and clipping treatment to the exact `fc2.bias` null direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9833, "parameters": 1603, "training_steps": 4999}



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
