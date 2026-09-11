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
verified_results: {"accuracy": 0.9975, "parameters": 984, "training_steps": 4999}
prior_hypothesis: Sharing final-LayerNorm scales for residual coordinates two and three will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient and tolerate zero companion-key contributions, while the sensitive later coordinates remain independent.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Tying final-LayerNorm scales for residual coordinates zero and one will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because those coordinates already share one learned MLP input coefficient and the verified 984-parameter model has a 0.75-point accuracy margin.
change: Store four final-LayerNorm scales and reconstruct scales as paired coordinates 0–1 and 2–3, independent coordinate 4, paired coordinates 5–6, and fixed coordinate 7.
mechanism: Pairwise final-normalization scale sharing
evidence_used: The current 984-parameter design achieved 99.75% accuracy after sharing coordinates two and three, showing that structurally aligned final-LayerNorm scale sharing can remove a parameter without approaching the 99% threshold.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3494, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s three farthest relative distances will reduce the verified 984-parameter model to 983 parameters while retaining at least 99% accuracy, because it extends an existing far-distance tie without imposing the final-LayerNorm coupling that previously caused accuracy to collapse.
change: Remove one head-0 relative-bias parameter and reconstruct the three farthest distances with the same fixed reference value.
mechanism: Extended far-distance relative-bias sharing
evidence_used: The 984-parameter design reached 99.75% accuracy, while tying final-LayerNorm coordinates zero and one fell to 34.94%; this motivates testing an orthogonal reduction in the already-compressed relative-position bias instead of further final-normalization sharing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9787, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one additional adjacent coefficient in head 1’s already-sparse companion key tail will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy.
change: Extend the fixed-zero prefix of head 1’s companion key-tail channel from two coefficients to three, updating initialization, reconstruction, gradient lifting, and optimizer projection consistently.
mechanism: Extended companion-key sparsification
evidence_used: The verified 984-parameter model reached 99.75% while already fixing this channel’s first two coefficients and the entire neighboring key tail; unlike the failed final-LayerNorm tie and far-distance bias extension, this incrementally extends a successful sparsity pattern.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing residual coordinate six’s MLP input coefficient with coordinates four and five will reduce the verified model from 984 to 983 parameters while retaining at least 99% accuracy, because coordinate six already shares its final-LayerNorm scale with coordinate five.
change: Store three MLP input coefficients instead of four, reconstructing the seven-coordinate input pattern as three groups: 0–1, 2–3, and 4–6, while preserving the original initialization RNG stream.
mechanism: Chained width-one MLP coordinate sharing
evidence_used: The current 984-parameter model achieved 99.75% accuracy with successful coordinate-aligned sharing, whereas reductions to final-LayerNorm scales, relative bias, and companion keys failed; this tests an orthogonal reduction in the width-one MLP along an existing final-normalization pairing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9823999999999999, "parameters": 983, "training_steps": 4999}



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
