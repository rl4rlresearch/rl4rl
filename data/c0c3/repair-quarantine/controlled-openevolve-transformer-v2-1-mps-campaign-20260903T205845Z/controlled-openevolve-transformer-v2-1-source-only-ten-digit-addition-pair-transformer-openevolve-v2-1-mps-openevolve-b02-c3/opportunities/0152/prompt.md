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
verified_results: {"accuracy": 0.9997, "parameters": 677, "training_steps": 4999}
prior_hypothesis: A 677-parameter transformer will retain at least 99% accuracy because extending the qualified second head’s two-way adjacent bias sharing to three adjacent long-distance biases preserves a learned routing signal while leaving the first head and all normalization scales unchanged.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987999999999999, "parameters": 670, "training_steps": 4999}
prior_hypothesis: The resulting 670-parameter transformer will retain at least 99% accuracy because it reproduces the qualified 671-parameter design and shares only one additional corresponding key coefficient, extending the same reduction that previously achieved 99.96%.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 671, "training_steps": 4999}
prior_hypothesis: Sharing one corresponding key-projection coefficient across attention heads will produce a 671-parameter transformer while retaining at least 99% accuracy, because head-specific queries and relative routing remain fully learned.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9922, "parameters": 671, "training_steps": 4999}
prior_hypothesis: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter architecture while fixing one query-bias coordinate that can be compensated through the learned second-head Q/K coordinate basis.

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
hypothesis: The resulting 670-parameter transformer will retain at least 99% accuracy because it preserves the qualified 671-parameter design while fixing only its final zero-initialized MLP output-bias coordinate.
change: Reproduce the 671-parameter reference’s relative-bias sharing and single fixed query-bias coordinate, then remove the sole remaining learned MLP output bias without creating a zero-length parameter.
mechanism: Complete MLP output-bias gauge fixing
evidence_used: Reference Design 2 retained 99.52% accuracy after fixing an additional MLP output-bias coordinate, and Reference Design 3 retained 99.22% after independently fixing one query-bias coordinate; this patch combines those verified reductions and tests the next MLP-bias reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.19510000000000002, "parameters": 670, "training_steps": 4999}

RECENT RESULT
hypothesis: A 670-parameter transformer will retain at least 99% accuracy because each attention head has its own learned Q/K coordinate basis, and the current 671-parameter design already achieved 99.22% after fixing the equivalent terminal query-bias coordinate in the second head.
change: Reshape the learned query bias by head and fix the final coordinate of each head at zero, removing one parameter while preserving the essential MLP output bias.
mechanism: Per-head query-bias gauge fixing
evidence_used: The current design verified at 99.22% with one query-bias coordinate fixed, whereas removing the remaining MLP output bias caused accuracy to collapse to 19.51%; this motivates an orthogonal, head-symmetric query-bias reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4562, "parameters": 670, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one corresponding key-projection coefficient across attention heads will produce a 671-parameter transformer while retaining at least 99% accuracy, because head-specific queries and relative routing remain fully learned.
change: Challenge the assumption that both heads require completely independent key projections by deduplicating one second-head key coefficient and reusing its first-head counterpart.
mechanism: Cross-head key-feature sharing
evidence_used: The current 672-parameter model reaches 99.52%, while removing another query-bias coordinate collapsed accuracy to 45.62%, indicating query biases are load-bearing. Qualified designs already tolerate identical fixed coefficients in corresponding key channels, motivating a less disruptive learned key sharing constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 671, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 670-parameter transformer will retain at least 99% accuracy because it reproduces the qualified 671-parameter design and shares only one additional corresponding key coefficient, extending the same reduction that previously achieved 99.96%.
change: Reproduce Reference Design 2’s fourth fixed MLP output-bias coefficient and first cross-head key tie, then deduplicate the adjacent key coefficient through the same learned source.
mechanism: Extended cross-head key-feature sharing
evidence_used: Reference Design 2 achieved 99.96% with one cross-head key coefficient shared, while alternative 670-parameter reductions to query bias and the remaining MLP output bias collapsed to 45.62% and 19.51%; extending the successful key-sharing mechanism is therefore the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 670, "training_steps": 4999}



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
