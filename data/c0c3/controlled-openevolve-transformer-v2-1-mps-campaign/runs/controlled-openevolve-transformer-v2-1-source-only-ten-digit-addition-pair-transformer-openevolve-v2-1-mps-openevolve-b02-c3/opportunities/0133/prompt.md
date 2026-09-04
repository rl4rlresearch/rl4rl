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
verified_results: {"accuracy": 0.9914000000000001, "parameters": 668, "training_steps": 4999}
prior_hypothesis: A 668-parameter transformer will retain at least 99% accuracy because it preserves the qualified 669-parameter architecture while fixing only the adjacent fifth-longest first-head relative-bias coefficient.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9997, "parameters": 677, "training_steps": 4999}
prior_hypothesis: A 677-parameter transformer will retain at least 99% accuracy because extending the qualified second head’s two-way adjacent bias sharing to three adjacent long-distance biases preserves a learned routing signal while leaving the first head and all normalization scales unchanged.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9911, "parameters": 669, "training_steps": 4999}
prior_hypothesis: A 669-parameter transformer will retain at least 99% accuracy because it preserves the qualified 670-parameter model while fixing only the adjacent fourth-longest first-head relative-bias coefficient.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9958, "parameters": 666, "training_steps": 4999}
prior_hypothesis: A 666-parameter transformer will retain at least 99% accuracy because extending the successful first-head boundary pair sharing to three adjacent biases preserves an adaptive long-distance routing signal.

## Recent verification evidence

RECENT RESULT
hypothesis: A 671-parameter transformer will retain at least 99% accuracy because sharing one direct input-code coordinate between two tokens preserves four-coordinate token codes, both nonlinear lifts, and the qualified 672-parameter attention/MLP architecture.
change: Replace the token embedding table with a one-scalar-smaller parameterization in which the first coordinate of the first two token codes shares one learned coefficient.
mechanism: Single-coordinate cross-token lexical sharing
evidence_used: The 672-parameter design achieved 99.52%, while collapsing every token to three coordinates fell to 39.64% and removing the remaining MLP output bias fell to 33.33%; this motivates an isolated one-scalar lexical tie that retains the full four-coordinate representation and leaves load-bearing attention and MLP paths unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5663, "parameters": 671, "training_steps": 4999}

RECENT RESULT
hypothesis: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter attention, lexical, and MLP architecture while fixing only one final LayerNorm scale at its unit initialization.
change: Reconstruct the qualified four-coordinate MLP bias fixing, then remove one learned final-normalization scale by extending the unit-valued fixed-scale suffix from three features to four.
mechanism: Final-normalization scale fixing
evidence_used: The 672-parameter design achieved 99.52%, while removing its remaining MLP output bias collapsed to 33.33% and tying a lexical coordinate collapsed to 56.63%; this motivates preserving those components and testing an isolated reduction in the previously untested final normalization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1441, "parameters": 671, "training_steps": 4999}

RECENT RESULT
hypothesis: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter architecture while fixing one query-bias coordinate that can be compensated through the learned second-head Q/K coordinate basis.
change: Reconstruct the qualified four-coordinate MLP output-bias fixing, then remove the final second-head query-bias scalar by fixing it at zero.
mechanism: Headwise query-coordinate gauge fixing
evidence_used: The 672-parameter design achieved 99.52%; eliminating its remaining MLP output bias, tying a lexical coordinate, or fixing another final-normalization scale failed, motivating an isolated reduction in the previously untested headwise query-coordinate gauge while preserving those components.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9922, "parameters": 671, "training_steps": 4999}

RECENT RESULT
hypothesis: A 670-parameter transformer will retain at least 99% accuracy because the qualified 671-parameter model already fixes one second-head query-bias coordinate, and the adjacent coordinate can likewise be absorbed by the learned second-head Q/K basis without changing attention routing, lexical codes, or MLP capacity.
change: Reconstruct the qualified 671-parameter architecture, then fix the penultimate second-head query-bias coordinate at zero alongside the already fixed final coordinate.
mechanism: Second-head query-bias gauge fixing
evidence_used: The 671-parameter query-gauge design achieved 99.22%, while reductions to its remaining MLP bias, lexical codes, or final normalization failed; this motivates continuing the successful isolated reduction within the same query-bias gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0233, "parameters": 670, "training_steps": 4999}

RECENT RESULT
hypothesis: A 670-parameter transformer will retain at least 99% accuracy because fixing the terminal query coordinate of the first head mirrors the successful terminal-coordinate fixing in the second head while preserving both remaining query-bias dimensions of the routing-sensitive second head.
change: Replace the five-scalar query bias with four learned scalars and reconstruct a query bias whose final coordinate in each head is fixed at zero.
mechanism: Per-head terminal query-coordinate gauge fixing
evidence_used: The 671-parameter model reached 99.22% after fixing the second head’s terminal query-bias coordinate, whereas fixing its adjacent coordinate collapsed to 2.33%; this motivates applying the successful coordinate-local constraint symmetrically to the other head instead of further reducing the second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4559, "parameters": 670, "training_steps": 4999}

RECENT RESULT
hypothesis: A 670-parameter transformer will retain at least 99% accuracy because fixing the first head’s third-longest relative-bias coefficient extends its existing fixed longest-distance suffix while preserving all frequently used routing biases and the qualified 671-parameter model’s query, lexical, normalization, and MLP parameters.
change: Remove one learned relative-bias scalar and reconstruct the first head’s third-longest distance bias as zero.
mechanism: First-head boundary-distance bias fixing
evidence_used: The current 671-parameter model achieved 99.22%, while further query-bias fixing failed; its two longest relative biases are already fixed for both heads, motivating an isolated extension of the first head’s sparsely used boundary-distance constraint without merging learned routing coefficients.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9945, "parameters": 670, "training_steps": 4999}

RECENT RESULT
hypothesis: A 669-parameter transformer will retain at least 99% accuracy because it preserves the qualified 670-parameter model while fixing only the adjacent fourth-longest first-head relative-bias coefficient.
change: Reconstruct the qualified second-head query-bias gauge fixing and first-head third-longest bias fixing, then extend the first head’s fixed boundary suffix by one additional relative-bias coefficient.
mechanism: First-head boundary-bias suffix extension
evidence_used: The 670-parameter design achieved 99.45% after fixing the first head’s third-longest relative bias, whereas additional query-bias constraints failed; this motivates continuing the successful boundary-distance reduction while preserving query, lexical, normalization, MLP, and second-head routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9911, "parameters": 669, "training_steps": 4999}

RECENT RESULT
hypothesis: A 668-parameter transformer will retain at least 99% accuracy because it preserves the qualified 669-parameter architecture while fixing only the adjacent fifth-longest first-head relative-bias coefficient.
change: Reconstruct the qualified 669-parameter query gauge, four-coordinate MLP bias fixing, and first-head boundary suffix, then extend that suffix by one relative-bias coefficient.
mechanism: First-head boundary-bias suffix extension
evidence_used: Successive first-head boundary reductions achieved 99.45% at 670 parameters and 99.11% at 669, while additional query, lexical, normalization, and MLP-bias constraints failed; this motivates continuing the only reduction mechanism that remained qualified.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9914000000000001, "parameters": 668, "training_steps": 4999}

RECENT RESULT
hypothesis: A 667-parameter transformer will retain at least 99% accuracy because it preserves the qualified 668-parameter architecture while fixing only the adjacent sixth-longest first-head relative-bias coefficient.
change: Extend the first head’s fixed long-distance relative-bias suffix from three to four coefficients, removing exactly one learned parameter.
mechanism: First-head boundary-bias suffix extension
evidence_used: Successive first-head boundary-bias reductions qualified at 670 parameters (99.45%), 669 parameters (99.11%), and 668 parameters (99.14%), while reductions in query bias, lexical codes, final normalization, and the remaining MLP output bias failed; continuing the only repeatedly successful reduction mechanism is the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9862000000000001, "parameters": 667, "training_steps": 4999}

RECENT RESULT
hypothesis: A 660-parameter transformer will retain at least 99% accuracy because each gated MLP unit can reuse its learned value/gate input directions to synthesize its output direction, eliminating 12 parameters while preserving the qualified 672-parameter model’s lexical, attention-routing, normalization, gating, and essential output-bias capacity.
change: Reconstruct the qualified 672-parameter attention backbone, then replace the independent 20-scalar MLP output matrix with eight learned mixing coefficients that combine each hidden unit’s value and gate input directions.
mechanism: Bidirectionally tied gated-MLP directions
evidence_used: The 672-parameter architecture achieved 99.52%, while deleting its last MLP output bias collapsed to 33.33% and reductions in lexical, normalization, and query capacity also failed. This motivates preserving those load-bearing components while challenging the untested assumption that the gated MLP requires input and output directions learned independently.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2253, "parameters": 660, "training_steps": 4999}

RECENT RESULT
hypothesis: A 667-parameter transformer will retain at least 99% accuracy because replacing the failed zeroing of the sixth-longest first-head bias with an adaptive tie to its adjacent bias preserves a learned long-distance routing signal.
change: Reconstruct the qualified 668-parameter three-zero first-head boundary suffix, then remove one additional parameter by sharing the two learned biases immediately preceding that suffix.
mechanism: First-head boundary-bias pair sharing
evidence_used: The 668-parameter design reached 99.14%, while extending its zero suffix to 667 parameters narrowly missed at 98.62%; adjacent relative-bias sharing is a stronger alternative because the 677-parameter second-head sharing design reached 99.97%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9944, "parameters": 667, "training_steps": 4999}

RECENT RESULT
hypothesis: A 666-parameter transformer will retain at least 99% accuracy because extending the successful first-head boundary pair sharing to three adjacent biases preserves an adaptive long-distance routing signal.
change: Remove one relative-bias parameter by sharing the three learned first-head biases immediately preceding its fixed three-bias suffix.
mechanism: First-head boundary-bias triplet sharing
evidence_used: The current pair-sharing design achieved 99.44% at 667 parameters, whereas fixing the same boundary bias at zero reached only 98.62%; the 99.97% second-head design also demonstrates that three-way adjacent bias sharing can preserve routing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9958, "parameters": 666, "training_steps": 4999}



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
