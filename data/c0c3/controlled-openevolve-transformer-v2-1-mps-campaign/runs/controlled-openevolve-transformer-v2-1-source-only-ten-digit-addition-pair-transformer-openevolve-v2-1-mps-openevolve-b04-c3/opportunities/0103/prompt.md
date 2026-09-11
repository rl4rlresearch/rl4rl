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
verified_results: {"accuracy": 0.9978, "parameters": 1138, "training_steps": 44999}
prior_hypothesis: Fixing a third trailing attention-output projection coefficient will reduce the model to 1,138 learned parameters while retaining at least 99% accuracy after 45,000 updates; computing shared keys and values before head replication will reduce redundant work that contributed to the previous verification timeout.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998, "parameters": 1140, "training_steps": 44999}
prior_hypothesis: Fixing one trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the verified 1,141-parameter model to 1,140 learned parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1212, "training_steps": 44999}
prior_hypothesis: Fixing a seventh scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,213 to 1,212 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1143, "training_steps": 44999}
prior_hypothesis: Fixing a third scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,144-parameter design to 1,143 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: The verified 1,142-parameter model will retain at least 99% accuracy after 45,000 updates when one trailing shared-value projection coefficient is fixed at zero, yielding 1,141 parameters.
change: Adopt the qualified four-scalar token constraint and remove one learned shared-value coefficient, reconstructing it as zero while preserving full initialization RNG consumption.
mechanism: Single-scalar shared-value basis gauge fixing
evidence_used: The four-scalar token design achieved 100% accuracy at 1,142 parameters. Fixing an entire 16-coefficient value block failed, motivating a minimal one-scalar probe that retains the shared value projection’s rank and flexible learned output projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1141, "training_steps": 44999}

RECENT RESULT
hypothesis: The verified 1,141-parameter model will retain at least 99% accuracy after 45,000 updates when a second trailing shared-value projection coefficient is fixed at zero, yielding 1,140 learned parameters.
change: Adopt the qualified shared-key/value attention backbone, four-scalar tied-token constraint, eight-scalar positional constraint, and two-coordinate attention-output-bias constraint, while reconstructing the final two shared-value coefficients as zeros and preserving full initialization RNG consumption.
mechanism: Two-scalar shared-value basis constraint
evidence_used: Fixing one shared-value coefficient achieved 100% accuracy at 1,141 parameters; because fixing an entire 16-coefficient value block failed, one additional coefficient is the smallest informative probe of this successful mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6931999999999999, "parameters": 1140, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing one trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the verified 1,141-parameter model to 1,140 learned parameters.
change: Learn 63 of the 64 attention-output projection weights, reconstruct the final coefficient as zero, and preserve full-matrix initialization RNG consumption.
mechanism: Output-side shared-value basis gauge fixing
evidence_used: One shared-value coefficient was removed successfully at 1,141 parameters, demonstrating redundancy in the value/output basis; because removing a second adjacent value coefficient failed, the complementary learned output projection is the most informative location for the next one-parameter constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1140, "training_steps": 44999}

RECENT RESULT
hypothesis: Extending the verified 1,140-parameter design by fixing a second trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the model to 1,139 learned parameters.
change: Adopt the qualified one-scalar shared-value constraint and reconstruct the final two attention-output projection weights as zeros, preserving full constructor initialization RNG consumption.
mechanism: Two-scalar output-side shared-value basis constraint
evidence_used: The combined one-scalar shared-value and one-scalar attention-output constraints achieved 99.98% accuracy with 1,140 parameters; since a second shared-value constraint failed, extending the independently successful output-side constraint by one scalar is the smallest informative next probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1139, "training_steps": 44999}

RECENT RESULT
hypothesis: Extending the verified 1,139-parameter design by fixing a third trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the model to 1,138 learned parameters.
change: Adopt the qualified four-scalar token-embedding and one-scalar shared-value constraints, then reconstruct the final three attention-output projection weights as zeros while preserving full constructor RNG consumption.
mechanism: Three-scalar output-side shared-value basis constraint
evidence_used: Reference Design 2 achieved 100% accuracy with one fixed shared-value coefficient and two fixed attention-output projection coefficients at 1,139 parameters; extending the independently successful output-side constraint by one scalar is the smallest informative next probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing a third trailing attention-output projection coefficient will reduce the model to 1,138 learned parameters while retaining at least 99% accuracy after 45,000 updates; computing shared keys and values before head replication will reduce redundant work that contributed to the previous verification timeout.
change: Reconstruct three trailing output-projection weights as zeros, preserve full constructor RNG consumption, and evaluate the mathematically equivalent shared key/value projections before duplicating them across heads.
mechanism: Three-scalar output-projection gauge constraint with factorized shared-KV evaluation
evidence_used: The verified 1,139-parameter model achieved 100% accuracy with two fixed output-projection coefficients. The three-coefficient attempt timed out rather than producing an accuracy failure, so repeating that incremental constraint with less redundant attention computation directly tests the unresolved capacity hypothesis.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1138, "training_steps": 44999}

RECENT RESULT
hypothesis: Extending the verified 1,138-parameter design by fixing a fourth trailing attention-output projection coefficient at zero will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,137 learned parameters.
change: Reconstruct the final four attention-output projection weights as zeros and use the qualified factorized shared-key/value evaluation to avoid redundant computation while preserving constructor RNG consumption.
mechanism: Four-scalar output-side shared-value basis constraint
evidence_used: The factorized three-coefficient design achieved 99.78% accuracy with 1,138 parameters; one additional output-projection constraint is the smallest direct probe of the remaining redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0592, "parameters": 1137, "training_steps": 44999}

RECENT RESULT
hypothesis: Tying the fourth trailing attention-output coefficient to the corresponding coefficient for the other head will reduce the model to 1,137 learned parameters while retaining at least 99% accuracy.
change: Keep the three successful trailing zeros, remove one additional learned projection scalar, and reconstruct the essential fourth coefficient from the other head’s learned coefficient for the same shared-value coordinate.
mechanism: Cross-head output-coefficient tying
evidence_used: The 1,138-parameter model reached 99.78%, whereas zeroing the fourth coefficient collapsed accuracy to 5.92%; because both heads share the value projection, tying their corresponding output coefficients preserves a learned nonzero route while testing a one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.867, "parameters": 1137, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing one trailing shared-key coefficient at 0.02 will preserve at least 99% accuracy after 45,000 updates while reducing the qualified 1,138-parameter model to 1,137 parameters.
change: Adopt the qualified three-scalar output-projection constraint and factorized shared-KV evaluation, then replace one learned shared-key coefficient with a nonzero canonical scale while preserving full initialization RNG consumption.
mechanism: Nonzero shared-key scale anchoring
evidence_used: The three-output-constraint design achieved 99.78% accuracy at 1,138 parameters. Unlike the failed fourth output zero, a nonzero key anchor preserves the attention route and exploits reciprocal query/key scaling already motivating the qualified query gauge constraints.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Coupling the MLP’s analysis and synthesis dictionaries through learned feature gains and an 8×8 channel metric will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,138 to 1,118 learned parameters.
change: Replace the independent 8×12 MLP output matrix with the transpose of the learned 12×8 input dictionary, twelve learned feature gains, and a learned 8×8 channel mixer initialized to identity.
mechanism: Metric-tied analysis–synthesis MLP
evidence_used: The 1,138-parameter backbone reached 99.78%, while further attention-output pruning, cross-head tying, embedding pruning, and output-bias pruning failed sharply. This leaves intact those sensitive paths and instead challenges the shared assumption that each nonlinear MLP feature requires unrelated input and output directions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing one trailing shared-key coefficient at 0.02 will preserve at least 99% accuracy at 1,137 learned parameters, while fusing the factorized query and shared-key/value projections will allow all 45,000 updates to finish within the verification limit.
change: Adopt the qualified three-scalar attention-output constraint, remove one shared-key scalar and reconstruct it as 0.02, then evaluate the compact query and shared key/value dictionaries with one linear operation.
mechanism: Nonzero shared-key gauge anchoring with fused compact QKV evaluation
evidence_used: The three-output-constraint design reached 99.78% accuracy at 1,138 parameters. The prior nonzero shared-key anchor timed out rather than failing accuracy, so preserving that unresolved one-parameter gauge probe while reducing projection overhead is the most direct next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the essential fourth trailing attention-output coefficient with a fixed nonzero 0.02 anchor will retain at least 99% accuracy while reducing the qualified 1,138-parameter model to 1,137 learned parameters.
change: Adopt the qualified factorized shared-key/value backbone, four-scalar token constraint, one-scalar value constraint, and three output zeros; additionally fix the preceding output coefficient at 0.02 while preserving full-matrix initialization RNG consumption.
mechanism: Nonzero output-side value-scale anchoring
evidence_used: The 1,138-parameter design reached 99.78%, while zeroing its fourth trailing output coefficient collapsed accuracy to 5.92% and cross-head tying reached only 86.7%. A nonzero anchor preserves that essential route while exploiting reciprocal scaling between the shared value coordinate and its output-projection columns.
result: training did not finish within the verification time limit



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
