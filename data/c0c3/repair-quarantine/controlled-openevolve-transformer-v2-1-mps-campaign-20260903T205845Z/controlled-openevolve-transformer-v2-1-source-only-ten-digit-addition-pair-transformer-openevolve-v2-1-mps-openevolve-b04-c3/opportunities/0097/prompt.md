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
verified_results: {"accuracy": 1.0, "parameters": 1139, "training_steps": 44999}
prior_hypothesis: Extending the verified 1,140-parameter design by fixing a second trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the model to 1,139 learned parameters.

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
hypothesis: The verified 1,145-parameter model will retain at least 99% accuracy after 45,000 updates when a second coordinate of the tied six-dimensional token code is fixed at zero, yielding 1,144 learned parameters.
change: Learn all but two scalars of the tied token embedding, reconstructing both fixed trailing scalars as zeros while preserving full embedding initialization RNG consumption.
mechanism: Two-scalar tied-token residual-basis gauge fixing
evidence_used: Fixing one tied-token scalar achieved 100% accuracy with 1,145 parameters, whereas further positional-mixer and attention-output-bias constraints failed; the smallest informative next probe is one additional constraint in the successful token-code mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1144, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a third scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,144-parameter design to 1,143 learned parameters.
change: Adopt the verified two-scalar token constraint and fix one additional trailing embedding scalar at zero, preserving full embedding initialization RNG consumption and tied input/output use.
mechanism: Three-scalar tied-token residual-basis constraint
evidence_used: Reference Design 1 achieved 100% accuracy with two fixed token-embedding scalars and 1,144 parameters, after the one-scalar version also achieved 100%; an incremental third constraint is the most direct evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1143, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fourth scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,143-parameter design to 1,142 learned parameters.
change: Adopt the qualified two-coordinate attention-output-bias constraint and reconstruct the final four token-embedding scalars as zeros, preserving full initialization RNG consumption and tied input/output use.
mechanism: Four-scalar tied-token residual-basis constraint
evidence_used: The 1,143-parameter reference achieved 100% accuracy with three fixed token-embedding scalars, after the one- and two-scalar versions also achieved 100%; one additional token constraint is the smallest evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1142, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fifth scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,142-parameter design to 1,141 learned parameters.
change: Adopt the qualified shared-key/value attention, two-coordinate output-bias constraint, and eight-scalar Fourier constraint, then reconstruct the final five token-embedding scalars as zeros while preserving full initialization RNG consumption and tied input/output use.
mechanism: Five-scalar tied-token residual-basis constraint
evidence_used: Reference Design 2 achieved 100% accuracy with four fixed token-embedding scalars and 1,142 parameters, after the one-, two-, and three-scalar constraints also achieved 100%; one additional token constraint is the smallest evidence-backed reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1431, "parameters": 1141, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing the final MLP-output bias coordinate at zero will retain at least 99% accuracy after 45,000 updates while reducing the verified model from 1,142 to 1,141 learned parameters.
change: Learn seven of the eight MLP-output bias coordinates and reconstruct the final coordinate as zero during the forward pass, preserving the qualified transformer backbone and constructor RNG consumption.
mechanism: One-coordinate MLP-output bias constraint
evidence_used: The current 1,142-parameter model achieved 100% accuracy, while fixing a fifth token-embedding scalar failed at 14.31%. Two attention-output bias coordinates were removable with 100% accuracy, motivating a one-scalar probe of the separate, previously unconstrained MLP-output bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6559, "parameters": 1141, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing the token-facing 4×4 block of the shared value projection to a scaled identity will preserve at least 99% accuracy after 45,000 updates with 1,126 parameters, because the dense output projection can absorb an invertible change of the four-dimensional value basis.
change: Replace the assumption that all 32 shared-value coefficients must be learned with a canonical 0.02-scaled passthrough basis plus 16 learned coefficients for the remaining input channels; also adopt the qualified four-scalar tied-token constraint.
mechanism: Canonical-basis shared-value attention
evidence_used: The shared-key/value model achieved 100% accuracy, and the four-scalar token constraint separately achieved 100% at 1,142 parameters. Unlike failed reductions of positional capacity, token rank, or output biases, this removes a 16-parameter value/output basis redundancy while retaining the full residual width, positional map, MLP, attention routing, and dense learned output projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.09269999999999999, "parameters": 1126, "training_steps": 44999}

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
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
