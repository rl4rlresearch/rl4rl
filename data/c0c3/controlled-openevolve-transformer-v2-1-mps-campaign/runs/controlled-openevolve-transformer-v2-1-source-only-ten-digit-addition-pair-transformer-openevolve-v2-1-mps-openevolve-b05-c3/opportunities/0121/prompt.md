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
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1124, "training_steps": 4999}
prior_hypothesis: Extending the qualified seven-way final-bias tie to all eight channels will reduce the model from 1,125 to 1,124 learned parameters while maintaining at least 99% accuracy, because the shared adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1124, "training_steps": 4999}
prior_hypothesis: Fixing the fully tied final normalization bias to zero will reduce the current model from 1,125 to 1,124 learned parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1127, "training_steps": 4999}
prior_hypothesis: Extending the qualified four-way final-bias tie to five channels will reduce the model from 1,128 to 1,127 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9975, "parameters": 1126, "training_steps": 4999}
prior_hypothesis: Extending the qualified five-way final-bias tie to six channels will reduce the model from 1,127 to 1,126 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying the final two bias coordinates will reduce the qualified 1,131-parameter model to 1,130 parameters while retaining at least 99% accuracy, because the adaptive normalization scale is preserved and only one latent output-bias degree of freedom is removed.
change: Reconstruct the zero-sum final LayerNorm bias from six learned coordinates with its final two coordinates tied, leaving attention, training, gauge-aware optimization, checkpoints, and decoding unchanged.
mechanism: Two-way final-LayerNorm bias tying
evidence_used: The current quotient design achieved 99.81% accuracy with 1,131 parameters, while sourcing its adaptive scale elsewhere collapsed to 88.17%; this motivates preserving the scale and testing an isolated final-bias tie instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1130, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the two independent four-feature value maps in the qualified 1,130-parameter quotient model with one learned map shared across the shifted attention heads will produce 1,102 parameters while retaining at least 99% accuracy, because the heads can reuse token features while their distinct routing patterns and output-projection slices preserve their roles.
change: Adopt the verified per-token embedding and tied zero-sum final-bias quotients, then broadcast one learned four-dimensional value representation to both attention heads instead of learning separate value encoders.
mechanism: Shared cross-head value encoder
evidence_used: The 1,130-parameter quotient design achieved 99.90%, whereas rank-seven token factorization collapsed to 7.18% and a fifth routing-logit tie collapsed to 91.63%; this motivates preserving token geometry and routing while challenging the shared assumption that shifted heads need independent value feature extractors.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.06309999999999999, "parameters": 1102, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-way final-bias tie to three channels will reduce the qualified 1,130-parameter model to 1,129 parameters while retaining at least 99% accuracy, because the learned shared normalization scale and all attention/value capacity remain unchanged.
change: Adopt the verified per-token embedding quotient and scalar final scale, then reconstruct the zero-sum final bias from five learned coordinates with its final three coordinates tied.
mechanism: Three-way final-bias tying with per-token embedding quotient
evidence_used: The two-way final-bias tie achieved 99.90% with 1,130 parameters, while relocating the adaptive scale collapsed to 88.17% and sharing value encoders collapsed to 6.31%; this motivates one further isolated final-bias tie while preserving those sensitive components.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1129, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified three-way final-bias tie to four channels will reduce the model from 1,129 to 1,128 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, and token-representation capacity remain unchanged.
change: Adopt the verified per-token embedding quotient and scalar final scale, then reconstruct the zero-sum final bias from four learned coordinates with its final four coordinates tied.
mechanism: Four-way final-LayerNorm bias tying with per-token embedding quotient
evidence_used: Three-way final-bias tying achieved 99.83% accuracy with 1,129 parameters after two-way tying achieved 99.90% with 1,130; this supports one further isolated bias tie while preserving components whose factorization or sharing caused severe accuracy collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1128, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified four-way final-bias tie to five channels will reduce the model from 1,128 to 1,127 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.
change: Reconstruct the zero-sum final LayerNorm bias from three learned coordinates with its final five coordinates tied.
mechanism: Five-way final-LayerNorm bias tying
evidence_used: Four-way final-bias tying achieved 99.83% accuracy with 1,128 parameters, following successful two-way and three-way ties at 99.90% and 99.83%; this motivates one further isolated bias tie while preserving components whose sharing or scale relocation caused severe accuracy collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1127, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified five-way final-bias tie to six channels will reduce the model from 1,127 to 1,126 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.
change: Reconstruct the zero-sum final LayerNorm bias from two learned coordinates with its final six coordinates tied.
mechanism: Six-way final-LayerNorm bias tying
evidence_used: Five-way final-bias tying achieved 99.90% accuracy with 1,127 parameters, following successful two-way through four-way ties; this motivates one further isolated bias tie while preserving components whose sharing or scale relocation caused severe accuracy collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1126, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified six-way final-bias tie to seven channels will reduce the model from 1,126 to 1,125 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.
change: Reconstruct the zero-sum final LayerNorm bias from one learned coordinate with its final seven coordinates tied.
mechanism: Seven-way final-LayerNorm bias tying
evidence_used: Six-way final-bias tying achieved 99.75% accuracy with 1,126 parameters after two-way through five-way ties all exceeded 99%, motivating one further isolated tie despite the reduced accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9971, "parameters": 1125, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified seven-way final-bias tie to all eight channels will reduce the model from 1,125 to 1,124 learned parameters while maintaining at least 99% accuracy, because the shared adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.
change: Adopt the qualified per-token embedding row-shift quotient and fix the fully tied zero-sum final bias to zero, eliminating its last learned coordinate.
mechanism: Eight-way final-LayerNorm bias tying
evidence_used: Seven-way final-bias tying achieved 99.71% accuracy with 1,125 parameters after every preceding two-way through six-way tie exceeded 99%, motivating the final isolated bias tie while preserving the adaptive scale and sensitive attention/value components.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1124, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the final two MLP output-bias channels will reduce the verified model from 1,124 to 1,123 learned parameters while retaining at least 99% accuracy, because it preserves the adaptive normalization scale and all attention, value, MLP-weight, and token-representation capacity.
change: Store six independent MLP output-bias coordinates and reconstruct the final two channels as a tied zero-valued gauge anchor.
mechanism: Two-way MLP output-bias tying
evidence_used: The current 1,124-parameter model achieved 99.85% accuracy, and progressive final-bias tying retained at least 99% through complete tying; this motivates testing another isolated hidden-bias degree while avoiding the value sharing and scale relocation that caused severe collapses.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9152, "parameters": 1123, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the fully tied final normalization bias to zero will reduce the current model from 1,125 to 1,124 learned parameters while maintaining at least 99% accuracy.
change: Remove the last learned final-LayerNorm bias coordinate and its gauge optimizer entry, while preserving the adaptive normalization scale and all attention, value, MLP, and embedding capacity.
mechanism: Fully tied zero-sum final LayerNorm bias
evidence_used: The identical eight-way final-bias design previously achieved 99.85% accuracy with 1,124 parameters; the subsequent MLP output-bias reduction failed, so 1,124 is the smallest qualified design supported by the supplied evidence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1124, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1,124-parameter fully tied final-bias design, tying the final two attention output-bias channels will reduce the model to 1,123 parameters while retaining at least 99% accuracy, because all value, routing, projection-weight, MLP, embedding, and adaptive normalization capacity remains intact.
change: Remove the current two learned final-LayerNorm bias coordinates to adopt the qualified fully tied zero-bias design, then store six attention output-bias coordinates and reconstruct the last two as a tied gauge anchor.
mechanism: Two-way attention output-bias tying
evidence_used: Fully tying the final normalization bias achieved 99.84%–99.85% accuracy at 1,124 parameters. Tying MLP output-bias channels instead fell to 91.52%, motivating an isolated test of the attention residual bias while preserving the sensitive MLP bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9397, "parameters": 1123, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified fully tied final-bias design, reducing the GELU MLP from 12 to 11 units will lower the model from 1,124 to 1,109 parameters while retaining at least 99% accuracy; the old designs assumed all 12 nonlinear units were load-bearing, while this tests a narrower learned representation without constraining the sensitive MLP output bias, attention bias, value maps, routing kernel, or adaptive scale.
change: Adopt the verified zero-bias final LayerNorm and reduce `d_ff` by one, removing 15 learned MLP parameters while leaving training, checkpoints, causal attention, and decoding unchanged.
mechanism: Eleven-unit learned nonlinear bottleneck
evidence_used: The fully tied final-bias model reached 99.84%–99.85% with 1,124 parameters. MLP output-bias tying collapsed to 91.52%, so this patch preserves every residual-bias coordinate and instead tests the previously unchallenged assumption that the pointwise computation requires a 12-unit hidden bank.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1109, "training_steps": 4999}



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
