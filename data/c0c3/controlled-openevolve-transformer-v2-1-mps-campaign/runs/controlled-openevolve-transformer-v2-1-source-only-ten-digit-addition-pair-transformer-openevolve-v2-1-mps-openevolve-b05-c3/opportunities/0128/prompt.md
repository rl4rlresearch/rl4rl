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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1127, "training_steps": 4999}
prior_hypothesis: Extending the qualified four-way final-bias tie to five channels will reduce the model from 1,128 to 1,127 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998, "parameters": 1121, "training_steps": 4999}
prior_hypothesis: Fixing one attention-projection coefficient to zero through a triangular value-basis shear will reduce the qualified model from 1,122 to 1,121 parameters while retaining at least 99% accuracy, because it preserves the learned attention function and avoids dividing by the unstable third coefficient that caused both three-scale-anchor failures.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1123, "training_steps": 4999}
prior_hypothesis: Fixing one attention-output projection coefficient and absorbing its scale into the corresponding learned value feature will reduce the qualified model from 1,124 to 1,123 parameters while retaining at least 99% accuracy, because this removes an exact factorization symmetry without tying or deleting any attention feature.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9975, "parameters": 1126, "training_steps": 4999}
prior_hypothesis: Extending the qualified five-way final-bias tie to six channels will reduce the model from 1,127 to 1,126 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing one attention-output projection coefficient and absorbing its scale into the corresponding learned value feature will reduce the qualified model from 1,124 to 1,123 parameters while retaining at least 99% accuracy, because this removes an exact factorization symmetry without tying or deleting any attention feature.
change: Canonically fix the first attention-projection coefficient, reconstruct the projection from 55 learned coordinates, rescale the matching value row at initialization and after optimizer steps, and adapt gauge-gradient reconstruction.
mechanism: Single-feature value/projection scale quotient
evidence_used: The 1,124-parameter model reached 99.84%–99.85%, while independently tying attention or MLP output biases failed and sharing whole value encoders collapsed to 6.31%; an exact one-coordinate value/projection scale quotient preserves those sensitive functional degrees of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1123, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified single-feature attention scale quotient to a second independent value feature will reduce the model from 1,123 to 1,122 learned parameters while retaining at least 99% accuracy, because both removed coefficients are exact multiplicative factorization gauges and no attention feature is tied or deleted.
change: Fix the first two attention-projection coefficients, absorb their scales into the corresponding learned value rows at initialization and after optimizer steps, and reconstruct the two omitted projection gradients.
mechanism: Two-feature value/projection scale quotient
evidence_used: The single-feature value/projection scale quotient achieved 99.96% accuracy with 1,123 parameters, while attention-bias tying and whole value-encoder sharing failed; this motivates extending the successful exact symmetry removal by one independent feature without reducing functional capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1122, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-feature attention scale quotient to a third independent value feature will reduce the model from 1,122 to 1,121 learned parameters while retaining at least 99% accuracy, because the removed coefficient is another exact multiplicative factorization gauge and no learned feature is tied or deleted.
change: Adopt the qualified fully zero-biased final LayerNorm, then fix three attention-projection coefficients and absorb their scales into the corresponding learned value rows during initialization and optimizer updates.
mechanism: Three-feature value/projection scale quotient
evidence_used: The two-feature quotient achieved 99.98% accuracy with 1,122 parameters, improving on the one-feature quotient’s 99.96% at 1,123; this supports testing one further independent exact scale gauge while preserving the bias and nonlinear capacity whose direct reductions failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1121, "training_steps": 4999}

RECENT RESULT
hypothesis: Directly initializing the third fixed projection anchor in its canonical chart will reduce the qualified 1,122-parameter design to 1,121 parameters while retaining at least 99% accuracy, avoiding the potentially ill-conditioned initialization rescaling used by the failed three-feature attempt.
change: Fully fix the final LayerNorm bias, remove three attention-projection coefficients, absorb their scales into the corresponding learned value features during optimization, preserve the qualified initialization for the first two features, and directly initialize the new third anchor.
mechanism: Conditioned three-feature value/projection scale quotient
evidence_used: The two-feature quotient achieved 99.98% accuracy with 1,122 parameters, whereas the three-feature extension collapsed to 0.0001 accuracy; changing only the new feature’s chart initialization tests whether that failure was optimization conditioning rather than loss of functional capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.42479999999999996, "parameters": 1121, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one attention-projection coefficient to zero through a triangular value-basis shear will reduce the qualified model from 1,122 to 1,121 parameters while retaining at least 99% accuracy, because it preserves the learned attention function and avoids dividing by the unstable third coefficient that caused both three-scale-anchor failures.
change: Preserve the two qualified value-scale anchors, remove the third projection coefficient, and absorb it into a same-head shear between value features zero and two during initialization and optimizer updates.
mechanism: Within-head value-basis shear quotient
evidence_used: The two-feature scale quotient achieved 99.98% accuracy at 1,122 parameters, while third scale anchors achieved only 0.01% and 42.48%; the proposed shear instead uses the already-fixed nonzero first anchor as its stable denominator and removes an exact within-head factorization gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1121, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-scale, one-shear 1,121-parameter design with a second shear anchored by the fixed nonzero first projection coefficient will reduce it to 1,120 parameters while retaining at least 99% accuracy, because the transformation exactly preserves the learned attention function without dividing by another unstable coefficient.
change: Adopt the qualified two projection-scale anchors and first within-head shear, then eliminate the fourth projection coefficient by shearing value feature three into feature zero during initialization and every projection optimizer step.
mechanism: Second stable within-head value-basis shear quotient
evidence_used: The first within-head shear achieved 99.98% accuracy with 1,121 parameters, while attempts to use the third coefficient as a scale anchor failed at 0.01% and 42.48%; another shear through the already-fixed 0.02 anchor tests the same successful, well-conditioned symmetry removal.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the qualified two-scale, one-shear design with a second shear through the fixed 0.02 projection anchor will reduce the model from 1,121 to 1,120 parameters while retaining at least 99% accuracy.
change: Adopt the qualified fully zero-biased final LayerNorm, preserve both projection-scale anchors and the first shear, then eliminate the fourth projection coefficient by shearing value feature three into feature zero during initialization and every projection update.
mechanism: Second stable within-head value-basis shear quotient
evidence_used: The first stable within-head shear achieved 99.98% accuracy with 1,121 parameters, while unstable third scale anchors failed; the previous second-shear implementation could not be verified, leaving a fully specified extension of the successful shear mechanism as the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1120, "training_steps": 4999}



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
