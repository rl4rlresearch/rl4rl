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
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1328, "training_steps": 4999}
prior_hypothesis: Gauge-fixing all eight attention-output projection columns will reduce the model from 1,334 to 1,328 learned parameters while retaining at least 99% accuracy, because each omitted coefficient contributes only a hidden-state-dependent channel-common residual shift removed by subsequent LayerNorms.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9959, "parameters": 1318, "training_steps": 4999}
prior_hypothesis: Extending the verified one-coordinate `ln1` scale quotient to two coordinates will reduce the model from 1,319 to 1,318 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb both fixed channel scales while six adaptive scales retain substantially more optimization flexibility than the failed fully non-affine design.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9965999999999999, "parameters": 1320, "training_steps": 4999}
prior_hypothesis: Starting from the verified 1,328-parameter design, removing the eight-parameter first LayerNorm bias will produce 1,320 learned parameters and retain at least 99% accuracy, because content-independent attention maps that bias to a position-independent output offset absorbable by the attention projection bias, modulo downstream LayerNorm-invariant common shifts.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1336, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,337-parameter content-independent attention model’s `fc2` gauge from ten to eleven columns will produce 1,336 learned parameters and retain at least 99% accuracy, because the additional omitted coefficient changes the residual only by a channel-common shift removed by the final LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified eight-column `fc2` gauge to a ninth adjacent hidden column will produce 1,467 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Adopt the qualified eight-column construction and gauge-fix the final output-row coefficients of the last nine `fc2` input columns while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Ninth final-MLP common-output column gauge
evidence_used: The eight-column quotient achieved 99.92% accuracy at 1,468 parameters after every one-through-seven-column variant also exceeded 99%; this is the next one-coordinate extension of the same repeatedly verified exact symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1467, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing token-dependent query/key dot products with two learned causal relative-lag attention distributions will retain at least 99% accuracy with 1,338 parameters, because the fixed-format task needs head-specific operand routes but may not require those routes to vary with token content.
change: Start from the qualified nine-column MLP gauge, replace QKV with a learned value-only projection, route attention solely through the existing head-specific lag tables, and gauge-fix the now-unshared attention output bias.
mechanism: Content-independent dual-head learned lag routing
evidence_used: Learned relative-lag routing achieved 99.96% at 1,476 parameters, while reducing token-representation rank collapsed to 5.06%; this motivates preserving full-rank token embeddings and testing the distinct assumption that content-dependent Q/K routing—not representation capacity—is expendable.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1338, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified nine-column `fc2` gauge to a tenth adjacent hidden column will reduce the model to 1,337 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Gauge-fix the final output-row coefficient of the tenth trailing `fc2` input column while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Tenth final-MLP common-output column gauge
evidence_used: The current content-independent attention model achieved 99.85% accuracy with nine `fc2` column quotients at 1,338 parameters, and every preceding one-through-nine-column extension exceeded 99%; this tests the next coordinate of the same exact symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1337, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,337-parameter content-independent attention model’s `fc2` gauge from ten to eleven columns will produce 1,336 learned parameters and retain at least 99% accuracy, because the additional omitted coefficient changes the residual only by a channel-common shift removed by the final LayerNorm.
change: Adopt the qualified value-only learned-lag attention architecture and gauge-fix the final output-row coefficients of the last eleven `fc2` columns, preserving full-space initialization, AdamW moments, weight decay, and gauge-aware clipping.
mechanism: Eleventh final-MLP common-output column gauge
evidence_used: Reference Design 1 achieved 99.87% accuracy with 1,337 parameters after quotienting ten `fc2` columns; every preceding one-through-nine-column extension also exceeded 99%, motivating the next one-coordinate extension of the same exact symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1336, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified eleven-column `fc2` gauge to all twelve hidden columns will produce 1,335 learned parameters and retain at least 99% accuracy, because the final omitted coefficient also contributes only a channel-common shift removed by the final LayerNorm.
change: Adopt the qualified value-only learned-lag attention design and gauge-fix every `fc2` output column while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware clipping.
mechanism: Complete final-MLP common-output column gauge
evidence_used: The eleven-column value-only design achieved 99.88% accuracy with 1,336 parameters, after every preceding `fc2` gauge extension also exceeded 99%; completing the same exact symmetry is the most directly supported next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1335, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying an exact one-coordinate common-shift gauge to Reference Design 2’s tied token embedding/head will reduce the model from 1,335 to 1,334 parameters while retaining at least 99% accuracy.
change: Adopt the qualified value-only learned-lag attention and complete twelve-column MLP gauge, then represent the tied embedding modulo one global scalar while preserving full-space initialization, AdamW moments, weight decay, and gradient clipping.
mechanism: Global tied-embedding common-shift quotient
evidence_used: Reference Design 2 achieved 99.87% accuracy with 1,335 parameters; subtracting one scalar from every embedding entry changes residual inputs only by LayerNorm-invariant channel shifts and changes every output logit by the same softmax-invariant scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1334, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing all eight attention-output projection columns will reduce the model from 1,334 to 1,328 learned parameters while retaining at least 99% accuracy, because each omitted coefficient contributes only a hidden-state-dependent channel-common residual shift removed by subsequent LayerNorms.
change: Extend the attention projection gauge from two columns to all eight, preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Complete attention-output common-shift quotient
evidence_used: The current two-column attention gauge achieves 99.87% accuracy at 1,334 parameters, and the analogous complete twelve-column MLP common-output quotient also achieves 99.87%; both rely on the same downstream LayerNorm invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1328, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,328-parameter design, removing the eight-parameter first LayerNorm bias will produce 1,320 learned parameters and retain at least 99% accuracy, because content-independent attention maps that bias to a position-independent output offset absorbable by the attention projection bias, modulo downstream LayerNorm-invariant common shifts.
change: Adopt the verified globally gauge-fixed tied embedding and complete attention-output gauge, then remove `ln1.bias` while retaining learned causal relative-lag attention and gauge-aware optimization.
mechanism: Pre-attention LayerNorm-bias absorption
evidence_used: Reference Design 2 achieved 99.84% accuracy with 1,328 parameters using the global tied-embedding gauge and all eight attention-output gauges; its row-normalized, content-independent attention makes the first LayerNorm bias an exact redundant offset coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965999999999999, "parameters": 1320, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the remaining eight `ln1` scale parameters from the qualified 1,320-parameter design will produce 1,312 learned parameters and retain at least 99% accuracy, because content-independent attention allows the scale to be absorbed exactly into the unrestricted learned value projection.
change: Adopt Reference Design 3’s global tied-embedding, complete attention/MLP projection gauges, and removed `ln1` bias, then make `ln1` entirely non-affine while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware clipping.
mechanism: Pre-attention LayerNorm affine absorption
evidence_used: Reference Design 3 achieved 99.66% accuracy with 1,320 parameters after removing `ln1.bias`; its attention routes are independent of content, so the remaining LayerNorm scale only right-scales the learned value matrix and adds no independent function-space capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4251, "parameters": 1312, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,320-parameter design, fixing one `ln1` scale coordinate at one will produce 1,319 parameters and retain at least 99% accuracy, because the learned value projection can absorb that channel scale while the other seven adaptive scales preserve the optimization flexibility lost by the failed fully non-affine design.
change: Adopt the verified global embedding, complete attention/MLP projection, and `ln1`-bias gauges, then quotient only the final `ln1` scale coordinate while preserving full-shape initialization and gauge-aware optimization elsewhere.
mechanism: Single-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The affine-without-bias design achieved 99.66% at 1,320 parameters, whereas removing all eight remaining `ln1` scales collapsed to 42.51%; a one-coordinate quotient directly tests whether seven adaptive scales suffice without repeating the destructive all-at-once reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9952, "parameters": 1319, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified one-coordinate `ln1` scale quotient to two coordinates will reduce the model from 1,319 to 1,318 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb both fixed channel scales while six adaptive scales retain substantially more optimization flexibility than the failed fully non-affine design.
change: Fix the final two `ln1` scale coordinates at one instead of only the final coordinate, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Two-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The one-coordinate quotient achieved 99.52% accuracy at 1,319 parameters, whereas removing all eight scales collapsed to 42.51%; the next single-coordinate extension most directly tests where the optimization-sensitive boundary begins.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9959, "parameters": 1318, "training_steps": 4999}

RECENT RESULT
hypothesis: The unrestricted per-lag lookup is not load-bearing: three learned Gaussian delay components per head will retain at least 99% accuracy while reducing routing parameters from `n_head * (max_seq_len - 1)` to `8 * n_head`, because fixed-format routing should require only a few stationary delay modes.
change: Replace each head’s dense relative-lag table with a learned three-component continuous delay mixture, retaining independent centers and widths per head and gauge-fixed mixture logits; leave token representation, value projection, MLP, checkpoints, and decoding unchanged.
mechanism: Learned Gaussian delay-mixture attention
evidence_used: Content-independent dual-head learned-lag routing reached 99.85%, showing stationary learned routes suffice, while reducing token-representation rank collapsed to 5.06%; this motivates preserving representation capacity while challenging the shared assumption that stationary routing needs an unrestricted logit for every lag.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4985, "parameters": 1292, "training_steps": 4999}



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
