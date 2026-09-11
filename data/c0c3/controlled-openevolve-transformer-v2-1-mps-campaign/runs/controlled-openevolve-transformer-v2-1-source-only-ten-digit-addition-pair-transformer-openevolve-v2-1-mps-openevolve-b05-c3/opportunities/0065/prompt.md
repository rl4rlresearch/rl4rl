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
verified_results: {"accuracy": 0.9987, "parameters": 1334, "training_steps": 4999}
prior_hypothesis: Applying an exact one-coordinate common-shift gauge to Reference Design 2’s tied token embedding/head will reduce the model from 1,335 to 1,334 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987, "parameters": 1337, "training_steps": 4999}
prior_hypothesis: Extending the verified nine-column `fc2` gauge to a tenth adjacent hidden column will reduce the model to 1,337 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987, "parameters": 1335, "training_steps": 4999}
prior_hypothesis: Extending the verified eleven-column `fc2` gauge to all twelve hidden columns will produce 1,335 learned parameters and retain at least 99% accuracy, because the final omitted coefficient also contributes only a channel-common shift removed by the final LayerNorm.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1336, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,337-parameter content-independent attention model’s `fc2` gauge from ten to eleven columns will produce 1,336 learned parameters and retain at least 99% accuracy, because the additional omitted coefficient changes the residual only by a channel-common shift removed by the final LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-fixing a third adjacent `fc2` output coefficient will reduce the model to 1,473 learned parameters while retaining at least 99% accuracy, because it contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Extend the verified two-column `fc2` gauge to the last three hidden columns while preserving full-shape initialization, AdamW moments, weight decay, and gradient clipping.
mechanism: Third final-MLP output-column shift quotient
evidence_used: The two-column `fc2` quotient achieved 99.93% accuracy at 1,474 parameters after the one-column version achieved 99.94%; this extends the same successful exact symmetry by one coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1473, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-column `fc2` gauge to a fourth adjacent hidden column will produce 1,472 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Omit the final output-row coefficients of the last four `fc2` input columns while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Fourth final-MLP output-column shift quotient
evidence_used: The three-column `fc2` quotient achieved 99.98% accuracy at 1,473 parameters after the one- and two-column versions achieved 99.94% and 99.93%; this applies the same repeatedly successful exact symmetry to one additional adjacent column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1472, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified four-column `fc2` gauge to a fifth adjacent hidden column will produce 1,471 learned parameters and retain at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Omit the final output-row coefficients of the last five `fc2` input columns while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Fifth final-MLP output-column shift quotient
evidence_used: The four-column `fc2` quotient achieved 99.92% accuracy at 1,472 parameters after the one-, two-, and three-column variants all exceeded 99%; this applies the same repeatedly verified symmetry to one additional adjacent column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1471, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified five-column `fc2` gauge to a sixth adjacent hidden column will produce 1,470 learned parameters and retain at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Replace absolute positional embeddings with the qualified learned relative-lag biases and gauge-fix the last six `fc2` input columns, preserving full-space initialization, AdamW moments, weight decay, and gauge-aware clipping.
mechanism: Sixth final-MLP common-output column gauge with learned relative-lag attention
evidence_used: The five-column quotient achieved 99.95% accuracy at 1,471 parameters after the one-through-four-column variants all exceeded 99%; this applies the same repeatedly verified exact symmetry to one additional adjacent column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1470, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified six-column `fc2` gauge to a seventh adjacent hidden column will produce 1,469 learned parameters and retain at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Gauge-fix the final output-row coefficients of the last seven `fc2` input columns while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Seventh final-MLP common-output column gauge
evidence_used: The six-column quotient achieved 99.97% accuracy at 1,470 parameters after every one-through-five-column variant also exceeded 99%; this is another one-coordinate extension of the same repeatedly verified symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1469, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified seven-column `fc2` gauge to an eighth adjacent hidden column will produce 1,468 learned parameters and retain at least 99% accuracy because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.
change: Adopt the qualified seven-column construction and gauge-fix the final output-row coefficients of the last eight `fc2` input columns while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.
mechanism: Eighth final-MLP common-output column gauge
evidence_used: The seven-column quotient achieved 99.95% accuracy at 1,469 parameters after every one-through-six-column variant also exceeded 99%; this is the smallest and most directly supported extension of the repeatedly verified symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1468, "training_steps": 4999}

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
