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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1036, "training_steps": 4999}
prior_hypothesis: Applying the qualified fixed positional amplitude and fixing one key-projection coefficient in each attention head will produce a 1,036-parameter transformer with at least 99% accuracy, because each head has an independent reciprocal query/key scaling gauge and the first such fix already achieved 99.94%.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9994, "parameters": 1002, "training_steps": 4999}
prior_hypothesis: Reducing the qualified 10-unit GELU MLP to nine units will produce a 1,002-parameter transformer with at least 99% accuracy, because the 10-unit model achieved 99.96% accuracy and retained substantial margin above the threshold.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9992, "parameters": 1032, "training_steps": 4999}
prior_hypothesis: Fixing a second zero-initialized final latent-bias coordinate at zero will reduce the model from 1,033 to 1,032 parameters while retaining at least 99% accuracy, because the remaining three learned latent-bias coordinates, trainable token codes, and upstream projections preserve sufficient output calibration.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1017, "training_steps": 4999}
prior_hypothesis: A 10-unit GELU MLP on the qualified two-coordinate final-bias reduction will produce a 1,017-parameter transformer with at least 99% accuracy, because the task may require fewer nonlinear features than the universally retained 11-unit width while preserving both independent attention heads and the five-dimensional token representation.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing separate per-head key/value projections with one learned four-dimensional key/value representation will reduce the qualified 1,038-parameter model to 982 parameters while retaining at least 99% accuracy, because head-specific queries, relative biases, attended summaries, and output mixing still provide distinct routing, while digit identity and position can share a common key/value representation.
change: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then replace the 24-output QKV projection with an eight-dimensional query projection and a shared eight-output key/value projection broadcast across both attention heads.
mechanism: Multi-query causal attention with shared key/value features
evidence_used: The 1,038-parameter design achieved 99.96%, whereas replacing factorized attention with direct quotient-space operators fell to 89.02%; this motivates preserving factorized dot-product attention and head-specific routing while challenging the shared assumption that each head needs independent key and value features. The failed four-coordinate lexical bottleneck also suggests moving compression away from token representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8368000000000001, "parameters": 982, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero key-projection coefficient while leaving its reciprocal query scale learned will reduce the qualified model from 1,038 to 1,037 parameters and retain at least 99% accuracy, because this removes one multiplicative Q/K factorization gauge without replacing the successfully optimized factorized attention.
change: Add a quotient-input QKV projection whose first key coefficient is a fixed initialization-scale buffer, and use it in causal self-attention.
mechanism: Single-coordinate Q/K scale gauge fixing
evidence_used: The 1,038-parameter factorized-attention design achieved 99.96%, while folding the attention factorizations into direct operators fell to 89.02%; this motivates an incremental gauge fix that preserves the factorized dot-product parameterization and its optimization behavior.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1037, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the qualified fixed positional amplitude and fixing one key-projection coefficient in each attention head will produce a 1,036-parameter transformer with at least 99% accuracy, because each head has an independent reciprocal query/key scaling gauge and the first such fix already achieved 99.94%.
change: Replace the 168 learned QKV coefficients with 166 learned coefficients plus one fixed nonpersistent key coefficient per head, and replace the learned positional-scale scalar with its qualified fixed value.
mechanism: Per-head Q/K scale gauge fixing
evidence_used: The 1,037-parameter design achieved 99.94% after fixing one key coefficient, while the fixed-position 1,038-parameter design achieved 99.96%; extending the same gauge fix to the untouched second head is the narrowest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1036, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified per-head Q/K gauge fixing to one additional key channel will produce a 1,035-parameter transformer with at least 99% accuracy, because its corresponding learned query channel and query bias retain the reciprocal scaling freedom.
change: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then replace three learned QKV coefficients—one per head plus a second channel in the first head—with fixed nonzero initialization-scale coefficients.
mechanism: Third reciprocal Q/K scale gauge fix
evidence_used: The 1,036-parameter design achieved 99.99% accuracy after fixing one key coefficient per head; fixing one additional channel is the narrowest extension of that successful gauge reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1035, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the second key-channel coefficient in the remaining attention head will produce a 1,034-parameter transformer with at least 99% accuracy because its corresponding learned query channel and bias preserve the reciprocal scaling freedom.
change: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then fix two nonzero key-projection coefficients per attention head.
mechanism: Symmetric fourth reciprocal Q/K scale gauge fix
evidence_used: The 1,036-parameter per-head gauge-fixed design achieved 99.99%, and extending it to three fixed coefficients produced a qualified 1,035-parameter model at 99.78%; symmetrically fixing the second channel of the other head is the narrowest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965, "parameters": 1034, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third key-channel coefficient in the first attention head will reduce the model from 1,034 to 1,033 parameters while retaining at least 99% accuracy, because that channel’s learned query projection and query bias preserve the reciprocal Q/K scaling freedom.
change: Extend the qualified symmetric four-coefficient gauge fixing by fixing one additional nonzero key-projection coefficient in the first attention head.
mechanism: Fifth reciprocal Q/K channel-scale gauge fix
evidence_used: The current model achieved 99.65% accuracy with two fixed key-channel coefficients per head, after the three-coefficient model achieved 99.78%; the next single-channel gauge fix is the narrowest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1033, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified four-key-fix model will retain at least 99% accuracy with 1,033 parameters when one query-bias coordinate is fixed at zero, because a per-head query/key change of basis can orient the learned query bias away from that coordinate while preserving attention scores.
change: Apply the qualified symmetric fourth key-coefficient fix and replace the eight-parameter query bias with seven learned coordinates plus one fixed zero coordinate.
mechanism: Single-coordinate query-bias gauge fixing
evidence_used: The four-key-fix design achieved 99.65% accuracy with 1,034 parameters, while a fifth key-coefficient fix collapsed to 0.01%; this motivates testing an orthogonal one-parameter gauge reduction whose fixed query-bias coordinate exactly matches the qualified model’s zero initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0901, "parameters": 1033, "training_steps": 4999}

RECENT RESULT
hypothesis: The qualified four-key-fix architecture will retain at least 99% accuracy with 1,033 parameters when one head’s maximum-distance relative-bias coefficient is fixed at zero, because that coefficient affects only one attention logit at the full context length.
change: Apply the qualified two-key-coefficient-per-head gauge fixing, then store all relative-bias coefficients except the second head’s farthest-distance coefficient and reconstruct that coefficient as a fixed zero.
mechanism: Single-head maximum-distance bias pruning
evidence_used: The four-key-fix design achieved 99.65% accuracy with 1,034 parameters, while two different 1,033-parameter reductions targeting Q/K or query-bias gauges collapsed; this motivates testing a distinct, narrowly supported routing parameter.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.055999999999999994, "parameters": 1033, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified four-key-fix architecture, fixing one zero-initialized final latent-bias coordinate at zero will produce a 1,033-parameter transformer with at least 99% accuracy because the trainable token codes, remaining four latent-bias coordinates, and upstream learned projections retain output-calibration freedom.
change: Apply two fixed key-projection coefficients per attention head, then replace the five-parameter final latent bias with four learned coordinates and one fixed zero coordinate.
mechanism: Final-classifier latent-bias coordinate fixing
evidence_used: The 1,034-parameter four-key-fix design achieved 99.65%; three failed 1,033-parameter attempts modified attention routing or Q/K optimization, motivating an equally narrow reduction in the separate output-calibration path while preserving the qualified model’s initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1033, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second zero-initialized final latent-bias coordinate at zero will reduce the model from 1,033 to 1,032 parameters while retaining at least 99% accuracy, because the remaining three learned latent-bias coordinates, trainable token codes, and upstream projections preserve sufficient output calibration.
change: Shorten the learned final latent bias from four coordinates to three and reconstruct the final two coordinates as fixed zeros.
mechanism: Second final-classifier latent-bias coordinate fixing
evidence_used: The immediately preceding one-coordinate final-bias fix achieved 99.77% accuracy with 1,033 parameters, whereas similarly sized reductions in attention routing, key scaling, and query bias collapsed; extending the successful output-path reduction by one coordinate is the narrowest informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1032, "training_steps": 4999}

RECENT RESULT
hypothesis: A 10-unit GELU MLP on the qualified two-coordinate final-bias reduction will produce a 1,017-parameter transformer with at least 99% accuracy, because the task may require fewer nonlinear features than the universally retained 11-unit width while preserving both independent attention heads and the five-dimensional token representation.
change: Start from the qualified 1,032-parameter output-bias design and reduce `d_ff` from 11 to 10, removing 15 additional parameters by narrowing both learned MLP projections.
mechanism: Ten-feature nonlinear carry bottleneck
evidence_used: The 1,032-parameter final-bias design achieved 99.92%, while the 982-parameter shared-key/value model failed at 83.68% and the four-coordinate lexical bottleneck also failed; this identifies attention-head specialization and token representation as load-bearing, but leaves the shared assumption that 11 independent GELU features are necessary untested.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1017, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the qualified 10-unit GELU MLP to nine units will produce a 1,002-parameter transformer with at least 99% accuracy, because the 10-unit model achieved 99.96% accuracy and retained substantial margin above the threshold.
change: Reconstruct the qualified four-key-fix, two-coordinate final-bias reduction, then reduce `d_ff` from 10 to 9, removing 15 additional MLP parameters.
mechanism: Nine-feature nonlinear carry bottleneck
evidence_used: The qualified 1,017-parameter model with `d_ff=10` achieved 99.96% accuracy; testing the adjacent width of nine is the narrowest informative continuation of the most successful large parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1002, "training_steps": 4999}



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
