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
verified_results: {"accuracy": 0.9983, "parameters": 1602, "training_steps": 4999}
prior_hypothesis: Learning five query-bias coordinates while retaining the qualified gauge-aware positional parameterization will produce 1602 parameters and maintain at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9992, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Removing the eight value-projection bias parameters in addition to the already qualified key-bias removal will produce 1628 parameters and retain at least 99% accuracy, because softmax-normalized attention makes value bias an input-independent vector that the existing output-projection bias can represent.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1600, "training_steps": 4999}
prior_hypothesis: Fixing one `fc1` bias coordinate at zero will produce 1600 learned parameters while retaining at least 99% accuracy, because all 12 nonlinear units and their input/output weights remain trainable, and the learned `ln2` affine offset can still shift the affected unit’s preactivation.

## Recent verification evidence

RECENT RESULT
hypothesis: Learning four query-bias coordinates while retaining the qualified gauge-aware positional parameterization will produce 1601 parameters and maintain at least 99% accuracy.
change: Fix the final four query-bias coordinates at zero while preserving all weight tensors, positional quotient updates, initialization streams, and training behavior.
mechanism: Fourth query-bias coordinate ablation
evidence_used: Five learned query-bias coordinates achieved 99.83% accuracy at 1602 parameters, after six coordinates achieved 99.94% at 1603; this supports testing the next one-scalar localized ablation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning three query-bias coordinates while retaining the qualified gauge-aware positional parameterization will produce 1600 parameters and maintain at least 99% accuracy.
change: Apply the proven seven-coordinate positional embedding and full-space-equivalent AdamW updates, remove key/value biases, and fix the final five query-bias coordinates at zero.
mechanism: Fifth query-bias coordinate ablation with gauge-quotient positional embeddings
evidence_used: Four learned query-bias coordinates achieved 99.96% accuracy at 1601 parameters, following successful reductions at five and six coordinates; this supports testing the next localized one-scalar ablation while preserving the qualified positional quotient and training dynamics.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7319, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the common-channel degree of freedom from each attention output-projection bias will produce 1600 parameters and retain at least 99% accuracy, because that shift passes unchanged through residual paths and is canceled by subsequent LayerNorms.
change: Represent each eight-coordinate attention projection bias with seven learned coordinates, reconstruct a zero final coordinate during forward, and include it with the positional embedding in gauge-aware AdamW and gradient clipping.
mechanism: Gauge-quotient attention output bias with full-space-equivalent optimization
evidence_used: The current four-query-bias design achieved 99.96% at 1601 parameters, while the positional quotient succeeded only after preserving full-space AdamW and clipping dynamics; this motivates applying the same qualified treatment to another exact LayerNorm-invisible direction instead of repeating the failed three-query-coordinate ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2711, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1601-parameter four-query-bias design, fixing one `ln2` bias coordinate at zero will produce 1600 parameters and retain at least 99% accuracy because `fc1.bias` can represent that coordinate’s downstream affine contribution.
change: Adopt the qualified four-coordinate query bias, then learn seven `ln2` bias coordinates and reconstruct the eighth as zero without changing initialization RNG streams or decoding.
mechanism: Single-coordinate LayerNorm-to-MLP bias absorption
evidence_used: Four query-bias coordinates achieved 99.96% at 1601 parameters. Removing all eight `ln2` biases failed at 17.27%, motivating a localized one-coordinate test that retains seven coordinates and the full MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.048600000000000004, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one of the 96 `fc2` weights at zero while retaining the qualified four-coordinate query bias and gauge-aware positional embedding will produce 1600 learned parameters and maintain at least 99% accuracy.
change: Adopt the verified 1601-parameter design, then represent `fc2.weight` with 95 learned values and one fixed-zero final connection while preserving baseline constructor and initialization RNG streams.
mechanism: Single MLP output-connection ablation
evidence_used: The four-query-bias positional-quotient design achieved 99.96% at 1601 parameters, whereas removing another query coordinate failed at 73.19%; this motivates preserving the proven attention path and testing a single localized MLP connection without reducing its 12-unit nonlinear capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5504, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining query-bias coordinates 1–3 while fixing coordinate 0 will produce 1600 parameters and maintain at least 99% accuracy, indicating that the previous three-coordinate failure was caused by removing load-bearing coordinate 3 rather than by insufficient bias dimensionality.
change: Replace the four-coordinate query bias with three learned coordinates occupying indices 1–3, preserving all weight tensors, initialization streams, positional quotient updates, and training behavior.
mechanism: Alternate query-bias coordinate ablation
evidence_used: Four query-bias coordinates achieved 99.96% at 1601 parameters, while learning only coordinates 0–2 fell to 73.19%; testing the complementary removal within the same proven four-coordinate head directly distinguishes a coordinate-specific failure from a three-parameter capacity limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7012999999999999, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the shared token-embedding/output matrix to learned rank seven will reduce the verified four-query-bias design from 1601 to 1543 parameters while retaining at least 99% accuracy, because the transformer’s eight-channel internal computation remains intact and only the least energetic singular direction of the tied vocabulary interface is removed.
change: Replace the full tied vocabulary matrix with learned rank-seven token codes and basis factors, initialized from the best rank-seven approximation of the same fresh baseline draw; retain the qualified positional quotient and four-coordinate query bias.
mechanism: Rank-seven tied vocabulary interface
evidence_used: The four-query-bias design achieved 99.96% at 1601 parameters, while several one-coordinate attention, LayerNorm, and MLP ablations failed sharply. This indicates those internal paths are load-bearing and motivates challenging the previously untouched assumption that the 114-by-8 tied vocabulary interface must be full rank.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1543, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the single all-entries shift symmetry of the tied token-embedding/output matrix will reduce the verified 1601-parameter design to 1600 parameters while retaining at least 99% accuracy, because the shift is invisible to every LayerNorm-mediated residual block and adds only a common offset to all output logits.
change: Adopt the qualified seven-coordinate positional embedding and four-coordinate query bias, then represent the tied vocabulary matrix with one global scalar omitted while preserving baseline initialization draws, full-space AdamW moments, and gradient clipping.
mechanism: Global tied-embedding shift quotient with gauge-aware AdamW
evidence_used: The four-query-bias positional-quotient design achieved 99.96% at 1601 parameters, and the positional quotient succeeded after full-space optimizer dynamics were preserved; the rank-seven tied-interface attempt failed at 0.01%, motivating removal of only an exact global shift redundancy rather than an embedding rank direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.20489999999999997, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `fc1` bias coordinate at zero will produce 1600 learned parameters while retaining at least 99% accuracy, because all 12 nonlinear units and their input/output weights remain trainable, and the learned `ln2` affine offset can still shift the affected unit’s preactivation.
change: Preserve the verified positional quotient and four-coordinate query bias, while representing `fc1.bias` with 11 learned coordinates and one fixed-zero trailing coordinate.
mechanism: Single MLP hidden-bias ablation
evidence_used: The 1601-parameter design reached 99.96%, while reductions in query bias, `ln2`, and `fc2` failed; this motivates preserving those load-bearing paths and testing the previously untouched, zero-initialized `fc1` bias with a localized one-scalar ablation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the verified four-coordinate query bias while fixing a second trailing `fc1` bias coordinate at zero will reduce the model to 1599 learned parameters and maintain at least 99% accuracy.
change: Use four learned query-bias coordinates and ten learned `fc1` bias coordinates, preserving all weight shapes, initialization streams, positional gauge optimization, and decoding behavior.
mechanism: Second localized MLP hidden-bias ablation
evidence_used: The 1600-parameter reference fixed one `fc1` bias coordinate and achieved 99.91% accuracy; this directly motivates testing one additional scalar removal in the same tolerant parameter group.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9856999999999999, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying hidden-bias coordinates 9 and 10 while retaining the verified fixed-zero coordinate 11 will produce 1599 parameters and at least 99% accuracy, because it preserves a trainable threshold for both units whereas the prior 1599 design forced both trailing thresholds to zero and narrowly missed at 98.57%.
change: Adopt the qualified positional gauge, four-coordinate query bias, and gauge-aware optimization, then represent the 12-coordinate `fc1` bias with ten learned values, using the final learned value for coordinates 9 and 10 and fixing coordinate 11 at zero.
mechanism: Shared-threshold pair in the MLP hidden bias
evidence_used: The 1600-parameter design with only coordinate 11 fixed achieved 99.91%, while additionally fixing coordinate 10 reached 98.57%; replacing that second zero constraint with a shared learned threshold is a lower-distortion test in the same nearly qualified parameter group.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5196000000000001, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing hidden-bias coordinates 0 and 11 while retaining coordinates 1–10 will produce 1599 learned parameters and at least 99% accuracy, indicating that the prior 98.57% result was specific to removing coordinate 10 rather than a general two-bias capacity limit.
change: Preserve the qualified positional gauge, four-coordinate query bias, initialization streams, optimization, and decoding, while replacing the 11 learned `fc1` bias values with ten values mapped to coordinates 1–10 and fixing coordinates 0 and 11 at zero.
mechanism: Alternate second hidden-bias coordinate ablation
evidence_used: Fixing coordinate 11 alone achieved 99.91% at 1600 parameters, while additionally fixing coordinate 10 narrowly missed at 98.57%; testing a nonadjacent second coordinate is the cleanest localized test of hidden-unit-specific sensitivity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.09970000000000001, "parameters": 1599, "training_steps": 4999}



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
