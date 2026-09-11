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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Constraining one token-embedding feature column to centered contrasts will produce a 1628-parameter model with at least 99% accuracy, because shifting the removed column mean into every positional embedding preserves all transformer inputs exactly, while the tied output logits change only by a class-independent offset.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one `ln2` bias coordinate at zero will reduce the model from 1636 to 1635 parameters while retaining at least 99% accuracy, because its effect is exactly representable by the full learned `fc1` bias and the initial model computation remains unchanged.
change: Replace the block’s second LayerNorm with an equivalent implementation containing seven learned bias coordinates and one fixed-zero coordinate, while retaining all gains and MLP parameters.
mechanism: Locally absorbable pre-MLP normalization bias
evidence_used: The 1636-parameter design reached 99.98%, while narrowing the MLP collapsed accuracy and attention-path bias reductions proved optimization-sensitive; this tests a new, strictly local redundancy immediately before the MLP’s learned affine bias without reducing MLP width.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln2` bias coordinate at zero will reduce the model from 1635 to 1634 parameters while retaining at least 99% accuracy, because the 1635-parameter design achieved 99.87% and both removed offsets are locally representable by the learned `fc1` bias.
change: Store six learned `ln2` bias coordinates instead of seven and append two fixed zeros during normalization.
mechanism: Incremental pre-MLP LayerNorm bias reduction
evidence_used: Fixing one `ln2` bias coordinate at zero retained 99.87% accuracy at 1635 parameters; this is the only tested single-parameter reduction that passed and directly motivates a conservative second-coordinate ablation in the same redundant pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7637, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the failed second fixed-coordinate ablation with six learned zero-mean contrasts across the seven active `ln2` bias coordinates will reduce the model to 1634 parameters while retaining at least 99% accuracy, because every active coordinate remains trainable and the removed common-offset direction is locally absorbable by `fc1.bias`.
change: Parameterize the first seven `ln2` bias coordinates as six centered contrasts while keeping the eighth coordinate fixed at zero.
mechanism: Balanced LayerNorm-bias contrast gauge
evidence_used: One fixed `ln2` bias coordinate achieved 99.87% at 1635 parameters, but fixing an adjacent second coordinate fell to 76.37%; this motivates distributing the second constraint across all seven active coordinates instead of eliminating another coordinate’s optimization path.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the eight `ln2` gains to seven learned mean-one contrasts will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy, because every effective gain remains trainable and the removed common-scale direction is locally absorbable by `fc1.weight` and `fc1.bias`.
change: Replace the eight independent `ln2` gain parameters with seven parameters that reconstruct eight mean-one gains, preserving the initial computation exactly.
mechanism: Balanced LayerNorm-gain contrast gauge
evidence_used: Centering six learned contrasts across seven active `ln2` bias coordinates achieved 99.96% at 1634 parameters after a fixed-coordinate reduction failed; this motivates applying the same balanced parameterization to the locally redundant common-gain direction instead of fixing one gain coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.34340000000000004, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `ln1` bias to seven learned mean-zero contrasts will reduce the model to 1633 parameters while retaining at least 99% accuracy, because all eight effective coordinates remain trainable and the removed common-offset direction is absorbable by query/value biases while its key-bias effect cancels under softmax.
change: Replace the eight independent `ln1` bias parameters with seven parameters that reconstruct eight centered biases, preserving initialization, gains, and all attention/MLP capacity.
mechanism: Centered attention-input LayerNorm bias gauge
evidence_used: The balanced `ln2` bias contrast parameterization achieved 99.96% at 1634 parameters, whereas the balanced gain constraint collapsed to 34.34%; this motivates applying the successful distributed bias constraint to `ln1` without altering normalization gains.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `ln1` bias to six centered contrasts over seven active coordinates will reduce the model to 1632 parameters while retaining at least 99% accuracy, because the current centered `ln1` bias achieved 99.94% and the identical two-constraint structure in `ln2` achieved 99.96%.
change: Store six learned `ln1` bias contrasts, center them across seven active coordinates, and keep the eighth coordinate fixed at zero.
mechanism: Balanced second attention LayerNorm-bias gauge
evidence_used: The 1633-parameter centered `ln1` bias design reached 99.94%; moreover, the balanced six-contrast `ln2` parameterization succeeded at 99.96% where directly fixing a second coordinate failed, motivating the same distributed reduction for `ln1`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc2` bias to seven learned zero-mean contrasts will reduce the model to 1631 parameters while retaining at least 99% accuracy, because its eliminated common-offset direction is exactly removed by the immediately following final LayerNorm.
change: Preserve `fc2` construction and initialization while replacing its eight-element bias with seven learned contrasts that reconstruct a centered eight-coordinate bias.
mechanism: Terminal LayerNorm-null MLP output-bias gauge
evidence_used: Balanced bias contrasts in `ln1` and `ln2` retained 99.93% and 99.96% accuracy; unlike the failed attention projection-bias reduction, `fc2` is terminal and its common bias shift reaches final LayerNorm without an intervening computation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering one `fc2` output column will reduce the model to 1630 parameters while retaining at least 99% accuracy, because the removed input-dependent common offset is exactly canceled by final LayerNorm and all eight effective column coordinates remain represented through seven balanced contrasts.
change: Replace the first `fc2` weight column with seven learned centered contrasts, retain the other eleven columns unchanged, and preserve the original initialization and RNG sequence.
mechanism: Terminal LayerNorm-null MLP weight-column gauge
evidence_used: Centering the terminal `fc2` bias retained 99.93% accuracy at 1631 parameters, showing that a balanced gauge immediately before final LayerNorm trains successfully; reducing only one weight-column gauge is the smallest extension of that proven mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1630, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering a second `fc2` output column will reduce the model from 1630 to 1629 parameters while retaining at least 99% accuracy, because its input-dependent common offset is exactly removed by final LayerNorm and all eight effective coordinates remain represented by seven balanced contrasts.
change: Represent the first two `fc2` weight columns using seven learned centered contrasts each, while preserving the remaining columns and the original initialization RNG sequence.
mechanism: Incremental terminal LayerNorm-null weight-column gauge
evidence_used: Centering the first `fc2` output column achieved 99.96% accuracy at 1630 parameters; applying the identical proven gauge to one additional column is the most conservative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9943000000000001, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering a third `fc2` output column will reduce the model from 1629 to 1628 parameters while retaining at least 99% accuracy, because its input-dependent common offset is exactly removed by final LayerNorm and seven balanced contrasts preserve every effective coordinate.
change: Represent the first three `fc2` weight columns with seven learned centered contrasts each, leaving the remaining nine columns unchanged and preserving the original initialization RNG sequence.
mechanism: Incremental terminal LayerNorm-null weight-column gauge
evidence_used: Centering the first two `fc2` columns produced passing models at 1630 and 1629 parameters; the current two-column design achieved 99.43%, motivating the identical one-column incremental reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5222, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the first positional-embedding row will reduce the model from 1629 to 1628 parameters while retaining at least 99% accuracy, because its removed featurewise common offset is invisible to every tokenwise LayerNorm and all seven effective contrasts remain learned.
change: Replace the first positional-embedding row with seven centered learned contrasts, retain all other rows unchanged, and preserve the original initialization RNG sequence.
mechanism: Tokenwise LayerNorm-null positional gauge
evidence_used: A third centered `fc2` column collapsed to 52.22%, while balanced bias gauges in `ln1`, `ln2`, and terminal `fc2` passed; this motivates testing the same distributed one-parameter gauge in a distinct exact LayerNorm-null pathway without further perturbing MLP hidden-unit optimization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6411, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining one token-embedding feature column to centered contrasts will produce a 1628-parameter model with at least 99% accuracy, because shifting the removed column mean into every positional embedding preserves all transformer inputs exactly, while the tied output logits change only by a class-independent offset.
change: Replace one token-embedding column with balanced learned contrasts, transfer its initialization mean to the matching positional column, and reconstruct the tied output weight dynamically while preserving the original RNG sequence.
mechanism: Joint token–position embedding gauge
evidence_used: The isolated positional-row gauge failed at 64.11%, whereas balanced terminal contrast gauges passed through 1629 parameters; this patch instead removes an exact joint token/position/output redundancy without perturbing the represented initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1628, "training_steps": 4999}



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
