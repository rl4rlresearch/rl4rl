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
verified_results: {"accuracy": 0.9983, "parameters": 1520, "training_steps": 4999}
prior_hypothesis: Fixing the penultimate attention value-bias coordinate at zero will reduce the model from 1,521 to 1,520 parameters while retaining at least 99% accuracy, because its position-independent projected contribution can be represented by the trainable attention projection bias.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the position-common component of the sole dense positional coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its input contribution transfers exactly to the corresponding token-embedding coordinate and adds only a vocabulary-common offset under the tied output projection.
change: Represent positional coordinate zero in the mean-free position basis, preserve its initialized common component in token embeddings, and leave no dense positional coordinate.
mechanism: Final positional common-mode quotient
evidence_used: Removing the same common mode from positional coordinate four achieved 99.88% accuracy at 1,525 parameters; this applies that already-successful exact quotient to the remaining dense coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing two of six arbitrary positional lookup channels with a fixed mean-free Fourier pair and two learned amplitudes will remove `2 * (max_seq_len - 2)` parameters while retaining at least 99% accuracy, because four learned channels preserve absolute-position flexibility while the Fourier pair supplies a translation-compatible routing signal.
change: Retain four learned mean-free positional profiles, replace the fifth and sixth profiles with analytic sine/cosine features, and initialize their learned amplitudes to match the discarded channels’ RMS scale.
mechanism: Hybrid learned-absolute and Fourier-relative position encoding
evidence_used: The 1,524-parameter design reached 99.98% after all positional common modes were removed, while recent one-parameter LayerNorm and attention reductions repeatedly failed. This suggests testing the load-bearing assumption that every remaining positional channel requires an independent lookup table rather than continuing along the exhausted scalar-quotient path.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7441, "parameters": 1482, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing only the sixth learned positional lookup profile with a scaled mean-free cosine will reduce parameters from 1,524 to 1,503 while retaining at least 99% accuracy, because five learned profiles preserve substantially more absolute-position flexibility than the failed four-profile design.
change: Retain five learned mean-free positional profiles, replace the final lookup profile with a fixed normalized cosine and one learned amplitude, and initialize that amplitude to the discarded profile’s RMS scale.
mechanism: Single-channel hybrid positional encoding
evidence_used: Replacing two learned profiles simultaneously reduced accuracy to 74.41%, while all six learned profiles reached 99.98%; isolating one replacement tests whether five learned profiles are sufficient and removes only half as many parameters as the failed change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9889, "parameters": 1503, "training_steps": 4999}

RECENT RESULT
hypothesis: Freezing only the final contrast coefficient of the sixth positional profile will reduce the model from 1,524 to 1,523 learned parameters while retaining at least 99% accuracy, because the model keeps the coefficient’s original initialized value and all other positional degrees of freedom remain trainable.
change: Store the sixth profile’s final mean-free basis coefficient as a checkpointed nontrainable buffer while retaining its remaining coefficients as learned parameters.
mechanism: Initialization-preserving frozen positional contrast
evidence_used: Replacing the entire sixth learned positional profile with one cosine amplitude nearly passed at 98.89%; freezing just one of that profile’s coefficients is substantially less restrictive and preserves the verified model’s exact initial positional embedding.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5315, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate three at zero will reduce the model from 1,524 to 1,523 parameters while retaining at least 99% accuracy, because its query and value effects can be absorbed by trainable QKV biases while its key effect is softmax-null.
change: Retain only the first three learned `ln1` bias coordinates, reconstruct coordinates three through seven as zeros, and use the renamed five-pruned LayerNorm.
mechanism: Fifth attention-input LayerNorm bias quotient
evidence_used: The verified 1,524-parameter design already fixes four `ln1` bias coordinates and reaches 99.98% accuracy; removing one additional zero-initialized bias is a smaller extension of that successful quotient than the failed positional-profile and LayerNorm-scale restrictions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate two at zero will reduce the model from 1,523 to 1,522 parameters while retaining at least 99% accuracy, because its query and value effects can be absorbed by trainable QKV biases and its position-common key effect is softmax-null.
change: Retain only the first two learned `ln1` bias coordinates, reconstruct coordinates two through seven as zeros, and rename the specialized LayerNorm accordingly.
mechanism: Sixth attention-input LayerNorm bias quotient
evidence_used: Fixing `ln1` bias coordinate three at zero preserved 99.91% accuracy at 1,523 parameters; this applies the same single-coordinate, zero-initialized quotient while avoiding the optimization disruption seen from larger simultaneous reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1522, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` bias coordinate one at zero will reduce the model from 1,522 to 1,521 parameters while retaining at least 99% accuracy, because its query and value effects can be absorbed by trainable QKV biases and its position-common key effect is softmax-null.
change: Retain only the first learned `ln1` bias coordinate, reconstruct coordinates one through seven as zeros, and rename the specialized LayerNorm accordingly.
mechanism: Seventh attention-input LayerNorm bias quotient
evidence_used: Successively fixing `ln1` bias coordinates three and two preserved 99.91% and 99.96% accuracy; this applies the same one-coordinate quotient while minimizing optimization disruption.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7253000000000001, "parameters": 1521, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final attention value-bias coordinate at zero will reduce the model from 1,522 to 1,521 parameters while retaining at least 99% accuracy, because softmax-normalized attention makes value bias position-independent and its projected contribution can be represented by the trainable attention projection bias.
change: Retain all query-bias coordinates and the first seven value-bias coordinates, reconstructing the final value-bias coordinate as zero during the forward pass.
mechanism: Single-coordinate value-bias/projection-bias quotient
evidence_used: Removing the next `ln1` bias coordinate collapsed accuracy to 72.53%, motivating a different one-parameter reduction. Unlike that optimization-sensitive LayerNorm restriction, this targets a direct redundancy between a value-bias coordinate and the existing learned projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1521, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the penultimate attention value-bias coordinate at zero will reduce the model from 1,521 to 1,520 parameters while retaining at least 99% accuracy, because its position-independent projected contribution can be represented by the trainable attention projection bias.
change: Retain all query-bias coordinates and the first six value-bias coordinates, reconstructing the final two value-bias coordinates as zeros during the forward pass.
mechanism: Second value-bias/projection-bias quotient
evidence_used: Fixing the final value-bias coordinate preserved 100% accuracy at 1,521 parameters; applying the same direct redundancy to one adjacent coordinate is the smallest informative extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1520, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the third-from-last attention value-bias coordinate at zero will reduce the model from 1,520 to 1,519 parameters while retaining at least 99% accuracy, because its position-independent projected contribution can be represented by the trainable attention projection bias.
change: Retain all query-bias coordinates and the first five value-bias coordinates, reconstructing the final three value-bias coordinates as zeros during the forward pass.
mechanism: Third value-bias/projection-bias quotient
evidence_used: Fixing the final two value-bias coordinates successively preserved 100% and 99.83% accuracy; this applies the same direct redundancy to one adjacent coordinate as the smallest informative extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9405, "parameters": 1519, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query projection input coefficient at zero will reduce the model from 1,520 to 1,519 parameters while retaining at least 99% accuracy, because causal attention receives mean-zero normalized activations and the removed coefficient can be absorbed into the other coefficients of that query row.
change: Replace the QKV projection with a specialized linear layer that omits the final input coefficient of its first query row, reconstructs it as zero, and canonicalizes initialization by subtracting the omitted coefficient from the retained row entries.
mechanism: LayerNorm-null query-weight quotient
evidence_used: The verified 1,520-parameter design already uses twelve analogous LayerNorm-input weight quotients in the MLP while reaching 99.83%; this tests that successful mechanism in one attention query row instead of extending the value-bias reduction whose third coordinate fell to 94.05%.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Constraining the token-content table to a learned six-dimensional subspace will reduce the verified model from 1,520 to 1,448 parameters while retaining at least 99% accuracy, because the shared projection can learn which single token-feature direction to discard while preserving all six learned positional profiles.
change: Factor the mean-free token-content embedding into per-token six-dimensional codes and a learned shared 6-by-7 projection, use the reconstructed embeddings for both input and tied output logits, and exempt both factors from weight decay.
mechanism: Adaptive low-rank tied token representation
evidence_used: Replacing positional lookup channels reduced accuracy to 98.89% or 74.41%, showing that positional flexibility is load-bearing; this instead preserves the verified attention, MLP, and positional computation while testing the unchallenged assumption that every token requires seven independent mean-free content coordinates.
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
