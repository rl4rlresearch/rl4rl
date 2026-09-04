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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1520, "training_steps": 4999}
prior_hypothesis: A 1,520-parameter model will retain at least 99% accuracy because the verified 1,521-parameter design achieved 99.88%, while a sixth positional row has the same exact pre-LayerNorm scalar-shift symmetry as the five successfully removed rows.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987, "parameters": 1523, "training_steps": 4999}
prior_hypothesis: A 1,523-parameter model will retain at least 99% accuracy because the verified 1,524-parameter design achieved 99.92%, while a third positional row has the same exact pre-LayerNorm scalar-shift symmetry as the two already removed.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9964, "parameters": 1385, "training_steps": 4999}
prior_hypothesis: A 1,385-parameter model will retain at least 99% accuracy because the verified 1,386-parameter relative-lag design achieved 99.50%, while removing the common output component of a seventh `fc2` column is the same exact pre-LayerNorm scalar-shift symmetry already used successfully on six columns.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.995, "parameters": 1386, "training_steps": 4999}
prior_hypothesis: A 1,386-parameter model will retain at least 99% accuracy because the verified 1,388-parameter relative-lag design reached 99.22%, while removing one additive constant from each head’s lag logits is an exact softmax symmetry with ambient AdamW dynamics preserved.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,525-parameter model will retain at least 99% accuracy because the qualified 1,526-parameter design reached 99.61%, while a second ln1 scale has the same exact q/k/v column-factorization symmetry and preserves full ambient AdamW dynamics.
change: Absorb all eight ln2 scales into fc1 and the final two ln1 scales into the corresponding q, k, and v columns, retaining virtual scale/weight gradients, clipping, moments, decay, and updates.
mechanism: Second ln1 scale absorption into query, key, and value projection columns
evidence_used: The 1,526-parameter design achieved 99.61% after absorbing all ln2 scales and one ln1 scale; extending that proven ln1 factorization by one coordinate is the smallest controlled reduction below the best qualified parameter count.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,524-parameter model will retain at least 99% accuracy because the verified 1,525-parameter design achieved 99.93%, while a third ln1 scale has the same exact q/k/v column-factorization symmetry as the two already absorbed.
change: Absorb all eight ln2 scales into fc1 and the final three ln1 scales into the corresponding query, key, and value columns, preserving ambient AdamW moments, clipping, decay, and updates.
mechanism: Third ln1 scale absorption into attention input columns
evidence_used: The 1,525-parameter two-ln1-scale design reached 99.93%; extending that successful factorization by one coordinate is the smallest controlled reduction below the best qualified count.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9884999999999999, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second positional-row scalar will produce a 1,524-parameter model with at least 99% accuracy because this exact pre-LayerNorm shift symmetry already succeeds for the first positional row.
change: Gauge-fix the second positional embedding row and include its omitted ambient coordinate in AdamW moments, gradient clipping, decay, and updates.
mechanism: Second independent positional scalar-shift gauge
evidence_used: The current 1,525-parameter design reached 99.93% with one positional scalar gauge, while the third ln1 absorption fell to 98.85%; this tests a distinct exact symmetry with a one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1524, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,523-parameter model will retain at least 99% accuracy because the verified 1,524-parameter design achieved 99.92%, while a third positional row has the same exact pre-LayerNorm scalar-shift symmetry as the two already removed.
change: Reproduce the qualified two-scale ln1 absorption and two positional gauges, then gauge-fix a third positional row while preserving all omitted coordinates in initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Third positional-row scalar-shift gauge with ambient AdamW
evidence_used: Reference Design 3 reached 99.92% with two absorbed ln1 scales and two positional scalar gauges; extending the successful positional symmetry is the smallest controlled reduction after a third ln1 absorption fell to 98.85%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1523, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,522-parameter model will retain at least 99% accuracy because the verified 1,523-parameter design achieved 99.87%, while a fourth positional row has the same exact pre-LayerNorm scalar-shift symmetry as the three already removed.
change: Reproduce the qualified two-scale `ln1` absorption, complete `ln2` absorption, and three positional gauges, then gauge-fix a fourth positional row while preserving omitted coordinates through initialization, clipping, AdamW moments, decay, and updates.
mechanism: Fourth positional-row scalar-shift gauge with qualified full LayerNorm scale absorption
evidence_used: The 1,523-parameter three-position-gauge design achieved 99.87%; extending this repeatedly successful symmetry is more strongly supported than a third `ln1` absorption, which fell to 98.85%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1522, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,521-parameter model will retain at least 99% accuracy because the verified 1,522-parameter design achieved 99.96%, while a fifth positional row has the same exact pre-LayerNorm scalar-shift symmetry as the four successfully removed rows.
change: Reproduce the qualified two-scale ln1 absorption, complete ln2 absorption, and four positional gauges, then gauge-fix a fifth positional row while preserving every omitted coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Fifth positional-row scalar-shift gauge with complete LayerNorm scale absorption
evidence_used: The 1,522-parameter four-position-gauge design reached 99.96%; extending this repeatedly successful exact symmetry is better supported than a third ln1 absorption, which reached only 98.85%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1521, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,520-parameter model will retain at least 99% accuracy because the verified 1,521-parameter design achieved 99.88%, while a sixth positional row has the same exact pre-LayerNorm scalar-shift symmetry as the five successfully removed rows.
change: Gauge-fix the sixth positional embedding row and preserve its omitted ambient coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Sixth positional-row scalar-shift gauge with ambient AdamW
evidence_used: The current design removed five positional-row scalars and reached 99.88% at 1,521 parameters; all four preceding one-row extensions also exceeded 99%, making the same controlled one-parameter reduction the strongest supported next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1520, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,519-parameter model will retain at least 99% accuracy because the verified 1,520-parameter six-gauge design achieved 99.96%, and a seventh positional row has the same exact pre-LayerNorm scalar-shift symmetry.
change: Extend the current four positional gauges through the verified fifth and sixth gauges, then gauge-fix the seventh row while preserving every omitted coordinate during initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Seventh positional-row scalar-shift gauge with ambient AdamW
evidence_used: Reference Design 2 achieved 99.96% accuracy with six positional-row gauges at 1,520 parameters; every prior one-row positional extension remained above 99%, making the seventh identical one-parameter reduction the strongest supported next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1519, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,518-parameter model will retain at least 99% accuracy because the verified seven-gauge design achieved 99.98%, while the eighth positional row has the same exact pre-LayerNorm scalar-shift symmetry.
change: Reproduce the qualified seven positional gauges, then gauge-fix the eighth row while preserving every omitted coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.
mechanism: Eighth positional-row scalar-shift gauge with ambient AdamW
evidence_used: Reference Design 1 achieved 99.98% accuracy with seven positional-row gauges at 1,519 parameters; every tested positional-gauge extension has remained above 99%, making the identical eighth-row reduction the strongest supported next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9809, "parameters": 1518, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing full-width absolute position embeddings with per-head learned relative-lag logits will retain at least 99% accuracy while removing `6 * max_seq_len - 2` parameters, because addition dependencies repeat at relative offsets and content-dependent query/key attention remains intact.
change: Remove absolute positions from the residual stream and let each causal attention head learn its own lag preference directly; simplify gauge optimization accordingly. This challenges the prior assumption that every position needs a full learned vector.
mechanism: Learned relative-lag attention routing
evidence_used: Absolute-position compression stalled at 97.06% for a rank-4 subspace and 94.19% for fixed Chebyshev features, whereas the shared-key/value attention design reached 99.98%. This motivates changing positional computation itself instead of further compressing the same absolute-table representation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9922, "parameters": 1388, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,386-parameter model will retain at least 99% accuracy because the verified 1,388-parameter relative-lag design reached 99.22%, while removing one additive constant from each head’s lag logits is an exact softmax symmetry with ambient AdamW dynamics preserved.
change: Store each head’s learned relative-lag vector with its final coordinate fixed to zero, reconstruct the full vectors during attention, and optimize both omitted coordinates through the existing gauge-aware clipping and AdamW path.
mechanism: Per-head relative-logit softmax gauge fixing
evidence_used: The current learned relative-lag design achieved 99.22% at 1,388 parameters, and prior positional/output gauge reductions repeatedly preserved accuracy when omitted coordinates retained full ambient optimizer dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.995, "parameters": 1386, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,385-parameter model will retain at least 99% accuracy because the verified 1,386-parameter relative-lag design achieved 99.50%, while removing the common output component of a seventh `fc2` column is the same exact pre-LayerNorm scalar-shift symmetry already used successfully on six columns.
change: Replace absolute positional embeddings with gauge-fixed learned per-head relative-lag logits, then extend the terminal projection’s ambient-AdamW output gauge from six to seven columns.
mechanism: Relative-lag attention with a seventh residual-output gauge
evidence_used: The gauge-fixed relative-lag design reached 99.50% with 1,386 parameters, substantially outperforming other positional compression approaches; its terminal projection already removes six equivalent column-wise output shifts, motivating one controlled additional reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1385, "training_steps": 4999}



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
