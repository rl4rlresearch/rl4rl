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
verified_results: {"accuracy": 0.9989, "parameters": 1580, "training_steps": 4999}
prior_hypothesis: Constraining the third `fc2` weight column to zero mean while reproducing the verified 1,581-parameter model’s initialization and RNG consumption will achieve at least 99% accuracy with 1,580 parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a second `ln1` bias coordinate to zero will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because query shifts can be absorbed by the learned query bias, key shifts cancel in attention softmax, and value shifts can be absorbed by the attention projection bias up to its irrelevant common mode.
change: Store six rather than seven learned `AnchoredLayerNorm` bias coordinates and reconstruct its final two coordinates as zeros.
mechanism: Incremental pre-attention LayerNorm bias absorption
evidence_used: Removing every `ln2` bias coordinate retained 99.96% accuracy, and the current 1,582-parameter design already fixes one `ln1` bias coordinate while achieving 99.91%; extending this affine redundancy by one coordinate is a controlled alternative to the positional, `fc2`, and attention-weight gauges whose next adjacent constraints failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln1` bias coordinate to zero will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because query shifts remain absorbable by query bias, key shifts cancel in attention softmax, and value shifts remain absorbable by the attention projection bias up to its irrelevant common mode.
change: Store five rather than six learned `AnchoredLayerNorm` bias coordinates and reconstruct its final three coordinates as zeros.
mechanism: Incremental pre-attention LayerNorm bias absorption
evidence_used: The immediately preceding identical extension from one to two fixed `ln1` bias coordinates achieved 99.92% accuracy at 1,581 parameters, making a third coordinate the closest controlled reduction after adjacent positional, `fc2`, and attention-weight constraints failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5136999999999999, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring a second coordinate of the reference token embedding will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because translating that coordinate for every token can be absorbed by the positional embeddings, while its output-logit effect is a class-independent softmax shift.
change: Store two fewer token-embedding scalars and reconstruct the final two coordinates as zeros.
mechanism: Joint token–position translation gauge
evidence_used: The current design achieves 99.92% accuracy with one reference-token coordinate already anchored; this tests an independent exact token–position gauge after the adjacent third `ln1` bias and ninth positional-row constraints failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9604, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final input weight of one query projection row will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because LayerNorm inputs lie on an affine hyperplane and the learned query bias can absorb the eliminated constant term.
change: Store seven weights for the first query row, reconstruct its final weight as zero, and initialize the compact row to preserve the function of a full projection on initially zero-mean LayerNorm outputs.
mechanism: Normalized-input query-weight gauge fixing
evidence_used: The analogous `NormalizedInputLinear` constraint removed the final input weight from all 12 `fc1` rows while achieving 99.93% accuracy at 1,585 parameters; applying it incrementally to one query row tests an independent redundancy after adjacent embedding and `ln1`-bias reductions failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5063, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the third `fc2` weight column to zero mean while reproducing the verified 1,581-parameter model’s initialization and RNG consumption will achieve at least 99% accuracy with 1,580 parameters.
change: Extend `OutputAnchoredLinear` to three zero-mean columns and initialize it by projecting the third column from the exact compact random draw used by the current two-column design.
mechanism: RNG-aligned terminal-MLP common-mode gauge fixing
evidence_used: Two constrained `fc2` columns achieved 99.97%, whereas the direct third-column extension fell to 73.46%; preserving the successful design’s initial function up to the exact LayerNorm-invisible common mode isolates the gauge reduction from compact-layout and downstream-RNG changes.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fourth `fc2` weight column to zero mean while preserving the verified 1,580-parameter model’s initial function and RNG consumption will achieve at least 99% accuracy with 1,579 parameters.
change: Extend `OutputAnchoredLinear` from three to four zero-mean columns and initialize both added compact columns by projecting the same 94-scalar baseline draw used by the current successful design.
mechanism: RNG-aligned fourth terminal-MLP common-mode gauge fixing
evidence_used: The RNG-aligned third-column constraint achieved 99.89% at 1,580 parameters after the direct third-column extension achieved only 73.46%, showing that preserving initialization and downstream RNG state is critical for this exact downstream LayerNorm gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7426999999999999, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring a second reference-token embedding coordinate while preserving every retained initialization value and the verified 1,580-parameter design’s downstream RNG state will achieve at least 99% accuracy with 1,579 parameters.
change: Store two fewer token-embedding scalars, reconstruct both as zeros, and initialize from a baseline draw matching the current one-anchor model before discarding the newly anchored scalar.
mechanism: RNG-aligned second reference-token embedding anchor
evidence_used: The direct second token-coordinate anchor reached 96.04%, while RNG-aligning the previously failed third `fc2` constraint improved accuracy from 73.46% to 99.89%; this tests whether initialization and downstream-RNG disruption caused the near-threshold token-anchor failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7134, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fifth attention projection weight column to zero mean while preserving the verified 1,580-parameter model’s initial function and downstream RNG state will retain at least 99% accuracy with 1,579 parameters.
change: Extend `AttentionGaugeLinear` from four to five zero-mean columns and initialize it by projecting the fifth column from the exact 60-scalar compact draw used by the current four-column design.
mechanism: RNG-aligned fifth attention-output common-mode gauge fixing
evidence_used: A direct fifth attention-column constraint previously failed, while RNG-aligning the failed third `fc2` constraint raised accuracy from 73.46% to 99.89%; this tests the same initialization-preserving remedy on an independent exact downstream-LayerNorm gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2661, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fifth `ln2` scale at one will reduce the model to 1,579 parameters while retaining at least 99% accuracy, because the following learned `fc1` weights can absorb that coordinate’s scale and the initialization remains functionally and RNG-identical.
change: Store three rather than four learned `BiasAnchoredLayerNorm` scales and reconstruct its final five scales as ones.
mechanism: Pre-MLP LayerNorm scale absorption
evidence_used: The verified 1,580-parameter design achieves 99.89% accuracy with four `ln2` scales already fixed at one, while the analogous normalized-input gauge removed one `fc1` weight per row and previously retained 99.93%; this makes one more pre-MLP scale constraint an independent, initialization-preserving reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.927, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Exactly compensating a second reference-token coordinate anchor in the positional embeddings, while preserving the verified 1,580-parameter initialization and RNG consumption, will retain at least 99% accuracy with 1,579 parameters.
change: Remove the second-last reference-token scalar, initialize from the current 111-scalar token draw, translate every token in that coordinate, and absorb the translation into every positional row modulo its LayerNorm-invisible common mode.
mechanism: Function-preserving joint token–position translation gauge
evidence_used: A direct second token-coordinate anchor reached 96.04%, but RNG alignment alone reached only 71.34%; unlike both attempts, this patch applies the full joint token–position gauge transformation and therefore preserves the successful model’s initial computation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7376999999999999, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the final `fc2` weight column to zero mean while preserving the verified 1,580-parameter model’s initial function and RNG consumption will achieve at least 99% accuracy with 1,579 parameters.
change: Keep the first three constrained `fc2` columns, constrain the nonadjacent final column, and initialize all retained coordinates from the same 94-scalar baseline draw used by the successful design.
mechanism: Nonadjacent terminal-MLP common-mode gauge fixing
evidence_used: Three constrained `fc2` columns achieved 99.89%, while constraining the adjacent fourth column fell to 74.27%; testing the final column distinguishes a hidden-unit-specific optimization failure from a general limit on this exact downstream-LayerNorm gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8787999999999999, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fourth `fc2` column while updating it through full-coordinate AdamW moments will produce a 1,579-parameter model with at least 99% accuracy by preserving the verified 1,580-parameter model’s functional optimization trajectory modulo the LayerNorm-invisible common mode.
change: Store the fourth `fc2` column in seven zero-mean coordinates and train those coordinates with a projected AdamW optimizer that maintains the eight coordinatewise moments used by the successful baseline.
mechanism: Optimizer-faithful projected fourth-column gauge
evidence_used: The RNG-aligned fourth-column constraint fell to 74.27% even though the same exact gauge succeeded for three columns at 99.89%, indicating that AdamW’s coordinate-dependent optimization geometry—not representational capacity or initialization—is the most relevant remaining difference.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3857, "parameters": 1579, "training_steps": 4999}



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
