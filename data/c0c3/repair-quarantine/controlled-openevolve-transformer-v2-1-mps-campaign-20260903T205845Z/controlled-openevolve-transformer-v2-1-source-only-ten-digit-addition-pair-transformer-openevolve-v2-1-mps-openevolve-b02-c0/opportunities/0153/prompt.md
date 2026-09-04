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
verified_results: {"accuracy": 0.9992, "parameters": 1576, "training_steps": 4999}
prior_hypothesis: Fixing `ln1` bias coordinate 5 at zero will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because the following QKV and attention-output affine transformations can absorb this bias and initialization remains unchanged.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Anchoring the next `ln1` bias coordinate at zero will reduce the model from 1,576 to 1,575 parameters while preserving at least 99% accuracy, because its downstream affine effect can be absorbed by the learned attention projections.
change: Reduce `AnchoredLayerNorm` from five to four learned bias coordinates and reconstruct the remaining four as zeros.
mechanism: Incremental LayerNorm-to-attention affine gauge elimination
evidence_used: The current 1,576-parameter design achieved 99.92% accuracy after anchoring one `ln1` bias coordinate, supporting another conservative one-coordinate reduction using the same redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7309, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining a fifth attention projection column to have zero-mean outputs will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because this removes an exact residual-stream common mode rather than another LayerNorm coordinate.
change: Represent the first five, instead of four, attention-output weight columns in the existing zero-mean basis.
mechanism: Extend the attention-output residual common-mode gauge
evidence_used: The 1,576-parameter model reached 99.92% with four attention columns already using this gauge, whereas removing another `ln1` bias coordinate reduced accuracy to 73.09%; this motivates extending the successful exact gauge instead of further restricting LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.79, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining positional row `max_seq_len - 12` to zero mean will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because position-specific all-feature shifts are removed by every pre-LayerNorm and the final LayerNorm.
change: Extend the positional embedding’s orthonormal zero-mean representation from eight consecutive rows to nine, preserving equivalent initialization in the reduced coordinates.
mechanism: Per-position residual common-mode gauge elimination
evidence_used: The 1,576-parameter model achieved 99.92% accuracy with eight positional rows already using this gauge; the two failed 1,575-parameter trials instead constrained LayerNorm or attention coordinates, motivating extension of the successful positional invariance.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.38170000000000004, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fourth MLP output-weight column to zero mean will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because the removed component only adds a per-token scalar erased by downstream LayerNorm, and the initializer preserves the successful model’s random draw sequence and identifiable initial weights.
change: Extend `OutputAnchoredLinear` from three to four zero-mean columns and project the fourth column during initialization while consuming the same 94 baseline random draws.
mechanism: RNG-preserving MLP residual common-mode gauge
evidence_used: The 1,576-parameter design achieved 99.92% accuracy, while three upstream 1,575-parameter changes failed; this targets a distinct exact residual-stream gauge in the final randomized projection and avoids shifting initialization of any downstream learned module.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7251000000000001, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one query-bias coordinate with the retained fifth `ln1` bias will reduce the model to 1,575 parameters while preserving at least 99% accuracy, because `ln1` bias already supplies a learned query-offset path and both scalars initialize identically at zero.
change: Remove the final independent QKV bias parameter, reconstruct that query-bias coordinate from `ln1.bias[-1]`, and pass the shared scalar through attention.
mechanism: LayerNorm–query-bias parameter sharing
evidence_used: The 1,576-parameter model achieved 99.92% accuracy, while fixing the fifth `ln1` bias at zero fell to 73.09%; sharing a downstream redundant query bias retains that empirically important LayerNorm coordinate instead of anchoring it.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5399, "parameters": 1575, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final feature of positional row `max_seq_len - 12` at zero will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because subtracting that coordinate from the entire row changes only a per-token scalar erased by LayerNorm, while the axis-aligned parameterization avoids the failed ninth orthonormal gauge’s optimizer geometry.
change: Store positional row `max_seq_len - 12` using seven coordinate differences and reconstruct its eighth coordinate as zero, while preserving the baseline random draws and initial learned function.
mechanism: Axis-anchored positional common-mode gauge
evidence_used: The 1,576-parameter model reached 99.92% with an existing axis-anchored positional row, whereas extending the orthonormal positional gauge to this row achieved only 38.17%; this tests the same exact invariance using the successful anchor-style coordinate system.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2609, "parameters": 1575, "training_steps": 4999}



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
