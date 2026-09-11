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
verified_results: {"accuracy": 0.9971, "parameters": 1380, "training_steps": 4999}
prior_hypothesis: A 1,380-parameter relative-lag model will retain at least 99% accuracy because the verified 1,381-parameter design achieved 99.75%, while gauge-fixing the twelfth and final `fc2` column applies the same exact pre-final-LayerNorm symmetry already verified for eleven columns.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9961, "parameters": 1376, "training_steps": 4999}
prior_hypothesis: A 1,376-parameter transformer will retain at least 99% accuracy because the current 1,377-parameter design achieved 100%, while fixing the corresponding initially zero coordinate in the other attention head leaves three learned query-bias coordinates per head and preserves all query-key projections and relative-lag routing.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9975, "parameters": 1381, "training_steps": 4999}
prior_hypothesis: A 1,381-parameter relative-lag model will retain at least 99% accuracy because the verified 1,382-parameter ten-column design achieved 99.45%, while gauge-fixing an eleventh `fc2` column applies the same exact pre-final-LayerNorm output-shift symmetry.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 1375, "training_steps": 4999}
prior_hypothesis: A 1,375-parameter transformer will retain at least 99% accuracy because the verified 1,376-parameter design achieved 99.61%, while fixing one additional initially zero query-bias coordinate preserves all projection weights, value paths, and relative-lag routing.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while gauge-fixing the untested attention column five with a zero-mean representative minimizes the numerically irrelevant common output shift that may have destabilized prior fifth-gauge trials.
change: Reproduce the qualified complete terminal gauge, four-column attention gauge, and three-scale `ln1` absorption, then remove one additional parameter by reconstructing attention column five in a centered scalar-shift gauge while retaining full ambient-gradient clipping and AdamW updates.
mechanism: Balanced fifth attention-output scalar-shift gauge
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters. Prior fifth-gauge attempts using last-coordinate-zero representatives collapsed on columns four and seven, motivating a centered representative on the untested interior column five to distinguish numerical gauge conditioning from a true capacity limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9329999999999999, "parameters": 1377, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while centering the first `fc1` weight row removes its exact LayerNorm-null component without the large invisible offset used by the failed last-coordinate-zero row gauge.
change: Reproduce the qualified complete terminal gauge, four-column attention gauge, and three-scale `ln1` absorption, then replace `fc1` with a one-row centered gauge and preserve full ambient weight/scale AdamW and clipping dynamics.
mechanism: Centered pre-MLP zero-mean row gauge
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters. The previous last-coordinate-zero `fc1` row gauge collapsed to 10.66%, while centering an analogous attention gauge substantially improved its result to 93.3%, motivating a centered representative for this distinct exact symmetry.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while fixing one key-projection coordinate through a shared orthogonal query-key rotation preserves attention logits exactly and avoids the unstable scale and output-shift mechanisms tested previously.
change: Replace the shared key projection with a 31-parameter Givens-gauge projection that fixes one coordinate to zero, rotates both query heads consistently at initialization, includes the omitted ambient gradient in clipping, and preserves the existing absorbed-scale optimizer path.
mechanism: Orthogonal query-key head-space gauge
evidence_used: The current three-scale-absorption design achieved 99.39% at 1,378 parameters, whereas extending scale absorption or attention-output gauges failed; this tests a distinct exact symmetry of the learned dot-product attention while preserving its initialized function.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while removing one key coordinate through a shared query-key Givens rotation preserves attention logits and tests a distinct symmetry from the failed scale, output-projection, and MLP gauges.
change: Reproduce the qualified four-column attention gauge and three-scale ln1 absorption, then replace the shared key projection with a 31-parameter gauge-fixed projection, rotate both query heads consistently at initialization, and include the omitted key gradient in clipping.
mechanism: Orthogonal query-key head-space gauge
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters. The prior orthogonal-gauge attempt was never verified because its search text matched multiple locations, so this exact-once patch tests that still-unresolved mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.10869999999999999, "parameters": 1377, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while an orthonormal mean-zero representation removes one exact LayerNorm-null `fc1` component without the poor conditioning of the failed last-coordinate-zero gauge.
change: Reproduce the qualified complete terminal gauge, four-column attention gauge, and three-scale `ln1` absorption, then replace `fc1` with a centered first-row gauge and update that row through the existing full ambient scale-factorized AdamW path.
mechanism: Orthonormal centered pre-MLP row gauge
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters. The last-coordinate-zero `fc1` gauge collapsed to 10.66%, while centering an attention gauge improved its result to 93.3%; the centered `fc1` trial was never verified, so an orthonormal implementation is the most informative unresolved reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9662000000000001, "parameters": 1377, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,356-parameter transformer will retain at least 99% accuracy because three learned value coordinates per head can transport digit information while the full four-dimensional query-key paths and dense learned relative-lag routing preserve the content-addressing mechanism shown to be load-bearing.
change: Reproduce the qualified all-column terminal gauge, then challenge the assumption that attention values require the same width as query-key vectors by narrowing only the shared learned value stream from four to three dimensions and projecting the resulting six head outputs back into the eight-dimensional residual stream.
mechanism: Asymmetric query-key/value head widths
evidence_used: The 1,380-parameter all-terminal-gauge reference reached 99.71%, while the 1,270-parameter content-independent design collapsed after removing query-key addressing. This motivates retaining full query-key and relative-lag computation while testing a narrower value-transport mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: A 1,377-parameter transformer will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while absorbing coordinate zero instead of extending the failed contiguous suffix tests whether the fourth-scale failure was coordinate-specific.
change: Reproduce the qualified complete terminal gauge, four-column attention-output gauge, and three absorbed `ln1` scales, then absorb the additional `ln1` scale at coordinate zero with full ambient-gradient clipping and AdamW updates.
mechanism: Noncontiguous fourth pre-attention LayerNorm scale absorption
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters using three-scale absorption; the contiguous fourth-scale trial fell to 49.67%, motivating a noncontiguous fourth coordinate while retaining the previously successful mechanism.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 1,377-parameter transformer will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while fixing one initially zero query-bias coordinate leaves seven learned bias coordinates, full query-key projections, and learned relative-lag routing intact.
change: Reproduce the qualified complete terminal gauge, four-column attention-output gauge, and three-scale `ln1` absorption, then remove the final scalar of the learned query bias by reconstructing it as zero during the forward pass.
mechanism: Single-coordinate query-bias pruning atop the qualified 1,378-parameter gauge design
evidence_used: Reference Design 3 achieved 99.39% at 1,378 parameters. Prior 1,377-parameter failures modified optimization-sensitive scale, projection, or MLP gauges; this tests a simpler initialization-preserving redundancy in the eight-coordinate query bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1377, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,376-parameter transformer will retain at least 99% accuracy because the current 1,377-parameter design achieved 100%, while fixing the corresponding initially zero coordinate in the other attention head leaves three learned query-bias coordinates per head and preserves all query-key projections and relative-lag routing.
change: Replace the seven-coordinate query bias with six learned coordinates, reconstructing the final coordinate of each query head as zero during every forward pass.
mechanism: Balanced per-head query-bias pruning
evidence_used: The current design reached 100% accuracy after pruning one initially zero query-bias coordinate, substantially outperforming prior 1,377-parameter projection, scale, and MLP reductions; distributing the same pruning mechanism symmetrically across both heads is the closest informative extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9961, "parameters": 1376, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,375-parameter transformer will retain at least 99% accuracy because the verified 1,376-parameter design achieved 99.61%, while fixing one additional initially zero query-bias coordinate preserves all projection weights, value paths, and relative-lag routing.
change: Replace the eight-coordinate query bias with five learned coordinates, reconstructing the final coordinate of both heads and the penultimate coordinate of the first head as zero.
mechanism: Asymmetric third query-bias coordinate pruning
evidence_used: Balanced per-head query-bias pruning reached 99.61% at 1,376 parameters, whereas reductions involving projection, scale, or MLP gauges failed; extending the successful bias-pruning mechanism by one scalar is the closest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1375, "training_steps": 4999}



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
