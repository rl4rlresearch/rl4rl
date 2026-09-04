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
verified_results: {"accuracy": 1.0, "parameters": 1350, "training_steps": 44999}
prior_hypothesis: Sharing a second corresponding query-bias coordinate across the two attention heads will preserve at least 99% accuracy after 45,000 steps while reducing the qualified learned-position model from 1,351 to 1,350 parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}
prior_hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1352, "training_steps": 44999}
prior_hypothesis: Removing the redundant eight-dimensional value bias will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,360 to 1,352 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1351, "training_steps": 44999}
prior_hypothesis: Combining the qualified rank-six learned-position model with the independently qualified one-coordinate query-bias sharing will achieve at least 99% accuracy after 45,000 updates with 1,351 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: A better-conditioned dyadic Fourier basis will let the two-coordinate-shared model reach at least 99% accuracy in 45,000 updates with 1,395 parameters.
change: Share two corresponding query-bias coordinates, replace poorly distributed sinusoidal frequencies with variance-matched dyadic Fourier features, and perform validation only at the final positive step.
mechanism: Variance-matched multiscale Fourier positions with two-coordinate query-bias sharing
evidence_used: One-coordinate sharing reached 99.97% with 1,396 parameters, while Reference Design 3 reached 99.92% in 45,000 steps with well-conditioned learned positions; the old sinusoidal two-coordinate model’s 11.6% result motivates improving positional conditioning instead of further increasing its learning rate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing one query-bias axis independently in each head will yield a 1,395-parameter model with at least 99% accuracy after 45,000 updates, while exact first-operand prefiltering and final-only validation will finish within the verification limit.
change: Remove one learned query-bias coordinate per head without cross-head coupling, replace batch-wide Python holdout checks with an exact rare-candidate prefilter, and train for 45,000 steps with validation only at the positive final step.
mechanism: Per-head query-bias rotational gauge fixing with allocation-light holdout filtering
evidence_used: The one-coordinate-sharing design achieved 99.97% with 1,396 parameters and reportedly 99.92% at 45,000 steps. The previous independent per-head 1,395-parameter gauge trial timed out at 75,000 steps rather than failing accuracy, motivating the shorter qualified-duration schedule and sampler optimization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1395, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing one scalar in the qualified shared-query sinusoidal model’s tied token embedding will retain at least 99% accuracy in 45,000 steps with 1,395 learned parameters.
change: Adopt the qualified 1,396-parameter architecture, replace its tied embedding matrix with an equivalent matrix having one fixed zero coordinate, preserve initialization RNG consumption, and validate only at the final positive step.
mechanism: Single-coordinate token-embedding anchoring
evidence_used: Reference Design 3 achieved 99.97% with 1,396 parameters. Prior 1,395-parameter failures constrained attention or MLP behavior directly; anchoring one of 912 tied embedding coordinates leaves the qualified attention mechanism intact, while the current 45,000-step run demonstrates that duration finishes verification.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.038900000000000004, "parameters": 1395, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing only the attention value matrix’s globally redundant norm, while preserving its exact initialized direction and keeping an independent learned positional gain, will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates.
change: Adopt the qualified sinusoidal/RMSNorm/shared-query architecture, encode the value matrix with 63 projective coordinates plus a fixed initialization-derived norm, exempt those coordinates from coordinate-dependent weight decay, accelerate exact holdout filtering, and validate only at the final positive step.
mechanism: Max-pivot normalized value/output scale gauge
evidence_used: The 1,396-parameter shared-query design reached 99.97%. The prior single-weight value anchor scored 0% after 45,000 steps; this patch instead preserves the complete initialized value matrix exactly, fixes the smooth global V/O scale redundancy, leaves positional gain independent, and restores the qualified 75,000-step schedule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing each head’s query bias to a nonzero canonical direction and reallocating six of the seven removed parameters to per-channel positional gains will reach at least 99% accuracy with 1,395 parameters after 45,000 steps.
change: Remove learned query biases using the exact per-head Q/K basis-change symmetry, add seven learned sinusoidal channel gains with one fixed gain, and use the completed-duration 45,000-step schedule.
mechanism: Canonical fixed query gauge with channel-adaptive Fourier positions
evidence_used: The 1,396-parameter shared-query model achieved 99.97%, while prior 1,395-parameter constraints retained the poorly conditioned single positional gain and failed or timed out. Reference Design 3 reached 99.92% in 45,000 steps with more adaptable positional representations, motivating reallocating redundant query coordinates to positional conditioning.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.039, "parameters": 1395, "training_steps": 44999}

RECENT RESULT
hypothesis: Two learned query heads can independently address operand positions through one shared learned key/value space, achieving at least 99% accuracy after 75,000 updates with 1,332 parameters.
change: Replace dense per-head Q/K/V projections with two query heads sharing a four-dimensional key/value projection, retain the qualified one-coordinate query-bias sharing, accelerate exact holdout rejection, and validate only at the final positive step.
mechanism: Multi-query causal attention with shared key/value representation
evidence_used: The 1,396-parameter shared-query design achieved 99.97%, while repeated attempts to remove isolated parameters from embeddings, biases, or matrix gauges failed or timed out. This motivates challenging the load-bearing assumption that both attention heads require independent key and value representations while preserving their separate learned queries and the successful sinusoidal backbone.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining the current learned positional table while constraining the tied token embedding to a six-dimensional learned subspace will achieve at least 99% accuracy after 45,000 updates with 1,360 learned parameters.
change: Replace the 8-dimensional tied token matrix with six learned coordinates padded into the 8-dimensional residual stream, preserve exact input/output weight tying, and validate only at the final positive step.
mechanism: Rank-six tied vocabulary embedding with a parameter-free channel lift
evidence_used: The current learned-position model reached 99.92% in 45,000 steps, whereas multiple smaller sinusoidal-position models failed at 45,000 steps or timed out at longer schedules; this motivates preserving the proven positional and optimization backbone while removing 228 vocabulary-embedding parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1360, "training_steps": 44999}

RECENT RESULT
hypothesis: Reducing the qualified rank-six token code to five learned dimensions will retain at least 99% accuracy after 45,000 steps while lowering the model from 1,360 to 1,246 parameters.
change: Preserve the qualified learned-position, RMSNorm, attention, and optimization design, but encode each token with five learned coordinates padded into the eight-dimensional residual stream.
mechanism: Rank-five tied vocabulary embedding with a parameter-free residual-stream lift
evidence_used: Reference Design 2 achieved 100% accuracy with a six-dimensional tied token subspace at 45,000 steps; removing one token coordinate is the most direct test of whether that subspace still contains excess capacity without reintroducing the slower sinusoidal-position training regime.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5453, "parameters": 1246, "training_steps": 44999}

RECENT RESULT
hypothesis: Removing the redundant eight-dimensional value bias will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,360 to 1,352 learned parameters.
change: Keep only the learned query bias in QKV; set both key and value biases to zero because the output-projection bias can represent the value bias exactly.
mechanism: Attention value-bias absorption into the affine output projection
evidence_used: The current rank-six learned-position model achieved 100% accuracy, and Reference Design 2 achieved 99.69% with value bias removed; this targets a proven exact redundancy without constraining the successful token or positional representations.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1352, "training_steps": 44999}

RECENT RESULT
hypothesis: Combining the qualified rank-six learned-position model with the independently qualified one-coordinate query-bias sharing will achieve at least 99% accuracy after 45,000 updates with 1,351 learned parameters.
change: Replace full-width token embeddings and fixed sinusoidal positions with six-dimensional tied token codes and learned positions, retain value-bias removal and one-coordinate query-bias sharing, and validate only at the final step of a 45,000-step run.
mechanism: Rank-six tied token code with shared query-bias gauge
evidence_used: The rank-six learned-position model reached 100% accuracy at 1,360 parameters, value-bias removal retained 100% at 1,352, and one-coordinate query-bias sharing independently reached 99.97%; this combines those orthogonal successful reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1351, "training_steps": 44999}

RECENT RESULT
hypothesis: Preserving the qualified rank-six token code and shared query-bias gauge while reducing learned positions from eight to seven dimensions will retain at least 99% accuracy after 45,000 updates with 1,328 parameters.
change: Adopt the qualified 1,351-parameter learned-position architecture, lift seven-dimensional learned position codes into the eight-dimensional residual stream, and perform validation only at the final training step.
mechanism: Rank-seven learned positional subspace
evidence_used: Reference Design 3 achieved 100% accuracy with 1,351 parameters, while reducing token rank from six to five fell to 54.53% and sinusoidal positions repeatedly failed or timed out; this preserves both proven components and tests the unexamined redundancy of one positional coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3442, "parameters": 1328, "training_steps": 44999}

RECENT RESULT
hypothesis: Sharing a second corresponding query-bias coordinate across the two attention heads will preserve at least 99% accuracy after 45,000 steps while reducing the qualified learned-position model from 1,351 to 1,350 parameters.
change: Reproduce the qualified rank-six tied-token, learned-position, RMSNorm architecture; remove key/value biases; share two query-bias coordinates across heads; and validate only at the final step.
mechanism: Two-coordinate cross-head query-bias gauge sharing
evidence_used: Reference Design 3 reached 100% accuracy with 1,351 parameters and one-coordinate sharing. Earlier two-coordinate failures used the poorly conditioned sinusoidal backbone, while reducing learned positional capacity failed, motivating testing the additional gauge constraint without changing the qualified learned positions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1350, "training_steps": 44999}



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
