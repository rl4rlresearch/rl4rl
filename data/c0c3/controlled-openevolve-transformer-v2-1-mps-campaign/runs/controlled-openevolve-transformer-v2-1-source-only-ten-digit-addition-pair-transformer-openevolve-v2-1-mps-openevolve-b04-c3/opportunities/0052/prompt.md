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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}
prior_hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1360, "training_steps": 44999}
prior_hypothesis: Retaining the current learned positional table while constraining the tied token embedding to a six-dimensional learned subspace will achieve at least 99% accuracy after 45,000 updates with 1,360 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 1396, "training_steps": 74999}
prior_hypothesis: Sharing one query-bias coordinate across the two attention heads will retain at least 99% accuracy after 75,000 steps while reducing the qualified sinusoidal design from 1,397 to 1,396 parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing learned additive position vectors with parameter-free rotary Q/K phases, while retaining the qualified one-coordinate query sharing, will achieve at least 99% accuracy after 45,000 steps with 1,395 learned parameters.
change: Replace the assumption that positions require residual-stream embeddings with relative rotary attention; repurpose the former value-bias parameters as useful key biases, eliminate value bias, share one query coordinate, and use positive-step endpoint validation.
mechanism: Rotary relative-position attention with affine Q/K routing
evidence_used: Fixed sinusoidal positions reached 99.69%, proving a learned position table is unnecessary; one-coordinate query sharing reached 99.97%; and the current 45,000-step schedule completed at 99.92%. Rotary phases provide a different, direct positional-routing mechanism without the sinusoidal model’s learned gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing one hidden-unit bias will reduce the qualified model to 1,395 parameters, while 45,000 steps at 5e-3—approximately preserving the qualified 75,000-step schedule’s cumulative learning rate—will achieve at least 99% accuracy and finish verification.
change: Tie the first two MLP hidden biases, compress training to 45,000 higher-learning-rate steps, and validate only at the final positive step.
mechanism: MLP hidden-bias sharing with time-compressed optimization
evidence_used: The current 1,396-parameter model achieved 99.97%, but 60,000–75,000-step 1,395-parameter trials repeatedly timed out and completed 45,000-step trials at the original learning rate scored 0%; this motivates testing a less attention-sensitive one-parameter constraint with a time-compressed optimizer schedule.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0333, "parameters": 1395, "training_steps": 44999}

RECENT RESULT
hypothesis: Sharing two corresponding query-bias coordinates will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates, while batch size 128 and exact first-operand-prefiltered rejection will finish within the verification limit.
change: Encode eight query-bias coordinates with six learned scalars, eliminate batch-wide Python holdout checks, reduce the batch size to 128, and validate only at the final positive step.
mechanism: Two-coordinate cross-head query-bias sharing with lightweight vectorized sampling
evidence_used: One-coordinate sharing achieved 99.97% with 1,396 parameters; two-coordinate trials produced no negative accuracy evidence at 60,000–75,000 steps but timed out even at batch size 256, whereas shortening to 45,000 steps failed. This motivates preserving 75,000 updates while halving the lightest attempted batch and removing its per-example sampler loop.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing one query-bias coordinate in each head will preserve the qualified sinusoidal model’s expressivity and achieve at least 99% accuracy with 1,395 parameters after 75,000 updates; compilation and allocation-light holdout filtering will allow that schedule to finish.
change: Replace learned position embeddings with learned-scale sinusoidal positions, remove value bias, represent each four-dimensional query bias with three learned coordinates, compile the training forward pass, prefilter holdout collisions by first operand, and validate only at the final positive step.
mechanism: Per-head query-bias rotational gauge fixing with compiled training
evidence_used: The one-coordinate-sharing sinusoidal design achieved 99.97% with 1,396 parameters, while prior 1,395-parameter 45,000-step runs were undertrained and 60,000–75,000-step runs timed out. Per-head zero-coordinate fixing uses the independent Q/K rotational gauge without coupling the heads, while compilation targets the remaining training-loop runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The 1,395-parameter sinusoidal model will reach at least 99% accuracy in 45,000 steps when trained at the qualified 3e-3 peak rate for 40,000 steps and annealed over the final 5,000 steps.
change: Use parameter-free RMS normalization, learned-scale sinusoidal positions, query-only attention bias with two coordinates shared across heads, a 40,000-step learning-rate plateau, and final-step validation.
mechanism: Two-coordinate query-bias sharing with late cosine annealing
evidence_used: One-coordinate sharing achieved 99.97% at 75,000 steps, while the two-coordinate 45,000-step cosine run scored 0% and longer runs timed out. The prior 5e-3 compression also failed, motivating more cumulative learning at the already-qualified 3e-3 peak without raising it.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.11599999999999999, "parameters": 1395, "training_steps": 44999}

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
