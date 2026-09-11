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
verified_results: {"accuracy": 0.9992, "parameters": 1588, "training_steps": 44999}
prior_hypothesis: Removing all eight redundant key-projection bias coordinates and training for 45,000 steps will retain at least 99% accuracy with 1,588 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 1396, "training_steps": 74999}
prior_hypothesis: Sharing one query-bias coordinate across the two attention heads will retain at least 99% accuracy after 75,000 steps while reducing the qualified sinusoidal design from 1,397 to 1,396 parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing a second corresponding query-bias coordinate will retain at least 99% accuracy with 1,395 parameters; using batch size 384 will preserve 75,000 optimization steps while completing within the verification time limit.
change: Represent eight query-bias coordinates with six learned scalars, reduce the default batch size from 512 to 384, and validate only at the positive final step.
mechanism: Two-coordinate cross-head query-bias sharing with lighter batches
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters at 74,999 steps. Both second-coordinate trials timed out at batch size 512, including a 70,000-step run, so reducing per-step computation while retaining the qualified 75,000-update schedule is the closest unresolved test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing two corresponding query-bias coordinates will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates, while batch size 256 and positive-step endpoint validation will finish within the verification limit.
change: Represent eight query-bias coordinates with six learned scalars, halve the training batch size, and validate only at the final positive step by default.
mechanism: Two-coordinate cross-head query-bias sharing with lighter batches
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters. Two-coordinate attempts yielded no accuracy result because they timed out even at batch size 384, motivating the same incremental constraint with a lighter batch while preserving the 75,000-update schedule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing two corresponding query-bias coordinates will retain at least 99% accuracy with 1,395 parameters after 75,000 updates, while removing the per-example Python holdout-check loop will let the previously timing-out experiment finish.
change: Adopt learned-scale sinusoidal positions and query-only QKV bias, encode eight query-bias coordinates with six learned scalars, evaluate only at positive interval endpoints, and accelerate strict holdout rejection using bulk tuple-set disjointness.
mechanism: Two-coordinate cross-head query-bias sharing with allocation-light holdout filtering
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters; every two-coordinate trial timed out even after reducing batch size, motivating preservation of the full qualified batch/update schedule while targeting the sampler’s 38.4-million-iteration Python loop.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing a second corresponding query-bias coordinate will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates, while a vectorized first-operand holdout prefilter will eliminate the sampler’s per-example Python loop and allow training to finish.
change: Encode eight query-bias coordinates with six learned scalars, check only samples whose first operand occurs in the holdout via `searchsorted`, and validate only at positive interval endpoints.
mechanism: Two-coordinate cross-head query-bias sharing with sorted holdout prefilter
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters. All second-coordinate experiments timed out rather than producing negative accuracy evidence; their remaining sampler still performed batch-wide Python work, motivating an exact tensor prefilter that reduces Python hash checks to the extremely rare first-operand matches.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing two corresponding query-bias coordinates will achieve at least 99% accuracy with 1,395 parameters after 60,000 updates, which is short enough to finish verification.
change: Represent eight query-bias coordinates with six learned scalars, reduce training to 60,000 steps, and evaluate only at positive interval endpoints or the final step.
mechanism: Two-coordinate cross-head query-bias sharing with shortened endpoint-only training
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters at 74,999 steps, showing substantial accuracy margin. All two-coordinate trials timed out, including at 70,000 steps, so a clean 60,000-step trial directly tests the unresolved constraint while reducing runtime without adding sampler overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Anchoring one value-projection weight while retaining the qualified learned sinusoidal gain and shared query-bias coordinate will achieve at least 99% accuracy with 1,395 parameters after 45,000 steps.
change: Adopt the qualified 1,396-parameter architecture, remove one redundant value-projection scalar through its exact inverse scaling symmetry with the attention output projection, and use positive-step endpoint validation.
mechanism: Value/output channel-scale gauge fixing
evidence_used: The shared-query sinusoidal design achieved 99.97% accuracy with 1,396 parameters. Unlike the failed dynamic Q/K energy-ratio gauge, this removes a scale degree of freedom entirely within the linear value/output path; the completed 45,000-step current run motivates the shorter schedule after repeated longer runs timed out.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1395, "training_steps": 44999}

RECENT RESULT
hypothesis: Sharing a second corresponding query-bias coordinate will retain at least 99% accuracy with 1,395 parameters after 45,000 steps, a completed-duration schedule supported by the 99.92%-accurate 45,000-step reference.
change: Encode eight query-bias coordinates with six learned scalars, train for 45,000 steps, and validate only at the final positive step.
mechanism: Two-coordinate cross-head query-bias sharing with shortened endpoint-only training
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters, while all two-coordinate runs at 60,000 or more steps timed out rather than yielding negative accuracy evidence; a separate 45,000-step run completed, and Reference Design 3 reached 99.92% at that duration.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1395, "training_steps": 44999}

RECENT RESULT
hypothesis: Tying the learned sinusoidal gain to the query-bias coordinate already shared across heads will preserve adaptive positional scaling and achieve at least 99% accuracy after 75,000 steps with 1,395 parameters.
change: Share one query-bias coordinate across attention heads, reuse that scalar as an additive positional gain offset, remove the separate positional-scale parameter, and validate only at the final positive step.
mechanism: Positional-scale/query-bias gauge tying
evidence_used: Cross-head sharing achieved 99.97% accuracy with 1,396 parameters, while fixing the positional gain produced 0% accuracy; reusing the shared adaptive scalar targets 1,395 parameters without imposing the failed fixed-gain constraint.
result: training did not finish within the verification time limit

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
