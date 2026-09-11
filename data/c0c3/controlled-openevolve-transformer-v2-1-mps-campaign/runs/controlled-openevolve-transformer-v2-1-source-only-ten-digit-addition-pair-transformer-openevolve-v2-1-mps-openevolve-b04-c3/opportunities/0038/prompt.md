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
verified_results: {"accuracy": 0.9997, "parameters": 1396, "training_steps": 74999}
prior_hypothesis: Sharing one query-bias coordinate across the two attention heads will retain at least 99% accuracy after 75,000 steps while reducing the qualified sinusoidal design from 1,397 to 1,396 parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}
prior_hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1588, "training_steps": 44999}
prior_hypothesis: Removing all eight redundant key-projection bias coordinates and training for 45,000 steps will retain at least 99% accuracy with 1,588 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one query-bias coordinate while retaining the learned sinusoidal gain will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps, and endpoint-only validation will let training finish within the time limit.
change: Reconstruct one fixed-zero query-bias coordinate, retain all other learned parameters, and evaluate only at positive interval steps and the final step with a 75,000-step default interval.
mechanism: Single-coordinate query-bias gauge fixing with endpoint validation
evidence_used: The qualified 1,397-parameter design achieved 99.69% accuracy at 74,999 steps. Removing six query-bias coordinates collapsed accuracy, but the prior one-coordinate experiment produced no accuracy result because it timed out; testing the smallest gauge reduction with validation overhead reduced to one final evaluation is the closest unresolved experiment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9745, "parameters": 1396, "training_steps": 74999}

RECENT RESULT
hypothesis: Fixing the sinusoidal amplitude at 0.02 will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps, while endpoint-only validation will allow verification to finish within the time limit.
change: Adopt query-only QKV bias, replace learned positional embeddings with a fixed sinusoidal buffer, remove the learned positional gain, and validate only at the final training step.
mechanism: Fixed-amplitude sinusoidal positions with endpoint-only validation
evidence_used: The learned-gain sinusoidal design achieved 99.69% accuracy with 1,397 parameters at 74,999 steps. Prior fixed-gain runs produced no accuracy evidence solely because they timed out, while endpoint-only validation allowed a separate 75,000-step 1,396-parameter experiment to finish.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1396, "training_steps": 74999}

RECENT RESULT
hypothesis: Sharing one query-bias coordinate across the two attention heads will retain at least 99% accuracy after 75,000 steps while reducing the qualified sinusoidal design from 1,397 to 1,396 parameters.
change: Adopt the qualified learned-scale sinusoidal representation and key/value-bias quotient, represent eight query-bias coordinates with seven learned scalars by sharing one coordinate across heads, and use endpoint-only validation to stay within the verification time limit.
mechanism: Cross-head query-bias sharing with learned-scale sinusoidal positions
evidence_used: The 1,397-parameter sinusoidal design achieved 99.69% accuracy, while clamping one query-bias coordinate reached 97.45%. Sharing instead of clamping tests the same one-parameter reduction while preserving an adaptive bias path in every head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1396, "training_steps": 74999}

RECENT RESULT
hypothesis: Sharing a second corresponding query-bias coordinate across the two attention heads will retain at least 99% accuracy after 75,000 steps while reducing the qualified 1,396-parameter design to 1,395 parameters.
change: Represent the eight query-bias coordinates with six learned scalars by sharing two corresponding coordinates across heads; retain the learned sinusoidal gain and endpoint-only validation.
mechanism: Incremental cross-head query-bias sharing
evidence_used: The current one-coordinate-sharing design achieved 99.97% accuracy with 1,396 parameters, substantially above the threshold, while clamping a coordinate achieved only 97.45%; this supports another incremental adaptive sharing constraint rather than fixing a coordinate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A four-dimensional learned token codebook with separate learned projections into and out of the eight-dimensional residual stream will reach at least 99% accuracy after 75,000 steps while reducing the model from 1,397 to 1,005 parameters.
change: Replace the full-width tied token embedding/head with a shared four-dimensional learned codebook and orthogonally initialized input/output projections; retain the qualified attention, MLP, learned sinusoidal gain, and use endpoint-only validation.
mechanism: Learned low-dimensional token codebook with asymmetric input/output lifts
evidence_used: The 1,397-parameter sinusoidal model reached 99.69%, while attempts to constrain its positional gain or query-bias mechanism failed or approached the threshold. This instead challenges the shared assumption that every vocabulary token needs an unconstrained eight-dimensional interface: the learned transformer still computes in eight dimensions, but token identity and logits pass through a compact learned code space.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing a second corresponding query-bias coordinate will retain at least 99% accuracy with 1,395 parameters after 70,000 steps.
change: Adopt learned-scale sinusoidal positions and query-only QKV bias, share two query-bias coordinates across heads, and use positive-step endpoint validation with a reduced training budget.
mechanism: Two-coordinate cross-head query-bias sharing
evidence_used: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters at 74,999 steps; the prior two-coordinate attempt yielded no accuracy evidence because it timed out, motivating the same incremental constraint with 70,000 steps and no step-zero validation.
result: training did not finish within the verification time limit

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
