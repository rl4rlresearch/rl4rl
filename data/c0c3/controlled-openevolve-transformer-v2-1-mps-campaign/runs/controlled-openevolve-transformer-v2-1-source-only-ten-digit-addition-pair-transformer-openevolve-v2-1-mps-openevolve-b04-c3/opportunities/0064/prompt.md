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
verified_results: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}
prior_hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 1.0, "parameters": 1345, "training_steps": 44999}
prior_hypothesis: Fixing one query-weight coordinate at zero on the qualified 1,346-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,345 parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1347, "training_steps": 44999}
prior_hypothesis: Fixing one coordinate of the fully shared query-bias vector at zero will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,348 to 1,347 parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1346, "training_steps": 44999}
prior_hypothesis: Combining the qualified three-coordinate shared query bias with one fixed output-projection bias coordinate will retain at least 99% accuracy after 45,000 steps with 1,346 parameters.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Sharing a third corresponding query-bias coordinate across the two heads will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,350 to 1,349 learned parameters.
change: Extend the successful cross-head query-bias sharing constraint from two coordinates to three without changing the qualified learned-position, rank-six token, attention, or training backbone.
mechanism: Three-coordinate cross-head query-bias gauge sharing
evidence_used: The current two-coordinate-sharing design achieved 100% accuracy with 1,350 parameters, after one-coordinate sharing also achieved 100%; this directly motivates testing the same one-parameter reduction on the next query-bias coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1349, "training_steps": 44999}

RECENT RESULT
hypothesis: Sharing all four corresponding query-bias coordinates across the two heads will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,349 to 1,348 learned parameters.
change: Extend the qualified three-coordinate sharing constraint to the fourth and final query-bias coordinate, leaving the learned positions, rank-six tied token code, attention weights, and training schedule unchanged.
mechanism: Full-vector cross-head query-bias gauge sharing
evidence_used: Three-coordinate sharing achieved 99.98% accuracy with 1,349 parameters after both one- and two-coordinate sharing achieved 100%; the remaining corresponding coordinate is governed by the same per-head Q/K basis-change symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1348, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing one coordinate of the fully shared query-bias vector at zero will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,348 to 1,347 parameters.
change: Adopt full cross-head query-bias sharing and learn only three of its four coordinates, padding the fourth with zero while preserving the qualified learned-position and rank-six token backbone.
mechanism: One-coordinate shared-query basis gauge
evidence_used: Reference Design 3 achieved 100% accuracy with 1,348 parameters and a fully shared four-coordinate query bias; the Q/K basis symmetry motivates testing a single zero-coordinate gauge while retaining three adaptable bias coordinates and the successful initialization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1347, "training_steps": 44999}

RECENT RESULT
hypothesis: A five-parameter-per-token angular-product lift will produce a generically full-rank eight-dimensional tied token code and achieve at least 99% accuracy after 45,000 steps with 1,233 learned parameters.
change: Replace the load-bearing assumption that token representations must occupy a learned linear subspace with a nonlinear homogeneous lift from five learned coordinates to all eight residual channels, while restoring the qualified learned positions and shared three-parameter query bias.
mechanism: Nonlinear full-rank tied token manifold
evidence_used: The rank-six linear token model reached 100%, whereas rank-five zero-padded tokens reached only 54.53%; that test simultaneously reduced learned coordinates and limited token injection and classification to rank five. The nonlinear lift isolates learned degrees of freedom from representational rank while retaining the qualified 1,347-parameter backbone.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5664, "parameters": 1233, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a second coordinate of the fully shared query-bias vector at zero will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,347 to 1,346 learned parameters.
change: Learn two shared query-bias coordinates and pad the remaining two with zeros, leaving the qualified rank-six token, learned-position, attention-weight, and training backbone unchanged.
mechanism: Two-coordinate shared-query basis gauge
evidence_used: The current design achieved 100% accuracy after reducing the fully shared query bias from four learned coordinates to three; the same Q/K basis symmetry motivates testing the next incremental one-parameter gauge constraint.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.747, "parameters": 1346, "training_steps": 44999}

RECENT RESULT
hypothesis: Combining the qualified three-coordinate shared query bias with one fixed output-projection bias coordinate will retain at least 99% accuracy after 45,000 steps with 1,346 parameters.
change: Adopt the proven three-parameter shared query bias and use the orthogonal freedom in the two token-free residual channels to fix the final attention projection-bias coordinate at zero.
mechanism: Residual-subspace rotational gauge fixing
evidence_used: The three-parameter shared query bias achieved 100% accuracy with 1,347 parameters, while reducing it to two parameters failed at 74.7%; this instead removes an independent parameter through the residual stream’s two-dimensional rotational symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1346, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing one query-weight coordinate at zero on the qualified 1,346-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,345 parameters.
change: Adopt the qualified three-parameter shared query bias and seven-parameter projection bias, then remove one query-weight scalar and reconstruct the full QKV matrix with a fixed zero while preserving initialization RNG consumption.
mechanism: Single-entry query/key basis gauge fixing
evidence_used: Reference Design 3 achieved 100% accuracy with 1,346 parameters. Since reducing the shared query bias further failed at 74.7%, this instead tests an independent Q/K basis-change redundancy while leaving the successful token, position, and optimization backbone intact.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1345, "training_steps": 44999}



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
