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
verified_results: {"accuracy": 1.0, "parameters": 1219, "training_steps": 44999}
prior_hypothesis: Replacing 98 independent eight-dimensional position vectors with a learned shared projection of generic sinusoidal coordinates will retain at least 99% accuracy after 45,000 updates while removing 720 positional parameters from the qualified seven-entry query-gauge backbone.

REFERENCE DESIGN 1
verified_results: {"accuracy": 1.0, "parameters": 1342, "training_steps": 44999}
prior_hypothesis: Fixing a fourth query-weight coordinate at zero on the qualified 1,343-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,342 learned parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1217, "training_steps": 44999}
prior_hypothesis: Fixing a second scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,218 to 1,217 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1339, "training_steps": 44999}
prior_hypothesis: Fixing a seventh leading query-weight coordinate at zero will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,340 to 1,339 learned parameters.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing a second query-weight coordinate at zero on the qualified 1,345-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,344 parameters.
change: Adopt the qualified rank-six tied-token and learned-position backbone, three-parameter shared query bias, seven-parameter projection bias, and fix the first two flattened query-weight entries at zero while preserving initialization RNG consumption.
mechanism: Two-entry query/key basis gauge fixing
evidence_used: Reference Design 1 achieved 100% accuracy with 1,345 parameters after fixing one query-weight coordinate; extending that same Q/K basis gauge by one scalar is the smallest direct reduction, while the failed two-parameter query-bias result argues against further constraining the load-bearing shared bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1344, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a third query-weight coordinate at zero on the qualified 1,344-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,343 parameters.
change: Extend the existing query-weight constraint from the first two flattened entries to the first three, reconstructing the full QKV matrix with three leading zeros while preserving initialization RNG consumption.
mechanism: Three-entry query/key basis gauge fixing
evidence_used: The current design achieved 100% accuracy with two fixed query-weight coordinates at 1,344 parameters, after the one-coordinate design also achieved 100%; this makes the next incremental coordinate of the same Q/K basis gauge the most direct informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1343, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fourth query-weight coordinate at zero on the qualified 1,343-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,342 learned parameters.
change: Extend the query-weight constraint from three to four leading flattened entries, reconstructing the full QKV matrix with four fixed zeros while preserving initialization RNG consumption.
mechanism: Four-entry query/key basis gauge fixing
evidence_used: The three-entry design achieved 100% accuracy with 1,343 parameters, following 100% results with one and two fixed entries; this supports testing the next incremental coordinate of the same Q/K basis gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1342, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fifth leading query-weight coordinate at zero on the qualified 1,342-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,341 learned parameters.
change: Extend the proven query-weight constraint from four to five leading flattened entries, reconstructing the full QKV matrix with five fixed zeros while preserving initialization RNG consumption.
mechanism: Five-entry query/key basis gauge fixing
evidence_used: The four-entry design achieved 100% accuracy with 1,342 parameters, following 100% results with one, two, and three fixed entries; this directly motivates testing the next scalar in the same Q/K basis gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1341, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a sixth leading query-weight coordinate at zero on the qualified 1,341-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,340 learned parameters.
change: Reproduce the qualified five-entry design, then extend its query-weight constraint to six leading flattened entries while preserving full-matrix initialization RNG consumption.
mechanism: Six-entry query/key basis gauge fixing
evidence_used: The five-entry design achieved 100% accuracy with 1,341 parameters, following 100% results with one through four fixed entries; this directly motivates testing the next scalar in the same Q/K basis gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1340, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a seventh leading query-weight coordinate at zero will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,340 to 1,339 learned parameters.
change: Extend the existing query-weight constraint from six to seven leading flattened entries, reconstructing the full QKV matrix with seven fixed zeros while preserving full-matrix initialization RNG consumption.
mechanism: Seven-entry query/key basis gauge fixing
evidence_used: The six-entry design achieved 100% accuracy with 1,340 parameters, following 100% results with one through five fixed entries; this directly motivates testing the next scalar in the same Q/K basis gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1339, "training_steps": 44999}

RECENT RESULT
hypothesis: Replacing 98 independent eight-dimensional position vectors with a learned shared projection of generic sinusoidal coordinates will retain at least 99% accuracy after 45,000 updates while removing 720 positional parameters from the qualified seven-entry query-gauge backbone.
change: Adopt the proven seven-zero query-weight constraint, preserve the rank-six tied token code and attention/MLP computation, and replace the learned absolute-position lookup table with a parameter-efficient 8×8 learned Fourier synthesizer.
mechanism: Shared Fourier positional synthesis
evidence_used: The seven-entry query-weight design achieved 100% accuracy with 1,339 parameters, while the five-coordinate nonlinear token lift failed at 56.64%, identifying token identity geometry as load-bearing. Every qualified design still assumes an independent learned vector for every position; addition instead presents repeated position-to-position routing for which shared sinusoidal coordinates are a plausible structured representation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1219, "training_steps": 44999}

RECENT RESULT
hypothesis: Replacing the qualified dense 8×8 Fourier synthesizer with a fixed generic orthogonal transform and eight learned channel gains will retain at least 99% accuracy after 45,000 steps while reducing the qualified model from 1,219 to 1,163 learned parameters.
change: Adopt the proven seven-entry query-weight gauge and Fourier positional backbone, then constrain positional synthesis to a fixed DCT basis with eight trainable gains while preserving initialization RNG consumption and the rank-six tied token code.
mechanism: Orthogonally mixed diagonal Fourier positional synthesis
evidence_used: Reference Design 3 achieved 100% accuracy with 1,219 parameters using generic sinusoidal coordinates and a learned 8×8 projection; this tests whether downstream learned attention and MLP projections can supply the required coordinate mixing while positional learning retains only per-channel amplitudes. The failed five-coordinate token lift motivates leaving the qualified token representation unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.46490000000000004, "parameters": 1163, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing one scalar in the qualified dense Fourier positional synthesizer at zero will retain at least 99% accuracy after 45,000 updates with 1,218 learned parameters.
change: Adopt the qualified seven-entry query-weight gauge and Fourier positional backbone, then learn 63 of the 64 positional-mixer weights while fixing its final entry at zero and preserving full-matrix initialization RNG consumption.
mechanism: Single-entry positional-mixer constraint
evidence_used: The dense 8×8 Fourier synthesizer achieved 100% accuracy with 1,219 parameters, while the aggressive eight-gain restriction failed at 46.49%; retaining 63 cross-channel coefficients tests the smallest informative reduction without repeating that severe loss of positional mixing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1218, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a second scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,218 to 1,217 learned parameters.
change: Learn 62 of the 64 positional-mixer weights, fixing its final two entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Two-entry positional-mixer constraint
evidence_used: The immediately preceding single-entry positional-mixer constraint achieved 100% accuracy with 1,218 parameters, while the eight-gain restriction failed at 46.49%; the smallest informative next test is one additional fixed mixer scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1217, "training_steps": 44999}



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
