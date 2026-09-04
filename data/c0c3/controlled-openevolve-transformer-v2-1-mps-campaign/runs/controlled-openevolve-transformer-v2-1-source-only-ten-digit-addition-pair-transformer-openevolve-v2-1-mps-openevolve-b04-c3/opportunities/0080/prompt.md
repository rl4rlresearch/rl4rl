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
verified_results: {"accuracy": 1.0, "parameters": 1217, "training_steps": 44999}
prior_hypothesis: Fixing a second scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,218 to 1,217 learned parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 1.0, "parameters": 1213, "training_steps": 44999}
prior_hypothesis: Fixing a sixth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,214 to 1,213 learned parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1212, "training_steps": 44999}
prior_hypothesis: Fixing a seventh scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,213 to 1,212 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1211, "training_steps": 44999}
prior_hypothesis: Fixing an eighth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,212 to 1,211 learned parameters.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing a third scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,217 to 1,216 learned parameters.
change: Adopt the qualified two-entry positional-mixer constraint, fix one additional final mixer weight at zero, and preserve full-matrix initialization RNG consumption.
mechanism: Three-entry positional-mixer constraint
evidence_used: Reference Design 2 achieved 100% accuracy with two fixed positional-mixer scalars and 1,217 parameters; since the aggressive eight-gain restriction failed at 46.49%, the most informative next change is one additional scalar constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1216, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fourth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,216 to 1,215 learned parameters.
change: Replace the absolute-position embedding with the qualified Fourier synthesizer, learn 60 of its 64 mixer weights, and fix the final four weights at zero while preserving full-matrix initialization RNG consumption.
mechanism: Four-entry positional-mixer constraint
evidence_used: Reference Design 3 achieved 100% accuracy with three fixed positional-mixer scalars and 1,216 parameters; the preceding one- and two-scalar constraints also achieved 100%, motivating the smallest incremental fourth constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1215, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fifth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,215 to 1,214 learned parameters.
change: Adopt the qualified seven-entry query-weight gauge and Fourier positional backbone, then learn 59 of 64 positional-mixer weights with the final five fixed at zero while preserving full-matrix initialization RNG consumption.
mechanism: Five-entry positional-mixer constraint
evidence_used: The four-entry positional-mixer design achieved 100% accuracy with 1,215 parameters, following 100% results for one through three fixed entries; this motivates the smallest incremental fifth constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1214, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a sixth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,214 to 1,213 learned parameters.
change: Learn 58 of the 64 positional-mixer weights, fixing the final six entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Six-entry positional-mixer constraint
evidence_used: The five-entry positional-mixer design achieved 100% accuracy with 1,214 parameters, following 100% results for one through four fixed entries; this motivates the smallest incremental sixth constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1213, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a seventh scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,213 to 1,212 learned parameters.
change: Learn 57 of the 64 positional-mixer weights, fixing the final seven entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Seven-entry positional-mixer constraint
evidence_used: Reference Design 1 achieved 100% accuracy with six fixed positional-mixer scalars and 1,213 parameters, following successful one-through-five-scalar constraints; this motivates the smallest incremental seventh constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1212, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing an eighth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,212 to 1,211 learned parameters.
change: Learn 56 of the 64 positional-mixer weights, fixing the final eight entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Eight-entry positional-mixer constraint
evidence_used: Reference Design 3 achieved 100% accuracy with seven fixed positional-mixer scalars and 1,212 parameters, following successful one-through-six-scalar constraints; this motivates the smallest incremental eighth constraint while retaining learned cross-channel positional mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1211, "training_steps": 44999}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
