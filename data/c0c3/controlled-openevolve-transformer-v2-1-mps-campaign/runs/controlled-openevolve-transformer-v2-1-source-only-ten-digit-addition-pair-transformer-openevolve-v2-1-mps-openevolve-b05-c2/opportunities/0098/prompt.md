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
verified_results: {"accuracy": 0.9987, "parameters": 1500, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified 1501-parameter attention-bias pattern and fixing reduced MLP output-bias coordinate 0 will yield 1500 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9994, "parameters": 1498, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1499-parameter design by fixing reduced MLP output-bias coordinate 2 will yield 1498 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1501, "training_steps": 4999}
prior_hypothesis: Fixing reduced attention-output bias coordinate 3 in the qualified 1502-parameter design will yield 1501 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1502, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1503-parameter design by fixing reduced attention-output bias coordinate 1 will yield 1502 parameters while maintaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Starting from the qualified 1506-parameter `q_bias[6]`-anchored design, fixing the remaining pre-attention LayerNorm scale at one will yield 1505 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified `q_bias[6]` anchor, then make `ln1` fully parameter-free by fixing its final learned scale, which the following learned QKV map can absorb.
mechanism: Final pre-attention LayerNorm scale absorption
evidence_used: Anchoring `q_bias[6]` achieved 99.78% at 1506 parameters, while removing `q_bias[7]` fell to 94.48% and additional value rotations also failed; all qualified designs already fix seven of eight `ln1` scales, motivating completion of that distinct gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6103999999999999, "parameters": 1505, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1506-parameter nine-value-gauge and `q_bias[6]`-anchored design, then fixing one zero-initialized attention-output bias coordinate, will yield 1505 parameters while retaining at least 99% accuracy.
change: Upgrade the current model to nine value-output rotations, anchor `q_bias[6]` as in the qualified design, and remove one reduced-coordinate bias parameter from the attention output projection.
mechanism: Single attention-output bias anchoring
evidence_used: The nine-value-gauge design with `q_bias[6]` fixed achieved 99.78% at 1506 parameters. Further query-bias, value-rotation, and LayerNorm constraints failed, motivating a distinct zero-initialized affine-bias reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1505, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1505-parameter design by fixing a second zero-initialized reduced attention-output bias coordinate will yield 1504 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified nine-value-rotation and `q_bias[6]`-anchored model, then retain only five of the seven reduced attention-output bias coordinates.
mechanism: Second attention-output bias anchoring
evidence_used: The first attention-output bias anchor achieved 99.95% accuracy at 1505 parameters, while reductions targeting query bias, LayerNorm scale, and additional value rotations failed; anchoring the adjacent zero-initialized bias coordinate is the closest supported next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third zero-initialized reduced attention-output bias coordinate will reduce the qualified model to 1503 parameters while retaining at least 99% accuracy.
change: Retain four of the seven reduced attention-output bias coordinates and reconstruct the other three as zero.
mechanism: Third attention-output bias anchoring
evidence_used: Successive first and second attention-output bias anchors achieved 99.95% at 1505 parameters and 99.97% at 1504 parameters, making the adjacent third anchor the closest supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.39490000000000003, "parameters": 1503, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second reduced attention-output bias coordinate will produce a 1504-parameter model with at least 99% accuracy.
change: Retain five of seven reduced attention-output bias coordinates, reconstruct the final two as zero, and use this twice-anchored projection in causal self-attention.
mechanism: Second attention-output bias anchoring
evidence_used: Reference Design 1 achieved 99.97% accuracy with 1504 parameters using this exact reduction, while fixing a third coordinate fell to 39.49%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the first reduced attention-output bias coordinate while retaining the two qualified final anchors will yield 1503 parameters and maintain at least 99% accuracy.
change: Replace the twice-anchored attention projection with a three-anchor variant that reconstructs reduced bias coordinates 0, 5, and 6 as zero while learning coordinates 1–4.
mechanism: Nonadjacent attention-output bias anchoring
evidence_used: Fixing reduced coordinates 5 and 6 achieved 99.97% accuracy at 1504 parameters, whereas additionally fixing adjacent coordinate 4 fell to 39.49%; testing coordinate 0 distinguishes coordinate sensitivity from a general 1503-parameter capacity limit.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1503, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1503-parameter design by fixing reduced attention-output bias coordinate 1 will yield 1502 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified `q_bias[6]` anchor and fix reduced projection-bias coordinates 0, 1, 5, and 6, leaving coordinates 2–4 learned.
mechanism: Low-side nonadjacent attention-output bias anchoring
evidence_used: Anchoring coordinates 0, 5, and 6 achieved 99.93% accuracy at 1503 parameters, while anchoring coordinate 4 with 5 and 6 failed at 39.49%; coordinate 1 is the closest untested extension on the successful low-coordinate side.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1502, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing reduced attention-output bias coordinate 2 will reduce the qualified model to 1501 parameters while maintaining at least 99% accuracy.
change: Reconstruct reduced projection-bias coordinates 0, 1, 2, 5, and 6 as zero, leaving coordinates 3 and 4 learned.
mechanism: Low-side fifth attention-output bias anchoring
evidence_used: Anchoring coordinates 0, 1, 5, and 6 achieved 99.95% accuracy at 1502 parameters, while anchoring coordinate 4 with 5 and 6 failed at 39.49%; coordinate 2 is the closest untested extension on the successful low-coordinate side and preserves coordinate 4.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7426999999999999, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing reduced attention-output bias coordinate 3 in the qualified 1502-parameter design will yield 1501 parameters while maintaining at least 99% accuracy.
change: Reconstruct reduced projection-bias coordinates 0, 1, 3, 5, and 6 as zero, leaving only coordinates 2 and 4 learned.
mechanism: Interior attention-output bias anchoring
evidence_used: The 1502-parameter design anchoring coordinates 0, 1, 5, and 6 achieved 99.95% accuracy; coordinate 2 then failed at 74.27%, and coordinate 4 was previously harmful, making coordinate 3 the only untested extension of that qualified bias pattern.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1501-parameter attention-bias pattern and fixing reduced MLP output-bias coordinate 0 will yield 1500 parameters while maintaining at least 99% accuracy.
change: Keep only attention projection-bias coordinates 2 and 4, then reconstruct MLP output-bias coordinate 0 as zero while learning the remaining six coordinates.
mechanism: Independent MLP output-bias coordinate anchoring
evidence_used: Reference Design 2 achieved 99.96% accuracy at 1501 parameters with attention-bias coordinates 0, 1, 3, 5, and 6 fixed; further attention-bias anchoring failed, motivating a distinct zero-initialized affine-bias reduction in the MLP output.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1500, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1500-parameter design by fixing reduced MLP output-bias coordinate 1 will yield 1499 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified attention projection with only reduced bias coordinates 2 and 4 learned, then reconstruct MLP output-bias coordinates 0 and 1 as zero.
mechanism: Second MLP output-bias coordinate anchoring
evidence_used: Reference Design 1 achieved 99.87% accuracy at 1500 parameters after fixing MLP output-bias coordinate 0; coordinate 1 is the closest untested extension of that successful independent bias reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1499, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1499-parameter design by fixing reduced MLP output-bias coordinate 2 will yield 1498 parameters while maintaining at least 99% accuracy.
change: Reconstruct MLP output-bias coordinates 0, 1, and 2 as zero while learning the remaining four reduced coordinates.
mechanism: Third MLP output-bias coordinate anchoring
evidence_used: Successive anchoring of MLP output-bias coordinates 0 and 1 retained 99.87% and 99.84% accuracy respectively; coordinate 2 is the closest untested continuation of this successful reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1498, "training_steps": 4999}



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
