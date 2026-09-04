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
verified_results: {"accuracy": 0.9983, "parameters": 1495, "training_steps": 4999}
prior_hypothesis: Fixing final LayerNorm bias coordinate 0 at zero will reduce the qualified model from 1496 to 1495 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9994, "parameters": 1498, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1499-parameter design by fixing reduced MLP output-bias coordinate 2 will yield 1498 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1501, "training_steps": 4999}
prior_hypothesis: Fixing reduced attention-output bias coordinate 3 in the qualified 1502-parameter design will yield 1501 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9981, "parameters": 1497, "training_steps": 4999}
prior_hypothesis: Fixing reduced MLP output-bias coordinate 5 while preserving coordinates 3 and 4 will yield 1497 parameters and maintain at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the qualified 1499-parameter design by fixing reduced MLP output-bias coordinate 2 will yield 1498 parameters while maintaining at least 99% accuracy.
change: Reconstruct MLP output-bias coordinates 0, 1, and 2 as zero while learning the remaining four reduced coordinates.
mechanism: Third MLP output-bias coordinate anchoring
evidence_used: Successive anchoring of MLP output-bias coordinates 0 and 1 retained 99.87% and 99.84% accuracy respectively; coordinate 2 is the closest untested continuation of this successful reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1498, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1498-parameter design by fixing reduced MLP output-bias coordinate 3 will yield 1497 parameters while maintaining at least 99% accuracy.
change: Reconstruct MLP output-bias coordinates 0–3 as zero while learning the remaining three reduced coordinates.
mechanism: Fourth MLP output-bias coordinate anchoring
evidence_used: Reference Design 1 achieved 99.94% accuracy at 1498 parameters after successive anchoring of reduced MLP output-bias coordinates 0, 1, and 2, motivating coordinate 3 as the closest supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4534, "parameters": 1497, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing reduced MLP output-bias coordinate 4 while retaining the successful anchors at coordinates 0–2 will yield 1497 parameters and maintain at least 99% accuracy.
change: Reproduce the qualified 1498-parameter MLP bias reduction, then reconstruct reduced bias coordinates 0, 1, 2, and 4 as zero while learning coordinates 3, 5, and 6.
mechanism: Nonadjacent MLP output-bias anchoring
evidence_used: Fixing MLP bias coordinates 0–2 achieved 99.94% accuracy at 1498 parameters, whereas additionally fixing coordinate 3 fell to 45.34%; testing coordinate 4 distinguishes coordinate sensitivity from a general 1497-parameter capacity limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7298, "parameters": 1497, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing reduced MLP output-bias coordinate 5 while preserving coordinates 3 and 4 will yield 1497 parameters and maintain at least 99% accuracy.
change: Reproduce the qualified 1498-parameter attention and MLP bias pattern, then reconstruct MLP output-bias coordinates 0, 1, 2, and 5 as zero while learning coordinates 3, 4, and 6.
mechanism: Nonadjacent high-side MLP output-bias anchoring
evidence_used: The 1498-parameter design achieved 99.94% accuracy with MLP bias coordinates 0–2 fixed; additionally fixing coordinates 3 or 4 failed, so coordinate 5 is the closest untested reduction that preserves both empirically sensitive coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1497, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing reduced MLP output-bias coordinate 6 while preserving sensitive coordinates 3 and 4 will yield 1496 parameters and maintain at least 99% accuracy.
change: Reconstruct MLP output-bias coordinates 0, 1, 2, 5, and 6 as zero, leaving only coordinates 3 and 4 learned.
mechanism: Final high-side MLP output-bias anchoring
evidence_used: The current 1497-parameter design achieved 99.81% accuracy after fixing coordinate 5, while fixing coordinates 3 or 4 caused failure; coordinate 6 is the only remaining untested bias coordinate whose removal preserves both sensitive coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2787, "parameters": 1496, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1497-parameter design, fixing MLP input-bias coordinate 0 will yield 1496 parameters while maintaining at least 99% accuracy.
change: Preserve the successful MLP output-bias anchors at coordinates 0, 1, 2, and 5, and independently reconstruct the first of twelve MLP hidden biases as zero.
mechanism: Independent MLP hidden-bias anchoring
evidence_used: Reference Design 3 achieved 99.81% accuracy at 1497 parameters, while removing its remaining output-bias coordinate 6 failed at 27.87%; prior successful reductions moved to a distinct zero-initialized affine bias after exhausting a sensitive bias family, motivating the larger MLP input-bias vector as the next independent target.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7073, "parameters": 1496, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1497-parameter design, fixing its sole remaining pre-attention LayerNorm scale at one will yield 1496 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified MLP output-bias anchors at coordinates 0, 1, 2, and 5, then remove the remaining learned `ln1` scale so all eight pre-attention scales are fixed at one.
mechanism: Final pre-attention LayerNorm scale anchoring
evidence_used: The 1497-parameter design achieved 99.81% accuracy, while removing output-bias coordinate 6 and hidden-bias coordinate 0 failed. All qualified designs already fix seven of eight `ln1` scales, motivating the remaining scale as an independent, initialization-preserving reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1496, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1496-parameter design and fixing final LayerNorm scale coordinate 0 at one will yield 1495 parameters while maintaining at least 99% accuracy.
change: Apply the qualified MLP output-bias and pre-attention LayerNorm reductions, then reconstruct the first final LayerNorm scale as a fixed one while learning its other seven scales and all biases.
mechanism: Independent final-normalization scale anchoring
evidence_used: Reference Design 1 achieved 99.96% accuracy at 1496 parameters after fixing the last learned pre-attention LayerNorm scale; testing the same initialization-preserving reduction in the distinct final-normalization scale family is the most direct next one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3757, "parameters": 1495, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing final LayerNorm bias coordinate 0 at zero will reduce the qualified model from 1496 to 1495 parameters while maintaining at least 99% accuracy.
change: Replace the final LayerNorm with an equivalent module whose scale and bias coordinates 1–7 remain learned while bias coordinate 0 is reconstructed as zero.
mechanism: Zero-initialized final-normalization bias anchoring
evidence_used: The current 1496-parameter design reached 99.96% accuracy; fixing final scale coordinate 0 failed, while earlier successful reductions anchored zero-initialized affine biases, motivating a direct test of the distinct final-normalization bias family.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1495, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1495-parameter design and fixing final LayerNorm bias coordinate 1 at zero will yield 1494 parameters while maintaining at least 99% accuracy.
change: Fix the remaining pre-attention LayerNorm scale at one and reconstruct final LayerNorm bias coordinates 0 and 1 as zero, leaving all eight final scales and bias coordinates 2–7 learned.
mechanism: Sequential final-normalization bias anchoring
evidence_used: Reference Design 1 achieved 99.83% accuracy at 1495 parameters after fixing final LayerNorm bias coordinate 0; coordinate 1 is the closest untested continuation in that successful zero-initialized bias family, while fixing a final scale failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7126, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing final LayerNorm bias coordinate 2 while preserving coordinate 1 will reduce the qualified 1495-parameter design to 1494 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified 1495-parameter MLP and pre-attention reductions, then reconstruct final LayerNorm bias coordinates 0 and 2 as zero.
mechanism: Nonadjacent final-normalization bias anchoring
evidence_used: Fixing final LayerNorm bias coordinate 0 achieved 99.83% accuracy at 1495 parameters, whereas additionally fixing coordinate 1 fell to 71.26%; testing coordinate 2 distinguishes coordinate sensitivity from a general 1494-parameter capacity limit.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9803000000000001, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1495-parameter design and fixing final LayerNorm bias coordinate 3 while preserving coordinates 1 and 2 will yield 1494 parameters and maintain at least 99% accuracy.
change: Restore the qualified MLP output-bias and fixed pre-attention normalization reductions, then reconstruct final LayerNorm bias coordinates 0 and 3 as zero.
mechanism: Nonadjacent final-normalization bias anchoring
evidence_used: Fixing final bias coordinate 0 achieved 99.83% at 1495 parameters; additionally fixing coordinate 2 reached 98.03%, substantially closer to qualification than coordinate 1 at 71.26%, motivating the adjacent untested coordinate 3 as the next coordinate-sensitivity probe.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4483, "parameters": 1494, "training_steps": 4999}



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
