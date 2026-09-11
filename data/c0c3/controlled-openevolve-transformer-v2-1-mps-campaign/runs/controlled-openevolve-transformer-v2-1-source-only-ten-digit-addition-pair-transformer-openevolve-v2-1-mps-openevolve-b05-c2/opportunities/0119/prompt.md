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
verified_results: {"accuracy": 0.9997, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1492-parameter design with a triangular shear that fixes first-head query row 3, input coordinate 1 will yield 1491 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9997, "parameters": 1492, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified 1493-parameter ten-value-gauge design and fixing first-head query row 2, input coordinate 2 through an exact query–key shear will yield 1492 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9992, "parameters": 1490, "training_steps": 4999}
prior_hypothesis: Extending the qualified 1491-parameter design with a shear that fixes first-head query row 3, input coordinate 2 will yield 1490 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 1493, "training_steps": 4999}
prior_hypothesis: Reproducing the qualified 1494-parameter design and fixing second-head value coordinate 2 through the remaining triangular value–output rotation will yield 1493 parameters while maintaining at least 99% accuracy.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Fixing final LayerNorm bias coordinate 4 while preserving coordinates 1–3 will reduce the qualified 1495-parameter model to 1494 parameters and maintain at least 99% accuracy.
change: Reconstruct final LayerNorm bias coordinates 0 and 4 as zero, leaving coordinates 1–3 and 5–7 learned.
mechanism: Nonadjacent final-normalization bias coordinate 4 anchoring
evidence_used: Fixing coordinate 0 achieved 99.83% accuracy; additional anchors at coordinates 1 and 3 failed substantially, while coordinate 2 reached 98.03%. Coordinate 4 is the next untested coordinate and extends the coordinate-sensitivity probe without removing any previously implicated coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2966, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing final LayerNorm bias coordinate 5 while preserving coordinates 1–4 will reduce the qualified model from 1495 to 1494 parameters and maintain at least 99% accuracy.
change: Reconstruct final LayerNorm bias coordinates 0 and 5 as zero, leaving coordinates 1–4 and 6–7 learned.
mechanism: Nonadjacent final-normalization bias coordinate 5 anchoring
evidence_used: The 1495-parameter design reached 99.83% accuracy with coordinate 0 fixed; additional anchors at coordinates 1–4 failed, while the analogous MLP search succeeded specifically at nonadjacent coordinate 5 after lower coordinates failed, making coordinate 5 the most informative untested final-bias reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6328, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the remaining second-head query-weight coordinate through a simultaneous query–key rotation will reduce the qualified model from 1495 to 1494 parameters while maintaining at least 99% accuracy.
change: Add the unused `(second_query, 2)` rotation and reconstruct that query row with three leading zeros instead of two.
mechanism: Twelfth within-head query–key rotation gauge fixing
evidence_used: The qualified 1495-parameter model already preserves accuracy with eleven query–key rotations; unlike the unsuccessful final-LayerNorm bias anchors, this change extends the same initialization-preserving attention symmetry by one coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1494-parameter design and fixing second-head value coordinate 2 through the remaining triangular value–output rotation will yield 1493 parameters while maintaining at least 99% accuracy.
change: Apply the qualified twelfth query rotation, fixed pre-attention scales, and final-bias anchor, then add `(second_query, 2)` to the value rotations and compensate the attention-output projection at initialization.
mechanism: Tenth value–output rotation gauge fixing
evidence_used: Reference Design 1 achieved 99.97% accuracy at 1494 parameters after completing the analogous twelfth query–key rotation; the second value head currently has the first five triangular rotations, making its sixth orthogonal rotation the closest symmetry-preserving reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1493, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the qualified 1493-parameter design and replacing the first value head’s four-coordinate gauge with five triangular rotations will yield 1492 parameters while maintaining at least 99% accuracy.
change: Apply the qualified MLP-bias, fixed pre-attention scale, final-bias, twelfth query-rotation, and tenth value-rotation reductions; then fix one additional first-head value coordinate with compensated orthogonal initialization.
mechanism: Eleventh value–output orthogonal gauge fixing via first-head triangular recanonicalization
evidence_used: Reference Design 3 achieved 99.97% accuracy at 1493 parameters after extending the same value–output symmetry to ten rotations. The first value head still uses only four orthogonal degrees of freedom, so recanonicalizing it with five triangular rotations tests another exact symmetry rather than removing functional capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3371, "parameters": 1492, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying one inverse-compensated first-head query–key shear will reduce the qualified model from 1493 to 1492 parameters while maintaining at least 99% accuracy.
change: After the twelve orthogonal query rotations, add a triangular shear that fixes first-head query row 2, input coordinate 2 at zero, applies the inverse-transpose shear to the corresponding key rows, and reconstructs the removed query coordinate during every forward pass.
mechanism: Within-head query–key shear gauge fixing
evidence_used: The twelfth query–key rotation preserved 99.97% accuracy at 1494 parameters, and the subsequent value–output rotation preserved 99.97% at 1493; the failed 1492 attempt recanonicalized the sensitive first value head, so testing an unused exact query–key GL symmetry is the most informative independent reduction.
result: the patch search text did not match the source

RECENT RESULT
hypothesis: Reproducing the qualified 1493-parameter ten-value-gauge design and fixing first-head query row 2, input coordinate 2 through an exact query–key shear will yield 1492 parameters while maintaining at least 99% accuracy.
change: Add the qualified tenth second-head value rotation with output-projection compensation, then remove one first-head query coordinate using a triangular shear and inverse-transpose key compensation.
mechanism: Inverse-compensated query–key shear gauge fixing
evidence_used: The ten-value-gauge reference achieved 99.97% accuracy at 1493 parameters; the prior 1492 value-head recanonicalization failed, while the independent query–key shear attempt was not tested because its patch did not apply.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1492, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1492-parameter design with a triangular shear that fixes first-head query row 3, input coordinate 1 will yield 1491 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified MLP, normalization, twelfth query-rotation, tenth value-rotation, and first query-shear reductions, then remove one additional query coordinate while applying the exact inverse-transpose compensation to its key row.
mechanism: Second inverse-compensated first-head query–key shear
evidence_used: The qualified 1492-parameter design achieved 99.97% accuracy using an inverse-compensated query–key shear, whereas the alternative eleventh value-gauge reduction fell to 33.71%; extending the successful query–key GL symmetry is therefore the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1491, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1491-parameter design with a shear that fixes first-head query row 3, input coordinate 2 will yield 1490 parameters while maintaining at least 99% accuracy.
change: Reproduce the qualified MLP, normalization, twelfth query rotation, tenth value rotation, and two query shears, then remove one additional query coordinate with exact inverse-transpose key compensation.
mechanism: Third inverse-compensated first-head query–key shear
evidence_used: The 1491-parameter design achieved 99.97% accuracy after two successive query–key shears, while the alternative additional value-head gauge failed at 33.71%; continuing the successful exact GL symmetry is the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1490, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1490-parameter design with a shear that fixes first-head query row 3, input coordinate 3 will yield 1489 parameters while maintaining at least 99% accuracy.
change: Remove query coordinate `(row 3, input 3)` by adding a multiple of row 0, apply the exact inverse-transpose compensation to key row 0, and reconstruct the removed coordinate as zero.
mechanism: Fourth inverse-compensated first-head query–key shear
evidence_used: The previous three successive first-head query–key shears preserved qualification, with the current 1490-parameter design reaching 99.92%; row 0 is already zero in coordinates 0–2, so this fourth shear fixes coordinate 3 without disturbing any earlier query anchors.
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
