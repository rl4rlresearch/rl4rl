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
verified_results: {"accuracy": 0.9995, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}
prior_hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9979, "parameters": 1486, "training_steps": 4999}
prior_hypothesis: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1485, "training_steps": 4999}
prior_hypothesis: Normalizing the remaining orthogonal first-head zero-bias query row with a max-pivot chart will reduce Reference Design 2 from 1486 to 1485 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Extending Reference Design 3’s verified 1485-parameter normalized query frames with the current design’s successful bias-preserving shear will remove one additional parameter while retaining at least 99% accuracy.
change: Port the verified first- and second-head query gauges, then shear the first head’s final bias-bearing query row against its normalized zero-bias target row, yielding 1484 parameters.
mechanism: First-head bias-preserving query/key shear
evidence_used: Reference Design 3 reached 0.9992 accuracy with 1485 parameters, while the current design reached 0.9995 using the same one-coordinate bias-preserving shear mechanism in the second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6739, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the anchor query row at the normalized target row’s stable pivot will reduce the model from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Shear the second head’s anchor query row against its zero-bias normalized target row, omit the resulting fixed coordinate, and apply the inverse shear to the corresponding key row.
mechanism: Second-head bias-preserving anchor-to-target query/key shear
evidence_used: The current 1485-parameter design achieved 0.9992 accuracy and already uses the second head’s max-pivot target shear successfully; the failed 1484 attempt instead modified the first-head frame, motivating an independent reduction within the proven second-head construction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9068, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying Reference Design 3’s verified neighbor normalization and shearing the first head’s anchor biased row against its freely biased neighbor will yield 1484 parameters while retaining at least 99% accuracy.
change: Normalize the first head’s orthogonal zero-bias neighbor and omit one coordinate from a max-pivot shear between its two bias-bearing query rows, with inverse transformations applied to the corresponding key rows.
mechanism: First-head biased-row shear plus normalized zero-bias frame
evidence_used: Reference Design 3 achieved 0.9992 accuracy at 1485 parameters using the normalized first-head frame and an analogous biased-anchor shear in the second head; the failed 1484 attempts instead sheared biased rows against normalized zero-bias targets.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7341, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
