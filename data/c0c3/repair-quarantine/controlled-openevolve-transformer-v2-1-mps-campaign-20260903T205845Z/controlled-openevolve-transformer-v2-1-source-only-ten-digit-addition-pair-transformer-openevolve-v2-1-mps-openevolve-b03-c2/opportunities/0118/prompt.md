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
verified_results: {"accuracy": 0.9979, "parameters": 1486, "training_steps": 4999}
prior_hypothesis: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}
prior_hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1485, "training_steps": 4999}
prior_hypothesis: Normalizing the remaining orthogonal first-head zero-bias query row with a max-pivot chart will reduce Reference Design 2 from 1486 to 1485 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Completing Reference Design 3’s verified normalization and applying the second head’s successful biased-query coordinate shear to the first head will reduce the model from 1486 to 1484 parameters while retaining at least 99% accuracy.
change: Normalize the first head’s orthogonal zero-bias neighbor, then omit one coordinate from its first biased query row and transfer the initialization shear into the paired key row.
mechanism: Max-pivot first-head biased-query shear atop the qualified orthogonal frame
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters, and Reference Design 1 achieved 99.99% with the analogous max-pivot shear between two biased queries. Unlike failed orthogonality and affine-normalization attempts, this adds only an exact coordinate Q/K basis shear.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7866, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the final two MLP hidden biases will reduce Reference Design 3 from 1485 to 1484 learned parameters while retaining at least 99% accuracy.
change: Reproduce Reference Design 3’s verified first-head normalized orthogonal query frame, then represent the final two `fc1` biases with one shared learned scalar.
mechanism: Shared terminal MLP hidden bias atop the normalized orthogonal-query frame
evidence_used: Reference Design 3 achieved 99.92% accuracy at 1485 parameters. Recent 1484 failures altered attention query/key or value/output geometry, so preserving that qualified attention design while testing a minimal, previously untested MLP sharing constraint is the most distinct next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7003, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing Reference Design 3 and tying two adjacent learned final-LayerNorm gains will reduce the verified 1485-parameter model to 1484 parameters while retaining at least 99% accuracy.
change: Add Reference Design 3’s complete first-head normalized orthogonal query frame and second-head construction, then represent two terminal LayerNorm gain coordinates with one shared learned scalar.
mechanism: Shared terminal LayerNorm gain atop the normalized orthogonal-query design
evidence_used: Reference Design 3 achieved 99.92% accuracy at 1485 parameters. Recent 1484 failures constrained attention geometry or MLP biases; sharing a terminal readout gain is a distinct, previously untested reduction that preserves the qualified attention and MLP parameterizations.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9891, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying two nonadjacent learned final-LayerNorm gains will reduce the model from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Store six learned final-LayerNorm gains, reuse the first gain for coordinate six, and retain the fixed final gain.
mechanism: Nonadjacent terminal LayerNorm gain sharing
evidence_used: Adjacent final-LayerNorm gain sharing reached 98.91%, only 0.09 percentage points below the requirement, while the unconstrained 1485-parameter design reached 99.92%; testing a distinct nonadjacent pair is the closest informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9517, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Completing Reference Design 3’s first-head normalization and removing only the feature-uniform final-LayerNorm bias direction with an orthonormal Householder chart will reduce the model from 1486 to 1484 parameters while retaining at least 99% accuracy.
change: Normalize the remaining first-head zero-bias query row and transfer its scale into the paired key row, then store the final LayerNorm bias in a seven-dimensional orthonormal centered chart.
mechanism: Orthogonal centered final-bias chart atop the normalized first-head query frame
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters. Final-LayerNorm gain sharing reached 98.91%, suggesting the terminal affine map can nearly tolerate another constraint; centering its bias preserves all seven relative bias directions and leaves the learned gains unrestricted.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7394, "parameters": 1484, "training_steps": 4999}

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
