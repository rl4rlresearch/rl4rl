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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1489, "training_steps": 4999}
prior_hypothesis: Shearing the untouched second-head bias-bearing query row against the already-successful dual-sheared bias-bearing row will reduce Reference Design 3 from 1490 to 1489 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9979, "parameters": 1486, "training_steps": 4999}
prior_hypothesis: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995, "parameters": 1491, "training_steps": 4999}
prior_hypothesis: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1485, "training_steps": 4999}
prior_hypothesis: Normalizing the remaining orthogonal first-head zero-bias query row with a max-pivot chart will reduce Reference Design 2 from 1486 to 1485 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing the first head’s two freely biased query coordinates to share one learned bias will reduce Reference Design 3 from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Reproduce the verified normalized orthogonal first-head query frame, then remove its unused antisymmetric biased-query direction by tying the two leading query biases.
mechanism: Balanced first-head query-bias direction gauge
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters; prior 1484 attempts constrained query or value/output weight geometry, whereas this untested gauge leaves those learned row directions unrestricted.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7049, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the already-sheared second-head anchor query and transferring its scale into the matching key row will reduce Reference Design 3 from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Reproduce the verified 1485-parameter normalized orthogonal-query design, then represent the second-head anchor row with a fixed-norm max-pivot chart and absorb its initialization scale into the corresponding key row.
mechanism: Max-pivot normalization of the sheared second-head anchor query
evidence_used: Reference Design 3 achieved 99.92% after max-pivot normalization of a sheared query row. The failed 1484 key-row normalization used the opposite query/key orientation, so query-side normalization of the already-qualified anchor shear is the closest distinct untested gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4139, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Projecting the first head’s second biased query row orthogonally to the first in joint weight-and-bias space will reduce the verified model from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Replace one biased-query weight coordinate with a stable projection chart that preserves both learned biases, and absorb the initialization shear into the corresponding key row.
mechanism: Augmented-affine biased-query orthogonality gauge
evidence_used: The weight-only biased-pair constraint reached 80.47% and tying the biases reached 70.49%, indicating that preserving the affine bias degree of freedom is important; the current unconstrained affine design reaches 99.92%.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Projecting the first head’s second biased query row orthogonally to the first in joint relative-weight-and-bias space will reduce the verified normalized-frame design to 1484 parameters while retaining at least 99% accuracy.
change: Complete the qualified normalization of the first-head zero-bias frame, then omit one weight coordinate from the second biased query row, reconstruct it from an affine orthogonality constraint, and apply the inverse initialization shear to its paired key row.
mechanism: Max-pivot augmented-affine biased-query gauge
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters; the weight-only biased-pair constraint achieved 80.47% and tying biases achieved 70.49%, motivating a gauge that preserves both independent biases. The prior augmented-affine attempt was not verifiable, so a max-weight-pivot implementation is the closest informative retry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9262999999999999, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding one bias-preserving shear from the first head’s second biased query row into its normalized zero-bias target will reduce Reference Design 3 from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Reproduce the qualified normalized orthogonal first-head query frame, then omit the biased row’s target-pivot coordinate, reconstruct it as zero, and transfer the inverse initialization shear into the matching key row.
mechanism: Max-pivot first-head biased-to-zero-bias query/key shear
evidence_used: Reference Design 3 reached 99.92% at 1485 parameters, while analogous biased-to-zero-bias shears already succeed in the second head. Unlike the failed first-head biased-pair constraints, this transformation preserves both query biases and uses the proven max-pivot normalized zero-bias row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.76, "parameters": 1484, "training_steps": 4999}

RECENT RESULT
hypothesis: Normalizing the twice-sheared second-head query in joint weight-and-bias space will reduce Reference Design 3 from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Reproduce the verified normalized orthogonal-query design, then encode the second head’s biased shear row as a fixed-norm augmented affine chart and transfer its omitted scale into the matching key row.
mechanism: Joint affine query/key scale quotient
evidence_used: Reference Design 3 achieved 99.92% at 1485 parameters. Earlier 1484 attempts imposed additional geometric restrictions or normalized only weights; this change preserves the query bias and removes only an exact diagonal query/key scale redundancy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Normalizing one first-head biased query in joint relative-weight-and-bias space and transferring its scale to the matching key will reduce the verified model from 1485 to 1484 parameters while retaining at least 99% accuracy.
change: Store the first biased query as a max-pivot fixed-norm affine chart, remove its bias from `qkv.bias`, omit its weight row from `qkv.weight`, and absorb its initialization scale into the paired key row.
mechanism: First-head joint affine query/key scale quotient
evidence_used: The current normalized-frame model achieved 99.92% at 1485 parameters. Tested 1484 changes that constrained weight geometry or tied biases failed, while augmented-affine orthogonality reached 92.63%; this tests the less restrictive exact scale redundancy while preserving both biased-query directions and independent biases.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.36700000000000005, "parameters": 1484, "training_steps": 4999}

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
