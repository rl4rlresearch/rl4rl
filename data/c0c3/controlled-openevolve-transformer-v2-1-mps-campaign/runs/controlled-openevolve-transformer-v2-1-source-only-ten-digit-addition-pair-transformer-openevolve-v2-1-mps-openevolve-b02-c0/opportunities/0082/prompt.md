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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9997, "parameters": 1584, "training_steps": 4999}
prior_hypothesis: Constraining a second `fc2` weight column to have zero mean will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy, because its removed common-mode output is canceled by downstream LayerNorms.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a sixth `ln2` bias coordinate to zero will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed constant contribution.
change: Store two rather than three `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final six coordinates as zeros.
mechanism: Incremental pre-MLP LayerNorm bias absorption
evidence_used: The three preceding identical `ln2` bias reductions achieved 99.97%, 99.95%, and 99.97% accuracy at 1,606, 1,605, and 1,604 parameters, making one further coordinate anchoring the closest controlled test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a seventh `ln2` bias coordinate to zero will reduce the model from 1,603 to 1,602 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed constant contribution.
change: Store one rather than two `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final seven coordinates as zeros.
mechanism: Incremental pre-MLP LayerNorm bias absorption
evidence_used: The four most recent identical `ln2` bias reductions retained 99.95%–99.97% accuracy, including 99.97% at 1,603 parameters after fixing six coordinates; extending the same constraint by one coordinate is the closest controlled test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final learned `ln2` bias coordinate will reduce the model from 1,602 to 1,601 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the LayerNorm bias contribution.
change: Make `BiasAnchoredLayerNorm` entirely bias-free by removing its final bias parameter and passing no additive bias to layer normalization.
mechanism: Complete pre-MLP LayerNorm bias absorption
evidence_used: Seven consecutive `ln2` bias-coordinate removals retained at least 99% accuracy, culminating in 99.90% at 1,602 parameters; removing the sole remaining coordinate is the closest controlled extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate to one will reduce the model from 1,601 to 1,600 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb that coordinate’s learned scaling.
change: Store seven learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the eighth as a fixed one.
mechanism: Incremental pre-MLP LayerNorm scale absorption
evidence_used: Removing all eight `ln2` bias coordinates retained 99.96% accuracy at 1,601 parameters; this tests the next affine redundancy incrementally, with the same downstream `fc1` providing exact multiplicative absorption through its corresponding weight column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln2` scale coordinate to one will reduce the model from 1,600 to 1,599 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb the removed scaling degree.
change: Store six learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final two coordinates as fixed ones.
mechanism: Incremental pre-MLP LayerNorm scale absorption
evidence_used: Fixing the first `ln2` scale coordinate to one retained 99.99% accuracy at 1,600 parameters; extending the identical constraint by one coordinate is the closest controlled test of this multiplicative redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln2` scale coordinate to one will reduce the model from 1,599 to 1,598 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb the removed scaling degree.
change: Store five learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final three coordinates as fixed ones.
mechanism: Incremental pre-MLP LayerNorm scale absorption
evidence_used: Fixing the second `ln2` scale coordinate retained 99.96% accuracy at 1,599 parameters; extending the identical constraint by one coordinate is the closest controlled test of this multiplicative redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fourth `ln2` scale coordinate to one will reduce the model from 1,598 to 1,597 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb the removed scaling degree.
change: Store four learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final four coordinates as fixed ones.
mechanism: Incremental pre-MLP LayerNorm scale absorption
evidence_used: Fixing the third `ln2` scale coordinate retained 99.94% accuracy at 1,598 parameters, after the preceding two identical reductions achieved 99.99% and 99.96%; extending the same constraint by one coordinate is the closest controlled test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9926999999999999, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fifth `ln2` scale coordinate to one will reduce the model from 1,597 to 1,596 parameters while retaining at least 99% accuracy, because the corresponding unrestricted downstream `fc1` weight column can absorb the removed scaling degree.
change: Store three learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final five coordinates as fixed ones.
mechanism: Incremental pre-MLP LayerNorm scale absorption
evidence_used: Fixing the fourth `ln2` scale coordinate retained 99.27% accuracy at 1,597 parameters, while the three preceding identical reductions retained 99.94%–99.99%; one further coordinate is the closest controlled test of this established redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6893, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final `fc1` weight in every output row to zero will reduce the model from 1,597 to 1,585 parameters while retaining at least 99% accuracy, because bias-free `ln2` produces normalized activations whose pre-scale coordinates sum to zero and whose final scale is fixed at one.
change: Replace `fc1` with a compact linear layer storing seven rather than eight weights per output, reconstruct its final column as zero, and initialize it to the same function distribution as a full linear layer on zero-mean inputs.
mechanism: Zero-mean LayerNorm input gauge fixing
evidence_used: The 1,597-parameter design still achieved 99.27%, while a fifth fixed `ln2` scale collapsed to 68.93%; this motivates preserving the four-scale design and removing an independent exact downstream linear redundancy instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1585, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining a fifth attention projection weight column to have zero mean will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy, because any common-mode attention output is removed by the downstream LayerNorms.
change: Extend `AttentionGaugeLinear`’s orthogonal zero-mean parameterization from its first four weight columns to its first five.
mechanism: Incremental attention-output common-mode gauge fixing
evidence_used: The 1,585-parameter design achieved 99.93% accuracy while already applying this gauge to four attention projection columns; extending the same constraint by one column is the closest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5546, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining a second `fc2` weight column to have zero mean will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy, because its removed common-mode output is canceled by downstream LayerNorms.
change: Extend `OutputAnchoredLinear`’s orthogonal zero-mean parameterization from its first weight column to its first two columns.
mechanism: Incremental terminal-MLP common-mode gauge fixing
evidence_used: The verified 1,585-parameter design achieved 99.93% accuracy while already constraining the first `fc2` column; the failed fifth attention-column constraint motivates testing the next independent downstream gauge without further restricting attention.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining a third `fc2` weight column to have zero mean will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because downstream LayerNorm removes the eliminated common-mode output.
change: Extend `OutputAnchoredLinear`’s zero-mean parameterization from its first two weight columns to its first three.
mechanism: Incremental terminal-MLP common-mode gauge fixing
evidence_used: The immediately preceding identical extension from one to two constrained `fc2` columns achieved 99.97% accuracy at 1,584 parameters, making a third column the closest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7345999999999999, "parameters": 1583, "training_steps": 4999}



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
