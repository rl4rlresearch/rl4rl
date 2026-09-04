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
verified_results: {"accuracy": 0.9992, "parameters": 1581, "training_steps": 4999}
prior_hypothesis: Fixing a second `ln1` bias coordinate to zero will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because query shifts can be absorbed by the learned query bias, key shifts cancel in attention softmax, and value shifts can be absorbed by the attention projection bias up to its irrelevant common mode.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Constraining one additional positional-embedding row to zero mean will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because a position-specific common-mode shift is ignored by every LayerNorm and removed before logits by the final LayerNorm.
change: Extend `PositionAnchoredEmbedding`’s orthogonal zero-mean parameterization from six positional rows to seven, including matching compact initialization and reconstruction.
mechanism: Per-position residual common-mode gauge fixing
evidence_used: The verified 1,584-parameter design achieved 99.97% accuracy with six positional rows already using this gauge, while extending `fc2` to a third constrained column collapsed accuracy to 73.46%; this tests an independent structural redundancy without further restricting `fc2`.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining an eighth positional-embedding row to zero mean will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy, because its position-specific common-mode shift is ignored by pre-LayerNorm sublayers and removed by the final LayerNorm.
change: Extend `PositionAnchoredEmbedding`’s orthogonal zero-mean parameterization backward by one additional positional row, including matching compact initialization and reconstruction.
mechanism: Per-position residual common-mode gauge fixing
evidence_used: The immediately preceding identical extension from six to seven constrained positional rows achieved 99.83% accuracy at 1,583 parameters, making an eighth row the closest controlled reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining a ninth positional-embedding row to zero mean will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because its position-specific common-mode shift is removed by every LayerNorm without changing the learned computation.
change: Extend `PositionAnchoredEmbedding`’s orthogonal zero-mean parameterization backward by one additional positional row, including compact reconstruction and initialization.
mechanism: Incremental per-position residual common-mode gauge fixing
evidence_used: Extending the same positional gauge from seven to eight rows achieved 99.91% accuracy at 1,582 parameters, making a ninth row the closest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5483, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln1` bias coordinate to zero will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because query shifts can be absorbed by the learned query bias, key shifts cancel in attention softmax, and value shifts can be absorbed by the attention projection bias up to its irrelevant common mode.
change: Store six rather than seven learned `AnchoredLayerNorm` bias coordinates and reconstruct its final two coordinates as zeros.
mechanism: Incremental pre-attention LayerNorm bias absorption
evidence_used: Removing every `ln2` bias coordinate retained 99.96% accuracy, and the current 1,582-parameter design already fixes one `ln1` bias coordinate while achieving 99.91%; extending this affine redundancy by one coordinate is a controlled alternative to the positional, `fc2`, and attention-weight gauges whose next adjacent constraints failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third `ln1` bias coordinate to zero will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because query shifts remain absorbable by query bias, key shifts cancel in attention softmax, and value shifts remain absorbable by the attention projection bias up to its irrelevant common mode.
change: Store five rather than six learned `AnchoredLayerNorm` bias coordinates and reconstruct its final three coordinates as zeros.
mechanism: Incremental pre-attention LayerNorm bias absorption
evidence_used: The immediately preceding identical extension from one to two fixed `ln1` bias coordinates achieved 99.92% accuracy at 1,581 parameters, making a third coordinate the closest controlled reduction after adjacent positional, `fc2`, and attention-weight constraints failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5136999999999999, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring a second coordinate of the reference token embedding will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because translating that coordinate for every token can be absorbed by the positional embeddings, while its output-logit effect is a class-independent softmax shift.
change: Store two fewer token-embedding scalars and reconstruct the final two coordinates as zeros.
mechanism: Joint token–position translation gauge
evidence_used: The current design achieves 99.92% accuracy with one reference-token coordinate already anchored; this tests an independent exact token–position gauge after the adjacent third `ln1` bias and ninth positional-row constraints failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9604, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final input weight of one query projection row will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because LayerNorm inputs lie on an affine hyperplane and the learned query bias can absorb the eliminated constant term.
change: Store seven weights for the first query row, reconstruct its final weight as zero, and initialize the compact row to preserve the function of a full projection on initially zero-mean LayerNorm outputs.
mechanism: Normalized-input query-weight gauge fixing
evidence_used: The analogous `NormalizedInputLinear` constraint removed the final input weight from all 12 `fc1` rows while achieving 99.93% accuracy at 1,585 parameters; applying it incrementally to one query row tests an independent redundancy after adjacent embedding and `ln1`-bias reductions failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5063, "parameters": 1580, "training_steps": 4999}



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
