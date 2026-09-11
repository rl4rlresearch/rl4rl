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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1447, "training_steps": 4999}
prior_hypothesis: Replacing the 184-parameter absolute-position table with 44 learned head-specific causal lag biases will retain at least 99% accuracy while reducing the model from 1587 to 1447 parameters, because the two attention heads can directly specialize to the previous-output and aligned-input offsets without consuming residual-stream dimensions.

## Recent verification evidence

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the seventh `fc2` input column will reduce the model from 1597 to 1596 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the first remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and include its coordinates in the ambient-state optimizer.
mechanism: Ambient-state AdamW for a ninth quotiented MLP channel
evidence_used: The adjacent sixth-column extension achieved 100% accuracy at 1597 parameters, while ambient AdamW has enabled every tested initialization-preserving quotient from the third through sixth columns to meet the threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the independent 8-parameter lookup at every position with generic multiscale Fourier features and four learned frequency gains will retain at least 99% accuracy while reducing parameters by `8 * max_seq_len - 4`, because attention needs a consistent positional geometry more than unrelated vectors for every position.
change: Preserve the verified transformer and MLP widths, but replace the learned absolute-position table with a compact shared Fourier encoder; preserve downstream seeded initialization draws for a clean comparison.
mechanism: Shared multiscale Fourier positional representation
evidence_used: Narrowing the MLP and adding recurrent refinement collapsed to 24.97%, while width-preserving reductions repeatedly retained at least 99%; this motivates preserving load-bearing channel capacity and challenging the shared assumption that every sequence position needs an independent learned vector.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.14400000000000002, "parameters": 1416, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the eighth `fc2` input column will reduce the model from 1596 to 1595 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the first remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and include its coordinates in the ambient-state optimizer.
mechanism: Ambient-state AdamW for a tenth quotiented MLP channel
evidence_used: The adjacent seventh-column extension achieved 99.95% accuracy at 1596 parameters, while ambient AdamW has enabled every sequential initialization-preserving quotient from the third through seventh columns to meet the accuracy threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the ninth `fc2` input column will reduce the model from 1595 to 1594 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the first remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and include its coordinates in the ambient-state optimizer.
mechanism: Ambient-state AdamW for an eleventh quotiented MLP channel
evidence_used: The adjacent eighth-column extension achieved 99.98% accuracy at 1595 parameters, while ambient AdamW has enabled every sequential initialization-preserving quotient from the third through eighth columns to meet the accuracy threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the tenth `fc2` input column will reduce the model from 1594 to 1593 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Replace the sole remaining dense `fc2` column with a Householder zero-mean parameterization, preserve its conceptual dense initialization, and add it to the ambient-state optimizer.
mechanism: Ambient-state AdamW for the final dense MLP channel
evidence_used: The ninth-column extension achieved 100% accuracy at 1594 parameters, and ambient AdamW has enabled every sequential initialization-preserving quotient from the third through ninth columns to meet the accuracy threshold; the adjacent tenth column is the closest supported continuation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the first attention output-projection column will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Replace one dense attention projection column with a Householder zero-mean parameterization, preserve the conceptual dense initialization and random-number stream, and include its coordinates in the ambient-state optimizer.
mechanism: Ambient-state quotient of an attention output-projection column
evidence_used: Ambient AdamW enabled every sequential initialization-preserving `fc2` column quotient through the final dense column to meet the threshold, with the latest 1593-parameter design reaching 100%; the attention output projection has the same final-LayerNorm shift symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the second attention output-projection column will reduce the model from 1592 to 1591 parameters while retaining at least 99% accuracy when both quotiented columns are optimized with eight-dimensional ambient AdamW.
change: Replace the second dense attention projection column with an independent seven-coordinate Householder zero-mean parameterization, preserve conceptual dense initialization, and include it in the ambient-state optimizer.
mechanism: Ambient-state quotient of a second attention output-projection column
evidence_used: The immediately preceding first-column attention quotient achieved 100% accuracy at 1592 parameters, while the same symmetry and ambient AdamW method successfully supported every tested `fc2` column quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the third attention output-projection column will reduce the model from 1591 to 1590 parameters while retaining at least 99% accuracy when all three quotiented columns use eight-dimensional ambient AdamW.
change: Replace the third dense attention projection column with a seven-coordinate Householder zero-mean parameterization, preserve conceptual dense initialization, and include its coordinates in the ambient optimizer.
mechanism: Ambient-state quotient of a third attention output-projection column
evidence_used: The immediately preceding second-column attention quotient achieved 100% accuracy at 1591 parameters, and the identical symmetry-preserving ambient optimization method has succeeded for every sequentially tested attention and `fc2` column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the fourth attention output-projection column will reduce the model from 1590 to 1589 parameters while retaining at least 99% accuracy when all four quotiented columns use eight-dimensional ambient AdamW.
change: Replace the fourth dense attention projection column with a seven-coordinate Householder zero-mean parameterization, preserve conceptual dense initialization and the random-number stream, and include its coordinates in the ambient optimizer.
mechanism: Ambient-state quotient of a fourth attention output-projection column
evidence_used: The immediately preceding third-column attention quotient achieved 100% accuracy at 1590 parameters, and the identical symmetry-preserving ambient optimization method has succeeded for every sequentially tested attention projection column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the fifth attention output-projection column will reduce the model from 1589 to 1588 parameters while retaining at least 99% accuracy when all five quotiented columns use eight-dimensional ambient AdamW.
change: Replace the fifth dense attention projection column with a seven-coordinate Householder zero-mean parameterization, preserve its conceptual initialization and random stream, and include its coordinates in the ambient optimizer.
mechanism: Ambient-state quotient of a fifth attention output-projection column
evidence_used: The immediately preceding fourth-column attention quotient achieved 100% accuracy at 1589 parameters, and the identical symmetry-preserving ambient optimization method has succeeded for every sequentially tested attention projection column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the sixth attention output-projection column will reduce the model from 1588 to 1587 parameters while retaining at least 99% accuracy when all six quotiented columns use eight-dimensional ambient AdamW.
change: Replace the sixth dense attention projection column with a seven-coordinate Householder zero-mean parameterization, preserve its conceptual initialization and random stream, and include its coordinates in the ambient optimizer.
mechanism: Ambient-state quotient of a sixth attention output-projection column
evidence_used: The immediately preceding fifth-column attention quotient achieved 99.99% accuracy at 1588 parameters, and the identical symmetry-preserving ambient optimization method has succeeded for every sequentially tested attention projection column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the 184-parameter absolute-position table with 44 learned head-specific causal lag biases will retain at least 99% accuracy while reducing the model from 1587 to 1447 parameters, because the two attention heads can directly specialize to the previous-output and aligned-input offsets without consuming residual-stream dimensions.
change: Remove absolute positional vectors, add a learned relative-lag bias to each attention head, fix the softmax-redundant zero-lag coordinate, and preserve the verified initialization random stream.
mechanism: Head-specific learned relative-lag attention
evidence_used: Fixed Fourier positions achieved only 14.4% accuracy, showing that generic positional features were inadequate, while width-preserving quotient changes consistently exceeded 99%. Learned lag biases test a different mechanism: task-relevant positional geometry is learned directly in attention logits while all eight content channels remain available.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1447, "training_steps": 4999}



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
