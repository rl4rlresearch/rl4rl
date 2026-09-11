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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1085, "training_steps": 4999}
prior_hypothesis: Reducing the tied token representation from six to five learned channels will lower the 1199-parameter model by one parameter per vocabulary token while retaining at least 99% accuracy, because the six-channel bottleneck achieved 99.99% accuracy with substantial margin and the eight-dimensional attention and MLP remain unchanged.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the 184-parameter absolute-position table with 44 learned head-specific causal lag biases will retain at least 99% accuracy while reducing the model from 1587 to 1447 parameters, because the two attention heads can directly specialize to the previous-output and aligned-input offsets without consuming residual-stream dimensions.
change: Remove absolute positional vectors, add a learned relative-lag bias to each attention head, fix the softmax-redundant zero-lag coordinate, and preserve the verified initialization random stream.
mechanism: Head-specific learned relative-lag attention
evidence_used: Fixed Fourier positions achieved only 14.4% accuracy, showing that generic positional features were inadequate, while width-preserving quotient changes consistently exceeded 99%. Learned lag biases test a different mechanism: task-relevant positional geometry is learned directly in attention logits while all eight content channels remain available.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1447, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the seventh attention output-projection column will reduce the current model from 1447 to 1446 parameters while retaining at least 99% accuracy.
change: Replace the seventh dense attention projection column with a seven-coordinate Householder zero-mean parameterization, preserve conceptual initialization, and optimize it with eight-dimensional ambient AdamW.
mechanism: Ambient-state quotient of the seventh attention output-projection column
evidence_used: Sequential quotienting of the first six attention projection columns retained 99.99–100% accuracy, and the subsequent relative-lag design retained 99.96%; extending the identical symmetry-preserving method to the adjacent seventh column is the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1446, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the eighth attention output-projection column will reduce the model from 1446 to 1445 parameters while retaining at least 99% accuracy.
change: Replace the remaining dense projection column with seven Householder zero-mean coordinates, preserve conceptual initialization, and optimize it using eight-dimensional ambient AdamW.
mechanism: Ambient-state quotient of the final attention output-projection column
evidence_used: Quotienting the adjacent seventh projection column retained 99.92% accuracy, while the same initialization-preserving ambient optimization succeeded for every previously tested attention projection column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1445, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight learned `ln2` scale parameters will reduce the model from 1445 to 1437 parameters while retaining at least 99% accuracy, because `fc1` can absorb any diagonal LayerNorm scaling and initialization remains functionally identical.
change: Make the MLP-input LayerNorm non-affine, eliminating eight redundant trainable scales without changing the initial model function or random stream.
mechanism: Absorbable MLP LayerNorm scale elimination
evidence_used: The 1445-parameter design achieved 99.91% accuracy after extensive symmetry quotienting; unlike earlier one-dimensional gauges, all eight `ln2` scales are algebraically redundant with the immediately following dense `fc1` weights.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1437, "training_steps": 4999}

RECENT RESULT
hypothesis: Making `ln1` non-affine will reduce the model from 1437 to 1430 learned parameters while retaining at least 99% accuracy, because the following `qkv` projection can absorb any diagonal LayerNorm scaling and initialization remains functionally identical.
change: Remove the seven learned `ln1` scale coordinates by replacing `ScaleFixedLayerNorm` with non-affine LayerNorm.
mechanism: Absorbable attention LayerNorm scale elimination
evidence_used: The analogous removal of all eight `ln2` scales retained 99.93% accuracy at 1437 parameters; `ln1` has the same absorbable-scale relationship with its immediately following dense `qkv` projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948999999999999, "parameters": 1430, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing every `qkv` and `fc1` weight row with seven learned coordinates will reduce the model from 1430 to 1394 parameters while retaining at least 99% accuracy, because non-affine `ln1` and `ln2` produce zero-mean inputs, making all 36 common-mode row coordinates unobservable, while eight-dimensional ambient AdamW preserves the established optimization geometry.
change: Replace the dense `qkv` and `fc1` weights with initialization-preserving zero-mean row parameterizations and extend the ambient optimizer to support their batched gauge coordinates.
mechanism: Ambient row-space quotienting after non-affine LayerNorm
evidence_used: Removing `ln2` affine scales retained 99.93% accuracy and removing `ln1` scales retained 99.49%; those changes expose exact rowwise input-shift symmetries, while ambient AdamW has already preserved accuracy across every tested initialization-preserving projection quotient.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting one `qkv` weight row will reduce the model from 1430 to 1429 parameters while retaining at least 99% accuracy, because non-affine `ln1` produces zero-mean inputs and scalar ambient AdamW preserves the established optimization geometry.
change: Replace the first `qkv` weight row with seven Householder zero-mean coordinates, preserve the conceptual initialization and random stream, and add those coordinates to the existing ambient optimizer.
mechanism: Single-row input-shift quotient for QKV
evidence_used: The 1430-parameter model reached 99.49%, and scalar Householder quotients with ambient AdamW succeeded for every tested attention-projection column; isolating one exact `qkv` row symmetry avoids the unverified batched implementation used by the unsuccessful 36-row attempt.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1429, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the second `qkv` weight row will reduce the model from 1429 to 1428 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes each row’s common-mode input weight unobservable and independent ambient AdamW already preserved 99.94% accuracy for the first row.
change: Add a seven-coordinate Householder parameterization for the second `qkv` row, retain dense conceptual initialization and random-stream order, and optimize both quotiented rows in eight-dimensional ambient space.
mechanism: Second-row input-shift quotient for QKV
evidence_used: The immediately preceding single-row `qkv` quotient achieved 99.94% accuracy at 1429 parameters; applying the identical symmetry and optimizer treatment to the adjacent row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1428, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the third `qkv` weight row will reduce the model from 1428 to 1427 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and the identical ambient AdamW treatment succeeded for the first two rows.
change: Add a seven-coordinate Householder parameterization for the third `qkv` row, preserve conceptual initialization and random-stream order, and optimize all three quotiented rows in eight-dimensional ambient space.
mechanism: Third-row QKV input-shift quotient
evidence_used: The immediately preceding second-row quotient achieved 99.97% accuracy at 1428 parameters; extending the same exact symmetry and optimizer treatment to the adjacent third row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1427, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the fourth `qkv` weight row will reduce the model from 1427 to 1426 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes the row’s common-mode input weight unobservable and the identical ambient AdamW treatment achieved 100% accuracy for the first three rows.
change: Add a seven-coordinate Householder parameterization for the fourth `qkv` row, preserve conceptual initialization and random-stream order, and optimize all four quotiented rows in eight-dimensional ambient space.
mechanism: Fourth-row QKV input-shift quotient
evidence_used: The immediately preceding third-row quotient achieved 100% accuracy at 1427 parameters; extending the same exact symmetry and optimizer treatment to the adjacent fourth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1426, "training_steps": 4999}

RECENT RESULT
hypothesis: Restricting the tied token embeddings to six learned channels while retaining the eight-channel transformer will reduce the model from 1426 to 1199 parameters and retain at least 99% accuracy, because learned relative-lag attention already supplies positional structure separately and addition-token identity should not require every residual channel.
change: Replace each full-width token vector with a learned six-dimensional vector padded into the eight-dimensional residual stream, use the same compact vectors for logits, and preserve the verified initialization random stream by reconstructing and truncating the former conceptual initialization.
mechanism: Six-dimensional learned token bottleneck with eight-dimensional internal computation
evidence_used: The learned relative-lag design reached 99.96% after removing the 184-parameter position table, and the current model reaches 99.93%; this challenges the shared assumption that all eight residual channels must be independently learned for every token while leaving the successful eight-dimensional attention and MLP computation intact.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1199, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the tied token representation from six to five learned channels will lower the 1199-parameter model by one parameter per vocabulary token while retaining at least 99% accuracy, because the six-channel bottleneck achieved 99.99% accuracy with substantial margin and the eight-dimensional attention and MLP remain unchanged.
change: Pad five-dimensional learned token embeddings with three fixed zero channels before transformer processing and use the same materialized vectors for tied output logits.
mechanism: Five-dimensional tied token bottleneck
evidence_used: The immediately preceding reduction from eight learned token channels to six achieved 99.99% accuracy at 1199 parameters, making the adjacent five-channel bottleneck the most informative supported test of remaining token-representation redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1085, "training_steps": 4999}



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
