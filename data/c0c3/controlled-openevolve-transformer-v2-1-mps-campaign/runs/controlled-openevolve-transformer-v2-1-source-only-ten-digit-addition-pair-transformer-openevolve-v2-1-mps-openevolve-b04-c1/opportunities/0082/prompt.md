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
verified_results: {"accuracy": 0.9984000000000001, "parameters": 964, "training_steps": 4999}
prior_hypothesis: Quotienting the eleventh `qkv` weight row will reduce the model from 965 to 964 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes the row’s common-mode input weight unobservable and the same ambient AdamW treatment succeeded for the first ten rows.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Reducing the tied token representation from five to four learned channels will lower the model from 1085 to 971 parameters while retaining at least 99% accuracy, because the five-channel design achieved 99.96% accuracy and the full eight-dimensional attention and MLP computation remains unchanged.
change: Pad four-dimensional learned token embeddings with four fixed zero channels before transformer processing and use the same materialized vectors for tied output logits.
mechanism: Four-dimensional tied token bottleneck
evidence_used: The immediately preceding reduction from six learned token channels to five achieved 99.96% accuracy at 1085 parameters, so the adjacent four-channel bottleneck is the most directly supported test of further token-representation redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 971, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the tied token representation from four to three learned channels will lower the 971-parameter model by one parameter per vocabulary token while retaining at least 99% accuracy, because the four-channel bottleneck achieved 99.83% accuracy and the full eight-dimensional attention and MLP computation remains unchanged.
change: Pad three-dimensional learned token embeddings with five fixed zero channels before transformer processing and use the same materialized vectors for tied output logits.
mechanism: Three-dimensional tied token bottleneck
evidence_used: The immediately preceding reduction from five learned token channels to four achieved 99.83% accuracy at 971 parameters, so the adjacent three-channel bottleneck is the most direct test of remaining token-representation redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9691, "parameters": 857, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the fifth `qkv` weight row will reduce the model from 971 to 970 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first four rows.
change: Add a seven-coordinate Householder parameterization for the fifth `qkv` row, preserve conceptual initialization, and optimize it in eight-dimensional ambient space.
mechanism: Fifth-row QKV input-shift quotient
evidence_used: The fourth-row quotient retained 99.93% accuracy, while the current four-channel token bottleneck retains 99.83%; this applies the same proven exact symmetry without narrowing the failed token representation further.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 970, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the sixth `qkv` weight row will reduce the model from 970 to 969 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first five rows.
change: Add a seven-coordinate Householder parameterization for the sixth `qkv` row, preserve conceptual initialization, and optimize it in eight-dimensional ambient space.
mechanism: Sixth-row QKV input-shift quotient
evidence_used: The fifth-row quotient retained 99.84% accuracy at 970 parameters; extending the identical exact symmetry to the adjacent row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 969, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the seventh `qkv` weight row will reduce the model from 969 to 968 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first six rows.
change: Add a seven-coordinate Householder parameterization for the seventh `qkv` row, preserve conceptual initialization and random-stream order, and optimize all seven quotiented rows in eight-dimensional ambient space.
mechanism: Seventh-row QKV input-shift quotient
evidence_used: The immediately preceding sixth-row quotient achieved 99.84% accuracy at 969 parameters; extending the same exact symmetry and optimizer treatment to the adjacent seventh row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 968, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the eighth `qkv` weight row will reduce the model from 968 to 967 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first seven rows.
change: Add a seven-coordinate Householder parameterization for the eighth `qkv` row, preserve conceptual initialization and random-stream order, and optimize all eight quotiented rows in eight-dimensional ambient space.
mechanism: Eighth-row QKV input-shift quotient
evidence_used: The immediately preceding seventh-row quotient achieved 99.83% accuracy at 968 parameters; extending the same exact symmetry and optimizer treatment to the adjacent eighth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 967, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the ninth `qkv` weight row will reduce the model from 967 to 966 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes every row’s common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first eight rows.
change: Add a seven-coordinate Householder parameterization for the ninth `qkv` row, preserve conceptual initialization and random-stream order, and include the new coordinates in the ambient optimizer.
mechanism: Ninth-row QKV input-shift quotient
evidence_used: The immediately preceding eighth-row quotient achieved 99.83% accuracy at 967 parameters; extending the same exact symmetry and optimizer treatment to the adjacent ninth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 966, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the tenth `qkv` weight row will reduce the model from 966 to 965 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes every row’s common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first nine rows.
change: Add a seven-coordinate Householder parameterization for the tenth `qkv` row, preserve conceptual initialization and random-stream order, and include the new coordinates in the ambient optimizer.
mechanism: Tenth-row QKV input-shift quotient
evidence_used: The immediately preceding ninth-row quotient achieved 99.83% accuracy at 966 parameters; extending the same exact symmetry and optimizer treatment to the adjacent tenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 965, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the 12 learned MLP feature directions as output directions, with one learned gain per feature, will reduce the model from 965 to 893 parameters while retaining at least 99% accuracy because attention and the load-bearing four-channel token representation remain unchanged.
change: Replace the independent 84-parameter MLP output matrix with the transpose of `fc1` and 12 learned channel gains, preserve the verified initialization random stream, and remove obsolete MLP gauge-optimizer entries.
mechanism: Tied feature-dictionary feed-forward readout
evidence_used: The four-channel token bottleneck retained 99.84% accuracy while three channels fell to 96.91%, motivating preservation of that interface while challenging the separate assumption that the MLP requires independently learned analysis and synthesis matrices.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3701, "parameters": 893, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the eleventh `qkv` weight row will reduce the model from 965 to 964 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes the row’s common-mode input weight unobservable and the same ambient AdamW treatment succeeded for the first ten rows.
change: Add a seven-coordinate Householder parameterization for the eleventh `qkv` row, preserve conceptual initialization and random-stream order, and include its coordinates in the ambient optimizer.
mechanism: Eleventh-row QKV input-shift quotient
evidence_used: Quotienting the tenth row retained 99.84% accuracy at 965 parameters; extending the identical exact symmetry to the adjacent eleventh row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 964, "training_steps": 4999}



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
