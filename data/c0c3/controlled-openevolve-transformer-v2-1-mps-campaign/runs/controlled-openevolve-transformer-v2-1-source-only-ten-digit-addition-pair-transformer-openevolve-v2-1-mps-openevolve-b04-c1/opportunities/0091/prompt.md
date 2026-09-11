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
verified_results: {"accuracy": 0.9992, "parameters": 895, "training_steps": 4999}
prior_hypothesis: Reducing each attention head’s learned query/key width from four dimensions to two will produce a 895-parameter transformer with at least 99% accuracy, because learned relative-lag biases can perform most positional routing while the unchanged four-dimensional value paths, token representation, and MLP retain the task’s load-bearing capacity.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Quotienting the twelfth `qkv` weight row will reduce the model from 964 to 963 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes the row’s common-mode input weight unobservable and the same ambient AdamW treatment succeeded for the first eleven rows.
change: Add a seven-coordinate Householder parameterization for the twelfth `qkv` row, preserve conceptual initialization and random-stream order, and include its coordinates in the ambient optimizer.
mechanism: Twelfth-row QKV input-shift quotient
evidence_used: Quotienting the eleventh row retained 99.84% accuracy at 964 parameters; extending the identical exact symmetry to the adjacent twelfth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 963, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the thirteenth `qkv` weight row will reduce the model from 963 to 962 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes the row’s common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first twelve rows.
change: Add a seven-coordinate Householder parameterization for the thirteenth `qkv` row, preserve conceptual initialization and random-stream order, and include its coordinates in the ambient optimizer.
mechanism: Thirteenth-row QKV input-shift quotient
evidence_used: Quotienting the twelfth row retained 99.82% accuracy at 963 parameters; extending the same exact symmetry to the adjacent thirteenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 962, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the fourteenth `qkv` weight row will reduce the model from 962 to 961 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes the row’s common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first thirteen rows.
change: Add a seven-coordinate Householder parameterization for the fourteenth `qkv` row, preserve conceptual initialization and random-stream order, and include its coordinates in the ambient optimizer.
mechanism: Fourteenth-row QKV input-shift quotient
evidence_used: Quotienting the thirteenth row retained 99.84% accuracy at 962 parameters; extending the same exact symmetry to the adjacent fourteenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 961, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the fifteenth `qkv` weight row will reduce the model from 961 to 960 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first fourteen rows.
change: Add a seven-coordinate Householder parameterization for the fifteenth `qkv` row, preserve conceptual initialization and random-stream order, and include its coordinates in the ambient optimizer.
mechanism: Fifteenth-row QKV input-shift quotient
evidence_used: Quotienting the fourteenth row retained 99.84% accuracy at 961 parameters; extending the same exact symmetry to the adjacent fifteenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 960, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the sixteenth `qkv` weight row will reduce the model from 960 to 959 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes its common-mode input weight unobservable and identical ambient AdamW treatment succeeded for the first fifteen rows.
change: Add a seven-coordinate Householder parameterization for the sixteenth `qkv` row, preserve conceptual initialization and random-stream order, and include its coordinates in the ambient optimizer.
mechanism: Sixteenth-row QKV input-shift quotient
evidence_used: Quotienting the fifteenth row retained 99.84% accuracy at 960 parameters; extending the same exact symmetry to the adjacent sixteenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 959, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the seventeenth `qkv` weight row will reduce the model from 959 to 958 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes that row’s common-mode input weight unobservable.
change: Add a seven-coordinate Householder parameterization for the seventeenth `qkv` row, preserve conceptual initialization and random-stream order, and optimize its coordinates in eight-dimensional ambient space.
mechanism: Seventeenth-row QKV input-shift quotient
evidence_used: Quotienting the sixteenth row retained 99.83% accuracy at 959 parameters; extending the identical exact symmetry and optimizer treatment to the adjacent seventeenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 958, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the eighteenth `qkv` weight row will reduce the model from 958 to 957 parameters while retaining at least 99% accuracy, because non-affine `ln1` makes that row’s common-mode input weight unobservable.
change: Add a seven-coordinate Householder parameterization for the eighteenth `qkv` row, preserve conceptual initialization and random-stream order, and optimize its coordinates in eight-dimensional ambient space.
mechanism: Eighteenth-row QKV input-shift quotient
evidence_used: Quotienting the seventeenth row retained 99.82% accuracy at 958 parameters; extending the identical exact symmetry and optimizer treatment to the adjacent eighteenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 957, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the nineteenth `qkv` weight row will reduce the model from 957 to 956 learned parameters while retaining at least 99% accuracy, because non-affine `ln1` makes that row’s common-mode input weight unobservable.
change: Add a seven-coordinate Householder parameterization for the nineteenth `qkv` row, preserve conceptual initialization and random-stream order, and optimize its coordinates in eight-dimensional ambient space.
mechanism: Nineteenth-row QKV input-shift quotient
evidence_used: Quotienting the eighteenth row retained 99.82% accuracy at 957 parameters; extending the identical exact symmetry and optimizer treatment to the adjacent nineteenth row is the closest supported incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 956, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing each attention head’s learned query/key width from four dimensions to two will produce a 895-parameter transformer with at least 99% accuracy, because learned relative-lag biases can perform most positional routing while the unchanged four-dimensional value paths, token representation, and MLP retain the task’s load-bearing capacity.
change: Replace the 24-output QKV projection with a 16-output projection containing two-dimensional queries and keys plus full-width values, preserving the conceptual initialization stream and input-shift gauge optimization.
mechanism: Narrow-content positional attention
evidence_used: The four-channel token bottleneck retained 99.84% accuracy while three channels reached only 96.91%, and tying the MLP readout collapsed accuracy to 37.01%; this motivates preserving token, value, and MLP capacity while testing whether full-width content addressing is unnecessary alongside the learned relative-lag table.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 895, "training_steps": 4999}



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
