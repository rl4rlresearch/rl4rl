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
verified_results: {"accuracy": 0.9991, "parameters": 1107, "training_steps": 4999}
prior_hypothesis: Tying both heads’ twenty-second-farthest attention biases will reduce the model from 1,108 to 1,107 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through twenty-first-farthest ties all succeeded without removing any routing distance.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying both heads’ sixteenth-farthest attention biases will reduce the model from 1,114 to 1,113 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- and fifteenth-farthest cross-head ties achieved 99.92% and 99.97% without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the sixteenth-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head sixteenth-distance bias tying
evidence_used: Cross-head tying at the fifteenth-farthest distance reduced parameters from 1,115 to 1,114 while improving verified accuracy to 99.97%; the immediately adjacent sixteenth-farthest distance is the smallest informative continuation of that successful compression pattern.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1113, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ seventeenth-farthest attention biases will reduce the model from 1,113 to 1,112 parameters while retaining at least 99% accuracy, because the adjacent fourteenth-, fifteenth-, and sixteenth-farthest ties all preserved at least 99.92% accuracy without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the seventeenth-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head seventeenth-distance bias tying
evidence_used: The immediately preceding sixteenth-farthest cross-head tie reduced parameters from 1,114 to 1,113 and achieved 99.93% accuracy, extending a three-step sequence of successful adjacent ties.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1112, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ eighteenth-farthest attention biases will reduce the model from 1,112 to 1,111 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through seventeenth-farthest ties all preserved at least 99.84% accuracy without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the eighteenth-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head eighteenth-distance bias tying
evidence_used: The immediately preceding seventeenth-farthest cross-head tie reduced parameters from 1,113 to 1,112 and achieved 99.84% accuracy, extending four consecutive successful adjacent bias ties.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1111, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ nineteenth-farthest attention biases will reduce the model from 1,111 to 1,110 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through eighteenth-farthest ties all succeeded without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the nineteenth-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head nineteenth-distance bias tying
evidence_used: The immediately preceding eighteenth-farthest cross-head tie reduced parameters from 1,112 to 1,111 and achieved 99.92% accuracy, extending five consecutive successful adjacent bias ties.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1110, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ twentieth-farthest attention biases will reduce the model from 1,110 to 1,109 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through nineteenth-farthest ties all succeeded without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the twentieth-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head twentieth-distance bias tying
evidence_used: The immediately preceding nineteenth-farthest cross-head tie reduced parameters from 1,111 to 1,110 and achieved 99.93% accuracy, extending six consecutive successful adjacent bias ties.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1109, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing each head’s separate query/key factors with its observable full-rank affine bilinear score operator will reduce parameters from 1,109 to 1,101 while retaining at least 99% accuracy, because it exactly preserves initialized attention scores, keeps head-specific routing, and enlarges rather than restricts the learned score-function class.
change: Compose initialized query, key, and query-bias parameters into independent learned 8-by-7 score operators per head; compute causal attention directly in the normalized seven-dimensional residual subspace while leaving values and all other computation unchanged.
mechanism: Direct full-rank affine bilinear attention scoring
evidence_used: The 1,109-parameter model achieved 99.93%, whereas shared keys collapsed to 56.39% and an earlier restrictive query/key refactor reached 90.12%. This indicates that expressive head-specific score maps are load-bearing; the proposed composite operators preserve them without retaining redundant internal query/key coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.30329999999999996, "parameters": 1101, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ twenty-first-farthest attention biases will reduce the model from 1,109 to 1,108 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through twentieth-farthest ties all succeeded without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-first-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head twenty-first-distance bias tying
evidence_used: The immediately preceding twentieth-farthest cross-head tie reduced parameters from 1,110 to 1,109 and achieved 99.93% accuracy, extending seven consecutive successful adjacent bias ties; the failed direct bilinear refactor shows that preserving the existing attention score structure is safer than compressing its query/key parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1108, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ twenty-second-farthest attention biases will reduce the model from 1,108 to 1,107 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through twenty-first-farthest ties all succeeded without removing any routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-second-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head twenty-second-distance bias tying
evidence_used: The immediately preceding twenty-first-farthest tie reduced parameters from 1,109 to 1,108 and achieved 99.96% accuracy, extending eight consecutive successful adjacent cross-head bias ties.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1107, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ twenty-third-farthest attention biases will reduce the model from 1,107 to 1,106 parameters while retaining at least 99% accuracy, because every adjacent cross-head tie from the fourteenth- through twenty-second-farthest distance preserved the accuracy requirement.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-third-farthest bias for both heads from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The immediately preceding twenty-second-farthest tie achieved 99.91% accuracy with 1,107 parameters, extending nine consecutive successful adjacent cross-head bias ties while preserving every routing distance.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Tying both heads’ twenty-third-farthest attention biases will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent ties from the fourteenth- through twenty-second-farthest distances succeeded while preserving every routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-third-farthest bias from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The twenty-second-farthest tie achieved 99.91% accuracy with 1,107 parameters; the subsequent twenty-third-distance implementation was unverifiable, so a complete, directly adjacent implementation is the most informative next test.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent ties through the twenty-second-farthest distance succeeded and this preserves every routing distance.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-third-farthest bias from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The twenty-second-farthest tie achieved 99.91% accuracy with 1,107 parameters; prior twenty-third-distance attempts were unverifiable or reproduced the unchanged implementation, providing no negative accuracy evidence against the adjacent tie.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Tying both heads’ twenty-third-farthest attention biases will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent cross-head ties through the twenty-second-farthest distance succeeded while preserving every routing distance.
change: Shorten the independent per-head relative-bias core by one column and reconstruct the twenty-third-farthest bias from one new shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The twenty-second-farthest tie achieved 99.91% accuracy with 1,107 parameters; previous twenty-third-distance submissions were unverifiable or reproduced the unchanged implementation, so correctly inserting the missing shared scalar is the most informative adjacent test.
result: the edit reproduced a previously verified implementation



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
