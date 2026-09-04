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

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance and reconstruct the newly removed distance from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: All nine adjacent cross-head ties from the fourteenth- through twenty-second-farthest distances preserved at least 99.84% accuracy; the latest achieved 99.91% with 1,107 parameters, while prior twenty-third-distance attempts did not actually produce a new verified implementation.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar for that distance, and insert it immediately before the existing twenty-second-distance scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: All nine adjacent cross-head ties from the fourteenth- through twenty-second-farthest distances met the accuracy requirement; the latest achieved 99.91% with 1,107 parameters, while prior twenty-third-distance attempts reproduced the unchanged implementation rather than testing the intended model.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar, and insert it immediately before the existing twenty-second-distance scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The adjacent twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters, while prior twenty-third-distance attempts yielded no negative accuracy result because they reproduced the old implementation or could not be verified.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Summing the two independently routed attention contexts before one shared learned output projection will reduce parameters from 1,107 to 1,079 while maintaining at least 99% accuracy, because the heads already share their value representation and decimal addition is symmetric in its operands.
change: Replace the separate learned output map for each attention head with one shared zero-mean map applied to their variance-preserving summed context, while preserving independent query/key routing and all relative-distance biases.
mechanism: Symmetric aggregation of head-routed values
evidence_used: The 1,107-parameter design achieved 99.91% accuracy while using a single shared value readout across both heads, showing that operand routes need not encode digits differently. The failed direct bilinear scorer and shared-key result indicate that head-specific routing is load-bearing, so this patch preserves both heads’ query/key systems and changes only the downstream interpretation of their already-shared semantic values.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1079, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar for the removed distance, and insert it immediately before the existing twenty-second-distance scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The adjacent twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters, completing nine consecutive successful cross-head distance ties; prior twenty-third-distance submissions produced no negative accuracy evidence because they were unverifiable or reproduced the unchanged implementation.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent cross-head ties through the twenty-second-farthest distance succeeded while preserving every routing distance.
change: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar for the removed twenty-third-farthest distance, and insert it immediately before the existing twenty-second-distance scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The adjacent twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters; previous twenty-third-distance submissions provided no negative accuracy evidence because they were unverifiable or reproduced the unchanged implementation.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar for the removed distance, and insert it immediately before the existing twenty-second-distance scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The adjacent twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters, completing nine consecutive successful cross-head distance ties; prior twenty-third-distance submissions yielded no negative accuracy evidence because they were unverifiable or reproduced the unchanged implementation.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Replacing the two head-specific biases at the twenty-third-farthest attention distance with one shared learned scalar will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the independent per-head relative-bias core by one distance, add one shared scalar for the removed distance, and prepend it to the existing sequence of shared distant biases.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: Sharing each adjacent distance from the fourteenth- through twenty-second-farthest preserved at least 99% accuracy; the latest verified design reached 99.91% with 1,107 parameters, while previous twenty-third-distance submissions produced no negative accuracy result because they were unverifiable or reproduced the unchanged model.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent cross-head distance ties through the twenty-second-farthest distance succeeded.
change: Remove the final column from the head-specific relative-bias core, replace it with one shared learned scalar, and insert that scalar immediately before the existing twenty-second-distance bias.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters; earlier twenty-third-distance submissions did not yield negative accuracy evidence because they reproduced the unchanged model or could not be verified.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing both heads’ twenty-third-farthest attention bias will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy, because all nine adjacent distance ties through the twenty-second-farthest distance succeeded.
change: Shorten the head-specific relative-bias core by one distance, add one shared learned scalar for the removed twenty-third-farthest distance, and insert it immediately before the existing twenty-second-distance scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters; prior twenty-third-distance submissions yielded no negative accuracy evidence because they reproduced the unchanged implementation or could not be verified.
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
