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
verified_results: {"accuracy": 0.9995, "parameters": 1045, "training_steps": 4999}
prior_hypothesis: Removing one redundant shared-value scale coordinate will reduce the verified model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because both independent head gains remain learned and can absorb any common rescaling of the shared value map.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing the two head-specific biases at the twenty-third-farthest attention distance with one shared learned scalar will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one column, add a shared twenty-third-distance scalar, expand it across heads, and insert it before the existing twenty-second-distance bias.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The adjacent twenty-second-distance tie achieved 99.91% accuracy with 1,107 parameters, while prior twenty-third-distance submissions supplied no negative accuracy result because they reproduced the unchanged implementation or could not be verified.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Replacing the two head-specific biases at the twenty-third-farthest attention distance with one shared learned scalar will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.
change: Shorten the independent per-head relative-bias core by one column, add an explicitly used shared scalar for the removed distance, and prepend it to the existing shared-distance sequence.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: Sharing every adjacent distance from the fourteenth- through twenty-second-farthest retained at least 99% accuracy, with the twenty-second tie reaching 99.91% at 1,107 parameters; prior attempts at this next tie did not produce adverse accuracy evidence because they reproduced the existing implementation or were unverifiable.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest attention bias across both heads will reduce the model from 1,107 to 1,106 parameters while retaining at least 99% accuracy.
change: Pack the remaining head-specific bias core and one explicitly shared boundary scalar into a single parameter, then reconstruct them separately during attention.
mechanism: Packed cross-head twenty-third-distance bias tying
evidence_used: Nine consecutive adjacent cross-head distance ties through the twenty-second-farthest distance retained at least 99% accuracy, most recently reaching 99.91%; prior attempts at this next tie produced no adverse accuracy evidence because they reproduced the old implementation or were unverifiable.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing dense attention output mixing with distinct residual slots and one learned gain per head will reduce the model from 1,107 to 1,046 parameters while retaining at least 99% accuracy, because preserving route identity—not dense post-attention mixing—is load-bearing.
change: Compress the initialized observable output maps into two energy-matched learned gains, remove the dense projection and its bias before training, and concatenate the independently routed head contexts directly into disjoint residual channels.
mechanism: Energy-matched fixed head-slot residual routing
evidence_used: Summing the routed contexts before a shared projection collapsed accuracy to 0.01%, showing that head identity is essential. Conversely, the current shared-value model reached 99.91%, indicating that both routes can use the same semantic value representation. Fixed disjoint slots preserve that identity while testing whether the 63-parameter dense mixer is unnecessary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1046, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the two learned head gains will reduce the verified model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because the successful design shows that preserving separate residual slots is load-bearing, whereas both routes already share their value representation and should not require independent scalar amplitudes.
change: Replace the two per-head output gains with one learned RMS energy-matched gain, while keeping each attention head’s context in its existing distinct residual-channel slot.
mechanism: Shared gain with disjoint head-slot routing
evidence_used: Summing the head contexts before projection collapsed accuracy to 0.01%, but disjoint fixed routing with per-head gains achieved 99.96% at 1,046 parameters. This isolates route identity as essential and makes scalar amplitude tying the smallest direct test of whether head-specific output scaling is also necessary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9525, "parameters": 1045, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one redundant shared-value scale coordinate will reduce the verified model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because both independent head gains remain learned and can absorb any common rescaling of the shared value map.
change: Fix the largest initialized shared-value coefficient as a buffer, learn the other 27 coefficients, and reconstruct the full value map during attention while preserving both head-specific gains.
mechanism: Common value-scale gauge fixing
evidence_used: Disjoint routing with two learned gains achieved 99.96% at 1,046 parameters, whereas tying those gains reduced accuracy to 95.25%; this patch preserves their independence and removes only the exact common scaling redundancy between the shared value map and both gains.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1045, "training_steps": 4999}



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
