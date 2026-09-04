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
verified_results: {"accuracy": 0.9989, "parameters": 606, "training_steps": 4999}
prior_hypothesis: Fixing an adaptive four-column key basis to identity in each attention head will reduce the verified 638-parameter model to 606 parameters while retaining at least 99% accuracy, because the transformation preserves every initialized attention score and the model’s full learned query/key function class.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified 606-parameter model to 605 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one column and reconstruct the removed column with one zero-initialized learned scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The current model achieved 99.89% accuracy, and sharing the nine immediately farther bias distances retained at least 99%, making the adjacent twenty-third-farthest distance the most evidence-supported next parameter tie.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Fixing one token’s phase to zero will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy, because a common phase shift is an exact rotational gauge of the four-dimensional harmonic lexical representation.
change: Remove one learned token-phase coordinate and rotate the lexical projection and bias at initialization to preserve the model function exactly.
mechanism: Global lexical-phase gauge fixing
evidence_used: The 606-parameter model achieved 99.89% accuracy, while removing the final LayerNorm gain failed catastrophically; this instead removes a mathematically redundant coordinate without reducing the learned lexical function class.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4429, "parameters": 605, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias table by one column and reconstruct that column from one zero-initialized scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing the nine adjacent farther-distance biases retained at least 99%; prior attempts at this boundary were unverifiable and therefore provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Remove the final two-value column from the head-specific relative-bias core and reconstruct it as one zero-initialized scalar shared by both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing the nine adjacent farther-distance biases retained at least 99%; the unverifiable attempts provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one column and reconstruct the removed column as one zero-initialized scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing the nine adjacent farther-distance biases retained at least 99%; prior attempts at this boundary were unverifiable rather than accuracy failures.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one column and reconstruct the removed column with one learned scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing the nine adjacent farther-distance biases retained at least 99%; prior attempts at this boundary were unverifiable rather than accuracy failures.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy.
change: Remove the final two-value column from the head-specific relative-bias table and reconstruct it as one learned scalar shared by both attention heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and the nine immediately farther relative biases are already shared successfully; prior attempts at this adjacent tie were unverifiable rather than accuracy failures.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Constraining the four harmonic amplitudes to their initialized ratios while learning one global amplitude will reduce the verified model from 606 to 603 parameters and retain at least 99% accuracy, because the learned query/key maps can still mix each positional axis independently and the initialized function is preserved.
change: Factor the four-parameter positional scale vector into a fixed normalized spectral shape and one learned global scale.
mechanism: Fixed-spectrum positional clock
evidence_used: The 606-parameter model reached 99.89% accuracy using fixed harmonic codes and fixed orthogonal positional directions with fully learned query/key maps. The catastrophic non-affine final-LayerNorm result shows lexical readout geometry is load-bearing, motivating compression of the more structurally redundant positional amplitudes instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.062, "parameters": 603, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias table by one column and reconstruct the removed column from one learned scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing the nine adjacent farther-distance biases retained at least 99%; prior attempts at this boundary were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified 606-parameter model to 605 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias table by one column and reconstruct the removed column from one zero-initialized learned scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, while sharing the nine adjacent farther-distance biases already retained at least 99%; prior attempts at this boundary were unverifiable rather than accuracy failures.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
