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
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance and reconstruct that distance from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias sharing
evidence_used: The current model achieved 99.89% accuracy, while nine consecutive adjacent cross-head bias ties through the twenty-second-farthest distance previously retained at least 99%; the prior twenty-third-distance attempts produced no accuracy result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing `d_ff` from 10 to 9 will lower the verified model from 606 to 591 learned parameters while retaining at least 99% accuracy.
change: Remove one hidden unit from the transformer MLP while preserving the verified lexical representation and attention architecture.
mechanism: One-unit MLP width reduction
evidence_used: The prior `d_ff` reduction from 11 to 10 retained 99.22% accuracy, and the current width-10 model reaches 99.89%, making another single-unit reduction the most direct test of remaining MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3946, "parameters": 591, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one globally redundant token-radius coordinate will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy, because virtual full-coordinate AdamW updates preserve the successful 606-parameter training trajectory up to an exact scale gauge.
change: Fix the first token radius as a buffer, learn the remaining radii, and train the gauge-fixed lexical parameters through virtual full-radius optimizer states mapped back into the 605-parameter model after every step.
mechanism: Radius-scale gauge fixing with ambient AdamW moments
evidence_used: The current 606-parameter model achieved 99.89% accuracy. The earlier phase gauge collapsed to 4.8%, showing optimizer-coordinate changes can be destructive; this patch therefore retains the omitted coordinate only as transient optimizer state and reproduces full-coordinate gradients, clipping, AdamW moments, and weight decay before exact gauge normalization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9018, "parameters": 605, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance and reconstruct the removed distance with one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing each of the nine adjacent farther-distance biases through the twenty-second-farthest distance retained at least 99%; prior attempts at this boundary produced no verification result and therefore provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance and reconstruct the removed distance with one shared learned scalar, keeping the relative-bias table length unchanged.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and cross-head sharing at each of the nine adjacent farther distances through the twenty-second-farthest distance previously retained at least 99%; the prior boundary attempts yielded no accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The ten GELU detectors are load-bearing, but their decoder directions need not be independently learned; decoding normalized detector directions through a shared learned 7×7 transport and ten learned gains will retain at least 99% accuracy while reducing parameters from 606 to 595.
change: Retain all ten nonlinear MLP units, but replace their independent 70-parameter output matrix with a 49-parameter shared transport and ten per-unit gains.
mechanism: Projectively tied associative MLP
evidence_used: Reducing `d_ff` from 10 to 9 collapsed accuracy to 39.46%, showing that removing a nonlinear detector is destructive. Conversely, the tied rank-four lexical input/output geometry reached 99.74%, and the current model reaches 99.89%, motivating an associative geometry that preserves detector count while challenging the assumption that every detector requires an unrelated output vector.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.019299999999999998, "parameters": 595, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Replace the two head-specific bias values at the twenty-third-farthest distance with one learned scalar while preserving the relative-bias table length.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing each of the nine adjacent farther-distance biases through the twenty-second-farthest distance retained at least 99%; prior attempts at this boundary yielded no accuracy result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy.
change: Replace the two head-specific values at the twenty-third-farthest distance with one learned scalar while preserving the relative-bias table length and initialization.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and nine consecutive farther-distance biases through the twenty-second-farthest distance have already been shared successfully; malformed boundary attempts supplied no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one column and reconstruct that column from one shared learned scalar, preserving table length and zero initialization.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, while sharing each of the nine adjacent farther-distance biases through the twenty-second-farthest distance retained at least 99%; prior malformed boundary attempts produced no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias table by one column and reconstruct that column from a zero-initialized shared scalar while preserving the full table length.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing each of the nine adjacent farther-distance biases through the twenty-second-farthest distance retained at least 99%; prior attempts at this boundary were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified model from 606 to 605 learned parameters while retaining at least 99% accuracy, given the current 99.89% result and nine successful adjacent far-distance ties.
change: Remove the final two-value column from the head-specific relative-bias core and reconstruct it as one zero-initialized scalar shared by both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter design achieved 99.89% accuracy, and sharing the nine immediately farther bias distances retained at least 99%; previous attempts at this boundary were unverifiable and therefore did not test its accuracy impact.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the final independently learned relative-bias distance across both attention heads will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one column and reconstruct the removed column with one zero-initialized learned scalar shared across both heads.
mechanism: Tenth cross-head far-distance bias tie
evidence_used: The 606-parameter model achieved 99.89% accuracy, and sharing each of the nine adjacent farther-distance biases through the twenty-second-farthest distance retained at least 99%; previous attempts at this next boundary were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified



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
