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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1109, "training_steps": 4999}
prior_hypothesis: Tying both heads’ twentieth-farthest attention biases will reduce the model from 1,110 to 1,109 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through nineteenth-farthest ties all succeeded without removing any routing distance.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one value-projection coefficient through an orthogonal value/output basis rotation will reduce the model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because it preserves the initialized function and all learned attention-score routing coordinates.
change: Rotate the shared value representation, compensate both attention-output projection blocks, omit the resulting fixed-zero value coefficient, and reconstruct it during forward passes.
mechanism: Orthogonal value/output latent-gauge fixing
evidence_used: The 1,119-parameter design achieved 99.53%, while pruning another routing bias fell to 73.42% and altering the query/key factorization reached only 90.12%; this motivates preserving those coordinates and removing an exact degree of freedom from the distinct consecutive linear value/output factorization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4181, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing `fc2.bias` to its seven observable zero-mean coordinates while emulating the original eight-coordinate AdamW updates will reduce the model from 1,119 to 1,118 parameters and retain at least 99% accuracy.
change: Reconstruct the full zero-mean MLP output bias from seven learned coordinates and train those coordinates using projected updates from virtual eight-dimensional Adam moments.
mechanism: Optimizer-equivariant LayerNorm bias gauge fixing
evidence_used: The current model achieved 99.53%, while the earlier orthonormal `fc2.bias` compression collapsed to 5.8%; because AdamW is not invariant to orthogonal reparameterization, preserving its original coordinate-wise moments directly tests whether optimization—not representational capacity—caused that failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9933, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the four-dimensional key representation across both heads will reduce the model from 1,118 to 1,090 parameters while retaining at least 99% accuracy, because head specialization can remain in the independent queries, output blocks, and load-bearing per-head relative biases.
change: Replace the two independent 28-parameter zero-mean key maps with one variance-preserving shared key map, producing multi-query attention while preserving independent queries and all routing biases.
mechanism: Multi-query attention with shared learned keys and values
evidence_used: The current design already succeeds with a shared value representation, while asymmetric endpoint pruning shows that head-specific distance routing is load-bearing. This challenges the shared assumption that semantic key features must also be head-specific without removing the demonstrated source of head specialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5639, "parameters": 1090, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the thirteenth-farthest bias only for the second head will reduce the model from 1,118 to 1,117 parameters while maintaining at least 99% accuracy, because removing that second-head bias caused the partial collapse to 73.42%, whereas the preceding distance showed that only one head needed its endpoint.
change: Shorten the shared per-head relative-bias core by one distance, retain the twelfth-farthest endpoint only for the first head, and retain the thirteenth-farthest endpoint only for the second head.
mechanism: Complementary head-specific thirteenth-distance routing
evidence_used: Pruning the second head’s thirteenth-farthest bias reduced accuracy to 73.42%, directly indicating that this endpoint should be restored; the complementary first-head pruning remains untested and mirrors the successful asymmetric pruning at the adjacent distance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948, "parameters": 1117, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the first head’s twelfth-farthest bias to the second head’s thirteenth-farthest bias will reduce the model from 1,117 to 1,116 parameters while maintaining at least 99% accuracy, because prior ablations show that both routes must remain present but do not establish that they require independent learned strengths.
change: Replace the two complementary head-specific endpoint parameters with one shared scalar used at both distances.
mechanism: Complementary endpoint bias tying
evidence_used: Removing the first head’s twelfth-farthest endpoint collapsed accuracy to 6.53%, while removing the second head’s thirteenth-farthest endpoint reduced it to 73.42%; retaining their complementary routes produced 99.48%, motivating parameter tying rather than another endpoint deletion.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1116, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ fourteenth-farthest attention biases will reduce the model from 1,116 to 1,115 parameters while retaining at least 99% accuracy, because complementary endpoint tying reached 99.87%, showing that distinct routing paths can share a learned strength without losing accuracy.
change: Remove the final column from each head’s relative-bias core and reconstruct that distance for both heads from one shared learned scalar.
mechanism: Cross-head fourteenth-distance bias tying
evidence_used: The current design tied two load-bearing, head-specific endpoint biases and achieved 99.87%; this supports testing the smallest adjacent sharing change while preserving every routing distance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1115, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying both heads’ fifteenth-farthest attention biases will reduce the model from 1,115 to 1,114 parameters while retaining at least 99% accuracy, because the adjacent fourteenth-farthest cross-head tie achieved 99.92% without removing any routing distance.
change: Remove the final column from each head’s relative-bias core and reconstruct that fifteenth-farthest distance from one new shared learned scalar.
mechanism: Cross-head fifteenth-distance bias tying
evidence_used: Cross-head tying at the immediately adjacent fourteenth-farthest distance reduced parameters from 1,116 to 1,115 and improved verified accuracy to 99.92%, motivating the same one-parameter sharing test while preserving all attention distances.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1114, "training_steps": 4999}

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

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
