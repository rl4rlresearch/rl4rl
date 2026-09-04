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
verified_results: {"accuracy": 0.9987, "parameters": 1116, "training_steps": 4999}
prior_hypothesis: Tying the first head’s twelfth-farthest bias to the second head’s thirteenth-farthest bias will reduce the model from 1,117 to 1,116 parameters while maintaining at least 99% accuracy, because prior ablations show that both routes must remain present but do not establish that they require independent learned strengths.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing each head’s eleventh-farthest attention bias to zero will reduce learned parameters from 1,122 to 1,120 while retaining at least 99% accuracy, because this distance affects only eleven causal query-key pairs per full sequence.
change: Learn relative biases only for distances 1 through `max_seq_len - 12`, with fixed zeros at distance zero and the eleven largest distances.
mechanism: Eleventh-farthest relative-bias pruning
evidence_used: The current 1,122-parameter harmonic-position design achieved 99.51%, while ten consecutive farthest-distance pruning steps retained at least 99.77%; this supports testing the adjacent two-parameter reduction without changing the newly compressed positional representation or known load-bearing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965, "parameters": 1120, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s twelfth-farthest attention bias to zero will reduce learned parameters from 1,120 to 1,118 while retaining at least 99% accuracy, because this distance affects only twelve causal query-key pairs per full sequence.
change: Learn relative biases only for distances 1 through `max_seq_len - 13`, with fixed zeros at distance zero and the twelve largest distances.
mechanism: Twelfth-farthest relative-bias pruning
evidence_used: The current 1,120-parameter design achieved 99.65% after eleven consecutive farthest-distance biases were fixed, motivating the adjacent two-parameter reduction while preserving all demonstrated lexical, positional, MLP, value, and routing capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0801, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing the MLP output bias from eight coordinates to the seven-dimensional zero-mean subspace will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because any discarded all-ones component is exactly removed by downstream LayerNorms.
change: Represent `fc2.bias` in the existing orthonormal zero-mean basis and reconstruct its full bias during the forward pass.
mechanism: Zero-mean MLP output-bias gauge fixing
evidence_used: The 1,120-parameter design reached 99.65%, whereas removing the twelfth-farthest attention biases collapsed accuracy to 8.01%; this motivates preserving the load-bearing routing endpoint and instead eliminating an exact LayerNorm-invariant MLP bias degree of freedom.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.057999999999999996, "parameters": 1119, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing only the first head’s twelfth-farthest attention bias will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because the joint two-head ablation may have collapsed by removing a route needed by only one head.
change: Split the relative biases into a shared-length per-head core and a one-parameter endpoint retained only for the second head, reconstructing the full bias table during the forward pass.
mechanism: Asymmetric single-head endpoint pruning
evidence_used: Removing both twelfth-farthest biases collapsed accuracy from 99.65% to 8.01%, while all eleven farther two-head removals succeeded; isolating one head is the smallest informative test of whether both parameters are necessary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0653, "parameters": 1119, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing only the second head’s twelfth-farthest attention bias will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because the failed complementary ablation indicates that the first head may be the sole head requiring this routing endpoint.
change: Split the relative biases into a per-head core and a one-parameter twelfth-farthest endpoint retained only for the first head, reconstructing the full bias table during the forward pass.
mechanism: Complementary single-head endpoint pruning
evidence_used: Pruning both twelfth-farthest biases collapsed accuracy to 8.01%, and pruning only the first head’s endpoint collapsed it to 6.53%; the untested complementary ablation directly determines whether the first head’s endpoint is load-bearing while the second head’s parameter is redundant.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9953, "parameters": 1119, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the second head’s thirteenth-farthest attention bias will reduce the model from 1,119 to 1,118 learned parameters while retaining at least 99% accuracy, because that head already retained 99.53% accuracy without its twelfth-farthest endpoint while the complementary first-head ablation collapsed.
change: Shorten the per-head relative-bias core by one distance and retain the two adjacent endpoints only for the first head.
mechanism: Cascaded asymmetric endpoint pruning
evidence_used: Pruning the second head’s twelfth-farthest endpoint achieved 99.53%, whereas pruning the first head’s corresponding endpoint achieved only 6.53%; this head-specific contrast motivates extending pruning by one adjacent distance only on the apparently less endpoint-dependent second head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7342, "parameters": 1118, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one key-projection coefficient through a compensating query/key basis transformation will reduce the model from 1,119 to 1,118 learned parameters while retaining at least 99% accuracy, because it preserves the initialized attention scores and removes only one exact latent-factorization degree of freedom.
change: Apply an invertible row transformation to the first head’s key projection that zeros one coefficient, apply the inverse-transpose transformation to its query projection and bias, omit the fixed zero from the learned parameter vector, and reconstruct it during forward passes.
mechanism: Single-coordinate query/key latent-gauge fixing
evidence_used: The 1,119-parameter design achieved 99.53%, while further second-head relative-bias pruning fell to 73.42% and MLP-bias gauge compression collapsed; this motivates preserving all demonstrated routing and MLP parameters while cautiously testing a single exact query/key factorization gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9012, "parameters": 1118, "training_steps": 4999}

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
