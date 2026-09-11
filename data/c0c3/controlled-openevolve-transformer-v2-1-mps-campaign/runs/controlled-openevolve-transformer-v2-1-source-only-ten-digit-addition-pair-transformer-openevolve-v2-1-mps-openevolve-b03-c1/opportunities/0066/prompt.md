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
verified_results: {"accuracy": 0.9953, "parameters": 1119, "training_steps": 4999}
prior_hypothesis: Fixing only the second head’s twelfth-farthest attention bias will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because the failed complementary ablation indicates that the first head may be the sole head requiring this routing endpoint.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing each head’s fifth-farthest attention bias to zero will reduce learned parameters from 1,158 to 1,156 while retaining at least 99% accuracy, because this distance affects only five causal query-key pairs per full sequence and pruning the four farther endpoints retained 99.94% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 6`, with fixed zeros at distance zero and the five largest distances.
mechanism: Fifth-farthest relative-bias pruning
evidence_used: The current 1,158-parameter design achieved 99.94% accuracy after successively fixing the four largest-distance biases, motivating the adjacent minimal reduction while preserving all demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1156, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s sixth-farthest attention bias to zero will reduce learned parameters from 1,156 to 1,154 while retaining at least 99% accuracy, because this distance affects only six causal query-key pairs per full sequence and pruning the five farther endpoints retained 99.92% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 7`, with fixed zeros at distance zero and the six largest distances.
mechanism: Sixth-farthest relative-bias pruning
evidence_used: The current 1,156-parameter design achieved 99.92% accuracy after successively fixing the five largest-distance biases, supporting the adjacent two-parameter reduction while preserving all demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1154, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s seventh-farthest attention bias to zero will reduce learned parameters from 1,154 to 1,152 while retaining at least 99% accuracy, because this distance affects only seven causal query-key pairs per full sequence and pruning the six farther endpoints retained 99.92% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 8`, with fixed zeros at distance zero and the seven largest distances.
mechanism: Seventh-farthest relative-bias pruning
evidence_used: The current 1,154-parameter design achieved 99.92% accuracy after successively fixing the six largest-distance biases, supporting the adjacent two-parameter reduction while preserving demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1152, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s eighth-farthest attention bias to zero will reduce learned parameters from 1,152 to 1,150 while retaining at least 99% accuracy, because this distance affects only eight causal query-key pairs per full sequence and pruning the seven farther endpoints retained 99.77% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 9`, with fixed zeros at distance zero and the eight largest distances.
mechanism: Eighth-farthest relative-bias pruning
evidence_used: The current 1,152-parameter design achieved 99.77% accuracy after successively fixing the seven largest-distance biases, supporting the adjacent two-parameter reduction while preserving demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1150, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s ninth-farthest attention bias to zero will reduce learned parameters from 1,150 to 1,148 while retaining at least 99% accuracy, because this distance affects only nine causal query-key pairs per full sequence and pruning the eight farther endpoints retained 99.87% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 10`, with fixed zeros at distance zero and the nine largest distances.
mechanism: Ninth-farthest relative-bias pruning
evidence_used: The current 1,150-parameter design achieved 99.87% accuracy after successively fixing the eight largest-distance biases, supporting the adjacent two-parameter reduction while preserving demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1148, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s tenth-farthest attention bias to zero will reduce learned parameters from 1,148 to 1,146 while retaining at least 99% accuracy, because this distance affects only ten causal query-key pairs per full sequence and pruning the nine farther endpoints retained 99.96% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 11`, with fixed zeros at distance zero and the ten largest distances.
mechanism: Tenth-farthest relative-bias pruning
evidence_used: The current 1,148-parameter design achieved 99.96% accuracy after successively fixing the nine largest-distance biases, supporting the adjacent two-parameter reduction while preserving all demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1146, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the 28-parameter dense positional readout with four learned harmonic amplitudes will reduce the model from 1,146 to 1,122 parameters while retaining at least 99% accuracy, because the independent learned query/key projections can mix a fixed orthogonal positional basis without requiring a separately learned direction for every harmonic-coordinate pair.
change: Represent the four fixed harmonics along deterministic orthogonal zero-mean residual directions, learning only one amplitude per harmonic while preserving the initialized positional magnitude and all lexical, MLP, and head-specific routing capacity.
mechanism: Canonical-basis harmonic position encoding
evidence_used: The 1,146-parameter design reaches 99.96% with fixed harmonic coordinates, showing that slot-specific learned position codes are unnecessary; meanwhile, `d_ff=10` and rank-five lexical compression failed, so this patch preserves those load-bearing capacities and tests the distinct assumption that dense learned positional orientation is necessary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9951000000000001, "parameters": 1122, "training_steps": 4999}

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
