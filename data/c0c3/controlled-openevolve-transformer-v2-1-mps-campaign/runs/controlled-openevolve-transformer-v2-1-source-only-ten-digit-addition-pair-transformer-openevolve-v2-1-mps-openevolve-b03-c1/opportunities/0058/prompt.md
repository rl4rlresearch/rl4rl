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
verified_results: {"accuracy": 0.9987, "parameters": 1150, "training_steps": 4999}
prior_hypothesis: Fixing each head’s eighth-farthest attention bias to zero will reduce learned parameters from 1,152 to 1,150 while retaining at least 99% accuracy, because this distance affects only eight causal query-key pairs per full sequence and pruning the seven farther endpoints retained 99.77% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the gauge-fixed tied lexical representation from rank six to rank five will reduce learned parameters from 1,209 to 1,097 while retaining at least 99% accuracy, because the successful rank-six model reached 99.96% and the full-width attention and residual pathways remain unchanged.
change: Use five-dimensional token codes, a fixed five-column identity chart, and a learned 5-by-3 lexical tail for both embeddings and logits.
mechanism: Rank-five tied lexical bottleneck
evidence_used: The gauge-fixed rank-six lexical bottleneck achieved 99.96% accuracy at 1,223 parameters, and subsequent independent MLP compression retained 99.89%; lowering lexical rank is the most direct test of whether six latent token features contain excess capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3787, "parameters": 1097, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower the model from 1,209 to 1,194 learned parameters while retaining at least 99% accuracy, because the successful zero-mean MLP output compression indicates remaining MLP redundancy, whereas the failed rank-five lexical bottleneck shows lexical rank is the riskier capacity target.
change: Change the default feed-forward width from 12 hidden units to 11, preserving the architecture, training schedule, and learned causal attention pathway.
mechanism: One-unit MLP width reduction
evidence_used: The 1,209-parameter model retained 99.89% accuracy after removing 12 MLP output-weight parameters, supporting a small additional MLP reduction; by contrast, reducing lexical rank caused accuracy to collapse to 37.87%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1194, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 11 to 10 will lower the model from 1,194 to 1,179 learned parameters while retaining at least 99% accuracy, because the preceding 12-to-11 reduction achieved 99.98% accuracy and showed substantial remaining MLP capacity.
change: Change the default feed-forward width from 11 hidden units to 10 while preserving all other architecture and training settings.
mechanism: One-unit MLP width reduction
evidence_used: The verified 1,194-parameter design with `d_ff=11` reached 99.98% accuracy, improving on the already successful 1,209-parameter `d_ff=12` design; this supports another conservative one-unit reduction, while the failed rank-five lexical bottleneck indicates lexical compression is substantially riskier.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1179, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s least-used maximum-distance attention bias to zero will reduce the model from 1,194 to 1,192 parameters while retaining at least 99% accuracy, because that bias affects only one causal query-key pair per full sequence.
change: Learn relative biases only for distances 1 through `max_seq_len - 2` and append fixed zeros at distances zero and `max_seq_len - 1`.
mechanism: Farthest-distance relative-bias pruning
evidence_used: The `d_ff=11` design reached 99.98%, whereas reducing `d_ff` again collapsed accuracy; this motivates preserving MLP capacity and testing a minimal reduction in the untouched 44-parameter relative-bias table.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1192, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one four-dimensional learned value projection across both heads will reduce parameters from 1,192 to 1,164 while retaining at least 99% accuracy, because addition requires distinct head-specific addressing but both routes can transport digits through a common learned value representation.
change: Preserve independent query and key projections and biases for each head, but replace the two value projections with one learned shared projection whose attended outputs remain separately mixed by the output projection.
mechanism: Shared-value, head-specific routing attention
evidence_used: Shared-key attention collapsed to 36.45%, identifying head-specific routing as load-bearing, while the current two-head model reaches 99.99%; this patch leaves that routing capacity intact and tests the distinct assumption that each route also needs its own semantic value coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1164, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s second-farthest attention bias to zero will reduce parameters from 1,164 to 1,162 while retaining at least 99% accuracy, because this distance affects only two causal query-key pairs per full sequence and pruning the even-farthest endpoint retained 99.99% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 3`, appending fixed zeros for distance zero and the two largest distances.
mechanism: Second-farthest relative-bias pruning
evidence_used: The 1,192-parameter design retained 99.99% accuracy after fixing the maximum-distance bias, and shared values subsequently retained 99.98%; this supports testing the adjacent, next-least-used bias while preserving all demonstrated routing and MLP capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1162, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s third-farthest attention bias to zero will reduce parameters from 1,162 to 1,160 while retaining at least 99% accuracy, because this distance affects only three causal query-key pairs per full sequence and pruning the two farther endpoints retained 99.95% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 4`, appending fixed zeros for distance zero and the three largest distances.
mechanism: Third-farthest relative-bias pruning
evidence_used: The current design achieved 99.95% accuracy after fixing the two largest-distance biases, while the immediately preceding one-endpoint design achieved 99.99%; this supports testing the adjacent next-least-used bias without reducing demonstrated MLP, lexical, or routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1160, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s fourth-farthest attention bias to zero will reduce parameters from 1,160 to 1,158 while retaining at least 99% accuracy, because this distance affects only four causal query-key pairs per full sequence and pruning the three farther endpoints retained 99.94% accuracy.
change: Learn relative biases only for distances 1 through `max_seq_len - 5`, appending fixed zeros for distance zero and the four largest distances.
mechanism: Fourth-farthest relative-bias pruning
evidence_used: The current 1,160-parameter design achieved 99.94% accuracy after successively fixing the three largest-distance biases; this tests the adjacent next-least-used bias while preserving all demonstrated lexical, MLP, value, and routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1158, "training_steps": 4999}

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
