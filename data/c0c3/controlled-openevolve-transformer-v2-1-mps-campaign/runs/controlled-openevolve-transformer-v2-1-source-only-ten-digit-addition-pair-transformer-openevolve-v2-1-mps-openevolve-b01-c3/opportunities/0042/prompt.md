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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9940000000000001, "parameters": 1405, "training_steps": 4999}
prior_hypothesis: Sharing MLP biases in two four-neuron clusters and two pairs will produce a 1,405-parameter model with at least 99% accuracy, because the verified 1,406-parameter design achieved 99.87% and this imposes only one additional scalar tie while preserving every neuron, weight, and learned threshold.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987, "parameters": 1406, "training_steps": 4999}
prior_hypothesis: Merging two of the six learned MLP bias pairs into one four-neuron cluster will produce a 1,406-parameter model with at least 99% accuracy, because it preserves every neuron, weight, and adaptive threshold while imposing only one additional scalar tie.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9956999999999999, "parameters": 1404, "training_steps": 4999}
prior_hypothesis: Sharing the twelve MLP hidden biases across three learned four-neuron clusters will produce a 1,404-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9952, "parameters": 1410, "training_steps": 4999}
prior_hypothesis: Sharing biases within three disjoint pairs of permutation-symmetric MLP neurons will produce a 1,410-parameter model with at least 99% accuracy, because the verified second disjoint tie preserved 99.96% accuracy while retaining all twelve neurons, independent weights, and learned thresholds.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing one learned value projection across both attention heads will reduce the verified fixed-bias design from 1,413 to 1,381 parameters while retaining at least 99% accuracy, because addition’s two operands use the same digit representation while independent query/key projections preserve each head’s load-bearing ability to route to different source positions.
change: Replace per-head value projections with one learned four-dimensional value map shared by both heads, retain full-width independent query/key maps, and apply the verified fixed 0.02 shared query bias.
mechanism: Shared value alphabet with independent attention routing
evidence_used: The 1,413-parameter fixed-bias design reached 99.96%, whereas reducing query/key routing width or positional capacity hurt accuracy. This suggests routing capacity is load-bearing, but does not establish that the two routed operands need separate value coordinate systems; a shared learned value alphabet directly tests that assumption while leaving routing intact.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6009, "parameters": 1381, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one learned threshold between two permutation-symmetric MLP neurons will produce a 1,412-parameter model with at least 99% accuracy, because it preserves all twelve neurons and keeps every bias position adaptive.
change: Apply the verified two-sided centered embedding and fixed 0.02 query bias, then replace the twelve independent MLP input biases with eleven learned values by sharing the final value across two neurons.
mechanism: Adaptive pairwise MLP bias sharing
evidence_used: The 1,413-parameter fixed-bias design achieved 99.96%; fixing one MLP bias at zero failed at 39.67%, motivating a learned tie that removes one parameter without permanently suppressing a neuron’s threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1412, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing biases within two disjoint pairs of permutation-symmetric MLP neurons will produce a 1,411-parameter model with at least 99% accuracy, because all twelve neurons and their weights remain independent while every threshold remains learned.
change: Apply two-sided centered tied embeddings and the verified fixed 0.02 query bias, then parameterize the twelve MLP biases with ten learned values shared across two neuron pairs.
mechanism: Two disjoint adaptive MLP bias ties
evidence_used: A single adaptive MLP bias tie achieved 99.96% with 1,412 parameters, while fixing one bias at zero collapsed to 39.67%; this indicates threshold adaptivity matters but full threshold independence may not, making a second disjoint learned tie the smallest supported next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1411, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing biases within three disjoint pairs of permutation-symmetric MLP neurons will produce a 1,410-parameter model with at least 99% accuracy, because the verified second disjoint tie preserved 99.96% accuracy while retaining all twelve neurons, independent weights, and learned thresholds.
change: Parameterize the twelve MLP hidden biases with nine learned values, using the final three values for three disjoint neuron pairs.
mechanism: Three disjoint adaptive MLP bias ties
evidence_used: The 1,411-parameter design with two disjoint learned bias ties achieved 99.96% accuracy, identical to the one-tie and untied verified designs; this directly motivates testing one additional incremental tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9952, "parameters": 1410, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing biases within four disjoint pairs of permutation-symmetric MLP neurons will produce a 1,409-parameter model with at least 99% accuracy, because the verified three-pair design reached 99.52% while retaining all twelve neurons, independent weights, and learned thresholds.
change: Parameterize the twelve MLP hidden biases with eight learned values, using the final four values for four disjoint neuron pairs.
mechanism: Four disjoint adaptive MLP bias ties
evidence_used: The 1,410-parameter design with three disjoint adaptive bias ties achieved 99.52%; extending the same successful reduction by one pair is the smallest informative next ablation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1409, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing biases within five disjoint pairs of permutation-symmetric MLP neurons will produce a 1,408-parameter model with at least 99% accuracy, because the verified four-pair design achieved 99.98% while preserving all twelve neurons, independent weights, and learned thresholds.
change: Parameterize the twelve MLP hidden biases with seven learned values, using the final five values for five disjoint neuron pairs.
mechanism: Five disjoint adaptive MLP bias ties
evidence_used: The 1,409-parameter design with four disjoint adaptive bias ties achieved 99.98% accuracy; extending the same successful reduction by one pair is the smallest informative next ablation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1408, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing biases across all six disjoint pairs of permutation-symmetric MLP neurons will produce a 1,407-parameter model with at least 99% accuracy, because the verified five-pair design achieved 99.96% while retaining all twelve neurons, independent weights, and learned thresholds.
change: Apply the verified fixed 0.02 query bias and parameterize the twelve MLP hidden biases as six independently learned pairwise-shared values.
mechanism: Complete pairwise sharing of adaptive MLP thresholds
evidence_used: The 1,408-parameter five-pair design reached 99.96% accuracy; tying its remaining two independent neuron biases is the smallest incremental reduction supported by the successful one-through-five-pair sequence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9908, "parameters": 1407, "training_steps": 4999}

RECENT RESULT
hypothesis: Merging two of the six learned MLP bias pairs into one four-neuron cluster will produce a 1,406-parameter model with at least 99% accuracy, because it preserves every neuron, weight, and adaptive threshold while imposing only one additional scalar tie.
change: Replace six pairwise-shared MLP biases with five learned biases: one shared across four neurons and four shared across two neurons each.
mechanism: Four-neuron adaptive threshold cluster
evidence_used: The 1,407-parameter design achieved 99.08% after all six disjoint bias ties; merging two existing bias groups is the smallest incremental reduction along the only demonstrated successful parameter-reduction sequence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1406, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing MLP biases in two four-neuron clusters and two pairs will produce a 1,405-parameter model with at least 99% accuracy, because the verified 1,406-parameter design achieved 99.87% and this imposes only one additional scalar tie while preserving every neuron, weight, and learned threshold.
change: Replace five learned MLP bias groups—one quartet and four pairs—with four learned groups comprising two quartets and two pairs.
mechanism: Dual four-neuron adaptive threshold clusters
evidence_used: The 1,406-parameter design with one four-neuron bias cluster and four bias pairs achieved 99.87%; merging two remaining pairs into a second quartet is the smallest incremental reduction along the demonstrated successful bias-sharing sequence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9940000000000001, "parameters": 1405, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twelve MLP hidden biases across three learned four-neuron clusters will produce a 1,404-parameter model with at least 99% accuracy.
change: Replace four independent biases and four pair-shared biases with three learned biases, each shared by four MLP neurons.
mechanism: Three four-neuron adaptive threshold clusters
evidence_used: The verified 1,405-parameter design achieved 99.40% using two quartets and two pairs; merging its two remaining pairs into a third quartet is the smallest next reduction along the consistently successful bias-sharing sequence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956999999999999, "parameters": 1404, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,340-parameter model will retain at least 99% accuracy because each attention head can route through a full-width learned similarity metric, while independent value projections and the fixed query bias preserve content capacity and directional asymmetry.
change: Start from the verified three-quartet MLP-bias design, then replace independent query and key projections with one shared learned projection while retaining separate full-width values.
mechanism: Shared query-key metric attention
evidence_used: The 1,404-parameter three-quartet design achieved 99.57%. Shared values failed at 60.09%, and reduced query/key width reportedly hurt, so this patch preserves independent values, both heads, and four routing dimensions per head while challenging only the untested assumption that query and key require separate coordinate maps.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9349, "parameters": 1340, "training_steps": 4999}

RECENT RESULT
hypothesis: Merging two of the three learned MLP bias quartets into one eight-neuron cluster will produce a 1,403-parameter model with at least 99% accuracy while preserving every hidden neuron and learned weight.
change: Replace three quartet-shared MLP biases with two learned biases: one shared across eight neurons and one shared across four.
mechanism: Learned octet-and-quartet MLP threshold sharing
evidence_used: The verified 1,404-parameter design achieved 99.57% accuracy with three learned bias quartets; merging two existing groups removes one scalar along the consistently successful incremental bias-sharing sequence.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9691, "parameters": 1403, "training_steps": 4999}



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
