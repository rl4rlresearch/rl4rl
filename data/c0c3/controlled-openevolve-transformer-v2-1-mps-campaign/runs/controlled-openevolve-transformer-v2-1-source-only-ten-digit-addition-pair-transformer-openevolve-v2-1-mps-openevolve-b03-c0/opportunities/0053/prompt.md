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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1599, "training_steps": 16999}
prior_hypothesis: Anchoring a second attention-projection input column with equivalent full eight-coordinate initialization and implicit AdamW updates will reduce the model from 1,600 to 1,599 parameters while retaining at least 99% accuracy after 17,000 steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing one globally shift-invariant positional-embedding coordinate and preserving its implicit AdamW state will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps.
change: Store the positional-embedding matrix as all but one anchored coordinate, reconstruct the final coordinate as zero, optimize it with the existing gauge-aware optimizer, and add 1,000 low-rate refinement steps.
mechanism: Gauge-aware anchored positional-embedding coordinate
evidence_used: Gauge-aware anchoring of the attention projection bias retained 99.89% at 1,607 parameters; positional embeddings have an independent exact uniform-shift invariance through the pre-normalized block and final LayerNorm, making this a direct test of the successful optimizer-preserving mechanism without repeating the failed MLP-bias reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5628, "parameters": 1606, "training_steps": 9999}

RECENT RESULT
hypothesis: Anchoring one coordinate of one attention-projection weight column and optimizing its seven relative coordinates with an implicit eight-coordinate AdamW state will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps.
change: Store the projection’s final input column as seven relative output coordinates with an implicit zero anchor, reconstruct the full weight during forward passes, include the new coordinates in gauge-aware optimization, and add 1,000 low-rate refinement steps.
mechanism: Gauge-aware anchored attention-projection weight column
evidence_used: Gauge-aware anchoring of the attention projection bias achieved 99.89% at 1,607 parameters, whereas applying the mechanism to the MLP bias or positional embeddings failed; testing another exact uniform-output-shift gauge within the already successful attention projection is the most directly supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.40049999999999997, "parameters": 1606, "training_steps": 9999}

RECENT RESULT
hypothesis: Anchoring one `fc1` weight coordinate at zero will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps, because LayerNorm confines each MLP input to a seven-dimensional affine hyperplane and the retained `fc1` bias preserves the omitted direction’s constant component.
change: Store 95 of the 96 `fc1` weights, reconstruct the final coordinate as zero during forward passes, and add 1,000 low-learning-rate refinement steps.
mechanism: LayerNorm-hyperplane anchoring of one MLP input weight
evidence_used: The current 1,607-parameter model achieved 99.89% after 9,000 steps. The failed attention-projection weight gauge motivates testing a distinct redundancy localized to one MLP neuron, whose LayerNorm-constrained input and retained bias preserve representational capacity without another gauge-aware optimizer.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1606, "training_steps": 9999}

RECENT RESULT
hypothesis: Anchoring the final input-weight coordinate of a second `fc1` neuron will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy after 11,000 steps, because every MLP neuron receives inputs confined by LayerNorm to the same seven-dimensional affine hyperplane and keeps an independent learned bias.
change: Store 94 of the 96 `fc1` weights, reconstruct coordinate 7 of the final two neuron rows as zero, and add 1,000 low-learning-rate refinement steps.
mechanism: Incremental LayerNorm-hyperplane anchoring
evidence_used: Anchoring this coordinate for one `fc1` neuron achieved 99.8% at 1,606 parameters after 10,000 steps, directly validating the same per-neuron redundancy and training strategy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8883, "parameters": 1605, "training_steps": 10999}

RECENT RESULT
hypothesis: Removing the final singleton value-bias coordinate will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy after 11,000 steps, because a position-independent value bias contributes only a constant after attention and can be represented by the learned output-projection bias.
change: Store 11 QKV bias parameters, reconstruct the final value-bias coordinate as zero while preserving every other validated bias mapping, and add 1,000 low-learning-rate refinement steps.
mechanism: Absorb a redundant value-bias coordinate into the attention output bias
evidence_used: The 1,606-parameter design reached 99.8%, while anchoring a second `fc1` neuron fell to 88.83%; the successful gauge-aware attention-output bias at 1,607 parameters motivates testing this distinct exact redundancy, since that bias can absorb the removed value offset.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1605, "training_steps": 10999}

RECENT RESULT
hypothesis: Removing the remaining singleton value-bias scalar will reduce the model from 1,605 to 1,604 parameters while retaining at least 99% accuracy after 12,000 steps, because its position-independent attention contribution can be absorbed by the learned output-projection bias.
change: Store 10 QKV bias parameters, reconstruct value coordinate 6 as zero while preserving all other bias mappings, and add 1,000 low-learning-rate refinement steps.
mechanism: Incremental absorption of a singleton value bias
evidence_used: Removing the previous singleton value-bias coordinate achieved 99.97% at 1,605 parameters after 11,000 steps, directly validating the same redundancy and incremental training strategy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9934999999999999, "parameters": 1604, "training_steps": 11999}

RECENT RESULT
hypothesis: Removing the value-bias scalar shared by coordinates 4 and 5 will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy after 13,000 steps, because its position-independent attention contribution can be absorbed by the learned output-projection bias.
change: Store nine QKV bias parameters, reconstruct value coordinates 4 and 5 as zero while preserving the remaining validated mappings, and add 1,000 low-learning-rate refinement steps.
mechanism: Incremental absorption of a tied value-bias group
evidence_used: Successive removal of two singleton value-bias parameters produced 99.97% at 1,605 parameters and 99.35% at 1,604 parameters; this directly supports testing the next incremental value-bias absorption rather than repeating the failed second-neuron anchoring.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1603, "training_steps": 12999}

RECENT RESULT
hypothesis: Removing the last learned value-bias scalar will reduce the model from 1,603 to 1,602 parameters while retaining at least 99% accuracy after 14,000 steps, because its position-independent contribution can be absorbed by the learned output-projection bias.
change: Store only the eight query-bias parameters, reconstruct the entire value bias as zero, and add 1,000 minimum-learning-rate refinement steps.
mechanism: Absorb the final tied value-bias group into the attention output bias
evidence_used: Removing the preceding tied value-bias group achieved 99.90% at 1,603 parameters after 13,000 steps, while earlier singleton removals also succeeded; the remaining scalar has the same redundant value-bias role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1602, "training_steps": 13999}

RECENT RESULT
hypothesis: Anchoring the final input-weight coordinate of one query-projection neuron will reduce the model from 1,602 to 1,601 parameters while retaining at least 99% accuracy after 15,000 steps, because LayerNorm confines its input to a seven-dimensional affine hyperplane and the neuron retains an independent query bias.
change: Store 191 of the 192 QKV weights, reconstruct coordinate 7 of the first query row as zero during forward passes, and add 1,000 low-rate refinement steps.
mechanism: LayerNorm-hyperplane anchoring of one query-projection weight
evidence_used: The analogous first-neuron `fc1` anchoring achieved 99.8% at 1,606 parameters, validating this LayerNorm-input redundancy; applying it to one query neuron tests the same mechanism in a different sublayer without repeating the failed second-`fc1` anchoring.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.39520000000000005, "parameters": 1601, "training_steps": 14999}

RECENT RESULT
hypothesis: Anchoring one scalar of the tied token-embedding/output matrix, while sampling an equivalent full initialization and preserving implicit full-coordinate AdamW updates, will reduce the model from 1,602 to 1,601 parameters and retain at least 99% accuracy after 15,000 steps.
change: Store 103 of the 104 tied embedding coordinates, reconstruct the final coordinate as zero for both input embedding and output logits, initialize the anchored representation from a full i.i.d. matrix, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.
mechanism: Initialization-preserving global token-embedding gauge
evidence_used: Gauge-aware anchoring of the attention projection bias achieved 99.89% at 1,607 parameters, whereas positional-embedding anchoring reached only 56.28%; preserving both the full initialization distribution and implicit optimizer state tests whether initialization geometry caused that failure. A global scalar shift of the tied token matrix is an exact symmetry because it propagates as a uniform residual shift removed by LayerNorm and adds only a common offset to all logits.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1601, "training_steps": 14999}

RECENT RESULT
hypothesis: Anchoring one uniform-output gauge coordinate in an attention-projection weight column, while preserving equivalent full initialization and implicit eight-coordinate AdamW updates, will reduce the model from 1,601 to 1,600 parameters and retain at least 99% accuracy after 16,000 steps.
change: Split the final attention-projection input column into seven learned relative coordinates plus a zero anchor, initialize it from an eight-coordinate i.i.d. column, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.
mechanism: Initialization-preserving attention-output column gauge
evidence_used: The earlier projection-column gauge reached only 40.05%, but the later initialization-preserving token-embedding gauge achieved 99.99% at 1,601 parameters; this tests whether transferring that validated full-initialization treatment rescues the exact uniform-output projection gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1600, "training_steps": 15999}

RECENT RESULT
hypothesis: Anchoring a second attention-projection input column with equivalent full eight-coordinate initialization and implicit AdamW updates will reduce the model from 1,600 to 1,599 parameters while retaining at least 99% accuracy after 17,000 steps.
change: Split another projection column into seven learned relative coordinates plus a zero anchor, initialize both anchored columns from full i.i.d. columns, optimize each with GaugeAdamW, and add 1,000 low-rate refinement steps.
mechanism: Incremental initialization-preserving attention-output column gauge
evidence_used: Initialization-preserving anchoring rescued the first projection-column gauge from 40.05% to 99.98% accuracy at 1,600 parameters, directly validating the same exact symmetry and optimizer treatment used here.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1599, "training_steps": 16999}



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
