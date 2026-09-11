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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1603, "training_steps": 12999}
prior_hypothesis: Removing the value-bias scalar shared by coordinates 4 and 5 will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy after 13,000 steps, because its position-independent attention contribution can be absorbed by the learned output-projection bias.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying the second head’s remaining key-bias coordinate to the validated coordinate-1 group and extending training with 1,000 minimum-learning-rate steps will reduce the model to 1,610 parameters while raising the prior 98.92% result above 99%.
change: Store 14 QKV bias parameters, reconstruct the second head’s final key bias from the shared coordinate-1 scalar, preserve the existing value-bias mapping, and train for 6,000 steps while retaining the original 5,000-step cosine schedule.
mechanism: Near-threshold key-bias merge with low-rate refinement
evidence_used: This exact key-bias tie reached 98.92% at 5,000 steps—only eight test examples below the threshold—whereas other 1,610-parameter ties achieved 68.48% or less; continued low-rate optimization is therefore the most targeted next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1610, "training_steps": 5999}

RECENT RESULT
hypothesis: Replacing the remaining independent key-bias group with detached views of the surviving learned key scalar and adding 1,000 low-rate steps will reduce the model from 1,610 to 1,609 parameters while retaining at least 99% accuracy, because the reconstructed offsets remain softmax-invariant without adding the failed group’s gradients to the surviving scalar.
change: Store 13 QKV bias parameters, reconstruct the removed key-bias group from detached views of the surviving key scalar, shift value-bias slices to preserve their mapping, and train for 7,000 steps on the existing 5,000-step cosine schedule.
mechanism: Stop-gradient key-bias group sharing
evidence_used: The current 1,610-parameter key-bias merge reached 99.88% after 1,000 low-rate refinement steps, while earlier direct key-group merges failed; stop-gradient sharing tests whether preserving the successful scalar’s existing gradient paths avoids that optimization failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1609, "training_steps": 6999}

RECENT RESULT
hypothesis: Reconstructing all remaining learned key-bias offsets from a detached query-bias scalar will reduce the model from 1,609 to 1,608 parameters while retaining at least 99% accuracy after 1,000 additional low-rate steps, because key offsets are softmax-invariant and detachment prevents the gradient coupling implicated by the failed direct query/key reuse.
change: Remove the final dedicated key-bias scalar, derive its key offsets from a detached query-bias view, shift value-bias slices to preserve their validated mapping, and train for 8,000 steps on the existing 5,000-step cosine schedule.
mechanism: Detached query-to-key bias reuse
evidence_used: Detached key-group sharing achieved 99.95% at 1,609 parameters after 6,999 steps, while non-detached query/key reuse reached only 12.53%; this directly motivates testing detached reuse of a query scalar with another 1,000 refinement steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1608, "training_steps": 7999}

RECENT RESULT
hypothesis: Representing the shift-invariant eight-dimensional attention projection bias with seven relative coordinates and updating them with an implicit eight-coordinate AdamW state will reduce the model from 1,608 to 1,607 parameters while retaining at least 99% accuracy after 9,000 steps.
change: Anchor the final projection-bias coordinate at zero, reconstruct it during inference, and use a gauge-aware optimizer that preserves the relative updates of the original eight-coordinate AdamW optimization.
mechanism: Gauge-aware anchored attention-output bias
evidence_used: The current 1,608-parameter model reached 99.91%, while naive zero-sum projection-bias reparameterization fell to 46.64%; this tests whether optimizer-geometry distortion, rather than loss of model capacity, caused that exact-gauge reduction to fail.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1607, "training_steps": 8999}

RECENT RESULT
hypothesis: Anchoring one coordinate of the shift-invariant MLP output bias and optimizing its seven relative coordinates with the proven implicit eight-coordinate AdamW state will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps.
change: Store seven `fc2` bias coordinates, reconstruct the eighth as zero, include this bias in gauge-aware optimization, and add 1,000 low-rate refinement steps.
mechanism: Gauge-aware anchored MLP output bias
evidence_used: Gauge-aware anchoring of the attention projection bias achieved 99.89% at 1,607 parameters, while the earlier learned cross-sublayer `fc2` bias tie failed at 49.09%; applying the successful optimizer-preserving gauge treatment avoids that harmful gradient coupling while exploiting the MLP bias’s equivalent uniform-shift invariance.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1103, "parameters": 1606, "training_steps": 9999}

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
