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
verified_results: {"accuracy": 0.9979, "parameters": 1044, "training_steps": 4999}
prior_hypothesis: Tying each head’s two farthest relative-distance biases will reduce the model from 1,046 to 1,044 parameters while retaining at least 99% accuracy, because it preserves independent, otherwise unconstrained positional tables and alters only the two sparsest causal distances.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the MLP width from 6 to 5 will lower the model from 1,128 to 1,113 learned parameters while retaining at least 99% accuracy, because six consecutive one-neuron reductions have preserved the requirement and the width-6 model achieved 99.88%.
change: Change the default feed-forward width from 6 to 5 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Seventh one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 6 each removed 15 parameters while maintaining at least 99.88% accuracy; continuing this demonstrated trend is the most direct test of the remaining MLP capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1113, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 5 to 4 will lower the model from 1,113 to 1,098 learned parameters while retaining at least 99% accuracy, because seven consecutive one-neuron reductions have met the requirement and the width-5 model achieved 99.92%.
change: Change the default feed-forward width from 5 to 4 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Eighth one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 5 each removed 15 parameters while maintaining at least 99.88% accuracy; the latest width-5 result reached 99.92%, making the next single-neuron ablation the most informative remaining capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1098, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 4 to 3 will lower the model from 1,098 to 1,083 learned parameters while retaining at least 99% accuracy, because eight consecutive one-neuron reductions have met the requirement and the width-4 model achieved 99.96%.
change: Change the default feed-forward width from 4 to 3 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Ninth one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 4 each removed 15 parameters while maintaining at least 99.88% accuracy; the latest width-4 result reached 99.96%, so the next single-neuron ablation is the most direct test of the remaining MLP capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1083, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 3 to 2 will lower the model from 1,083 to 1,068 learned parameters while retaining at least 99% accuracy, because nine consecutive one-neuron reductions have met the requirement and the width-3 model achieved 99.94%.
change: Change the default feed-forward width from 3 to 2 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Tenth one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 3 each removed 15 parameters while maintaining at least 99.88% accuracy; the latest width-3 result reached 99.94%, making the next single-neuron ablation the most informative remaining capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1068, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 2 to 1 will lower the model from 1,068 to 1,053 learned parameters while retaining at least 99% accuracy, because all ten preceding one-neuron reductions met the requirement and the width-2 model achieved 99.78%.
change: Change the default feed-forward width from 2 to 1 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Eleventh one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 2 each removed 15 parameters while maintaining at least 99.78% accuracy; this uninterrupted trend makes the final single-neuron ablation the most informative capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1053, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the final one-neuron MLP with its learned residual bias will reduce parameters from 1,053 to 1,038 while retaining at least 99% accuracy, because every preceding one-neuron ablation succeeded and the width-1 model achieved 100%.
change: Remove the last nonlinear feed-forward unit while retaining its seven-parameter residual bias, preserve the initialization RNG stream, and remove the deleted projection from the quotient optimizer.
mechanism: Bias-only feed-forward limit
evidence_used: The width-1 design achieved 100% accuracy after eleven consecutive successful width reductions; a bias-only branch continues the same 15-parameter ablation while isolating whether any nonlinear MLP unit remains necessary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2329, "parameters": 1038, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing each head’s unconstrained relative-distance table with its own learned affine pointer and learned sharpness will reduce parameters by `2*(INPUT_LEN-1)-6` while retaining at least 99% accuracy; head independence, rather than a free parameter at every distance, is the load-bearing positional requirement.
change: Each attention head learns a Gaussian-shaped positional focus whose center moves affinely with the query position; remove the relative-bias tables and their quotient-optimizer handling while preserving all token, value, nonlinear, and decoding pathways.
mechanism: Independent learned affine positional pointers
evidence_used: Sharing one signed positional table collapsed accuracy to 30.51%, showing that distinct head roles are essential. The width-1 model nevertheless reached 100%, suggesting the successful computation is highly structured; independent learned affine pointers preserve separate operand-alignment roles while testing whether arbitrary per-distance tables are unnecessary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1015, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the seven-parameter MLP residual bias will reduce the model from 1,053 to 1,046 parameters while retaining at least 99% accuracy, because the width-1 model achieved 100% and the failed bias-only limit indicates the nonlinear neuron—not the standalone bias—is the load-bearing component.
change: Delete the learned residual bias from the final one-neuron MLP while preserving its learned GELU input and output projections.
mechanism: Complementary nonlinear-branch isolation
evidence_used: The width-1 MLP reached 100% accuracy, whereas retaining only its residual bias collapsed accuracy to 23.29%; the complementary ablation directly tests whether that bias contributes anything when the necessary nonlinear pathway remains intact.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1046, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final MLP neuron’s scalar input bias will reduce the model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because the nonlinear pathway is necessary but the latest successful result shows its seven-parameter residual bias is not.
change: Make the width-one MLP input projection bias-free while consuming the removed constructor draw so every remaining parameter keeps the current initialization stream.
mechanism: Bias-free nonlinear bottleneck
evidence_used: The width-one MLP achieved 100% accuracy, the bias-only replacement collapsed to 23.29%, and removing the nonlinear branch’s residual bias still achieved 99.95%; this isolates the only remaining scalar bias in the load-bearing nonlinear unit with the smallest possible ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.37810000000000005, "parameters": 1045, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the seven learned final LayerNorm bias coordinates will reduce the model from 1,046 to 1,039 parameters while retaining at least 99% accuracy, because the successful residual-bias ablation indicates constant feature offsets are dispensable, while the failed MLP input-bias ablation specifically preserves evidence that offsets before the nonlinear bottleneck are essential.
change: Make the final LayerNorm bias-free and remove its deleted bias from the quotient optimizer.
mechanism: Bias-free final normalization
evidence_used: Removing the seven-parameter MLP residual bias retained 99.95% accuracy, whereas removing the scalar MLP input bias collapsed accuracy to 37.81%; this motivates testing another post-computation feature offset without disturbing the load-bearing nonlinear bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1564, "parameters": 1039, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one final LayerNorm bias coordinate to its common-mode reference will reduce the model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because it preserves six learned feature offsets rather than deleting the entire essential bias pathway.
change: Store six final LayerNorm bias coordinates and pad two reference-valued coordinates; the existing quotient optimizer automatically trains the resulting seven-coordinate virtual bias representation.
mechanism: Single-coordinate final-normalization bias tie
evidence_used: Removing all seven final LayerNorm bias coordinates collapsed accuracy to 15.64%, while removing the seven-parameter MLP residual bias retained 99.95%; this makes a one-coordinate final-bias ablation the most informative test of whether the essential offset pathway contains any residual redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1967, "parameters": 1045, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying each head’s two farthest relative-distance biases will reduce the model from 1,046 to 1,044 parameters while retaining at least 99% accuracy, because it preserves independent, otherwise unconstrained positional tables and alters only the two sparsest causal distances.
change: Store two fewer relative-bias coordinates and reconstruct both farthest distances from each head’s existing omitted reference; the quotient optimizer continues to train the resulting per-head virtual reference automatically.
mechanism: Per-head farthest-distance bias tying
evidence_used: Independent affine positional pointers failed at 0% accuracy, showing that aggressive table compression loses essential positional structure, while the current independent-table model achieved 99.95%; a single boundary tie per head is the smallest structured positional ablation that preserves head independence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1044, "training_steps": 4999}



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
