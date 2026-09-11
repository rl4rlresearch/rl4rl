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
verified_results: {"accuracy": 0.9994, "parameters": 847, "training_steps": 4999}
prior_hypothesis: Replacing the twelve-unit GELU MLP with a seven-unit learned GLU will reduce the model from 866 to 847 parameters while retaining at least 99% accuracy, because seven independent output directions span the entire LayerNorm-visible residual quotient and multiplicative gates provide greater nonlinear efficiency per direction.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing each attention head’s learned query/key width from four dimensions to two will produce a 895-parameter transformer with at least 99% accuracy, because learned relative-lag biases can perform most positional routing while the unchanged four-dimensional value paths, token representation, and MLP retain the task’s load-bearing capacity.
change: Replace the 24-output QKV projection with a 16-output projection containing two-dimensional queries and keys plus full-width values, preserving the conceptual initialization stream and input-shift gauge optimization.
mechanism: Narrow-content positional attention
evidence_used: The four-channel token bottleneck retained 99.84% accuracy while three channels reached only 96.91%, and tying the MLP readout collapsed accuracy to 37.01%; this motivates preserving token, value, and MLP capacity while testing whether full-width content addressing is unnecessary alongside the learned relative-lag table.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 895, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final LayerNorm affine values on its last four channels will reduce the model from 895 to 887 parameters while retaining at least 99% accuracy, because the compact tied output embedding has exact zeros in those channels, so their learned scales and biases cannot affect any token logit.
change: Replace the final full affine LayerNorm with a compact affine LayerNorm that learns only the four logit-visible channel scales and biases while preserving normalization across all eight channels.
mechanism: Dead output-channel affine pruning
evidence_used: The 895-parameter design achieved 99.92% accuracy with a four-channel token embedding padded by four exact zeros; unlike the failed MLP tying experiment, this removes only parameters annihilated by the existing output projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 887, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing all 12 `fc1` weight rows will reduce the model from 887 to 875 learned parameters while retaining at least 99% accuracy, because non-affine `ln2` produces zero-mean inputs, making each row’s common-mode weight exactly unobservable.
change: Replace `fc1` with a row-wise Householder gauge-fixed linear layer, preserve the original conceptual initialization stream and bias, and optimize its coordinates with ambient AdamW.
mechanism: Full MLP input-shift quotient
evidence_used: Gauge-fixing every narrow-QKV row retained 99.92% accuracy in the current design; `fc1` has the same exact input-shift symmetry because it consumes non-affine LayerNorm output.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 875, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the three learned value-bias parameters will reduce the model from 875 to 872 parameters while retaining at least 99% accuracy, because normalized attention passes value bias through as a position-independent vector that the existing projection bias can represent modulo a common residual-stream shift annihilated by downstream LayerNorm.
change: Remove `v_bias` and its addition to the projected values, leaving the seven-parameter attention output bias as the nonredundant branch-bias representation.
mechanism: Value-to-output bias quotient
evidence_used: The 875-parameter design retained 99.92% accuracy after exact LayerNorm-induced gauge fixing; this applies another exact functional quotient rather than reducing the demonstrated token, value-projection, or MLP capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 872, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the sole biased two-dimensional attention head to one query-bias coordinate will reduce the model from 872 to 871 parameters while retaining at least 99% accuracy, because a joint orthogonal rotation of that head’s query and key channels can align any query-bias vector with one axis without changing attention logits.
change: Replace the two learned query-bias coordinates with one scalar coordinate; the existing padding keeps all other query-bias coordinates fixed at zero while preserving initialization and random-stream order.
mechanism: Query-bias rotational gauge fixing
evidence_used: The two-dimensional-query/key design reached 99.92% accuracy, and the current exact-quotient design reached 99.98% at 872 parameters; this removes only the remaining rotationally redundant direction rather than reducing attention capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 871, "training_steps": 4999}

RECENT RESULT
hypothesis: Canonicalizing one query-weight coordinate in the unbiased second attention head will reduce the model from 871 to 870 parameters while retaining at least 99% accuracy, because a joint rotation of that head’s query and key channels preserves every attention logit.
change: Remove one coordinate from the second head’s second query row, jointly rotate its initialized Q/K row pairs into the resulting gauge, and extend ambient AdamW to support the additional fixed coordinate.
mechanism: Unbiased-head Q/K rotational gauge fixing
evidence_used: The previous query-bias rotational gauge retained 99.89% accuracy at 871 parameters, while the narrow two-dimensional Q/K design retained 99.92%; the unbiased head has the same exact joint Q/K rotational symmetry without a bias vector restricting it.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 870, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second query-weight coordinate in the unbiased attention head through a compensating Q/K shear will reduce the model from 870 to 869 learned parameters while retaining at least 99% accuracy, because the inverse-transpose key transformation preserves every attention logit.
change: Triangularize two coordinates of the unbiased head’s query rows, omit the resulting zero coordinate, and generalize ambient AdamW to optimize noncontiguous gauge coordinates.
mechanism: Unbiased-head Q/K shear gauge fixing
evidence_used: The preceding joint Q/K rotational gauge retained 99.90% accuracy at 870 parameters; a shear is another exact dimension of the same Q/K change-of-basis invariance and removes no attention function capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 869, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting one query-weight coordinate in the biased attention head will reduce the model from 869 to 868 learned parameters while retaining at least 99% accuracy, because a Q/K shear that leaves its one-axis query bias unchanged preserves every attention logit.
change: Triangularize one coordinate of the biased head’s first query row using its unbiased second query row, apply the compensating inverse-transpose transformation to its keys, and omit the resulting zero coordinate from training.
mechanism: Biased-head stabilizer shear gauge
evidence_used: The unbiased-head shear gauge retained 99.93% accuracy at 869 parameters; the biased head admits the same exact shear within the subgroup that preserves its existing one-coordinate query bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 868, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero coefficient of the biased head’s first query row will reduce the model from 868 to 867 learned parameters while retaining at least 99% accuracy, because scaling that query channel and its bias while inversely scaling the corresponding key channel preserves every attention logit.
change: Preserve the initialized value of the largest surviving query coefficient as a checkpointed buffer, omit it from learned coordinates, reconstruct it during forward passes, and include the fixed base value in ambient AdamW updates.
mechanism: Biased-head diagonal Q/K scale gauge
evidence_used: The biased-head stabilizer shear retained 99.78% accuracy at 868 parameters; diagonal scaling is another exact Q/K change-of-basis symmetry that preserves the established attention capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 867, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero coefficient of the biased head’s second query row will reduce the model from 867 to 866 learned parameters while retaining at least 99% accuracy, because scaling that query channel while inversely scaling its key channel preserves every attention logit and leaves the one-axis query bias unchanged.
change: Preserve the initialized largest coefficient of the biased head’s second query row as a checkpointed buffer, omit it from learned coordinates, reconstruct it during forward passes, and include the fixed value in ambient AdamW updates.
mechanism: Biased-head second-channel diagonal Q/K scale gauge
evidence_used: Fixing the corresponding scale of the biased head’s first query channel produced the current 867-parameter model at 99.96% accuracy; the remaining independently scalable, bias-free second channel admits the same exact quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 866, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the twelve-unit GELU MLP with a seven-unit learned GLU will reduce the model from 866 to 847 parameters while retaining at least 99% accuracy, because seven independent output directions span the entire LayerNorm-visible residual quotient and multiplicative gates provide greater nonlinear efficiency per direction.
change: Add a generic gauge-fixed output projection, replace the GELU MLP with seven learned value/gate pairs, update ambient optimization for its columns, and set the default gated width to seven.
mechanism: Rank-complete sigmoid-gated MLP bottleneck
evidence_used: Tying the MLP readout collapsed accuracy to 37.01%, showing that independent output directions are load-bearing; this design preserves seven independent directions—the maximum observable rank after common-mode quotienting—while challenging the assumption that twelve separate GELU features are required.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 847, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero weight coordinate in each of the seven GLU value rows will reduce the model from 847 to 840 learned parameters while retaining at least 99% accuracy, because independently scaling a value row and inversely scaling its output-projection column leaves the learned MLP function unchanged.
change: Gauge-fix the seven GLU value-row scales at their largest initialized coordinates, reconstruct those coordinates from checkpointed buffers, and optimize the remaining weights in ambient space.
mechanism: GLU value-channel scale quotient
evidence_used: The seven-unit GLU achieved 99.94% accuracy at 847 parameters while preserving seven independent output directions; the proposed change retains every direction and removes only seven independent continuous scaling redundancies between value rows and their corresponding output columns.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4512, "parameters": 840, "training_steps": 4999}



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
