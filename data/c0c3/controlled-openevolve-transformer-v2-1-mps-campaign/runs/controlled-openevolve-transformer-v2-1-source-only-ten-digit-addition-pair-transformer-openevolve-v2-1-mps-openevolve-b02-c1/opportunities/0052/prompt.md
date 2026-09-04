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
verified_results: {"accuracy": 0.9995, "parameters": 1046, "training_steps": 4999}
prior_hypothesis: Removing the seven-parameter MLP residual bias will reduce the model from 1,053 to 1,046 parameters while retaining at least 99% accuracy, because the width-1 model achieved 100% and the failed bias-only limit indicates the nonlinear neuron—not the standalone bias—is the load-bearing component.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing the two independent per-head distance tables with one learned positional potential, read positively by one head and with a learned initially negative scale by the other, will reduce the model from 1,173 to 1,152 parameters while retaining at least 99% accuracy because its maxima and minima can encode complementary attention landmarks.
change: Challenge the assumption that each attention head requires an unconstrained positional table: share one gauge-fixed relative-bias vector, fix the first head’s scale to one, and learn the second head’s polarity and magnitude.
mechanism: Complementary signed relative-position potential
evidence_used: The current two-head model achieves 99.98% with only nine MLP units, indicating optimization margin. The successful Q/K and value/output basis quotients also show that attention-head roles survive substantial structural reparameterization, motivating a direct test of complementary rather than independent positional specialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.30510000000000004, "parameters": 1152, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 9 to 8 will lower the model from 1,173 to 1,158 learned parameters while retaining at least 99% accuracy, because the width-9 model achieved 99.98% and three consecutive one-neuron reductions preserved the required accuracy.
change: Change the default feed-forward width from 9 to 8 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Fourth one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 to 11, 11 to 10, and 10 to 9 each removed 15 parameters and achieved 99.91%, 99.96%, and 99.98% accuracy; unlike the failed shared-relative-bias experiment, this continues the strongest demonstrated parameter-reduction trend.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1158, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 8 to 7 will lower the model from 1,158 to 1,143 learned parameters while retaining at least 99% accuracy, because four consecutive one-neuron reductions preserved at least 99.91% accuracy and the width-8 model achieved 99.99%.
change: Change the default feed-forward width from 8 to 7 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Fifth one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 8 each removed 15 parameters and achieved 99.91%, 99.96%, 99.98%, and 99.99% accuracy; this uninterrupted trend makes the next one-neuron ablation the most informative capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1143, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 7 to 6 will lower the model from 1,143 to 1,128 learned parameters while retaining at least 99% accuracy, because the width-7 model achieved 100% accuracy after five consecutive successful one-neuron reductions.
change: Change the default feed-forward width from 7 to 6 while leaving attention, optimization, checkpointing, and protected generation unchanged.
mechanism: Sixth one-neuron feed-forward width ablation
evidence_used: Width reductions from 12 through 7 each removed 15 parameters while maintaining at least 99.91% accuracy, and the latest width-7 model achieved 100%; this uninterrupted trend makes the next single-neuron ablation the most informative capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1128, "training_steps": 4999}

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
