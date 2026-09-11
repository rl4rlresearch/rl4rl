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
verified_results: {"accuracy": 0.9997, "parameters": 1607, "training_steps": 4999}
prior_hypothesis: Gauge-fixing key row `d_model + 2` will reduce the verified model from 1608 to 1607 parameters while retaining at least 99% accuracy, because the identical virtual-optimizer gauge already succeeds across six key rows.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 2 will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its constant downstream contribution can be absorbed by `fc1.bias`.
change: Remove LayerNorm-2 bias coordinate 2 alongside coordinates 0, 3, and 4, reconstructing all four as zeros during forward passes.
mechanism: Fourth LayerNorm-bias/downstream-affine gauge
evidence_used: The current three-coordinate bias quotient achieved 99.86% at 1614 parameters. Coordinate 2 is the most informative untested bias coordinate, while coordinate 1 already has direct negative evidence at 87.11%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 6 will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy, because its constant contribution can be absorbed by `fc1.bias`.
change: Remove LayerNorm-2 bias coordinate 6 alongside coordinates 0, 2, 3, and 4, reconstructing all five as zeros during forward passes.
mechanism: Fifth LayerNorm-bias/downstream-affine gauge at coordinate 6
evidence_used: Successive LayerNorm-2 bias reductions reached 99.96% at 1613 parameters; coordinate 6 is an untested retained coordinate that avoids feature 5’s empirically essential value-bias pathway and feature 7’s use as the embedding and QKV gauge anchor.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.732, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing LayerNorm-2 bias coordinate 6 while preserving its omitted gradient, AdamW moments, and downstream `fc1.bias` compensation will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy.
change: Remove LayerNorm-2 bias coordinate 6, reconstruct it as zero, and optimize its virtual coordinate through the exact downstream-affine gauge.
mechanism: Virtual AdamW for an affine LayerNorm-bias gauge
evidence_used: Directly fixing coordinate 6 reduced accuracy to 73.2%, while the current four-coordinate quotient reaches 99.96%; restoring the removed coordinate’s optimization pathway directly tests whether optimizer dynamics caused that failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Virtually optimizing omitted LayerNorm-2 bias coordinates 1 and 6 will reduce the model from 1612 to 1611 parameters while retaining at least 99% accuracy, because their learned updates remain represented through independent AdamW moments and compensation in `fc1.bias`.
change: Remove LayerNorm-2 bias coordinate 1, generalize the existing affine-gauge optimizer and gradient clipping to two omitted coordinates, and transfer both virtual updates through the downstream affine bias.
mechanism: Dual virtual-AdamW LayerNorm-bias affine gauge
evidence_used: Virtual optimization rescued coordinate 6 from 73.2% to 99.86%; coordinate 1’s simple fixing reached 87.11%, while related coordinate-1 value-bias and LayerNorm-scale reductions reached 99.98% and 99.81%, making optimizer restoration the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Virtually optimizing omitted LayerNorm-2 bias coordinate 5 alongside coordinates 1 and 6 will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy.
change: Remove LayerNorm-2 bias coordinate 5, reconstruct only coordinate 7 explicitly, and extend the downstream `fc1.bias` compensation and virtual optimizer state to coordinates 1, 5, and 6.
mechanism: Triple virtual-AdamW LayerNorm-bias affine gauge
evidence_used: Dual virtual optimization of coordinates 1 and 6 achieved 99.98%, and virtual optimization rescued coordinate 6 from 73.2% to 99.86%. Coordinate 5 is the remaining non-anchor bias coordinate; its analogous value-bias pathway was empirically essential, motivating preserved virtual optimization instead of simple fixing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7412000000000001, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing `fc2.weight[4, 11]` while retaining its virtual gradient and AdamW moments will reduce the model from 1611 to 1610 parameters and preserve at least 99% accuracy, because shifting that entire input column equally across output features is invisible to the final LayerNorm.
change: Gauge-fix one MLP output-weight coordinate, reconstruct the full projection during inference, and train the quotient with full-coordinate AdamW dynamics and gauge-aware clipping.
mechanism: Virtual-AdamW quotient of an MLP common-output weight direction
evidence_used: The 1611-parameter design achieved 99.98% while already quotienting the analogous common-output bias direction; virtual AdamW also rescued an omitted LayerNorm-bias coordinate from 73.2% to 99.86%. This tests a distinct exact null direction instead of extending the triple LayerNorm-bias change that fell to 74.12%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc2.weight[4, 10]` alongside `[4, 11]` will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because each affected input column can shift all output features equally and is therefore independently invisible to the final LayerNorm.
change: Remove a second MLP output-weight coordinate, reconstruct both fixed coordinates during inference, and generalize virtual AdamW moments and gradient clipping to both common-output directions.
mechanism: Dual virtual-AdamW quotient of independent MLP common-output weight directions
evidence_used: Removing `fc2.weight[4, 11]` with virtual-gradient and AdamW-state preservation achieved 99.96% accuracy at 1610 parameters; applying the same verified exact gauge to the adjacent independent input column is the most direct next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7297, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `attn.proj.weight[4, 7]` will reduce the verified model from 1610 to 1609 parameters while retaining at least 99% accuracy, because a common shift across the projection’s output features is removed by subsequent LayerNorms.
change: Replace the attention output projection with a one-coordinate gauge-fixed projection and train its omitted coordinate using the existing virtual-gradient, AdamW-moment, and gauge-aware clipping machinery.
mechanism: Virtual-AdamW quotient of an attention common-output weight direction
evidence_used: The analogous single-coordinate MLP output-weight quotient achieved 99.96% at 1610 parameters, while adding a second quotient within that same MLP matrix failed at 72.97%; testing the same exact null direction in the distinct attention projection is therefore the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6715000000000001, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the sole learned value-bias coordinate while preserving its virtual gradient, AdamW moments, gauge-aware clipping, and equivalent attention-projection bias updates will reduce the model from 1610 to 1609 parameters while maintaining at least 99% accuracy.
change: Eliminate `v_bias[5]`, reconstruct a zero value bias, and train its omitted coordinate virtually by absorbing each update into the gauge-fixed attention projection bias.
mechanism: Virtual-AdamW quotient of the remaining attention value bias
evidence_used: Feature 5 is the only retained and empirically essential value-bias pathway, so simple removal is risky; virtual optimization previously rescued an omitted LayerNorm-bias coordinate from 73.2% to 99.86%, supporting preservation of optimizer dynamics for this exact downstream-affine gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing key row `d_model + 1` will reduce the verified model from 1609 to 1608 parameters while retaining at least 99% accuracy, because it uses the same exact LayerNorm-null key-weight gauge and virtual optimizer already applied successfully to five other key rows.
change: Add the second row of the first attention head to `GaugeFixedQKV.fixed_rows`; the existing generalized reconstruction, virtual-gradient recovery, AdamW moments, and gauge-aware clipping automatically cover it.
mechanism: Sixth virtual-AdamW key-projection LayerNorm-null quotient
evidence_used: The verified 1609-parameter model achieved 99.97% accuracy while quotienting five key-projection rows with this machinery, making another independent remaining key row the most direct evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1608, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing key row `d_model + 2` will reduce the verified model from 1608 to 1607 parameters while retaining at least 99% accuracy, because the identical virtual-optimizer gauge already succeeds across six key rows.
change: Add the third row of the first attention head to `GaugeFixedQKV.fixed_rows`; existing reconstruction, virtual-gradient recovery, AdamW moments, and gauge-aware clipping automatically generalize to it.
mechanism: Seventh virtual-AdamW key-projection LayerNorm-null quotient
evidence_used: Adding key row `d_model + 1` reduced the model from 1609 to 1608 parameters with 99.95% accuracy, directly validating another independent application of this exact gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1607, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing key row `d_model + 3` will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the identical virtual-optimizer gauge has succeeded across seven key rows, including the three adjacent rows of the first attention head.
change: Add the remaining row of the first attention head to `GaugeFixedQKV.fixed_rows`; the existing reconstruction, virtual-gradient recovery, AdamW moments, and gauge-aware clipping automatically cover it.
mechanism: Eighth virtual-AdamW key-projection LayerNorm-null quotient
evidence_used: Adding key row `d_model + 2` reduced the model from 1608 to 1607 parameters with 99.97% accuracy, directly validating the same gauge on the adjacent first-head key row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.46340000000000003, "parameters": 1606, "training_steps": 4999}



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
