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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1621, "training_steps": 4999}
prior_hypothesis: Fixing MLP output-bias index 4 will reduce the model from 1622 to 1621 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing value-bias index 4 in addition to indices 0, 1, 2, 6, and 7 will reduce the model from 1624 to 1623 parameters while maintaining at least 99% accuracy.
change: Store only value-bias coordinates 3 and 5, reconstructing the eight-dimensional bias with zeros at indices 0, 1, 2, 4, 6, and 7.
mechanism: Sixth value/output-bias quotient coordinate
evidence_used: Among the remaining coordinates, index 4 produced the strongest prior unsuccessful removal at 94.20%, compared with 30.65% for index 3 and 4.06% for index 5; later successful removals of indices 1, 2, and 0 may provide the optimization geometry needed for this exact gauge to train successfully.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing value-bias index 3 in addition to indices 0, 1, 2, 4, 6, and 7 will reduce the model from 1623 to 1622 parameters while maintaining at least 99% accuracy.
change: Store only value-bias coordinate 5 and reconstruct the eight-dimensional bias with zeros at every other index.
mechanism: Seventh value/output-bias quotient coordinate
evidence_used: Index 3 is the strongest remaining candidate: its prior removal reached 30.65%, versus 4.06% for index 5, and the successful index-4 removal showed that later quotient reductions can improve optimization enough for a previously unsuccessful coordinate to train.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the MLP output bias will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because an all-coordinate bias shift survives only in the residual stream and is removed by subsequent LayerNorms.
change: Store seven MLP output-bias coordinates, reconstruct the eighth as zero during the learned projection, and train the reduced bias with the existing virtual-coordinate AdamW and gradient-clipping logic.
mechanism: Post-MLP residual common-shift quotient
evidence_used: The 1622-parameter design reached 99.92%, while additional first-head key gauges repeatedly failed and the remaining value-bias coordinate was historically the weakest removal candidate; the current successful attention projection-bias quotient already uses the same LayerNorm-null common-shift mechanism and optimizer treatment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.078, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final learned value-bias coordinate will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because position-independent value bias is functionally absorbable into the attention output bias.
change: Remove the remaining scalar `v_bias` parameter and omit value-bias addition in attention.
mechanism: Complete value/output-bias quotient
evidence_used: Successively fixing seven of eight value-bias coordinates reached 99.92% accuracy at 1622 parameters; despite index 5 being the historically weakest candidate, the later successes of previously failing indices 3 and 4 make completing this proven quotient mechanism more informative than retrying the MLP-bias or first-head key gauges that collapsed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.882, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one all-coordinate shift per positional embedding row will reduce the model by `max_seq_len` learned parameters while retaining at least 99% accuracy, because with zero dropout each shift passes unchanged through residual connections and is removed by every subsequent LayerNorm.
change: Replace the positional embedding with row-wise gauge-fixed parameters and train each row using the existing virtual-coordinate AdamW and gradient-clipping logic.
mechanism: Per-position residual-stream shift quotient
evidence_used: The 1622-parameter model achieved 99.92% using the same virtual-coordinate optimizer for successful exact shift quotients; unlike the failed final value-bias and MLP-bias reductions, every positional-row shift is independently invisible to the pre-LayerNorm blocks and final LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7341, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one second-LayerNorm bias coordinate at zero will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because the following MLP input projection’s full bias can exactly absorb that coordinate.
change: Replace the second LayerNorm with an equivalent gauge-fixed LayerNorm storing seven bias coordinates and padding the eighth with zero.
mechanism: Downstream-affine LayerNorm bias absorption
evidence_used: The 1622-parameter design reached 99.92% accuracy; unlike the failed MLP output-bias reduction, this removes an upstream bias coordinate that is directly and exactly representable by the full `fc1` bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.374, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing MLP output-bias index 1 instead of the previously failed final coordinate will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy.
change: Store the seven MLP output-bias coordinates other than index 1, reconstruct index 1 as zero, and train the reduced bias with the existing virtual-coordinate AdamW and gradient-clipping logic.
mechanism: Coordinate-specific post-MLP residual-shift quotient
evidence_used: The final-coordinate MLP-bias gauge failed at 7.8%, but value-bias reductions showed extreme coordinate sensitivity: index 1 succeeded at 99.98% while several other coordinates failed. This directly motivates testing index 1 for the same exact residual-shift quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4989, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final value-bias scalar while retaining its adaptive gradient moment as a virtual quotient direction will reduce the model to 1621 parameters and restore at least 99% accuracy.
change: Eliminate `v_bias`; reconstruct its gradient from the projection column, maintain scalar Adam moments without a trainable parameter, and continuously fold each update into the gauge-fixed projection bias.
mechanism: Virtual Adam coordinate for the absorbed value bias
evidence_used: Directly removing the scalar reached 88.2% versus 99.92% with it present, despite its exact absorbability into projection bias, indicating lost optimization geometry rather than lost model capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.07139999999999999, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing MLP output-bias index 0 will reduce the verified model from 1622 to 1621 parameters while maintaining at least 99% accuracy; value-bias evidence identifies index 0 as substantially more trainable than the previously failed MLP coordinates 1 and 7.
change: Store MLP output-bias coordinates 1–7, reconstruct coordinate 0 as zero, and include the reduced bias in the existing virtual-coordinate AdamW and gradient-clipping path.
mechanism: Coordinate-specific post-MLP residual-shift quotient
evidence_used: MLP output-bias removals at indices 7 and 1 reached only 7.8% and 49.89%, but analogous value-bias gauges were strongly coordinate-sensitive: index 0 reached 96.92% in its earlier adverse configuration and later supported 99.94%, motivating a coordinate-specific test rather than abandoning this exact quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4067, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing positional coordinate `[0, 0]` and folding its virtual optimizer update through the token/position additive symmetry will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy.
change: Replace the positional embedding with a one-coordinate gauge-fixed embedding, preserve the original functional initialization by transferring its anchor into the tied token embedding, and reconstruct the omitted coordinate’s gradient and AdamW update during training.
mechanism: Token–position additive gauge with a virtual AdamW coordinate
evidence_used: The 1622-parameter design achieved 99.92%; the 1599-parameter positional-row experiment removed 23 coordinates simultaneously and reached 73.41%, motivating a single exact positional gauge with retained virtual optimization geometry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6657, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing MLP output-bias index 4 will reduce the model from 1622 to 1621 parameters while retaining at least 99% accuracy.
change: Replace the MLP output projection with an initialization-preserving projection whose eight-dimensional bias is reconstructed with coordinate 4 fixed to zero, and optimize its seven learned coordinates through the existing virtual-coordinate AdamW path.
mechanism: Coordinate-4 post-MLP residual-shift quotient
evidence_used: The previously tested MLP coordinates 0, 1, and 7 failed, demonstrating coordinate sensitivity; value-bias coordinate 4 is the strongest untested analogue because its later removal achieved 100% accuracy despite an earlier adverse result.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the all-coordinate shift of positional row 0 will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because the shift is removed by every subsequent LayerNorm and its virtual AdamW coordinate preserves full-row optimization geometry.
change: Replace positional row 0 with seven gauge-fixed coordinates, retain ordinary parameters for all other rows, and optimize the reduced row through the existing virtual-coordinate AdamW and gradient-clipping path.
mechanism: Single-position residual-stream shift quotient
evidence_used: The 1621-parameter MLP gauge reached 99.93%, proving another exact one-coordinate quotient can train successfully; the 1599-parameter positional experiment removed all 23 row shifts simultaneously and reached 73.41%, motivating the more informative test of one isolated positional-row quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.37729999999999997, "parameters": 1620, "training_steps": 4999}



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
