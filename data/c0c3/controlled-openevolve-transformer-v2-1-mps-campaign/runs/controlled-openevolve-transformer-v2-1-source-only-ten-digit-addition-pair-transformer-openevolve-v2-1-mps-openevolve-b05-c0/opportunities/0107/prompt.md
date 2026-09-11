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
verified_results: {"accuracy": 0.9992, "parameters": 1604, "training_steps": 4999}
prior_hypothesis: Gauge-fixing attention-output column 4 instead of the previously failed column 5 will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because columns 7 and 6 already support this exact symmetry and column 4 avoids overlap with the virtual value-bias pathway on feature 5.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing one common-output coordinate from the attention projection will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because token-wise uniform residual shifts are eliminated by subsequent LayerNorms and the proven virtual-AdamW common-output gauge preserves full-coordinate optimization.
change: Gauge-fix `attn.proj.weight[4, 7]`, optimize its omitted coordinate with the existing common-output optimizer, and reconstruct the full projection dynamically for attention and value-bias compensation.
mechanism: Attention-output residual-shift quotient
evidence_used: The verified 1607-parameter model reached 99.97% accuracy with the identical common-output quotient on the MLP projection. Applying it to a separate attention-projection parameter tests the same proven symmetry without the multi-coordinate generalization whose second-MLP-column trial timed out.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second common-output coordinate from the attention projection will reduce the verified model from 1606 to 1605 parameters while retaining at least 99% accuracy, because each projection input column has an independent residual-stream shift symmetry and the first attention-output quotient achieved 99.98% accuracy.
change: Gauge-fix row 4 in the final two attention-output projection columns and generalize virtual AdamW and gradient clipping to reconstruct and optimize multiple omitted common-output coordinates per projection.
mechanism: Second attention-output common-shift quotient
evidence_used: The latest verified design reached 99.98% accuracy after removing one attention-output coordinate. The only prior two-coordinate common-output trial timed out rather than failing accuracy, so extending the newly successful attention-specific quotient is the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1605, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third common-output coordinate from the attention projection will reduce the verified model from 1605 to 1604 parameters while retaining at least 99% accuracy, because each projection input column has an independent residual-stream shift symmetry and the first two attention-output quotients achieved 99.98% and 99.99% accuracy.
change: Gauge-fix row 4 in the final three attention-output projection columns; the existing generic reconstruction, virtual AdamW, and gradient-clipping logic already supports multiple omitted coordinates.
mechanism: Third attention-output common-shift quotient
evidence_used: Extending the attention-output quotient from one to two coordinates reduced parameters from 1606 to 1605 and improved verified accuracy from 99.98% to 99.99%, directly supporting another independent column in the same projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7298, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing attention-output column 4 instead of the previously failed column 5 will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because columns 7 and 6 already support this exact symmetry and column 4 avoids overlap with the virtual value-bias pathway on feature 5.
change: Add a third attention-output projection gauge at row 4, column 4 while leaving the existing generic reconstruction, virtual AdamW, and gradient-clipping machinery unchanged.
mechanism: Alternate-column third attention-output common-shift quotient
evidence_used: Two attention-output quotients reached 99.99% accuracy, but extending them to the final three columns failed at 72.98%; the failed third coordinate was column 5, which the current design also uses for virtual value-bias compensation, motivating an alternate independent column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing attention-output column 3 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because columns 7, 6, and 4 already support independent residual-shift symmetries and column 3 avoids the feature-5 virtual value-bias pathway implicated in the failed column-5 trial.
change: Add a fourth attention-output projection gauge at row 4, column 3 using the existing generic reconstruction, virtual AdamW, and gradient-clipping machinery.
mechanism: Fourth attention-output common-shift quotient
evidence_used: Three attention-output gauges at columns 7, 6, and 4 achieved 99.92% accuracy with 1604 parameters, whereas using column 5 as the third gauge achieved only 72.98%; testing the next non-feature-5 column is the closest controlled extension of the successful design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gauge-fixing attention-output column 3 will reduce the model from 1604 to 1603 parameters while retaining at least 99% accuracy; evaluating only after the final update will avoid the timeout seen in the previous column-3 trial without changing its 5000-step optimization.
change: Add a fourth residual-shift gauge at row 4, column 3 and remove intermediate validation passes while still producing both final-step best and last checkpoints.
mechanism: Fourth attention-output common-shift quotient with final-only validation
evidence_used: Three attention-output gauges at columns 7, 6, and 4 achieved 99.92% accuracy. The direct column-3 extension timed out rather than failing accuracy, so completing that controlled extension with lower validation overhead is the most informative next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gauge-fixing attention-output column 2 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because columns 7, 6, and 4 already support independent residual-shift quotients and column 2 avoids both the feature-5 value-bias pathway and the repeatedly timed-out column-3 configuration.
change: Add a fourth attention-output projection gauge at row 4, column 2 using the existing generic reconstruction, virtual AdamW, and gradient-clipping machinery.
mechanism: Alternate-column fourth attention-output common-shift quotient
evidence_used: Three attention-output gauges at columns 7, 6, and 4 achieved 99.92% accuracy with 1604 parameters; column 5 failed at 72.98%, while column 3 produced two timeouts rather than an accuracy result, motivating the next unused column that does not overlap feature 5.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gauge-fixing one uniform scalar direction of the positional embedding will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because the shift is exactly removed by the pre-norm/final-LayerNorm computation and full-coordinate AdamW moments are preserved virtually.
change: Replace the positional embedding with a one-coordinate gauge-fixed embedding and extend the coupled token-position optimizer and gradient clipping to reconstruct and optimize its omitted coordinate.
mechanism: Virtual-AdamW positional common-shift quotient
evidence_used: The 1604-parameter design achieved 99.92% accuracy using an analogous global-shift quotient in the token embedding and virtual-AdamW common-shift quotients in learned projections; this applies the same proven invariance to the previously untouched positional table without adding another attention-output quotient, whose fourth-coordinate trials timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing LayerNorm-2 scale coordinate 2 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because that scale can be absorbed into the downstream `fc1` weight column and three coordinates already train successfully under the same gauge.
change: Remove LayerNorm-2 scale coordinate 2 by reconstructing it as one, leaving trainable scales only at features 3, 5, 6, and 7.
mechanism: Fourth LayerNorm-scale absorption gauge
evidence_used: The 1604-parameter design achieved 99.92% accuracy with LayerNorm-2 scales 0, 1, and 4 already fixed; extending this proven gauge family avoids another attention-output coordinate after three fourth-coordinate trials timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing LayerNorm-1 bias feature 0 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because its query effect is representable by `q_bias`, its constant-key effect is softmax-invisible, and its value effect is representable by the attention projection bias.
change: Replace the first block’s LayerNorm with a gauge-fixed variant that reconstructs bias feature 0 as zero while retaining all scale coordinates and the other seven learned bias coordinates.
mechanism: LayerNorm-1 bias absorption through attention biases
evidence_used: The 1604-parameter design achieved 99.92% accuracy while fixing six LayerNorm-2 bias coordinates, including feature 0. This tests the same single-coordinate bias gauge in LayerNorm-1 without extending the attention-output gauges whose fourth-coordinate trials repeatedly timed out or touching feature 5’s sensitive value-bias pathway.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Gauge-fixing LayerNorm-2 bias feature 7 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because its effect is exactly absorbable into `fc1.bias` and the existing virtual-AdamW quotient already supports this gauge family.
change: Remove LayerNorm-2 bias feature 7, retain feature 5, and extend the existing affine-bias optimizer mapping to reconstruct and optimize the omitted feature virtually.
mechanism: Seventh LayerNorm-2 bias absorption gauge
evidence_used: The 1604-parameter model achieved 99.92% accuracy with six of eight LayerNorm-2 bias coordinates already fixed; extending that proven absorption mechanism avoids feature 5, which prior evidence identifies as optimization-sensitive.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing LayerNorm-2 bias feature 7 without adding another virtual optimizer coordinate will reduce the model from 1604 to 1603 parameters, finish within the time limit, and retain at least 99% accuracy because its constant downstream effect is absorbable by `fc1.bias`.
change: Retain only feature 5 of the LayerNorm-2 bias and reconstruct feature 7 as zero, leaving the existing optimizer machinery unchanged.
mechanism: Direct LayerNorm-2 bias absorption gauge
evidence_used: The 1604-parameter model reached 99.92% while LayerNorm-2 bias features 0, 2, 3, and 4 were already fixed without virtual-coordinate optimization; the feature-7 trial timed out only after extending that machinery, motivating a direct omission of the same absorbable coordinate.
result: training did not finish within the verification time limit



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
