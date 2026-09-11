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

RECENT RESULT
hypothesis: Gauge-fixing token-embedding feature 0 alongside feature 4 will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because shifting an entire token-embedding feature into the matching positional-embedding feature leaves both model inputs and softmax probabilities unchanged, while the generalized optimizer preserves the omitted coordinate’s gradient and AdamW moments.
change: Remove the last-token coordinate of embedding feature 0, reconstruct it as zero, transfer its initialization and virtual optimizer updates into positional feature 0, and generalize gauge-aware clipping to both token–position transfer coordinates.
mechanism: Second virtual-AdamW token–position transfer gauge
evidence_used: The verified 1607-parameter design reaches 99.97% while already using this exact virtual token–position gauge for feature 4; applying it to independent feature 0 tests a distinct symmetry instead of extending the key-row family whose eighth coordinate failed at 46.34%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7336, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing LayerNorm-2 scale coordinate 2 will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because its bias is already fixed at zero and its scale can be absorbed exactly into the corresponding `fc1` input column.
change: Remove LayerNorm-2 scale coordinate 2 and reconstruct it as one, extending the existing fixed scale set from coordinates 0, 1, and 4 to 0, 1, 2, and 4.
mechanism: Fourth downstream-absorbable LayerNorm-scale gauge
evidence_used: The 1607-parameter model achieved 99.97% accuracy while fixing three LayerNorm-2 scale coordinates, and the cited coordinate-1 LayerNorm-scale reduction independently reached 99.81%; coordinate 2 also has zero fixed bias, making it the cleanest remaining multiplicative downstream gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5781000000000001, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing one positional-embedding coordinate will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because adding a common feature shift at one position is exactly invisible to the pre-norm block and final LayerNorm, while reconstructed gradients and full-coordinate AdamW moments preserve optimizer dynamics.
change: Replace the positional embedding with a one-coordinate gauge-fixed embedding and extend the coupled token-position optimizer and gradient clipping to virtually optimize and project the omitted positional coordinate.
mechanism: Virtual-AdamW quotient of a per-position residual-stream shift
evidence_used: The 1607-parameter design achieved 99.97% using several virtual-AdamW quotients, while attempts to extend the key-row, token-transfer, and LayerNorm-scale reductions failed. This tests a distinct exact symmetry with the optimizer-state preservation that previously rescued otherwise destructive coordinate removal.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Gauge-fixing positional embedding coordinate `[0, 7]` will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because a uniform feature shift at position 0 is removed by every pre-norm and the final LayerNorm, while virtual gradients and full-coordinate AdamW moments preserve training dynamics.
change: Replace the positional embedding with a one-coordinate gauge-fixed embedding, reconstruct its omitted gradient, and extend the coupled token-position optimizer and clipping logic to project both the existing transfer gauges and the positional row-shift gauge.
mechanism: Virtual-AdamW quotient of a per-position residual-stream shift
evidence_used: The 1607-parameter design achieved 99.97% with virtual-AdamW gauge quotients, including a token-position embedding gauge; the prior positional-gauge submission was not evaluated because its SEARCH text matched multiple times, so this distinct exact symmetry remains the most informative untested reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5864, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing `fc1.weight[0, 7]` and absorbing its virtual updates into `fc1.bias` will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because this is the first application of the proven LayerNorm-null weight symmetry to the MLP input projection rather than an eighth quotient in the saturated key projection.
change: Replace `fc1` with a one-coordinate gauge-fixed affine layer, reconstruct the omitted gradient, retain full-coordinate AdamW moments and clipping, and compensate each omitted-coordinate update through the affine bias.
mechanism: Virtual-AdamW quotient of one LayerNorm-null MLP input weight
evidence_used: Seven virtual-AdamW key-projection LayerNorm-null quotients reached 99.97% at 1607 parameters, while adding the eighth key quotient failed at 46.34%; applying the same exact symmetry to a distinct affine projection is the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6416, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing key row `d_model + 3` at feature 4 instead of the previously failed feature 7 will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the key-row symmetry is exact but AdamW is coordinate-dependent, and feature 4 already supports successful token-position and LayerNorm-scale gauges.
change: Add the eighth key-row quotient using feature 4 as its anchor and generalize reconstruction, virtual gradients, AdamW projection, and clipping to support per-row anchor columns.
mechanism: Alternate-anchor virtual-AdamW quotient for the final key row
evidence_used: Seven key rows anchored at feature 7 reached 99.97%, while the eighth fell to 46.34%; feature 4 remains a particularly evidence-backed alternative because existing feature-4 token-position and LayerNorm-scale quotients both train successfully.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8759, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing LayerNorm-2 bias coordinate 7 and absorbing its virtual AdamW updates into `fc1.bias` will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy.
change: Remove `ln2.bias[7]`, reconstruct it as zero, and extend the existing affine-bias virtual optimizer and gradient-clipping machinery to preserve its gradient, moments, and downstream effect.
mechanism: Virtual-AdamW quotient of LayerNorm-2 bias feature 7
evidence_used: Virtual optimization previously rescued an omitted LayerNorm-bias coordinate to 99.86%, and the current 1607-parameter design already fixes six LayerNorm-2 bias coordinates. Fixing feature 7 extends that proven gauge while retaining feature 5, the feature index with the strongest evidence of bias-pathway sensitivity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.45990000000000003, "parameters": 1606, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing token-embedding feature 1 alongside feature 4 will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the added quotient is an exact token–position translation symmetry and feature 1 has already tolerated successful LayerNorm scale and bias gauges.
change: Remove the last-token coordinate of embedding feature 1, transfer its initialization into positional feature 1, and generalize gradient reconstruction, AdamW moments, quotient updates, and clipping from one transfer feature to two.
mechanism: Second virtual-AdamW token–position embedding translation quotient
evidence_used: The feature-4 token–position quotient is present in the verified 1607-parameter model at 99.97%; the failed feature-0 extension shows coordinate-dependent optimization sensitivity, while the prior successful feature-1 LayerNorm-scale reduction provides specific evidence that feature 1 is a better-supported alternative.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8789, "parameters": 1606, "training_steps": 4999}



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
