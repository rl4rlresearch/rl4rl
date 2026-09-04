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

RECENT RESULT
hypothesis: Removing a second MLP output-weight coordinate will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because each hidden-feature column has an independent softmax-invisible common-output direction and the first such virtual-AdamW quotient already succeeds.
change: Gauge-fix row 4 in the final two MLP output columns, extending reconstruction, virtual AdamW moments, quotient updates, and gradient clipping from one omitted coordinate to two.
mechanism: Second common-output MLP projection gauge
evidence_used: The verified 1607-parameter model achieves 99.97% accuracy with one common-output MLP weight quotient. A second independent column applies that proven symmetry without extending the key-row family whose eighth quotient failed.
result: training did not finish within the verification time limit

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
