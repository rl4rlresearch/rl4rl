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
verified_results: {"accuracy": 0.9998, "parameters": 1251, "training_steps": 4999}
prior_hypothesis: Canonicalizing each head’s rank-four key projection to an identity basis will reduce the model from 1,275 to 1,251 parameters while retaining at least 99% accuracy, because it preserves the exact attention scores and reconstructs the successful virtual Q/K AdamW gradients through the gauge transformation.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the single global additive degree of freedom shared by every tied embedding coordinate will reduce the model from 1,447 to 1,446 parameters while retaining at least 99% accuracy, because it changes input residuals only by LayerNorm-invisible scalar shifts and output logits only by a softmax-invisible common shift.
change: Store all but one flattened embedding coordinate as differences from a fixed reference, reconstruct the tied input/output weight dynamically, preserve the original initialization RNG stream, and include the omitted coordinate in quotient-aware AdamW moments and gradient clipping.
mechanism: Optimizer-preserving tied-embedding common-mode gauge quotient
evidence_used: The 1,447-parameter model reached 99.97%, and optimizer-preserving gauge quotients successfully removed both MLP residual and relative-attention common modes where direct reparameterizations disrupted training; this applies the same proven optimization treatment to another exact model-wide invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1446, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight LayerNorm-invisible common-mode weights from the attention output projection will reduce the model from 1,446 to 1,438 parameters while retaining at least 99% accuracy when full-width initialization and virtual AdamW moments are preserved.
change: Store seven output rows for the attention projection, reconstruct an eighth zero row, retain its full-width shared value/output bias, and train the omitted row through the existing quotient-aware optimizer.
mechanism: Optimizer-preserving attention residual gauge quotient
evidence_used: Quotient-aware optimization let the seven-row MLP projection reach 99.75% after direct reparameterizations failed at 93.33% and 12.75%; the earlier combined attention/MLP reduction lacked this optimizer-preserving treatment, while the current 1,446-parameter model provides 99.93% accuracy headroom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1438, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing all key projections on seven zero-mean LayerNorm coordinates will reduce the model from 1,438 to 1,430 parameters while retaining at least 99% accuracy, because key constants are softmax-invisible and virtual full-width AdamW moments preserve the successful optimization geometry.
change: Replace the full-width key portion of QKV with an eight-by-seven learned projection over non-affine LayerNorm outputs, retain full-width query/value projections and `ln1` affine parameters, and add the omitted key coordinates to quotient-aware optimization.
mechanism: Optimizer-preserving key-projection LayerNorm quotient
evidence_used: Quotient-aware optimization preserved 99.75% for the MLP residual quotient and 99.93% for the attention-output quotient, whereas removing `ln1` scales failed near 75%; this isolates an exact key-only redundancy without constraining the load-bearing query/value affine pathways.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1430, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing queries on seven zero-mean LayerNorm coordinates will reduce the model from 1,430 to 1,422 parameters while maintaining at least 99% accuracy, because the learned affine transformation can be absorbed into query weights and the independent query bias, while virtual full-width AdamW moments preserve optimization geometry.
change: Split QKV storage into seven-coordinate query and key projections plus a full-width affine value projection, and add the omitted query coordinates to quotient-aware optimization.
mechanism: Optimizer-preserving query-projection LayerNorm quotient
evidence_used: The analogous key-projection quotient reduced the 1,438-parameter model to 1,430 parameters with 99.95% accuracy. Unlike the load-bearing value pathway with its shared value/output bias, queries already have an independent bias that can absorb the LayerNorm affine offset.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1422, "training_steps": 4999}

RECENT RESULT
hypothesis: Collapsing the affine first LayerNorm and full value projection into 56 zero-mean-coordinate weights plus an 8-coordinate value constant will reduce the model from 1,422 to 1,406 parameters while maintaining at least 99% accuracy, because virtual full-width AdamW updates preserve the successful value-path optimization geometry.
change: Replace the affine value pathway with its exact composite representation, retain the shared value/output bias, and train the compressed value parameters through virtual full value weights and LayerNorm scale/shift states.
mechanism: Optimizer-preserving value/LayerNorm composite quotient
evidence_used: Directly fixing one `ln1` scale coordinate fell to 74.8%, while optimizer-preserving query and key LayerNorm quotients reached 99.98% and 99.95%; this motivates quotienting the remaining value-only affine parameters without directly changing their AdamW dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1406, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the shared attention projection bias while compensating it in the value constant will reduce the model from 1,406 to 1,405 parameters and maintain at least 99% accuracy, because the transformation changes the residual only by a LayerNorm-invisible common shift and virtual full-width AdamW preserves the successful optimization geometry.
change: Store seven projection-bias differences, reconstruct the eighth as zero, compensate its virtual reference coordinate in the value bias, and train both original full bias pathways with reconstructed AdamW gradients and moments.
mechanism: Optimizer-preserving coupled value/output-bias gauge quotient
evidence_used: The 1,406-parameter composite value/LayerNorm quotient achieved 99.99%, while direct single-coordinate gauge fixing previously failed; optimizer-preserving quotients repeatedly retained at least 99.75%, motivating this exact one-dimensional coupled quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1405, "training_steps": 4999}

RECENT RESULT
hypothesis: Folding the eight-coordinate value/output constant into the seven observable residual-bias coordinates will reduce the model from 1,405 to 1,397 parameters while retaining at least 99% accuracy, because virtual full-width AdamW gradients preserve the successful optimization trajectory.
change: Remove the learned QKV value bias, use the residual projection bias as the folded effective constant, and reconstruct gradients and updates for the original value/LayerNorm and projection-bias pathways inside the quotient optimizer.
mechanism: Optimizer-preserving attention constant folding
evidence_used: The 1,405-parameter coupled value/output-bias quotient achieved 99.97%, and prior optimizer-preserving quotients consistently retained high accuracy. In that design the value constant and full projection bias affect the residual only through `W_out(c + p) + p`, which has seven observable coordinates.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Folding the value/output constant into the seven observable residual-bias coordinates will produce a 1,397-parameter model with at least 99% accuracy when gradients for the virtual value, LayerNorm, projection-weight, and shared-bias parameters are reconstructed jointly.
change: Remove the eight-parameter value bias, store only the folded residual constant, and extend quotient-aware AdamW to reconstruct and update every virtual parameter contributing to that constant.
mechanism: Optimizer-preserving attention constant folding with joint chain-rule gradients
evidence_used: The 1,405-parameter coupled value/output-bias quotient achieved 99.97%; the previous 1,397 constant-folding implementation could not be verified and therefore provides no accuracy counterevidence. Completing the joint chain rule preserves the successful model’s virtual optimization geometry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1397, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the final hidden state before the tied output projection makes each token embedding’s scalar row offset exactly unobservable, allowing 113 embedding parameters to be removed while the remaining seven-dimensional learned contrasts retain at least 99% accuracy; the resulting model has 1,284 parameters.
change: Store seven within-row embedding differences per token instead of one globally gauge-fixed flattened table, center the learned final representation before computing logits, and preserve full-width AdamW geometry with one virtual reference coordinate per token.
mechanism: Row-centered tied-embedding contrast classifier
evidence_used: The 1,397-parameter model reached 99.93%, while optimizer-preserving quotients repeatedly retained high accuracy. Unlike removing load-bearing LayerNorm scales, this keeps the final affine LayerNorm and instead challenges the assumption that token-specific embedding common modes are needed by making the classifier operate entirely on learned residual contrasts.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1284, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one common-mode coordinate of the final LayerNorm bias will reduce the model from 1,284 to 1,283 parameters while retaining at least 99% accuracy, because the centered tied classifier exactly annihilates uniform hidden-state shifts.
change: Store seven final-LayerNorm bias differences, reconstruct an eighth zero reference coordinate during inference, and preserve full-width AdamW moments through the existing quotient optimizer.
mechanism: Optimizer-preserving final-LayerNorm bias gauge quotient
evidence_used: The 1,284-parameter row-centered tied-embedding model achieved 100% accuracy, and prior optimizer-preserving quotients retained at least 99.75% accuracy across exact residual and attention invariances.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1283, "training_steps": 4999}

RECENT RESULT
hypothesis: Folding query weights, key weights, and query bias into learned bilinear attention scores will reduce the model from 1,283 to 1,275 parameters while retaining at least 99% accuracy, because it preserves the exact attention function and reconstructs the successful full-width AdamW updates through the composite’s chain rule.
change: Replace explicit query/key projections with per-head bilinear and linear score parameters, while maintaining virtual query, key, and query-bias states solely inside the quotient-aware optimizer.
mechanism: Optimizer-preserving Q/K bilinear score folding
evidence_used: The 1,397-parameter attention constant-folding model retained 99.93% after joint chain-rule reconstruction, and the current 1,283-parameter model retained 99.93% with optimizer-preserving quotients; this applies the same demonstrated technique to the exact Q/K factorization redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1275, "training_steps": 4999}

RECENT RESULT
hypothesis: Canonicalizing each head’s rank-four key projection to an identity basis will reduce the model from 1,275 to 1,251 parameters while retaining at least 99% accuracy, because it preserves the exact attention scores and reconstructs the successful virtual Q/K AdamW gradients through the gauge transformation.
change: Replace the 112 stored bilinear-score parameters with 88 canonical query, key-tail, and query-bias parameters, while retaining full virtual query/key optimizer states and recanonicalizing after every update.
mechanism: Optimizer-preserving Q/K basis gauge quotient
evidence_used: The 1,275-parameter folded Q/K model achieved 99.99% using virtual factor updates; the proposed 24-parameter reduction removes only the exact per-head 4-by-4 change-of-basis redundancy of those same factors.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1251, "training_steps": 4999}



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
