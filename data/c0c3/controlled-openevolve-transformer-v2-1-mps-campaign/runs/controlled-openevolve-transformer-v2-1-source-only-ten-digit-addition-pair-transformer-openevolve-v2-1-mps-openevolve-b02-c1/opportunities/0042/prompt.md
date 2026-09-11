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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1158, "training_steps": 4999}
prior_hypothesis: Reducing the MLP width from 9 to 8 will lower the model from 1,173 to 1,158 learned parameters while retaining at least 99% accuracy, because the width-9 model achieved 99.98% and three consecutive one-neuron reductions preserved the required accuracy.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Canonicalizing each head’s rank-four value projection to an identity basis will reduce the model from 1,251 to 1,219 parameters while retaining at least 99% accuracy, because it preserves the exact attention output and reconstructs the successful virtual value/output AdamW gradients through the gauge transformation.
change: Replace the 56 stored value-projection parameters with 24 canonical tail parameters, transform each head’s output-projection columns into the matching basis, and retain/update the original virtual value and output factors inside the quotient-aware optimizer.
mechanism: Optimizer-preserving per-head value/output basis gauge quotient
evidence_used: The analogous optimizer-preserving Q/K basis canonicalization removed 24 parameters and achieved 99.98% accuracy; the value/output factors have the same exact per-head four-dimensional change-of-basis invariance, while the 1,397-parameter joint attention folding result shows that reconstructing coupled value, projection, and bias gradients can retain 99.93% accuracy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Canonicalizing each head’s value projection while retaining the pre-canonical projection differences and reconstructing their joint virtual gradients will reduce the model from 1,251 to 1,219 parameters while maintaining at least 99% accuracy.
change: Store only each value head’s three-column canonical tail, absorb its learned four-dimensional basis into the corresponding output-projection columns, and update the original virtual value, LayerNorm, projection, and bias parameters before recanonicalizing after every optimizer step.
mechanism: Optimizer-preserving per-head value/output basis gauge quotient
evidence_used: Q/K basis canonicalization removed 24 parameters and achieved 99.98%, while joint attention folding achieved 99.93%. The earlier 1,219-parameter value/output attempt could not be verified and therefore supplies no accuracy counterevidence; this patch explicitly retains the virtual pre-canonical projection state needed for the coupled chain rule.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1219, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final LayerNorm’s eighth scale coordinate to one will reduce the model from 1,219 to 1,218 parameters while retaining at least 99% accuracy, because any positive value of that coordinate can be absorbed by globally scaling the tied embedding and residual-branch outputs while inversely scaling the final LayerNorm affine output.
change: Store seven learned final-LayerNorm scale coordinates and reconstruct the eighth as the constant one; leave the protected decoding path and all other learned modules unchanged.
mechanism: Global residual-stream scale gauge fixing
evidence_used: The current 1,219-parameter model achieved 99.97%, and the centered tied-embedding design previously preserved 100% accuracy after removing exact final-classifier gauges. This change removes another exact representational gauge rather than constraining attention, MLP, or embedding contrasts.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1218, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 12 to 11 will lower the model from 1,218 to 1,203 learned parameters while retaining at least 99% accuracy, because the current model reaches 99.99% and the change removes only one of twelve nonlinear hidden units without altering attention or decoding.
change: Change the default feed-forward width to 11; training, optimizer behavior, checkpointing, and the protected generation interface remain unchanged.
mechanism: One-neuron feed-forward width ablation
evidence_used: The 1,218-parameter design achieved 99.99% accuracy after 4,999 steps, providing the strongest available evidence that a conservative capacity reduction can preserve the required 99% threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1203, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 11 to 10 will lower the model from 1,203 to 1,188 learned parameters while retaining at least 99% accuracy, because the width-11 model achieved 99.91% without changing the attention or decoding pathways.
change: Change the default feed-forward width from 11 to 10; leave training, optimizer behavior, checkpointing, and protected generation unchanged.
mechanism: Second one-neuron feed-forward width ablation
evidence_used: The immediately preceding width reduction from 12 to 11 removed 15 parameters and still achieved 99.91% accuracy, making the next single-neuron ablation the most direct capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1188, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the MLP width from 10 to 9 will lower the model from 1,188 to 1,173 learned parameters while retaining at least 99% accuracy, because the width-10 model achieved 99.96% without changing the attention or decoding pathways.
change: Change the default feed-forward width from 10 to 9; leave training, optimizer behavior, checkpointing, and protected generation unchanged.
mechanism: Third one-neuron feed-forward width ablation
evidence_used: Consecutive reductions from width 12 to 11 and then 10 each removed 15 parameters while achieving 99.91% and 99.96% accuracy, making the next single-neuron ablation the clearest capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1173, "training_steps": 4999}

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
