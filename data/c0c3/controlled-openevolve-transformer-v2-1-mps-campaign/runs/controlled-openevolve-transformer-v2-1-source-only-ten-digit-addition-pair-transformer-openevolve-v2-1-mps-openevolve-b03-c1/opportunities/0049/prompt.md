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
verified_results: {"accuracy": 0.9998, "parameters": 1194, "training_steps": 4999}
prior_hypothesis: Reducing `d_ff` from 12 to 11 will lower the model from 1,209 to 1,194 learned parameters while retaining at least 99% accuracy, because the successful zero-mean MLP output compression indicates remaining MLP redundancy, whereas the failed rank-five lexical bottleneck shows lexical rank is the riskier capacity target.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining every value-projection row to the zero-mean input subspace will reduce the model from 1,462 to 1,454 learned parameters while retaining at least 99% accuracy, because the removed component is position-independent after `ln1` and is exactly representable by the existing value bias.
change: Store each 8-by-8 value projection as 8-by-7 coefficients in the existing orthonormal zero-mean basis and reconstruct its centered weight during every forward pass.
mechanism: LayerNorm-nullspace value-weight centering
evidence_used: The analogous key-weight centering reached 100% accuracy at 1,462 parameters, and MLP input-weight centering reached 99.95%; value centering uses the same LayerNorm-null direction while retaining a full value bias to represent its constant contribution.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1454, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the eight-dimensional value bias with a seven-dimensional zero-mean attention output bias will reduce learned parameters from 1,454 to 1,453 while retaining at least 99% accuracy, because zero-dropout attention weights sum to one and the centered output projection maps the value bias to only seven observable residual-stream dimensions.
change: Retain only the six query-bias parameters, learn seven coefficients for a centered attention output bias, and add that bias after the value aggregation and output projection.
mechanism: Direct zero-mean attention output bias
evidence_used: Attention-output centering achieved 99.93% accuracy at 1,470 parameters, establishing that this projection emits only into the seven-dimensional zero-mean subspace; the current successful 1,454-parameter design retains an eight-dimensional value bias whose position-independent effect passes through that centered projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1453, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing `ln1_bias`, restoring all eight direct query-bias coordinates, and centering every query-weight row will reduce learned parameters from 1,453 to 1,439 while retaining at least 99% accuracy, because the attention input then comes directly from zero-mean LayerNorm, while the removed constant query and value effects remain representable by the existing direct query and centered output biases.
change: Delete the redundant eight-parameter pre-attention bias, expand the query bias from six to eight parameters, and store query weights in the same seven-dimensional zero-mean basis already used successfully for key and value weights.
mechanism: LayerNorm-null query projection with absorbed attention input bias
evidence_used: Zero-mean input parameterizations retained 100% accuracy for key weights and 99.95% for MLP input weights, while the current direct centered attention output bias reached 99.97%; together these results support applying the same LayerNorm-nullspace constraint to queries after absorbing the redundant pre-attention affine bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1439, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the dense 114-by-8 tied token table with a learned rank-six factorization will reduce parameters from 1,439 to 1,259 while retaining at least 99% accuracy, because token identity and output classification can share a six-dimensional learned latent space while the full-width attention pathway remains intact.
change: Initialize the existing dense tied token table, truncate it by SVD into learned token codes and a learned 6-by-8 projection, and use their product for both input embeddings and output logits.
mechanism: Rank-six tied lexical bottleneck
evidence_used: The rank-four positional representation achieved 99.96%, showing that useful task representations tolerate substantial latent bottlenecks; conversely, shared-key attention collapsed to 36.45%, motivating compression of the much larger lexical table while preserving all query, key, and value routing capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1259, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the 36-dimensional factorization gauge from the successful rank-six tied token table will reduce learned parameters from 1,259 to 1,223 while retaining at least 99% accuracy, because the reconstructed token table has unchanged rank-six expressivity and is initialized identically.
change: Select the best-conditioned six feature columns as a fixed coordinate chart, absorb their invertible projection into the token codes, learn only the remaining 6-by-2 projection, and reconstruct the full tied token table during forward passes.
mechanism: Gauge-fixed rank-six lexical chart
evidence_used: The rank-six tied lexical bottleneck achieved 99.87% accuracy at 1,259 parameters; its two learned factors contain an exact 6-by-6 change-of-basis redundancy, so gauge-fixing that redundancy is a more conservative next reduction than lowering lexical rank.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1223, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing four well-conditioned key coordinates to the identity in each attention head will reduce learned parameters from 1,223 to 1,191 while retaining at least 99% accuracy, because compensating transformations of the corresponding queries and query biases preserve every initialized attention score exactly without reducing query/key rank.
change: For each head, select an invertible 4-by-4 key pivot, absorb it into the query projection and bias, store only the remaining 4-by-3 key coefficients, and reconstruct the full key projection during forward passes.
mechanism: Per-head query–key gauge-fixed coordinate charts
evidence_used: Removing the 36-dimensional rank-six lexical factorization gauge preserved 99.96% accuracy at 1,223 parameters; this applies the same best-conditioned coordinate-chart strategy to the exact per-head query–key change-of-basis redundancy while preserving the full routing capacity whose importance was demonstrated by the failed shared-key design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9592, "parameters": 1191, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s key pivot to its original initialized matrix will reduce learned parameters from 1,223 to 1,191 while retaining at least 99% accuracy, because it removes the same exact 32-dimensional query–key gauge as the failed identity chart without rescaling the initialized query and key tensors.
change: Select a well-conditioned four-column key pivot per head, preserve that initialized pivot as a fixed buffer, learn only the remaining key coefficients, and reconstruct the full key projections during forward passes.
mechanism: Natural-scale per-head query–key gauge charts
evidence_used: The identity-pivot query–key chart reached only 95.92% at 1,191 parameters despite preserving initial attention scores, whereas the analogous lexical gauge chart reached 99.96%; retaining the key pivots at their small natural initialization directly tests whether the failure was caused by the identity chart’s severe query/key rescaling rather than the gauge reduction itself.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1191, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing the final eight-dimensional LayerNorm bias as a six-dimensional token-feature bias will reduce parameters from 1,223 to 1,221 while retaining at least 99% accuracy, because logits observe the final hidden state only through the full-rank six-dimensional lexical projection.
change: Remove the final LayerNorm’s redundant eight-coordinate bias and add a learned six-coordinate bias immediately after projection into token-feature space.
mechanism: Rank-aligned final LayerNorm bias
evidence_used: The gauge-fixed rank-six lexical bottleneck achieved 99.96% accuracy at 1,223 parameters, and the successful direct attention-output-bias reparameterization showed that moving a bias into its observable projected space can preserve accuracy; this targets only the lexical bottleneck’s exact two-dimensional output nullspace.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1221, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining only the MLP output weight to the seven-dimensional zero-mean residual subspace will reduce parameters from 1,221 to 1,209 while retaining at least 99% accuracy, because each removed all-ones output component is erased by the final LayerNorm while the full eight-dimensional MLP output bias remains unchanged.
change: Store `fc2.weight` as 7-by-12 coefficients, reconstruct its centered 8-by-12 weight during forward passes, and preserve its initialized observable function.
mechanism: Final-LayerNorm-null MLP output-weight centering
evidence_used: Jointly centering the MLP output weight and bias failed at 84.22%, but centering the bias alone also failed at 79.56%; therefore the weight-only constraint is the missing isolation experiment. The same orthonormal output-subspace parameterization succeeded for the attention projection at 99.93%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1209, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the gauge-fixed tied lexical representation from rank six to rank five will reduce learned parameters from 1,209 to 1,097 while retaining at least 99% accuracy, because the successful rank-six model reached 99.96% and the full-width attention and residual pathways remain unchanged.
change: Use five-dimensional token codes, a fixed five-column identity chart, and a learned 5-by-3 lexical tail for both embeddings and logits.
mechanism: Rank-five tied lexical bottleneck
evidence_used: The gauge-fixed rank-six lexical bottleneck achieved 99.96% accuracy at 1,223 parameters, and subsequent independent MLP compression retained 99.89%; lowering lexical rank is the most direct test of whether six latent token features contain excess capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3787, "parameters": 1097, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower the model from 1,209 to 1,194 learned parameters while retaining at least 99% accuracy, because the successful zero-mean MLP output compression indicates remaining MLP redundancy, whereas the failed rank-five lexical bottleneck shows lexical rank is the riskier capacity target.
change: Change the default feed-forward width from 12 hidden units to 11, preserving the architecture, training schedule, and learned causal attention pathway.
mechanism: One-unit MLP width reduction
evidence_used: The 1,209-parameter model retained 99.89% accuracy after removing 12 MLP output-weight parameters, supporting a small additional MLP reduction; by contrast, reducing lexical rank caused accuracy to collapse to 37.87%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1194, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 11 to 10 will lower the model from 1,194 to 1,179 learned parameters while retaining at least 99% accuracy, because the preceding 12-to-11 reduction achieved 99.98% accuracy and showed substantial remaining MLP capacity.
change: Change the default feed-forward width from 11 hidden units to 10 while preserving all other architecture and training settings.
mechanism: One-unit MLP width reduction
evidence_used: The verified 1,194-parameter design with `d_ff=11` reached 99.98% accuracy, improving on the already successful 1,209-parameter `d_ff=12` design; this supports another conservative one-unit reduction, while the failed rank-five lexical bottleneck indicates lexical compression is substantially riskier.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1179, "training_steps": 4999}



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
