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
verified_results: {"accuracy": 1.0, "parameters": 1272, "training_steps": 49999}
prior_hypothesis: Fixing one token-embedding entry to zero will reduce the model from 1,273 to 1,272 parameters while preserving at least 99% accuracy, because subtracting that entry uniformly from every token-embedding coordinate changes residual inputs only by a LayerNorm-invisible all-ones shift and changes every output logit by the same amount.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing final LayerNorm scale coordinate 3 at one, while retaining learned coordinate 5, will reduce the model from 1,305 to 1,304 parameters and preserve at least 99% accuracy.
change: Learn final LayerNorm scale coordinates 0–2 and 5, fix coordinates 3–4 and 6–7 at one, and preserve the essential learned bias.
mechanism: Fourth coordinate-specific final LayerNorm scale restriction
evidence_used: Fixing coordinate 4 in addition to coordinates 6–7 retained 100% accuracy at 1,305 parameters, while fixing coordinate 5 caused accuracy to fall to 75.89%; testing adjacent coordinate 3 is the smallest informative extension of the successful coordinate-specific restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1304, "training_steps": 49999}

RECENT RESULT
hypothesis: Replacing the independent positional table with a normalized fixed eight-dimensional sinusoidal basis and a learned 8×8 projection will remove `8 * INPUT_LEN - 64` parameters while retaining at least 99% accuracy, because it preserves full positional rank and trainable orientation.
change: Generate generic content-independent sinusoidal position codes, normalize them to match the existing embedding initialization scale after projection, and learn only their shared projection into the residual stream.
mechanism: Full-rank sinusoidal positional subspace with learned orientation
evidence_used: The learned rank-six positional factorization reached only 74.72%, identifying positional dimensionality as load-bearing, but it did not test whether independent vectors are necessary. This design restores all eight dimensions while challenging the assumption that every position needs separately learned parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9763, "parameters": 1184, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing final LayerNorm scale coordinate 2 at one while retaining learned coordinate 5 will reduce the model from 1,304 to 1,303 parameters and preserve at least 99% accuracy.
change: Learn final LayerNorm scale coordinates 0–1 and 5, fix coordinates 2–4 and 6–7 at one, and preserve the essential learned bias.
mechanism: Fifth coordinate-specific final LayerNorm scale restriction
evidence_used: Fixing coordinates 3–4 and 6–7 retained 100% accuracy at 1,304 parameters, whereas fixing coordinate 5 reduced accuracy to 75.89%; testing adjacent coordinate 2 is the smallest informative extension of the successful coordinate-specific restriction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5529999999999999, "parameters": 1303, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing final LayerNorm scale coordinate 1 while restoring learned coordinate 2 will reduce the model from 1,304 to 1,303 parameters and retain at least 99% accuracy.
change: Learn final LayerNorm scale coordinates 0, 2, and 5; fix coordinates 1, 3–4, and 6–7 at one; preserve the essential learned bias.
mechanism: Coordinate-specific final LayerNorm scale swap
evidence_used: Fixing coordinate 2 caused accuracy to fall to 55.30%, while fixing coordinates 3–4 and 6–7 achieved 100% accuracy at 1,304 parameters. Restoring coordinate 2 and testing adjacent coordinate 1 isolates whether the failure was coordinate-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1303, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing final LayerNorm scale coordinate 0 at one while retaining learned coordinates 2 and 5 will reduce the model from 1,303 to 1,302 parameters and preserve at least 99% accuracy.
change: Learn only final LayerNorm scale coordinates 2 and 5; fix coordinates 0–1, 3–4, and 6–7 at one while preserving the essential learned bias.
mechanism: Sixth coordinate-specific final LayerNorm scale restriction
evidence_used: The 1,303-parameter design achieved 100% accuracy with learned coordinates 0, 2, and 5. Coordinate 2 previously proved essential, coordinate 5 is load-bearing, and coordinate 1 was safely fixed; testing coordinate 0 is the smallest remaining coordinate-specific reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.24109999999999998, "parameters": 1302, "training_steps": 49999}

RECENT RESULT
hypothesis: Adding independent per-position corrections on residual coordinates 2 and 5 to the 97.63%-accurate sinusoidal design will raise accuracy to at least 99% while reducing the current model from 1,303 to 1,229 parameters.
change: Replace the 23×8 learned positional table with a normalized fixed sinusoidal basis, a learned 8×8 projection, and a learned 23×2 positional residual injected into coordinates 2 and 5.
mechanism: Rank-two learned residual over a full-rank sinusoidal positional basis
evidence_used: The full-rank sinusoidal design reached 97.63% with only 64 positional parameters, showing that structured positions nearly suffice; coordinates 2 and 5 are also the final-normalization scale coordinates that could not safely be fixed, motivating a small unstructured correction on those channels.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.306, "parameters": 1229, "training_steps": 49999}

RECENT RESULT
hypothesis: Removing one redundant common-mode coordinate from every positional embedding will reduce the model from 1,303 to 1,280 parameters while retaining at least 99% accuracy, because all downstream LayerNorms are invariant to per-position shifts along the all-ones direction.
change: Learn an independent seven-coordinate embedding for each position and append a fixed-zero eighth coordinate before adding it to the token embedding.
mechanism: Gauge-fixed positional embeddings
evidence_used: Structured sinusoidal positions failed despite full dimensionality, indicating that independent positional freedom matters; this patch preserves seven independent degrees of freedom per position and removes only the common-shift degree that LayerNorm cannot observe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1280, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing position 0’s seven positional coordinates to zero will reduce the model from 1,280 to 1,273 parameters while retaining at least 99% accuracy, because any removed position-origin vector can be absorbed uniformly into the tied token embeddings, preserving inputs while changing output logits only by a class-common offset.
change: Store learned seven-coordinate embeddings only for positions 1 through `max_seq_len - 1` and prepend a fixed-zero embedding for position 0 during the forward pass.
mechanism: Anchored relative positional embeddings
evidence_used: Gauge-fixing each positional embedding’s unobservable common-mode coordinate retained 100% accuracy at 1,280 parameters; anchoring the shared positional origin removes another exact positional/token-embedding translation redundancy without restricting relative positional freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1273, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing one token-embedding entry to zero will reduce the model from 1,273 to 1,272 parameters while preserving at least 99% accuracy, because subtracting that entry uniformly from every token-embedding coordinate changes residual inputs only by a LayerNorm-invisible all-ones shift and changes every output logit by the same amount.
change: Store the tied token-embedding matrix in a one-parameter-smaller chart whose first entry is fixed at zero, reconstruct it for both input lookup and output projection, and otherwise leave the model and training unchanged.
mechanism: Gauge-fixed tied token embeddings
evidence_used: Anchoring position 0 by removing the exact positional/token-embedding translation redundancy retained 100% accuracy at 1,273 parameters; this applies the remaining scalar translation symmetry of the tied token embeddings without restricting observable model functions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1272, "training_steps": 49999}

RECENT RESULT
hypothesis: Removing six rotationally redundant query/key parameters will reduce the model from 1,272 to 1,266 parameters while preserving at least 99% accuracy, because an orthogonal rotation of each head’s mean-zero query coordinates preserves every tied query-key dot product.
change: Replace the dense tied query/key projection with an equivalent Helmert-basis chart whose three-dimensional centered component is upper triangular over its first three input coordinates; retain an unrestricted common component and value projection.
mechanism: Within-head rotational gauge fixing for tied queries and keys
evidence_used: The 1,272-parameter design achieved 100% accuracy after an exact embedding gauge reduction. Here, the supplied model uses identical queries and keys and adds the attention-bias scalar equally to every head coordinate, leaving a three-dimensional orthogonal symmetry per four-dimensional head and therefore three removable parameters per head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5534, "parameters": 1266, "training_steps": 49999}

RECENT RESULT
hypothesis: Removing the eight row-common degrees of freedom from the attention output matrix will reduce the model from 1,272 to 1,264 parameters while preserving at least 99% accuracy, because the removed component adds only a position-dependent all-ones residual shift that every downstream LayerNorm eliminates.
change: Replace the attention output projection’s dense 8×8 weight with a learned 7×8 matrix padded by a fixed-zero eighth output row, while retaining its unrestricted eight-coordinate bias and all existing bias sharing.
mechanism: Common-mode-free attention output projection
evidence_used: The 1,272-parameter design retained 100% accuracy after exact embedding gauge reductions, and the current MLP already uses this same seven-output-plus-zero-padding gauge successfully. Unlike the failed rotational query/key chart, this is a direct linear common-mode elimination already represented elsewhere in the verified architecture.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7464, "parameters": 1264, "training_steps": 49999}

RECENT RESULT
hypothesis: A full-rank sinusoidal positional basis plus per-head learned causal-distance biases will reach at least 99% accuracy with 1,226 parameters, because attention can learn operand alignments directly instead of reconstructing them from independently learned absolute vectors.
change: Replace the 154-parameter independent positional table with a learned 8×8 projection of fixed anchored sinusoidal codes and add 44 gauge-fixed per-head relative-attention biases.
mechanism: Sinusoidal absolute positions with learned relative-lag attention
evidence_used: The sinusoidal projection reached 97.63%, showing structured absolute positions retain nearly all required information, while a learned residual correction failed at 30.60%. Injecting the missing positional flexibility directly into attention logits tests a different mechanism tailored to pairwise alignment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6472, "parameters": 1226, "training_steps": 49999}



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
