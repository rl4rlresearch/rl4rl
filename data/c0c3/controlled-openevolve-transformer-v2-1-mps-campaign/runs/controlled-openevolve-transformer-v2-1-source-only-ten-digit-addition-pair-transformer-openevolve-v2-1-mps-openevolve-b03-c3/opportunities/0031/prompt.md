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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9995, "parameters": 1626, "training_steps": 4999}
prior_hypothesis: Gauge-fixing the first positional embedding and terminal MLP output bias while reproducing both full eight-coordinate AdamW updates will achieve at least 99% accuracy with 1,626 parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Retaining query and value biases while removing only the key bias will preserve at least 99% accuracy with 1,636 parameters, because a position-independent key bias adds the same scalar to every attention logit in a softmax row.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1488, "training_steps": 4999}
prior_hypothesis: Replacing full-width absolute position embeddings with a learned two-head relative-lag bias will achieve at least 99% accuracy with 1,488 parameters, because addition’s repeated sequence relationships can be routed by learned relative offsets while retaining the load-bearing lexical, query, value, and MLP capacity.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-fixing the attention output-projection bias’s all-ones component in addition to the two successful gauges will achieve at least 99% accuracy with 1,625 parameters, because this scalar is erased by the downstream LayerNorms while ambient-coordinate AdamW preserves all eight original optimization directions.
change: Reproduce the qualified 1,626-parameter positional and terminal-bias gauges, then represent each attention output bias with seven learned differences and update all three gauge groups using full eight-coordinate AdamW moments and clipping.
mechanism: Triple ambient-Adam gauge fixing
evidence_used: Dual ambient-Adam gauge fixing achieved 99.95% accuracy with 1,626 parameters, whereas deleting the entire attention output bias collapsed accuracy to 4.18%; this motivates removing only its exact scalar gauge while retaining its seven functional coordinates and ambient optimizer geometry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5821999999999999, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the full-rank tied 114-by-8 token embedding with a learned rank-seven factorization will reduce the model from 1,626 to 1,568 parameters while retaining at least 99% accuracy, because it preserves seven adaptable lexical features and the entire proven attention/MLP computation.
change: Factor the shared input/output embedding into learned vocabulary codes and a learned orthonormal-initialized feature basis, using the same factors for token lookup and logit projection.
mechanism: Rank-seven learned token-logit factorization
evidence_used: The 1,626-parameter design reached 99.95%, while repeated one-parameter gauge extensions were brittle and the gated-MLP alternative reached only 70.49%; this preserves every load-bearing transformer-block parameter and instead challenges the untested assumption that the tied lexical interface needs eight independent dimensions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.037599999999999995, "parameters": 1568, "training_steps": 4999}

RECENT RESULT
hypothesis: Adding a global scalar gauge to the qualified positional and terminal-bias gauges will produce a 1,625-parameter model with at least 99% accuracy, because a uniform shift of the tied embedding is removed from inputs by LayerNorm and contributes only a common output-logit shift, while ambient AdamW preserves the original coordinate-wise optimization dynamics.
change: Gauge-fix one global coordinate of the tied token embedding, add the qualified terminal MLP-bias gauge, and manually optimize all three gauge groups through their full ambient-coordinate AdamW moments and clipping.
mechanism: Global tied-embedding gauge with ambient AdamW
evidence_used: The positional-plus-terminal-bias ambient gauge achieved 99.95% accuracy at 1,626 parameters, whereas extending the attention-output bias gauge failed at 58.22%; this tests a different exact invariance whose output effect is softmax-invariant.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9876, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the all-ones output component of one `fc2` weight column while reproducing its full eight-coordinate AdamW dynamics will yield 1,625 parameters and retain at least 99% accuracy.
change: Gauge-fix one terminal MLP output-weight column, preserve its original full-space initialization, and optimize its seven stored differences alongside the successful positional and terminal-bias gauges.
mechanism: Ambient-Adam terminal MLP weight-column gauge
evidence_used: The 1,626-parameter design achieved 99.95% using an ambient-Adam gauge for the terminal MLP bias. A common offset across an `fc2` weight column likewise produces only an input-dependent all-ones residual immediately erased by the final LayerNorm, making this the closest untested extension of the successful bias gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7281, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Making both block LayerNorms bias-free will yield 1,620 parameters and retain at least 99% accuracy, because the retained query/value biases can absorb the first LayerNorm’s query/value offsets while its position-independent key offset cancels in attention softmax.
change: Remove the learned biases from `ln1` and `ln2` while retaining query and value attention biases and the full width-12 MLP.
mechanism: Pre-attention bias absorption
evidence_used: Removing `ln2` bias while retaining query/value biases achieved 99.95% at 1,628 parameters; the query-bias ablation failed at 48.92%, motivating preservation of both attention biases while eliminating the redundant upstream bias instead.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Centering the tied embedding in the mean-zero gauge and reproducing its full 912-coordinate AdamW update will achieve at least 99% accuracy with 1,625 parameters, because it removes the same exact global-shift invariance as the 98.76% near-miss while avoiding an asymmetric fixed-coordinate representation.
change: Replace the tied token embedding with 911 learned mean-zero coordinates, preserve the original initialization RNG sequence, use the reconstructed weight for lookup and logits, and optimize it through full ambient moments and clipping alongside the two qualified gauges.
mechanism: Symmetric mean-zero tied-embedding gauge with ambient AdamW
evidence_used: The qualified positional-plus-terminal-bias design achieved 99.95% with 1,626 parameters, and the prior global tied-embedding gauge reached 98.76% with 1,625—far closer to the threshold than the other one-parameter extensions—making a numerically symmetric parameterization the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8715999999999999, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the terminal MLP output bias alongside the existing positional gauge, while preserving both full eight-coordinate AdamW dynamics, will reproduce the qualified 1,626-parameter design with at least 99% accuracy.
change: Replace the eight-parameter `fc2` bias with seven learned differences and manually optimize both gauge-fixed vectors using their ambient gradients, moments, weight decay, and clipping contributions.
mechanism: Dual ambient-Adam gauge fixing
evidence_used: Reference Design 3 achieved 99.95% accuracy with 1,626 parameters using exactly these two ambient-Adam gauges, while adding a third gauge caused accuracy to collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the tied embedding’s exact global-shift scalar while matching the qualified 1,626-parameter model’s initialization stream and full-coordinate AdamW dynamics will achieve at least 99% accuracy with 1,625 parameters.
change: Reproduce the qualified positional and terminal-bias gauges, add an anchored 911-coordinate tied embedding, and manually optimize all compact gauges using their ambient gradients, moments, clipping, and weight decay.
mechanism: RNG-matched global tied-embedding gauge
evidence_used: The prior global tied-embedding gauge reached 98.76% at 1,625 parameters—much closer than other 1,625-parameter attempts—while the dual ambient-gauge design reached 100%; preserving its initialization draws as well as optimizer geometry is the most focused next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7579, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one value-bias coordinate at zero will produce a 1,625-parameter model with at least 99% accuracy, because attention weights sum to one and the resulting constant value contribution can be represented by the retained output-projection bias.
change: Store seven learned value-bias coordinates, reconstruct the eighth as zero during attention, and leave the qualified dual ambient-Adam gauges unchanged.
mechanism: One-coordinate value-bias absorption
evidence_used: The current 1,626-parameter design achieved up to 100% accuracy. Unlike failed 1,625-parameter gauges that perturbed sensitive positional, embedding, or residual-output optimization, this removes one coordinate from a value bias that is functionally absorbed by the preserved projection bias; the attention-output-bias ablation cautions against removing that projection bias itself.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5522, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one value-bias coordinate while transferring its full eight-coordinate AdamW update through the learned output projection will produce a 1,625-parameter model with at least 99% accuracy.
change: Represent the value bias with seven learned coordinates, reconstruct an anchored eighth coordinate, and manually optimize the value and projection biases with full ambient AdamW moments while transferring the omitted value update into the projection bias.
mechanism: Coupled value–projection ambient AdamW gauge
evidence_used: The naïve value-bias gauge failed at 55.22%, closely matching the naïve terminal-bias gauge’s 54.75% failure; preserving ambient optimizer geometry raised the latter to at least 99.95%, motivating the analogous coupled treatment here.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final value-bias coordinate from each attention head while transferring both omitted full-coordinate AdamW updates into the learned projection bias will produce a 1,624-parameter model with at least 99% accuracy.
change: Reproduce the qualified positional and terminal-bias gauges, make `ln2` bias-free, and compact the value bias to three learned coordinates per head while preserving full eight-coordinate moments, clipping, weight decay, and projection-bias effects.
mechanism: Symmetric per-head value–projection ambient AdamW gauge
evidence_used: Reference Design 3 achieved 99.93% accuracy at 1,625 parameters after removing one value-bias coordinate with coupled ambient optimization; symmetrically removing the corresponding coordinate from the other head is the smallest direct extension of that successful mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5296, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing full-width absolute position embeddings with a learned two-head relative-lag bias will achieve at least 99% accuracy with 1,488 parameters, because addition’s repeated sequence relationships can be routed by learned relative offsets while retaining the load-bearing lexical, query, value, and MLP capacity.
change: Remove the 183-parameter absolute positional representation and inject a 46-parameter learned relative-distance bias directly into causal attention, preserving the established initialization stream and existing successful gauge optimizers.
mechanism: Learned relative-lag attention routing
evidence_used: The 1,625-parameter current model reached 99.93%, while rank-seven token factorization collapsed to 3.76% and query-bias removal collapsed to 48.92%. This identifies the lexical interface and content-based attention as load-bearing, motivating a different reduction of positional representation rather than another coordinate gauge or embedding-rank reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1488, "training_steps": 4999}



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
