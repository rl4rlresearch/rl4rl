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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1486, "training_steps": 4999}
prior_hypothesis: Removing each relative-lag bias head’s softmax-invariant scalar while reproducing all 23-coordinate AdamW updates will retain at least 99% accuracy with 1,486 parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9994, "parameters": 1470, "training_steps": 4999}
prior_hypothesis: Removing `ln1`’s eight zero-initialized biases will reduce the qualified model to 1,470 parameters while retaining at least 99% accuracy, because query and value biases preserve the same function class and key offsets are softmax-invariant.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1488, "training_steps": 4999}
prior_hypothesis: Replacing full-width absolute position embeddings with a learned two-head relative-lag bias will achieve at least 99% accuracy with 1,488 parameters, because addition’s repeated sequence relationships can be routed by learned relative offsets while retaining the load-bearing lexical, query, value, and MLP capacity.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing each relative-lag bias head’s softmax-invariant scalar while reproducing all 23-coordinate AdamW updates will retain at least 99% accuracy with 1,486 parameters.
change: Port the qualified learned relative-lag design, compact each head’s 23 lag biases to 22 differences, and preserve its MLP-bias and coupled value–projection ambient optimizers.
mechanism: Ambient-Adam per-head relative-bias gauge fixing
evidence_used: Reference Design 3 achieved 99.98% accuracy with 1,488 parameters; unlike prior brittle functional approximations, the proposed two-parameter reduction removes exact per-head attention-softmax invariances while preserving full-coordinate initialization, clipping, moments, decay, and updates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1486, "training_steps": 4999}

RECENT RESULT
hypothesis: Folding the bias-free second LayerNorm’s eight learned scales into the first MLP weight matrix will reduce the qualified 1,486-parameter model to 1,478 parameters while retaining at least 99% accuracy, because the fold is functionally exact and full-space AdamW dynamics are reproduced during training.
change: Port the qualified gauge-fixed relative-position design, remove `ln2`’s affine parameters, and manually optimize virtual LayerNorm scales and MLP weights before folding their product into the stored learned weight after every step.
mechanism: Ambient-Adam LayerNorm-to-MLP scale folding
evidence_used: The 1,486-parameter relative-lag gauge design achieved 99.90% accuracy. Its `ln2` is already bias-free, so each remaining scale is exactly multiplicatively redundant with the corresponding `fc1` weight column; ambient-coordinate optimization follows the successful optimizer-preserving gauge strategy used by that design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1478, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one softmax-independent row-shift coordinate from each of the 12 `fc1` rows will reduce the qualified model from 1,478 to 1,466 learned parameters while retaining at least 99% accuracy, because bias-free LayerNorm produces mean-zero inputs and full eight-coordinate MLP-weight and LayerNorm-scale AdamW dynamics remain represented during training.
change: Store each folded `fc1` row as seven differences with an anchored eighth coordinate, preserve its original full-width initialization and optimizer state in non-parameter tensors, and canonicalize the folded ambient product after every update.
mechanism: Ambient-Adam row-gauge fixing of the folded MLP input weight
evidence_used: The current 1,478-parameter design achieved 99.88% after exactly folding `ln2` scales into `fc1` with ambient-coordinate AdamW; the proposed reduction acts on that same qualified interface and removes an exact null direction induced by its bias-free LayerNorm, avoiding the capacity-reducing approximations that failed in prior trials.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Folding `ln2` scales into `fc1` and removing one exact mean-zero row-shift coordinate will produce a 1,477-parameter model with at least 99% accuracy while avoiding the overhead that prevented the 12-row version from finishing.
change: Port the qualified LayerNorm-to-MLP fold, compact only the first `fc1` row to seven differences, and preserve full-width initialization, gradients, clipping, AdamW moments, decay, and updates in ambient coordinates.
mechanism: Single-row ambient-Adam gauge on the folded MLP input weight
evidence_used: The full-width fold achieved 99.88% at 1,478 parameters. The 1,466-parameter extension timed out rather than failing accuracy, so testing one of its exact row gauges is the smallest informative reduction with substantially less compact-parameter overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Directly optimizing the 84-dimensional orthonormal quotient of `fc1` will complete within the time limit and retain at least 99% accuracy with 1,466 parameters, because it preserves every linear function on bias-free LayerNorm outputs while avoiding the expensive ambient-coordinate machinery that caused the prior 1,466- and 1,477-parameter attempts to time out.
change: Port the qualified relative-bias gauge, remove redundant `ln2` affine scales, and represent every eight-coordinate `fc1` row by seven learned coefficients in a fixed orthonormal basis of the mean-zero subspace.
mechanism: Orthonormal quotient coordinates for mean-zero MLP inputs
evidence_used: The 1,478-parameter LayerNorm-to-MLP fold achieved 99.88%, while the exact 1,466-parameter row-gauge extension timed out rather than failing accuracy; using compact quotient coordinates directly targets that implementation overhead while preserving the same function class.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing `ln1`’s eight zero-initialized biases will reduce the qualified model to 1,470 parameters while retaining at least 99% accuracy, because query and value biases preserve the same function class and key offsets are softmax-invariant.
change: Make the pre-attention LayerNorm bias-free while retaining its learned scales and all attention biases.
mechanism: Pre-attention LayerNorm bias elimination
evidence_used: The 1,478-parameter model achieved 99.88%; unlike the timed-out `fc1` quotient attempts, this exact eight-parameter redundancy requires no reconstructed weights or extra per-step optimizer work. Prior query-bias failures motivate retaining the dedicated query bias rather than removing attention bias capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1470, "training_steps": 4999}



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
