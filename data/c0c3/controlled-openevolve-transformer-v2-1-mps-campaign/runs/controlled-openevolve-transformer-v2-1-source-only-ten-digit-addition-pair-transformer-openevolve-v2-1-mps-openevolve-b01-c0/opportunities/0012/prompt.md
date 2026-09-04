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
verified_results: {"accuracy": 0.9987, "parameters": 1635, "training_steps": 4999}
prior_hypothesis: Fixing one `ln2` bias coordinate at zero will reduce the model from 1636 to 1635 parameters while retaining at least 99% accuracy, because its effect is exactly representable by the full learned `fc1` bias and the initial model computation remains unchanged.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower parameters from 1644 to 1576 while retaining at least 99% accuracy.
change: Reduce the single transformer block’s default MLP width by one third.
mechanism: Narrower learned feed-forward bottleneck
evidence_used: The current 1644-parameter model reaches 99.96% accuracy, providing substantial accuracy margin for a conservative 68-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.46740000000000004, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the attention key bias will reduce parameters from 1644 to 1636 without reducing accuracy below 99%, because a shared key bias adds the same scalar to every unmasked attention logit for each query and therefore cancels exactly under softmax.
change: Replace the fused QKV linear bias with separate learned query and value biases while leaving keys unbiased.
mechanism: Eliminate softmax-invariant key bias
evidence_used: Narrowing `d_ff` from 12 to 8 reduced parameters to 1576 but collapsed accuracy to 46.74%, showing that learned MLP capacity is essential; this patch instead removes eight attention parameters that are functionally redundant.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9826999999999999, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight redundant key-bias parameters while preserving the baseline’s constructor RNG consumption and fused QKV computation will retain at least 99% accuracy with 1636 parameters.
change: Keep the full fused QKV weight initialization, replace its 24-element bias with learned query/value biases, and inject a fixed zero key bias during the fused projection.
mechanism: Initialization-preserving elimination of softmax-invariant key bias
evidence_used: The 1644-parameter baseline reached 99.96%, while the prior key-bias removal reached 98.27%; because key bias cancels under softmax, preserving baseline initialization and fused projection isolates initialization sensitivity as the likely cause.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eight value-bias parameters will reduce the model from 1636 to 1628 parameters while retaining at least 99% accuracy, because attention-weighted value bias is a position-independent offset already representable by the output projection bias.
change: Preserve constructor RNG consumption and fused QKV computation, but retain only the learned query bias and inject fixed zero key/value biases.
mechanism: Eliminate value/output bias redundancy
evidence_used: Initialization-preserving key-bias elimination achieved 99.98% accuracy at 1636 parameters, while narrowing the MLP failed; this motivates removing another analytically redundant attention bias without reducing learned MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4107, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only one of the eight value-bias parameters will reduce the model to 1635 parameters while retaining at least 99% accuracy, because the 1636-parameter design has 99.98% accuracy and preserves seven of the eight optimization-relevant value offsets whose complete removal caused collapse.
change: Shorten the combined query/value bias by one scalar and append a fixed zero to reconstruct the final value-bias coordinate during fused QKV projection.
mechanism: Granular value-bias ablation
evidence_used: Removing all eight value-bias parameters reduced accuracy to 41.07%, while retaining them produced 99.98%; a one-coordinate ablation is the smallest informative test of how much of that redundant parameterization training requires.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3148, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final value-bias coordinate with a projection-bias coordinate will reduce the model to 1635 parameters while retaining at least 99% accuracy, because the coordinate remains trainable through both redundant bias paths instead of being fixed at zero.
change: Shorten the query/value bias parameter by one scalar and reconstruct the missing value-bias coordinate from the learned attention output-projection bias.
mechanism: Cross-path bias tying
evidence_used: The 1636-parameter model reached 99.98%, but fixing one value-bias coordinate at zero collapsed accuracy to 31.48%; tying that coordinate to another learned, functionally redundant bias tests whether preserving its trainable optimization path avoids the collapse.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8578, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one attention output-projection bias scalar will reduce the model to 1635 parameters while retaining at least 99% accuracy, because the full learned value bias remains available to represent the missing output offset.
change: Replace the eight-element projection bias with seven learned elements plus one fixed zero while preserving constructor RNG consumption.
mechanism: Granular output-bias ablation with preserved value-bias pathway
evidence_used: The 1636-parameter model achieved 99.98%, whereas removing or tying a value-bias coordinate failed sharply; this motivates reducing the redundant projection-bias side while leaving every optimization-sensitive value-bias coordinate intact.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6844, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` gain coordinate at its initialized value of 1 will reduce the model from 1636 to 1635 learned parameters while retaining at least 99% accuracy, because that coordinate’s scale is exactly absorbable by the corresponding QKV weight column and the model begins with identical activations and RNG state.
change: Replace the first block LayerNorm with an equivalent implementation whose final gain coordinate is fixed at 1 while its other gains and all biases remain learned.
mechanism: Absorbable attention-input normalization scale
evidence_used: Initialization-preserving removal of all eight softmax-redundant key biases achieved 99.98%, whereas changing value or projection bias pathways failed; this motivates a minimal reduction in a different exactly redundant parameterization while preserving initial computation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7279000000000001, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Representing the attention projection bias with seven parameters and updating them as differences of the original eight-coordinate AdamW trajectory will produce at least 99% accuracy with 1635 parameters.
change: Fix the final projection-bias coordinate at zero, reconstruct the full bias during inference, and preserve the eliminated coordinate’s optimization effect through non-model AdamW moment state and gauge-aware gradient clipping.
mechanism: LayerNorm-null bias gauge with quotient-preserving AdamW
evidence_used: The 1636-parameter model reached 99.98%, while ordinary seven-coordinate projection-bias training reached only 68.44%; because a common projection-bias shift is removed by downstream LayerNorm, the failure motivates preserving the original optimizer trajectory on the seven-dimensional quotient rather than deleting one coordinate’s update.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5243, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` bias coordinate at zero will reduce the model from 1636 to 1635 parameters while retaining at least 99% accuracy, because its effect is exactly representable by the full learned `fc1` bias and the initial model computation remains unchanged.
change: Replace the block’s second LayerNorm with an equivalent implementation containing seven learned bias coordinates and one fixed-zero coordinate, while retaining all gains and MLP parameters.
mechanism: Locally absorbable pre-MLP normalization bias
evidence_used: The 1636-parameter design reached 99.98%, while narrowing the MLP collapsed accuracy and attention-path bias reductions proved optimization-sensitive; this tests a new, strictly local redundancy immediately before the MLP’s learned affine bias without reducing MLP width.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second `ln2` bias coordinate at zero will reduce the model from 1635 to 1634 parameters while retaining at least 99% accuracy, because the 1635-parameter design achieved 99.87% and both removed offsets are locally representable by the learned `fc1` bias.
change: Store six learned `ln2` bias coordinates instead of seven and append two fixed zeros during normalization.
mechanism: Incremental pre-MLP LayerNorm bias reduction
evidence_used: Fixing one `ln2` bias coordinate at zero retained 99.87% accuracy at 1635 parameters; this is the only tested single-parameter reduction that passed and directly motivates a conservative second-coordinate ablation in the same redundant pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7637, "parameters": 1634, "training_steps": 4999}



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
