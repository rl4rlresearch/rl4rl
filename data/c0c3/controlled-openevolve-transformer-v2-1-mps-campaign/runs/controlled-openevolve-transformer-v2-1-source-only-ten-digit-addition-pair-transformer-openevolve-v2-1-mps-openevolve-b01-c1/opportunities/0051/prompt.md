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
verified_results: {"accuracy": 0.9992, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Removing one `fc1` bias scalar will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy, because `ln2`’s six learned bias coordinates can generate the omitted neuron offset through its learned weight row, while the other `fc1` biases cancel collateral offsets.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing a fourth final-LayerNorm bias direction while exempting the tied token embedding from weight decay will achieve at least 99% accuracy with 1,582 parameters, because the token-row means that absorb omitted output biases are input-side invisible and should not be penalized toward zero.
change: Reduce the learned final-LayerNorm bias basis from five to four coordinates and use a zero-weight-decay optimizer group for the tied token embedding while retaining configured weight decay elsewhere.
mechanism: Weight-decay-free output-bias channel
evidence_used: The four-direction quotient narrowly missed at 98.28%, whereas the three-direction quotient reached 99.91%; amplifying the fixed bias failed, motivating an optimizer-level change that directly supports the demonstrated token-row-mean absorption channel without changing its activation scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing each head’s exact GL(4) query/key basis redundancy will reduce the model from 1,582 to 1,550 parameters while retaining at least 99% accuracy, because it preserves the full learned attention-score family and exactly preserves conventional Q/K initialization logits.
change: Replace unconstrained query and key projections with a gauge-fixed factorization whose query anchor submatrix is fixed and whose learned key factor absorbs the inverse basis transformation; retain independent values, query biases, heads, and all other successful settings.
mechanism: Gauge-fixed headwise Q/K factorization
evidence_used: The 1,582-parameter model reaches 99.95% with full Q/K factors, while deleting one key direction fell to 41.04%; this instead removes only 32 functionally redundant factor-basis coordinates without deleting an addressing direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7898000000000001, "parameters": 1550, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fifth final-LayerNorm bias direction will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because exempting the tied token embedding from weight decay raised the four-direction quotient from 98.28% to 99.95%, demonstrating that its token-row means can reliably learn the replacement output-bias channel.
change: Reduce the final-LayerNorm bias basis from four to three learned coordinates while preserving the successful zero-weight-decay embedding group and all attention, MLP, initialization, and training settings.
mechanism: Incremental decay-free final-output bias absorption
evidence_used: The immediately preceding four-direction quotient achieved 99.95% at 1,582 parameters only after token-embedding weight decay was removed; this strongly motivates testing one additional coordinate through the same demonstrated absorption mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a sixth final-LayerNorm bias direction will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because the zero-weight-decay token-row-mean channel achieved 99.96% accuracy after absorbing five bias directions.
change: Reduce the final-LayerNorm learned bias basis from three to two coordinates while preserving the successful embedding optimizer group and all attention, MLP, initialization, and decoding behavior.
mechanism: Incremental decay-free final-output bias absorption
evidence_used: The immediately preceding five-direction quotient achieved 99.96% accuracy at 1,581 parameters, improving slightly on the four-direction quotient’s 99.95%; this makes one additional coordinate removal the most direct test of the demonstrated absorption mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9913, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Orthogonally isolating the decay-free token-row-mean output-bias coordinates before removing a seventh final-LayerNorm bias direction will produce a 1,579-parameter model with at least 99% accuracy.
change: Reparameterize the unchanged globally mean-free tied embedding into row-centered content and explicit mean-free row offsets, keep both components free of weight decay, and reduce the final-LayerNorm learned bias basis from two coordinates to one.
mechanism: Optimizer-aligned token-row-mean quotient
evidence_used: Removing embedding weight decay raised the four-direction quotient from 98.28% to 99.95%, and the current six-direction quotient still reaches 99.13%; this indicates that optimization of the token-row-mean replacement channel is decisive and motivates aligning that channel with independent Adam coordinates before pruning another bias direction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the eighth and final learned final-LayerNorm bias direction will produce a 1,578-parameter model with at least 99% accuracy, because the optimizer-aligned token-row-mean channel reached 100% accuracy after absorbing the first seven directions.
change: Replace the one-coordinate quotient final LayerNorm with a fixed common bias only, eliminating its last learned bias parameter while preserving the decay-free explicit token-row means.
mechanism: Complete decay-free final-output bias absorption
evidence_used: Reparameterizing token-row means as independent Adam coordinates allowed the seven-direction quotient to achieve 100% accuracy at 1,579 parameters; this directly motivates testing complete absorption of the remaining bias direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4171, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the scaled final-LayerNorm activation before adding its fixed common component will let the explicit token-row means replace the last learned final-bias direction, producing a 1,578-parameter model with at least 99% accuracy.
change: Remove the final LayerNorm’s last learned bias parameter and reserve its common activation direction exclusively for the decay-free token-row-mean output-bias channel.
mechanism: Activation-isolated token-row output-bias quotient
evidence_used: The optimizer-aligned row means reached 100% accuracy with one learned final-bias coordinate at 1,579 parameters, but simply deleting that coordinate fell to 41.71%; activation centering removes the context-dependent common component that otherwise interferes with row means while preserving the original initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6175, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` bias scalar will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy, because `ln2`’s six learned bias coordinates can generate the omitted neuron offset through its learned weight row, while the other `fc1` biases cancel collateral offsets.
change: Replace the MLP’s first linear layer with an otherwise identical layer whose final output-bias coordinate is fixed at zero, preserving all successful attention, embedding, final-LayerNorm, initialization, and optimizer settings.
mechanism: LayerNorm-beta/downstream-bias gauge reorientation
evidence_used: Removing a third `ln2` bias coordinate collapsed accuracy to 40.61%, while the current design with six learned `ln2` bias coordinates reaches 100%; this motivates preserving that optimization pathway and instead removing one downstream bias coordinate from the same exact affine redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a second `fc1` bias scalar will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because the preceding `ln2` retains six learned bias coordinates that can supply both omitted neuron offsets through `fc1`’s learned weight rows.
change: Replace the one-pruned MLP input linear layer with a two-pruned version that fixes its final two output-bias coordinates at zero while preserving its full weight matrix and all other model and training settings.
mechanism: Incremental LayerNorm-beta/downstream-bias gauge reorientation
evidence_used: Removing the first `fc1` bias scalar achieved 99.92% accuracy at 1,578 parameters, whereas pruning an additional `ln2` bias coordinate collapsed accuracy; this directly supports extending the successful downstream-bias removal while preserving the LayerNorm optimization pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8448, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln1` scale coordinate at 1 will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because that coordinate’s bias is already fixed at zero and the following QKV weight column can absorb its learned scale without reducing the attention function family or changing initialization outputs.
change: Add a two-bias-pruned LayerNorm with one fixed unit scale coordinate and use it only for `ln1`, leaving the successful `ln2` and one-pruned `fc1` pathway unchanged.
mechanism: Zero-beta LayerNorm scale gauge
evidence_used: The one-pruned `fc1` design achieved 99.92% at 1,578 parameters, while removing a second `fc1` bias fell to 84.48%; this motivates a single exact gauge removal elsewhere that preserves the six learned `ln2` bias coordinates implicated in the successful result.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8689, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one value-projection bias scalar at zero will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because causal softmax rows sum to one, making every value bias a context-independent offset that the learned attention output-projection bias can exactly absorb.
change: Remove the final value-bias coordinate from `qkv.bias` and restore it as a fixed zero during the forward pass, preserving all weights, query biases, initialization behavior, and successful training settings.
mechanism: Value-bias/output-bias affine quotient
evidence_used: The current 1,578-parameter model achieves 99.92% accuracy. Unlike the unsuccessful second `fc1`-bias and `ln1`-scale removals, this quotient lies entirely within two consecutive affine attention operations: the mean-free output projection maps the omitted value offset into its own representable seven-dimensional bias space.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7924, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing seven learned coordinates per absolute position with two learned causal-distance logits—one per attention head—will reduce the 1,578-parameter model by `5 * INPUT_LEN` parameters while retaining at least 99% accuracy, because fixed-width addition repeatedly addresses operand digits through relative offsets rather than requiring independent absolute-position vectors.
change: Challenge the assumption that absolute position embeddings must represent both place and attention addressing. Remove them from the residual stream and let each attention head directly learn a bias for every causal relative distance, while preserving the successful initialization stream and all other model, optimizer, and decoding behavior.
mechanism: Learned relative-offset attention without absolute position embeddings
evidence_used: The current two-head model reaches 99.92%, showing its attention and MLP have sufficient computational capacity, while three different one-scalar affine quotients fell to 84.48%, 86.89%, and 79.24%. This motivates seeking larger savings through a different representation: directly learned relative addressing, rather than another fragile affine-coordinate removal.
result: the implementation could not be verified



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
