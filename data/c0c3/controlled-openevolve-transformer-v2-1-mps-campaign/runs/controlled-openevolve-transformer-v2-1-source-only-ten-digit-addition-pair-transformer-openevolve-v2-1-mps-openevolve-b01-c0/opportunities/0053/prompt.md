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
verified_results: {"accuracy": 1.0, "parameters": 1612, "training_steps": 4999}
prior_hypothesis: Tying the remaining third-coordinate `ln1` bias contrast to the third query-bias parameter will reduce the model to 1612 parameters while retaining at least 99% accuracy, because the contrast remains learned and its projection-space effect is gauge-coupled to the query/value biases.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one key-row coordinate to zero will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because LayerNorm confines inputs to an affine hyperplane and the omitted direction changes that key coordinate only by a position-independent offset, which causal softmax removes.
change: Replace the first key-projection row with seven directly represented coordinates and a fixed zero eighth coordinate, reconstructing a functionally equivalent initialization while preserving the original RNG sequence.
mechanism: Coordinate-anchored softmax-null key-row gauge
evidence_used: The centered key-row gauge tested the same exact redundancy but reached only 70.16%; its zero-sum reconstruction couples every learned coordinate to the eighth weight. This coordinate-anchored basis removes that coupling while retaining the verified 1619-parameter design’s successful elimination of all key biases.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7401000000000001, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third `ln2` bias degree of freedom will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because its zero-initialized additive effect is representable by the downstream `fc1` bias without the multiplicative optimization disruption seen when fixing an `ln2` scale.
change: Store five learned `ln2` bias contrasts and reconstruct two anchored coordinates before centering the seven active coordinates; leave `ln1` unchanged.
mechanism: Pre-MLP LayerNorm bias/input-bias redundancy
evidence_used: Balanced LayerNorm common-bias gauges previously passed, whereas fixing the eighth `ln2` scale reached only 73.64%; this extends the successful additive-bias mechanism while preserving the initial function and RNG sequence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1618, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth `ln2` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its zero-initialized additive contribution remains representable by the downstream `fc1` bias.
change: Store four learned `ln2` bias contrasts instead of five and reconstruct three anchored coordinates before centering the seven active coordinates.
mechanism: Fourth pre-MLP LayerNorm bias/input-bias redundancy
evidence_used: Removing the third `ln2` bias degree achieved 99.93% accuracy at 1618 parameters, while fixing an `ln2` scale achieved only 73.64%; this supports extending the successful additive-bias reduction by one coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.374, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third `ln1` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its additive effect can be absorbed by the query bias, is softmax-null for keys, and is representable through the value/output bias path.
change: Store five learned `ln1` bias contrasts instead of six and reconstruct two anchored coordinates before centering the seven active coordinates.
mechanism: Pre-attention LayerNorm bias/projection-bias redundancy
evidence_used: Removing a third `ln2` bias degree achieved 99.93% at 1618 parameters, establishing that an additional additive LayerNorm-bias gauge can train successfully; the fourth `ln2` reduction collapsed, so testing the analogous third reduction in the distinct pre-attention LayerNorm is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1617, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth `ln1` bias degree of freedom will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because its additive effect remains absorbable by the query and value/output bias paths while remaining softmax-null for keys.
change: Store four learned `ln1` bias contrasts instead of five and reconstruct three anchored coordinates before centering the seven active coordinates.
mechanism: Fourth pre-attention LayerNorm bias/projection-bias redundancy
evidence_used: Removing the third `ln1` bias degree achieved 99.87% accuracy at 1617 parameters; extending that successful pre-attention mechanism by one coordinate is the smallest informative reduction, particularly because the analogous fourth `ln2` reduction failed and establishes that success is pathway-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9945999999999999, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fifth `ln1` bias degree of freedom will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its additive effect remains absorbable by the query and value/output bias paths and is softmax-null for keys.
change: Store three learned `ln1` bias contrasts instead of four and reconstruct four anchored coordinates before centering the seven active coordinates.
mechanism: Fifth pre-attention LayerNorm bias/projection-bias redundancy
evidence_used: Removing the fourth `ln1` bias degree achieved 99.46% accuracy at 1616 parameters, so extending the same pathway-specific mechanism by one coordinate is the smallest evidence-backed next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a sixth `ln1` bias degree of freedom will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because its additive effect remains absorbable by the query and value/output bias paths and remains softmax-null for keys.
change: Store two learned `ln1` bias contrasts instead of three and reconstruct five anchored coordinates before centering the seven active coordinates.
mechanism: Sixth pre-attention LayerNorm bias/projection-bias redundancy
evidence_used: Removing the fifth `ln1` bias degree achieved 99.92% accuracy at 1615 parameters, making one further reduction along the same successful pathway the smallest evidence-backed next change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9677, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `ln1` to two learned bias contrasts will retain at least 99% accuracy when the previously discarded third active coordinate is preserved instead of the second, indicating that the 96.77% result was coordinate-specific rather than an intrinsic two-degree capacity limit.
change: Store two `ln1` bias parameters and place them in the first and third active coordinates, fixing the second and remaining coordinates before centering.
mechanism: Coordinate-swapped sixth pre-attention LayerNorm bias reduction
evidence_used: The prior sixth `ln1` reduction reached 96.77%, substantially closer to the requirement than other one-parameter reductions that collapsed to 16.94–74.01%; testing a minimally changed contrast basis is therefore the most informative next 1614-parameter design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining only the third active `ln1` bias coordinate will reduce the model to 1613 parameters while maintaining at least 99% accuracy.
change: Remove the first learned `ln1` bias contrast and preserve the third-coordinate contrast before centering.
mechanism: Third-coordinate-only pre-attention LayerNorm bias
evidence_used: Two contrasts in the first and third coordinates achieved 99.97% at 1614 parameters, while first and second achieved 96.77%; this identifies the third coordinate as the strongest candidate to retain in a one-contrast design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final learned `ln1` bias contrast will reduce the model to 1612 parameters while retaining at least 99% accuracy, because the additive offset remains absorbable by the attention projection biases and softmax-null key pathway.
change: Eliminate the third-coordinate `ln1` bias parameter and use a fixed zero bias in the pre-attention LayerNorm.
mechanism: Bias-free pre-attention LayerNorm
evidence_used: The third-coordinate-only `ln1` design achieved 99.90% accuracy at 1613 parameters, after successive reductions from five contrasts continued to pass; removing its final contrast is the smallest and most informative continuation of that successful pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9743, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering the third `fc2` output column will reduce the model to 1612 parameters while retaining at least 99% accuracy, because its eliminated all-ones component is removed by the final LayerNorm.
change: Represent the third `fc2` column with seven learned coordinates and reconstruct its eighth coordinate so the column sums to zero.
mechanism: Third-column zero-mean MLP output gauge
evidence_used: The current design already centers the first two `fc2` columns and achieves 99.90% at 1613 parameters; although centering the fourth column collapsed to 72.40%, the intervening third column remains the most informative untested one-parameter extension of this mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5824, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the remaining third-coordinate `ln1` bias contrast to the third query-bias parameter will reduce the model to 1612 parameters while retaining at least 99% accuracy, because the contrast remains learned and its projection-space effect is gauge-coupled to the query/value biases.
change: Remove the standalone `ln1` bias parameter and reuse `attn.qv_bias[2]` as its learned third-coordinate contrast.
mechanism: Shared pre-attention/query-bias gauge
evidence_used: The standalone third-coordinate contrast achieved 99.90% at 1613 parameters, whereas fixing it to zero reached 97.43%; sharing it with an existing zero-initialized query bias tests whether retaining an adaptive contrast, rather than an independent degree of freedom, supplies the optimization benefit.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1612, "training_steps": 4999}



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
