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
verified_results: {"accuracy": 0.9976, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Tying the attention value bias to the output-projection bias will reduce parameters from 1636 to 1628 while retaining at least 99% accuracy, because it preserves both bias pathways whose isolated removal failed while eliminating their redundant independent parameterization.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower parameters from 1644 to 1576 while retaining at least 99% accuracy because the current model reaches 99.96%, leaving substantial accuracy margin.
change: Reduce the single transformer block’s default feed-forward width by one third while leaving attention, embeddings, training, and decoding unchanged.
mechanism: Narrower nonlinear feed-forward bottleneck
evidence_used: The verified `d_model=8, d_ff=12` baseline achieves 0.9996 accuracy with 1644 parameters; a controlled feed-forward-width reduction is the most direct low-risk capacity ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6964, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing reparameterizable pre-LayerNorm affine parameters and redundant key/value biases will reduce parameters from 1644 to 1596 while retaining at least 99% accuracy, because it preserves width and representational capacity unlike the failed `d_ff=8` ablation.
change: Make both pre-LayerNorms non-affine and replace the combined QKV projection with separate projections that retain only the functionally relevant query bias.
mechanism: Gauge-free pre-normalization with bias-minimal attention
evidence_used: The 1644-parameter `d_ff=12` model achieved 99.96%, while narrowing `d_ff` to 8 produced only 69.64%; this motivates preserving the nonlinear width and instead removing parameterization redundancies.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.042300000000000004, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the attention key bias will reduce parameters from 1644 to 1636 while retaining at least 99% accuracy, because a shared key bias adds the same query-dependent constant to every unmasked attention logit and therefore cancels exactly in softmax.
change: Replace the combined QKV bias with learned query and value biases, preserving the combined QKV weight, all LayerNorm affine parameters, and the original feed-forward width.
mechanism: Exact softmax-invariant key-bias elimination
evidence_used: The 1644-parameter baseline reached 99.96%, whereas simultaneously removing LayerNorm affine parameters and key/value biases fell to 4.23%; this isolates the mathematically redundant key bias without repeating the disruptive LayerNorm or value-bias removals.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the attention value bias will reduce parameters from 1636 to 1628 while retaining at least 99% accuracy, because softmax attention weights sum to one, making the shared value bias a constant that the existing output-projection bias can represent exactly.
change: Delete the learned value bias and its addition while preserving the query bias, projection bias, LayerNorm parameters, feed-forward width, training, and decoding.
mechanism: Exact value/output-bias redundancy elimination
evidence_used: The isolated removal of the mathematically redundant key bias retained 99.95% accuracy at 1636 parameters; the failed 1596-parameter experiment confounded value-bias removal with disruptive non-affine LayerNorms, so isolating the other exact attention-bias redundancy is the most informative next ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.18289999999999998, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the attention value bias to the output-projection bias will reduce parameters from 1636 to 1628 while retaining at least 99% accuracy, because it preserves both bias pathways whose isolated removal failed while eliminating their redundant independent parameterization.
change: Make the projection bias and value bias reference the same learned 8-element parameter, leaving all computations, widths, training, and decoding unchanged.
mechanism: Shared value/output bias parameterization
evidence_used: Removing the value bias alone reduced accuracy to 18.29%, indicating that its optimization pathway matters despite functional redundancy with the projection bias; sharing the biases preserves that pathway and the direct projection-bias pathway while deduplicating eight parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the second pre-LayerNorm bias to the MLP output bias will reduce parameters from 1628 to 1620 while retaining at least 99% accuracy, because the MLP input bias can absorb any LayerNorm-bias effect while both optimization pathways remain active.
change: Make `ln2.bias` and `mlp.fc2.bias` reference the same learned 8-element parameter.
mechanism: Shared reparameterizable MLP/LayerNorm bias
evidence_used: Sharing the redundant attention value/output biases achieved 99.76% at 1628 parameters, whereas deleting the value bias collapsed accuracy; this motivates another pathway-preserving tie instead of removing a redundant bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1124, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower parameters from 1628 to 1611 while retaining at least 99% accuracy, because the failed width-8 model removed four neurons at once and does not establish that the immediately adjacent width 11 is below the capacity threshold.
change: Reduce the single transformer block’s default feed-forward width by one neuron while preserving attention, bias sharing, training, and decoding.
mechanism: Single-neuron feed-forward width ablation
evidence_used: The width-12 design achieves 99.76% at 1628 parameters, while width 8 achieves only 69.64%; testing width 11 is the smallest controlled step toward locating the feed-forward capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.727, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing `ln2.bias` will reduce parameters from 1628 to 1620 while retaining at least 99% accuracy, because its contribution is exactly absorbable by the independent `fc1.bias` before GELU.
change: Disable only the second LayerNorm’s bias while preserving its learned scale and all MLP parameters.
mechanism: Pre-MLP bias reparameterization
evidence_used: Tying `ln2.bias` to the post-nonlinearity `fc2.bias` failed at 11.24%; this tests the mathematically valid redundancy with the pre-nonlinearity `fc1.bias` without narrowing the MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1487, "parameters": 1620, "training_steps": 4999}



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
