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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1600, "training_steps": 4999}
prior_hypothesis: Because non-affine LayerNorm outputs sum to zero across the eight residual dimensions, the MLP input projection needs only seven coordinates; reducing `fc1` by 12 parameters to produce a 1,600-parameter model will preserve at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Because the 1,644-parameter baseline reaches 99.96% accuracy, constraining only its positional embeddings to four active dimensions will preserve at least 99% accuracy while reducing the model to 1,152 parameters.
change: Replace each eight-dimensional positional embedding with a learned four-dimensional embedding, zero-padded to the unchanged eight-dimensional residual stream.
mechanism: Four-dimensional learned positional subspace
evidence_used: The current one-layer, width-eight model achieves 0.9996 accuracy at 1,644 parameters, indicating enough accuracy margin to compress positional representation without reducing attention or MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7399, "parameters": 1552, "training_steps": 4999}

RECENT RESULT
hypothesis: Restoring full eight-dimensional positional embeddings while factorizing only the token/input-output embedding through six learned dimensions will retain at least 99% accuracy and reduce parameters from 1,644 to 1,484.
change: Replace the width-eight tied embedding/head with a learned width-six embedding projected into the unchanged width-eight residual stream and reuse its projected weights for logits.
mechanism: Six-dimensional factorized tied token embeddings
evidence_used: Compressing positional embeddings to four dimensions reduced accuracy to 73.99%, showing that positional capacity must be preserved; the 99.96% baseline margin motivates testing compression in the distinct token-representation component instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0421, "parameters": 1464, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the attention key bias will preserve at least 99% accuracy while reducing parameters from 1,644 to 1,636, because a shared key bias adds the same score offset to every visible key and therefore cancels exactly in the attention softmax.
change: Replace the fused QKV bias with learned query and value biases, leaving keys unbiased while preserving the attention computation and residual width.
mechanism: Softmax-invariant key-bias elimination
evidence_used: Compressing positional embeddings collapsed accuracy to 73.99%, and factorizing token embeddings collapsed it to 4.21%; this motivates retaining both learned representations and removing an analytically inactive attention parameter instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9939, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the shared attention value bias will reduce parameters from 1,636 to 1,628 while retaining at least 99% accuracy, because attention weights sum to one and the existing output-projection bias can represent the value bias’s constant contribution exactly.
change: Replace the joint query/value bias with a query-only bias and remove value-bias addition from attention.
mechanism: Softmax-normalized value-bias absorption
evidence_used: Removing the analytically inactive key bias retained 99.39% accuracy, while compressing positional or token representations failed; this motivates another function-class-preserving bias elimination instead of reducing learned representation capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9899, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the learned output-projection bias as the attention value bias will reduce parameters from 1,636 to 1,628 while reaching at least 99% accuracy, because it restores the value-bias optimization path without adding an independently learned vector.
change: Replace the independent query/value bias matrix with a query-only bias and use the projection layer’s existing bias as the value bias.
mechanism: Shared value/output-projection bias
evidence_used: Removing the value bias entirely reached 98.99%, only one correct test example below the requirement, while retaining it reached 99.39%; sharing it with the representationally redundant projection bias preserves both bias pathways at the smaller parameter count.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the second LayerNorm’s 16 affine parameters will reduce the model from 1,628 to 1,612 parameters while retaining at least 99% accuracy, because its learned scale and shift can be absorbed exactly into the following MLP input weights and bias.
change: Make the LayerNorm immediately before the MLP non-affine; leave all attention, embeddings, residual width, and MLP capacity unchanged.
mechanism: MLP pre-normalization affine absorption
evidence_used: The 1,628-parameter shared-bias design reached 99.99% accuracy, whereas reducing positional or token representation capacity failed. This motivates eliminating an exact affine reparameterization rather than removing representational dimensions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Because non-affine LayerNorm outputs sum to zero across the eight residual dimensions, the MLP input projection needs only seven coordinates; reducing `fc1` by 12 parameters to produce a 1,600-parameter model will preserve at least 99% accuracy.
change: Replace the MLP’s eight-input projection with a seven-input projection and omit the linearly dependent final normalized coordinate.
mechanism: Zero-mean LayerNorm coordinate elimination
evidence_used: Making the MLP-preceding LayerNorm non-affine retained 99.92% accuracy at 1,612 parameters, supporting a further exact reparameterization at the same interface rather than compressing positional or token representations, which failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1600, "training_steps": 4999}



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
