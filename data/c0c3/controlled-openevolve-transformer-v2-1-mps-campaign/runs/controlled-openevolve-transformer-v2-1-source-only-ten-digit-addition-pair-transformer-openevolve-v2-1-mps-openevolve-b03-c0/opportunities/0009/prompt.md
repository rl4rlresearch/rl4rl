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
verified_results: {"accuracy": 0.9940000000000001, "parameters": 1620, "training_steps": 4999}
prior_hypothesis: Tying `ln_f.bias` to the already shared block LayerNorm bias will reduce deduplicated parameters from 1,628 to 1,620 while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 8 will lower parameters from 1,644 to 1,576 while retaining at least 99% accuracy.
change: Change the default MLP hidden width from 12 to 8 without altering attention or training.
mechanism: Narrower feed-forward bottleneck
evidence_used: The current 1,644-parameter model achieves 99.96% accuracy, leaving substantial margin above the 99% requirement for an isolated 68-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9390000000000001, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: A `d_ff` width of 10 will retain at least 99% accuracy while reducing learned parameters from 1,644 to 1,610.
change: Change the default MLP hidden width from 12 to 10 while leaving attention and training unchanged.
mechanism: Intermediate feed-forward bottleneck width
evidence_used: Width 12 achieved 99.96% accuracy, while width 8 fell to 93.9%; testing the midpoint isolates whether the capacity threshold lies above or below width 10.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2343, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the 24 QKV bias parameters will reduce the model from 1,644 to 1,620 parameters while retaining at least 99% accuracy because LayerNorm already provides learned channel offsets without reducing attention or MLP width.
change: Disable the bias on the combined query, key, and value projection while leaving all model widths and training settings unchanged.
mechanism: Bias-free QKV projection
evidence_used: The 1,644-parameter baseline reached 99.96%, whereas reducing `d_ff` to 10 or 8 failed; this motivates preserving feed-forward capacity and instead testing a redundant affine component.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4986, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: A `d_ff` width of 11 will retain at least 99% accuracy while reducing parameters from 1,644 to 1,627.
change: Change the default MLP hidden width from 12 to 11 while preserving attention, biases, and training settings.
mechanism: Single-unit feed-forward bottleneck reduction
evidence_used: Width 12 achieved 99.96% accuracy, while widths 10 and 8 failed; width 11 is the most informative remaining capacity boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.718, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the two block LayerNorm bias vectors will reduce deduplicated parameters from 1,644 to 1,636 while retaining at least 99% accuracy because both normalization sites keep learned affine transformations and all attention and MLP capacity remains intact.
change: Tie `ln2.bias` to `ln1.bias` within the transformer block, leaving their scale parameters and all training settings unchanged.
mechanism: Shared pre-sublayer normalization offset
evidence_used: The baseline reached 99.96%, while narrowing the MLP and deleting QKV biases caused large failures; this motivates preserving core capacity and biases while testing a minimal eight-parameter sharing constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln2.weight` to `ln1.weight` will reduce deduplicated parameters from 1,636 to 1,628 while retaining at least 99% accuracy because the validated shared-bias design indicates both normalization sites can share affine structure without removing attention or MLP capacity.
change: Tie both the scale and bias parameters of the block’s two LayerNorm modules.
mechanism: Shared pre-sublayer normalization affine parameters
evidence_used: Sharing the two LayerNorm biases achieved 99.81% accuracy with 1,636 parameters, while narrowing the MLP or removing QKV biases failed; this motivates another isolated eight-parameter sharing constraint within the already validated normalization pair.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9959, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln_f.bias` to the already shared block LayerNorm bias will reduce deduplicated parameters from 1,628 to 1,620 while retaining at least 99% accuracy.
change: Share the final LayerNorm bias with `blocks[0].ln1.bias`, preserving separate normalization scales and all attention and MLP capacity.
mechanism: Shared normalization offset across block and output norms
evidence_used: Sharing both block LayerNorm affine parameters achieved 99.59% accuracy at 1,628 parameters; this motivates an isolated eight-parameter bias-sharing extension rather than reducing the capacity components that previously failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9940000000000001, "parameters": 1620, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln_f.weight` to the shared block LayerNorm weight will reduce deduplicated parameters from 1,620 to 1,612 while retaining at least 99% accuracy.
change: Share both the final LayerNorm scale and bias with `blocks[0].ln1`, preserving all attention and MLP capacity.
mechanism: Shared normalization scale across all LayerNorm sites
evidence_used: Sharing the block LayerNorm scale and bias achieved 99.59% accuracy, and extending bias sharing to the final LayerNorm still achieved 99.40%; this motivates testing the remaining isolated eight-parameter normalization sharing constraint.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2707, "parameters": 1612, "training_steps": 4999}



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
