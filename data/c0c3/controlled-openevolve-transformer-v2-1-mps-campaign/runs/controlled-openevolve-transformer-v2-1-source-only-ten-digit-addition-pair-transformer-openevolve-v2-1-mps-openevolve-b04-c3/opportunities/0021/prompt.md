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
verified_results: {"accuracy": 0.9992, "parameters": 1588, "training_steps": 44999}
prior_hypothesis: Removing all eight redundant key-projection bias coordinates and training for 45,000 steps will retain at least 99% accuracy with 1,588 learned parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}
prior_hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1589, "training_steps": 39999}
prior_hypothesis: Removing a seventh redundant key-projection bias coordinate and training for 40,000 steps will retain at least 99% accuracy with 1,589 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing two redundant key-projection bias coordinates and training for 10,000 steps will achieve at least 99% accuracy with 1,642 parameters.
change: Preserve the original constructor RNG stream, reconstruct two fixed-zero key-bias coordinates in the fused QKV operation, and increase default training from 5,000 to 10,000 steps.
mechanism: Two-coordinate softmax-invariant key-bias elimination
evidence_used: The one-coordinate reduction reached 96.07% at 4,999 steps and 99.98% at 7,499 steps with 1,643 parameters, showing that incremental key-bias elimination preserves capacity but needs longer optimization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1642, "training_steps": 9999}

RECENT RESULT
hypothesis: Replacing both affine pre-LayerNorms with parameter-free RMS normalization will preserve at least 99% accuracy after 10,000 training steps while reducing learned parameters from 1,644 to 1,612.
change: Preserve the full-width embeddings, attention, MLP, and affine final LayerNorm, but normalize each residual branch by RMS without learned scale or bias and extend training to 10,000 steps.
mechanism: Parameter-free RMS pre-normalization
evidence_used: Reducing `d_ff` removed an activation unit and collapsed accuracy to 74.93%, showing that representational width is load-bearing. RMS normalization instead removes 32 affine parameters while retaining the activation mean as an additional usable dimension; the 1,642-parameter reference also establishes that 10,000 steps can recover convergence in a perturbed parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1612, "training_steps": 9999}

RECENT RESULT
hypothesis: Replacing the affine final LayerNorm with parameter-free RMS normalization will retain at least 99% accuracy after 10,000 training steps while reducing learned parameters from 1,612 to 1,596.
change: Use the existing parameter-free RMSNorm for the model’s final normalization, removing its eight learned scales and eight learned biases.
mechanism: Parameter-free RMS output normalization
evidence_used: The current design replaced both affine pre-LayerNorms with RMSNorm, removed 32 parameters, and still achieved 99.98% accuracy at 9,999 steps; this directly motivates testing the same normalization simplification at the output.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9901000000000001, "parameters": 1596, "training_steps": 9999}

RECENT RESULT
hypothesis: Combining the proven two-coordinate key-bias elimination with parameter-free RMS normalization and training for 15,000 steps will achieve at least 99% accuracy with 1,594 parameters.
change: Replace all three affine LayerNorms with parameter-free RMSNorm while retaining the current two-coordinate key-bias reduction, and extend training from 10,000 to 15,000 steps.
mechanism: Parameter-free RMS normalization combined with redundant key-bias elimination
evidence_used: The all-RMSNorm design achieved 99.01% accuracy with 1,596 parameters at 10,000 steps, while the current two-coordinate key-bias reduction achieved 99.94%; additional training previously recovered reduced key-bias models from sub-threshold accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1594, "training_steps": 14999}

RECENT RESULT
hypothesis: Removing a third softmax-invariant key-bias coordinate and training for 20,000 steps will retain at least 99% accuracy with 1,593 learned parameters.
change: Replace all affine LayerNorms with parameter-free RMSNorm, reconstruct three fixed-zero key-bias coordinates, and extend training from 7,500 to 20,000 steps.
mechanism: Incremental key-bias quotient with parameter-free RMS normalization
evidence_used: The 1,594-parameter design combining RMSNorm with two removed key-bias coordinates achieved 99.96% accuracy at 14,999 steps; the earlier progression from one to two removed coordinates showed that additional training recovers convergence after incremental redundant-coordinate removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1593, "training_steps": 19999}

RECENT RESULT
hypothesis: Removing a fourth softmax-invariant key-bias coordinate and training for 25,000 steps will retain at least 99% accuracy with 1,592 learned parameters.
change: Reconstruct four fixed-zero key-bias coordinates in the fused QKV projection and extend default training from 20,000 to 25,000 steps.
mechanism: Incremental fourth-coordinate key-bias quotient
evidence_used: The three-coordinate design achieved 99.96% accuracy with 1,593 parameters at 19,999 steps, continuing the successful progression from two removed coordinates at 14,999 steps; another single-coordinate reduction with 5,000 additional steps is the closest supported extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1592, "training_steps": 24999}

RECENT RESULT
hypothesis: Removing a fifth redundant key-projection bias coordinate and training for 30,000 steps will retain at least 99% accuracy with 1,591 learned parameters.
change: Reconstruct five fixed-zero key-bias coordinates in the fused QKV projection and increase default training from 15,000 to 30,000 steps.
mechanism: Fifth-coordinate softmax-invariant key-bias quotient
evidence_used: The four-coordinate design achieved 99.95% accuracy with 1,592 parameters at 24,999 steps, extending the successful progression in which each additional removed coordinate was paired with 5,000 more training steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1591, "training_steps": 29999}

RECENT RESULT
hypothesis: Removing a sixth redundant key-projection bias coordinate and training for 35,000 steps will retain at least 99% accuracy with 1,590 learned parameters.
change: Reconstruct six fixed-zero key-bias coordinates in the fused QKV projection while preserving constructor RNG consumption, and increase default training from 10,000 to 35,000 steps.
mechanism: Sixth-coordinate softmax-invariant key-bias quotient
evidence_used: The five-coordinate design achieved 99.85% accuracy with 1,591 parameters at 29,999 steps; the successful progression from two through five removed coordinates supports testing one additional coordinate with 5,000 more steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1590, "training_steps": 34999}

RECENT RESULT
hypothesis: Removing a seventh redundant key-projection bias coordinate and training for 40,000 steps will retain at least 99% accuracy with 1,589 learned parameters.
change: Reconstruct seven fixed-zero key-bias coordinates in the fused QKV projection while preserving constructor RNG consumption, and increase default training from 35,000 to 40,000 steps.
mechanism: Seventh-coordinate softmax-invariant key-bias quotient
evidence_used: The six-coordinate design achieved 100% accuracy with 1,590 parameters at 34,999 steps; the successful progression from two through six removed coordinates supports one further coordinate reduction paired with 5,000 additional training steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1589, "training_steps": 39999}

RECENT RESULT
hypothesis: Removing all eight redundant key-projection bias coordinates and training for 45,000 steps will retain at least 99% accuracy with 1,588 learned parameters.
change: Reconstruct the entire key-bias vector as fixed zeros while preserving constructor RNG consumption, and increase default training from 30,000 to 45,000 steps.
mechanism: Complete softmax-invariant key-bias elimination
evidence_used: The seven-coordinate design achieved 99.96% accuracy with 1,589 parameters at 39,999 steps; completing the exact key-bias quotient with 5,000 additional steps is the closest supported extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1588, "training_steps": 44999}

RECENT RESULT
hypothesis: Removing all key-projection biases and the value-projection bias will retain at least 99% accuracy after 50,000 steps with 1,580 learned parameters, because key bias is softmax-invariant and value bias can be absorbed exactly by the retained attention output bias.
change: Store only the eight query-bias parameters, reconstruct fixed-zero key and value biases in the fused QKV projection, and increase default training from 25,000 to 50,000 steps.
mechanism: Joint key/value bias quotient in causal attention
evidence_used: The 1,588-parameter design removing every key bias achieved 99.92% accuracy at 44,999 steps. The next eight parameters are functionally redundant: attention weights sum to one, so value bias adds the same vector at every position, which the learned output-projection bias can represent directly.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1580, "training_steps": 49999}

RECENT RESULT
hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.
change: Replace per-position learned vectors with generic sinusoidal features scaled by one learned parameter, and extend training from 50,000 to 75,000 steps.
mechanism: Learned-scale sinusoidal positional representation
evidence_used: Every qualified design retains a separate learned vector for every sequence position, while the current 1,580-parameter model reaches 99.93% accuracy. This tests whether the learned attention and MLP can instead route information through a compact relative-offset-friendly positional basis, targeting substantially more parameters than further bias removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}



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
