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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1589, "training_steps": 39999}
prior_hypothesis: Removing a seventh redundant key-projection bias coordinate and training for 40,000 steps will retain at least 99% accuracy with 1,589 learned parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9969, "parameters": 1397, "training_steps": 74999}
prior_hypothesis: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1588, "training_steps": 44999}
prior_hypothesis: Removing all eight redundant key-projection bias coordinates and training for 45,000 steps will retain at least 99% accuracy with 1,588 learned parameters.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing the sinusoidal position gain will retain at least 99% accuracy after 90,000 steps while reducing the qualified 1,397-parameter design to 1,396 parameters.
change: Adopt the qualified query-only QKV bias and sinusoidal positional basis, fix its amplitude at 0.02 instead of learning one scalar, and train for 90,000 steps.
mechanism: Fixed-amplitude sinusoidal positional representation
evidence_used: The 1,397-parameter learned-scale sinusoidal design achieved 99.69% accuracy at 74,999 steps; fixing its single gain at its initial value is the smallest controlled reduction and preserves all positional frequencies.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Because each attention head’s Q/K coordinates can be jointly rotated without changing attention logits, retaining one learned query-bias coordinate per head will preserve at least 99% accuracy after 75,000 steps while reducing the qualified 1,397-parameter sinusoidal design to 1,391 parameters.
change: Adopt the qualified learned-scale sinusoidal positional representation and exact key/value-bias quotient, represent each four-dimensional query bias with one learned coordinate, and train for 75,000 steps.
mechanism: Per-head query-bias gauge fixing with learned-scale sinusoidal positions
evidence_used: The 1,397-parameter learned-scale sinusoidal design achieved 99.69% accuracy at 74,999 steps. This patch preserves its learned positional gain and removes only six within-head query-bias orientation degrees that can be absorbed by jointly rotating the learned query and key projections.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1531, "parameters": 1391, "training_steps": 74999}

RECENT RESULT
hypothesis: Fixing the sinusoidal position amplitude at 0.02 will retain at least 99% accuracy after 75,000 steps while reducing the qualified 1,397-parameter design to 1,396 parameters.
change: Remove the learned positional gain and add the existing fixed-amplitude sinusoidal buffer directly to token embeddings.
mechanism: Fixed-amplitude sinusoidal positional representation
evidence_used: The learned-scale sinusoidal design achieved 99.69% accuracy at 74,999 steps; the prior fixed-amplitude attempt produced no accuracy evidence because 90,000-step training exceeded the time limit, so testing the same one-parameter reduction at the qualified 75,000-step budget is the closest controlled experiment.
result: training did not finish within the verification time limit
reported_values: {"training_steps": 74999.0}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate while retaining the qualified learned positional gain will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps.
change: Adopt query-only QKV bias and learned-scale sinusoidal positions, remove one query-bias coordinate through an exact per-head rotational gauge, train for 75,000 steps, and reduce validation frequency to remain within the verification time limit.
mechanism: Incremental single-coordinate query-bias gauge fixing with learned-scale sinusoidal positions
evidence_used: The 1,397-parameter sinusoidal design achieved 99.69% accuracy at 74,999 steps, whereas removing six query-bias coordinates at once collapsed to 15.31%; removing only one coordinate is the closest informative intermediate test. Recent 75,000-step timeouts motivate reducing validation overhead without changing optimization updates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Encoding the sinusoidal position gain in the exact reciprocal Q/K scaling gauge will retain at least 99% accuracy after 75,000 steps with 1,396 parameters.
change: Adopt the qualified query-only QKV bias and sinusoidal positions, derive their gain from the learned Q/K weight-energy ratio instead of a separate parameter, and use sparse validation to reduce verification time.
mechanism: Q/K scale-gauge positional gain
evidence_used: The learned-gain sinusoidal design achieved 99.69% accuracy with 1,397 parameters at 74,999 steps. Fixed-gain variants produced no accuracy evidence because they timed out, while aggressive query-bias gauge fixing collapsed accuracy; recycling the smooth global Q/K scaling symmetry preserves adaptive positional amplitude without fixing query-bias coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1396, "training_steps": 74999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate while retaining the learned sinusoidal gain will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps, and endpoint-only validation will let training finish within the time limit.
change: Reconstruct one fixed-zero query-bias coordinate, retain all other learned parameters, and evaluate only at positive interval steps and the final step with a 75,000-step default interval.
mechanism: Single-coordinate query-bias gauge fixing with endpoint validation
evidence_used: The qualified 1,397-parameter design achieved 99.69% accuracy at 74,999 steps. Removing six query-bias coordinates collapsed accuracy, but the prior one-coordinate experiment produced no accuracy result because it timed out; testing the smallest gauge reduction with validation overhead reduced to one final evaluation is the closest unresolved experiment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9745, "parameters": 1396, "training_steps": 74999}

RECENT RESULT
hypothesis: Fixing the sinusoidal amplitude at 0.02 will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps, while endpoint-only validation will allow verification to finish within the time limit.
change: Adopt query-only QKV bias, replace learned positional embeddings with a fixed sinusoidal buffer, remove the learned positional gain, and validate only at the final training step.
mechanism: Fixed-amplitude sinusoidal positions with endpoint-only validation
evidence_used: The learned-gain sinusoidal design achieved 99.69% accuracy with 1,397 parameters at 74,999 steps. Prior fixed-gain runs produced no accuracy evidence solely because they timed out, while endpoint-only validation allowed a separate 75,000-step 1,396-parameter experiment to finish.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1396, "training_steps": 74999}



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
