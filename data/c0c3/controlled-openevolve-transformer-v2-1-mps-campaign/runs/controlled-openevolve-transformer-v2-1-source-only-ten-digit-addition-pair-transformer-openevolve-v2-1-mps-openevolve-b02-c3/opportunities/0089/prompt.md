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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 804, "training_steps": 4999}
prior_hypothesis: A 804-parameter transformer will retain at least 99% accuracy because fixing the final learned value-feature intercept at zero preserves all four gated features, every input-dependent projection weight, and the qualified lexical and attention paths.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9993000000000001, "parameters": 805, "training_steps": 4999}
prior_hypothesis: A 805-parameter transformer will retain at least 99% accuracy because fixing the second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9995999999999999, "parameters": 806, "training_steps": 4999}
prior_hypothesis: A 806-parameter transformer will retain at least 99% accuracy because fixing a second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9993000000000001, "parameters": 807, "training_steps": 4999}
prior_hypothesis: A 807-parameter transformer will retain at least 99% accuracy because fixing the fourth value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

## Recent verification evidence

RECENT RESULT
hypothesis: An 821-parameter transformer will retain at least 99% accuracy because extending the successful maximum-distance sharing to the adjacent distance preserves a learned bias at every causal distance while tying only the two sparsest query-key diagonals across heads.
change: Reduce the head-specific relative-bias table by one column and replace the shared scalar tail with two learned biases shared across both attention heads.
mechanism: Cross-head sharing of the two sparsest relative-distance biases
evidence_used: The current 822-parameter model achieved 99.94% after sharing the farthest-distance bias, while further value/output constraints failed; extending the same successful, orthogonal sharing mechanism to the next-sparsest distance is the narrowest adjacent reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0546, "parameters": 821, "training_steps": 4999}

RECENT RESULT
hypothesis: An 821-parameter transformer will retain at least 99% accuracy because sharing the maximum-distance bias preserves the qualified 822-parameter design, while fixing the remaining first-head key-channel scale uses a reciprocal Q/K basis degree without tying the functionally important adjacent-distance biases.
change: Reconstruct the qualified shared maximum-distance relative bias and fix the third first-head key channel’s leading coefficient at 0.02, reducing the current 823-parameter model by two parameters.
mechanism: First-head key-scale gauge completion with maximum-distance bias sharing
evidence_used: Maximum-distance sharing achieved 99.94% at 822 parameters, whereas extending sharing to the adjacent distance collapsed to 5.46%; the successful 99.98% fixing of all three first-head value-channel scales motivates testing the analogous remaining scale constraint on the first-head key basis instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0557, "parameters": 821, "training_steps": 4999}

RECENT RESULT
hypothesis: An 821-parameter transformer will retain at least 99% accuracy because fixing the already-shared maximum-distance bias at zero changes only the single sparsest causal query-key pair per head while preserving independent learned biases at every nearer distance.
change: Reconstruct the qualified 822-parameter value-path constraints and replace its shared learned farthest-distance bias with a fixed zero.
mechanism: Fixed maximum-distance attention bias
evidence_used: The 822-parameter design reached 99.94% when only the farthest-distance bias was shared, whereas sharing the adjacent distance collapsed to 5.46%; this isolates the next reduction to the successful maximum-distance mechanism without constraining the important neighboring bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 821, "training_steps": 4999}

RECENT RESULT
hypothesis: A 812-parameter transformer will retain at least 99% accuracy because four multiplicatively gated MLP features can replace seven independent GELU features while preserving the qualified 821-parameter model’s load-bearing lexical rank and attention routing.
change: Reconstruct the qualified 821-parameter attention design, then replace its seven-unit additive GELU MLP with a four-unit sigmoid-gated MLP, reducing the nonlinear sublayer by nine parameters.
mechanism: Compact sigmoid-gated nonlinear carry state
evidence_used: The 821-parameter reference achieved 99.96%; lexical-rank reduction reached only 95.12% and adjacent-distance sharing collapsed to 5.46%, so this patch preserves both mechanisms and instead challenges the untested assumption that carry-related nonlinear computation requires seven independent additive features.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9929000000000001, "parameters": 812, "training_steps": 4999}

RECENT RESULT
hypothesis: An 811-parameter transformer will retain at least 99% accuracy because fixing one sigmoid-gate bias at its zero initialization preserves all four learned gated features, their input-dependent gates, and the qualified model’s attention and lexical capacity.
change: Replace the gated MLP input projection’s final learned bias—the fourth gate intercept—with a fixed zero, removing one parameter while leaving every projection weight and the other seven biases learned.
mechanism: Single fixed gate intercept
evidence_used: The 812-parameter four-feature gated MLP achieved 99.29% accuracy; this motivates testing a one-parameter reduction inside that successful mechanism instead of reducing feature count, lexical rank, or additional relative-attention biases, which prior evidence identifies as fragile.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 811, "training_steps": 4999}

RECENT RESULT
hypothesis: A 810-parameter transformer will retain at least 99% accuracy because fixing a second sigmoid-gate bias at zero preserves all four nonlinear features, every input-dependent gate, and the qualified lexical and attention paths.
change: Reconstruct the qualified four-feature gated MLP, but learn only its first six input-projection biases and fix the final two gate intercepts at zero.
mechanism: Second fixed gate intercept
evidence_used: The 811-parameter design achieved 99.78% after fixing one gate intercept, while reductions to lexical rank or additional relative-attention biases failed; this motivates the narrowest adjacent one-parameter reduction within the successful gated-MLP mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 810, "training_steps": 4999}

RECENT RESULT
hypothesis: A 809-parameter transformer will retain at least 99% accuracy because fixing a third gate bias at zero preserves all four gated nonlinear features and their input-dependent weights while leaving the qualified lexical and attention paths unchanged.
change: Reconstruct the qualified 810-parameter design and fix the final three gated-MLP input-projection biases at zero instead of the final two.
mechanism: Third fixed sigmoid-gate intercept
evidence_used: The 810-parameter design achieved 99.82% after two gate intercepts were fixed, improving over the 812-parameter gated baseline’s 99.29%; the narrowest informative next reduction is one more intercept within this repeatedly successful mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 809, "training_steps": 4999}

RECENT RESULT
hypothesis: An 808-parameter transformer will retain at least 99% accuracy because fixing the fourth and final sigmoid-gate bias at zero preserves all four learned gated features and every input-dependent gate weight.
change: Reconstruct the qualified 809-parameter gated design, then learn only the four value-feature biases while fixing all four sigmoid-gate intercepts at zero.
mechanism: Complete sigmoid-gate intercept fixing
evidence_used: The 809-parameter design achieved 99.94% after three gate intercepts were fixed, improving on the qualified 810- and 811-parameter designs; fixing the sole remaining gate intercept is the narrowest continuation of this repeatedly successful reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 808, "training_steps": 4999}

RECENT RESULT
hypothesis: A 807-parameter transformer will retain at least 99% accuracy because fixing the fourth value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.
change: Learn only the first three gated-MLP input-projection biases, fixing the fourth value intercept and all four sigmoid-gate intercepts at zero.
mechanism: Single fixed value-feature intercept
evidence_used: The 808-parameter design achieved 99.91% after all four gate intercepts were fixed; its substantial margin motivates the narrowest adjacent reduction, removing one intercept while retaining every nonlinear feature and projection weight.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 807, "training_steps": 4999}

RECENT RESULT
hypothesis: A 806-parameter transformer will retain at least 99% accuracy because fixing a second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.
change: Reconstruct the qualified 807-parameter bias constraints and additionally fix the third value-feature intercept, leaving only the first two MLP input-projection biases learned.
mechanism: Second fixed value-feature intercept
evidence_used: The 807-parameter design achieved 99.93% after fixing the fourth value intercept and all gate intercepts; its margin motivates the narrowest adjacent reduction within the same repeatedly successful bias-fixing mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 806, "training_steps": 4999}

RECENT RESULT
hypothesis: A 805-parameter transformer will retain at least 99% accuracy because fixing the second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.
change: Extend the qualified 806-parameter MLP bias constraint by learning only its first input-projection bias and fixing the remaining seven intercepts at zero.
mechanism: Third fixed value-feature intercept
evidence_used: The 806-parameter design achieved 99.96% after fixing two value-feature and all four gate intercepts; its strong margin motivates the narrowest adjacent one-parameter reduction within the repeatedly successful bias-fixing mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 805, "training_steps": 4999}

RECENT RESULT
hypothesis: A 804-parameter transformer will retain at least 99% accuracy because fixing the final learned value-feature intercept at zero preserves all four gated features, every input-dependent projection weight, and the qualified lexical and attention paths.
change: Disable the gated MLP input projection bias, fixing all eight value and gate intercepts at zero and removing seven parameters from the current 811-parameter implementation.
mechanism: Complete gated-MLP intercept fixing
evidence_used: The 805-parameter design achieved 99.93% with only the first value-feature bias learned; fixing that sole remaining intercept is the narrowest continuation of the repeatedly successful bias-removal sequence from 811 through 805 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 804, "training_steps": 4999}



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
