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
verified_results: {"accuracy": 0.9981, "parameters": 803, "training_steps": 4999}
prior_hypothesis: A 803-parameter transformer will retain at least 99% accuracy because fixing one final classifier-offset coefficient preserves the full-rank tied token embedding, all attention routing, and all four gated nonlinear features.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9993000000000001, "parameters": 805, "training_steps": 4999}
prior_hypothesis: A 805-parameter transformer will retain at least 99% accuracy because fixing the second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9992, "parameters": 802, "training_steps": 4999}
prior_hypothesis: A 802-parameter transformer will retain at least 99% accuracy because fixing a second final classifier-offset coefficient preserves the full-rank tied token embedding, all attention routing, and all four gated nonlinear features.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9951000000000001, "parameters": 801, "training_steps": 4999}
prior_hypothesis: A 801-parameter transformer will retain at least 99% accuracy because fixing the final classifier-offset coefficient at zero preserves the full-rank tied token embedding, attention routing, all four gated nonlinear features, and all learned final normalization scales.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 803-parameter transformer will retain at least 99% accuracy because fixing one final classifier-offset coefficient preserves the full-rank tied token embedding, all attention routing, and all four gated nonlinear features.
change: Learn two rather than three final hidden-space bias coefficients, fixing the third coefficient at zero.
mechanism: Single fixed classifier-bias direction
evidence_used: The 804-parameter design achieved 99.96% accuracy after all gated-MLP intercepts were removed; its margin supports the narrowest one-parameter reduction, while preserving lexical rank because the prior lexical-rank reduction reached only 95.12%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 803, "training_steps": 4999}

RECENT RESULT
hypothesis: A `803 - VOCAB_SIZE`-parameter transformer will retain at least 99% accuracy because a parameter-free quadratic lift restores five-dimensional token and classifier geometry from four learned coordinates, avoiding the linear-rank bottleneck that hurt the prior lexical reduction.
change: Reconstruct the qualified 803-parameter bias constraints, then replace each five-coordinate token code with four learned coordinates whose pairwise products supply a fifth nonlinear feature shared by input embeddings and output logits.
mechanism: Quadratically lifted rank-four tied lexical manifold
evidence_used: The linear lexical-rank reduction still reached 95.12%, suggesting compact token codes retain most of the task but losing a classifier direction is load-bearing. The successful multiplicatively gated MLP shows learned low-dimensional coordinates can recover useful capacity through parameter-free products.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.41969999999999996, "parameters": 689, "training_steps": 4999}

RECENT RESULT
hypothesis: A 802-parameter transformer will retain at least 99% accuracy because fixing a second final classifier-offset coefficient preserves the full-rank tied token embedding, all attention routing, and all four gated nonlinear features.
change: Reconstruct the qualified 803-parameter bias-free gated MLP and additionally learn only one final hidden-space bias coefficient, fixing the remaining four coefficients at zero.
mechanism: Second fixed classifier-bias direction
evidence_used: The 803-parameter design achieved 99.81% accuracy with two learned final-bias coefficients; removing one more coefficient is the narrowest reduction from the smallest qualified design, while the failed 689-parameter result warns against reducing lexical rank.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 802, "training_steps": 4999}

RECENT RESULT
hypothesis: A 801-parameter transformer will retain at least 99% accuracy because fixing the final classifier-offset coefficient at zero preserves the full-rank tied token embedding, attention routing, all four gated nonlinear features, and all learned final normalization scales.
change: Reconstruct the qualified bias-free gated MLP and remove the sole classifier-offset coefficient remaining in the 802-parameter design.
mechanism: Complete classifier-offset fixing
evidence_used: The 802-parameter design achieved 99.92% after reducing the classifier offset to one learned coefficient; removing that last coefficient is the narrowest reduction from the smallest qualified design, while the failed 689-parameter experiment warns against reducing lexical rank.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9951000000000001, "parameters": 801, "training_steps": 4999}

RECENT RESULT
hypothesis: A 800-parameter transformer will retain at least 99% accuracy because fixing the third classifier-relevant LayerNorm scale at one preserves two learned scales, the full-rank tied lexical embedding, attention routing, and all four gated nonlinear features.
change: Learn only the first two final LayerNorm scales and fix the remaining four scales at one, removing one parameter.
mechanism: Single fixed final-normalization scale
evidence_used: The 801-parameter design achieved 99.51% accuracy after eliminating the classifier offset; its remaining margin motivates the narrowest one-parameter reduction while avoiding the lexical-rank reduction that fell to 41.97%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.10869999999999999, "parameters": 800, "training_steps": 4999}

RECENT RESULT
hypothesis: A 800-parameter transformer will retain at least 99% accuracy because fixing each head’s remaining key-coordinate anchor exploits Q/K basis-change freedom while preserving the qualified classifier offset, all three final normalization scales, full-rank lexical embedding, and attention routing capacity.
change: Extend the existing Q/K gauge constraint from the first two key channels to all three channels in each head, fixing two additional QKV coefficients at 0.02.
mechanism: Complete per-head key-vector gauge anchoring
evidence_used: The 802-parameter design achieved 99.92%, whereas removing a final normalization scale caused accuracy to collapse to 10.87%; this motivates preserving the classifier path and extending the already-qualified per-head Q/K gauge anchoring instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3151, "parameters": 800, "training_steps": 4999}



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
