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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 681, "training_steps": 4999}
prior_hypothesis: A 681-parameter transformer will retain at least 99% accuracy because fixing the second head’s sixth-longest relative-position bias preserves the qualified asymmetric nonlinear lexical lifts, all three load-bearing final-normalization scales, and the first head’s long-distance routing coefficients.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9994, "parameters": 679, "training_steps": 4999}
prior_hypothesis: A correctly constructed 679-parameter transformer will retain at least 99% accuracy because it fixes only the second head’s eighth-longest relative-position bias, whereas the failed 678-parameter trial removed two parameters and therefore did not test this intermediate design.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9982, "parameters": 800, "training_steps": 4999}
prior_hypothesis: A 800-parameter transformer will retain at least 99% accuracy because fixing one head’s sparsely used longest learned relative-position bias preserves the other head’s coefficient, while the qualified 801-parameter model’s full-rank lexical path, attention projections, MLP features, and normalization scales remain intact.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 682, "training_steps": 4999}
prior_hypothesis: A 682-parameter transformer will retain at least 99% accuracy because fixing the second head’s fifth-longest relative-position bias preserves the qualified nonlinear lexical lifts, all three final-normalization scales, and the first head’s long-distance routing capacity.

## Recent verification evidence

RECENT RESULT
hypothesis: A 800-parameter transformer will retain at least 99% accuracy because fixing each head’s remaining key-coordinate anchor exploits Q/K basis-change freedom while preserving the qualified classifier offset, all three final normalization scales, full-rank lexical embedding, and attention routing capacity.
change: Extend the existing Q/K gauge constraint from the first two key channels to all three channels in each head, fixing two additional QKV coefficients at 0.02.
mechanism: Complete per-head key-vector gauge anchoring
evidence_used: The 802-parameter design achieved 99.92%, whereas removing a final normalization scale caused accuracy to collapse to 10.87%; this motivates preserving the classifier path and extending the already-qualified per-head Q/K gauge anchoring instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3151, "parameters": 800, "training_steps": 4999}

RECENT RESULT
hypothesis: A 800-parameter transformer will retain at least 99% accuracy because fixing one head’s sparsely used longest learned relative-position bias preserves the other head’s coefficient, while the qualified 801-parameter model’s full-rank lexical path, attention projections, MLP features, and normalization scales remain intact.
change: Reconstruct the qualified 801-parameter design by removing the classifier offset, then fix only the last head’s longest currently learned relative-attention bias at zero.
mechanism: Asymmetric longest-learned-distance bias fixing
evidence_used: The 801-parameter design achieved 99.51% accuracy; unlike the failed 800-parameter changes to final normalization and Q/K projections, this removes one coefficient from the next-sparsest relative distance, adjacent to the already-fixed uniquely sparse maximum distance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 800, "training_steps": 4999}

RECENT RESULT
hypothesis: A 799-parameter transformer will retain at least 99% accuracy because fixing both heads’ sparsely used next-longest relative-position bias preserves the qualified model’s full-rank lexical path, normalization scales, attention projections, and nonlinear features.
change: Reconstruct the qualified 800-parameter design, then fix the first head’s next-longest relative-attention bias in addition to the already qualified fixed coefficient for the second head.
mechanism: Symmetric next-longest-distance attention-bias fixing
evidence_used: The 800-parameter reference achieved 99.82% after fixing one head’s next-longest relative-position coefficient; applying the same constraint to the corresponding equally sparse coefficient of the other head is the narrowest adjacent reduction, while prior reductions to final normalization or Q/K projections failed sharply.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 799, "training_steps": 4999}

RECENT RESULT
hypothesis: A 798-parameter transformer will retain at least 99% accuracy because fixing one head’s third-longest relative-position bias preserves the full-rank lexical path, normalization scales, attention projections, nonlinear features, and the other head’s coefficient at that distance.
change: Flatten the learned relative-bias coefficients, remove the last head’s longest remaining learned coefficient, and reconstruct it as zero during the forward pass.
mechanism: Asymmetric third-longest-distance attention-bias fixing
evidence_used: The 799-parameter design achieved 99.75% after fixing both heads’ two sparsest maximum-distance biases; its margin motivates the narrowest adjacent reduction at the next-sparsest distance while avoiding the final-normalization and Q/K constraints that failed sharply.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965999999999999, "parameters": 798, "training_steps": 4999}

RECENT RESULT
hypothesis: A 797-parameter transformer will retain at least 99% accuracy because fixing the first head’s third-longest relative-position bias preserves the full-rank lexical path, normalization scales, attention projections, nonlinear features, and the qualified second head’s corresponding constraint.
change: Reconstruct the qualified 798-parameter relative-bias layout and fix the corresponding third-longest coefficient for the first head, reducing the model by one parameter.
mechanism: Symmetric third-longest-distance attention-bias fixing
evidence_used: The 798-parameter design achieved 99.66% accuracy after asymmetrically fixing one third-longest-distance bias; its margin supports applying the same constraint to the corresponding equally sparse coefficient of the other head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9207, "parameters": 797, "training_steps": 4999}

RECENT RESULT
hypothesis: A 797-parameter transformer will retain at least 99% accuracy because further pruning the already-tolerant second head preserves the first head’s load-bearing third-longest relative-position coefficient.
change: Reconstruct the qualified 798-parameter relative-bias layout, then fix the second head’s fourth-longest-distance coefficient at zero.
mechanism: Deeper asymmetric long-distance attention-bias fixing
evidence_used: The asymmetric 798-parameter design achieved 99.66%, while symmetrically fixing the first head’s corresponding coefficient reduced accuracy to 92.07%; this motivates preserving that first-head coefficient and testing the next-sparsest coefficient only in the second head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 797, "training_steps": 4999}

RECENT RESULT
hypothesis: A 683-parameter transformer will retain at least 99% accuracy because four learned token coordinates can produce distinct five-dimensional input and classifier geometries through scale-homogeneous quadratic lifts, while preserving all three load-bearing final-normalization scales and the qualified 797-parameter attention layout.
change: Reconstruct the qualified 797-parameter relative-bias and classifier configuration, then replace the shared five-coordinate token table with four-coordinate codes whose input and output paths use different RMS-normalized quadratic fifth coordinates.
mechanism: Asymmetric homogeneous nonlinear lexical lifts
evidence_used: The fixed shared quadratic lift reached only 41.97%, despite the linear rank-four model reaching 95.12%, while removing one final LayerNorm scale separately collapsed accuracy to 10.87%. This patch tests whether the failure came from forcing both lexical paths onto the same poorly scaled nonlinear chart: it keeps five-dimensional geometry on both paths, preserves all three normalization scales, and starts from the 797-parameter attention design that achieved 99.83%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 683, "training_steps": 4999}

RECENT RESULT
hypothesis: A 682-parameter transformer will retain at least 99% accuracy because fixing the second head’s fifth-longest relative-position bias preserves the qualified nonlinear lexical lifts, all three final-normalization scales, and the first head’s long-distance routing capacity.
change: Remove the next-longest learned relative-bias coefficient from the already more heavily pruned second attention head and reconstruct it as zero during the forward pass.
mechanism: Deeper asymmetric long-distance attention-bias fixing
evidence_used: The current 683-parameter design achieved 99.92% accuracy, and the 797-parameter design achieved 99.83% after deeper pruning of the tolerant second head while the corresponding first-head constraint failed at 92.07%; this motivates the narrowest adjacent reduction in the second head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 682, "training_steps": 4999}

RECENT RESULT
hypothesis: A 681-parameter transformer will retain at least 99% accuracy because fixing the second head’s sixth-longest relative-position bias preserves the qualified asymmetric nonlinear lexical lifts, all three load-bearing final-normalization scales, and the first head’s long-distance routing coefficients.
change: Reconstruct the qualified 682-parameter nonlinear lexical design from the current 797-parameter source, then remove one additional relative-bias coefficient from the more heavily pruned second attention head.
mechanism: Sixth-longest tolerant-head relative-bias fixing
evidence_used: The 683-parameter nonlinear-lift design achieved 99.92%, and pruning the second head’s fifth-longest bias still achieved 99.91% at 682 parameters; this motivates the narrowest adjacent one-parameter reduction while preserving the final-normalization scales whose reduction previously collapsed accuracy to 10.87%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 681, "training_steps": 4999}

RECENT RESULT
hypothesis: A 680-parameter transformer will retain at least 99% accuracy because fixing the second head’s seventh-longest relative-position bias preserves the qualified asymmetric nonlinear lexical lifts, all three load-bearing final-normalization scales, and every first-head routing coefficient.
change: Reconstruct the qualified 681-parameter nonlinear lexical design, then remove one additional relative-bias coefficient from the more heavily pruned second attention head.
mechanism: Seventh-longest tolerant-head relative-bias fixing
evidence_used: The 681-parameter design achieved 99.96% after sequentially pruning the second head through its sixth-longest bias, while pruning the corresponding first-head coefficient previously fell to 92.07%; this supports one more adjacent second-head reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 680, "training_steps": 4999}

RECENT RESULT
hypothesis: A 679-parameter transformer will retain at least 99% accuracy because fixing the second head’s eighth-longest relative-position bias preserves the qualified asymmetric nonlinear lexical lifts, all three final-normalization scales, and every first-head routing coefficient.
change: Reconstruct the qualified 680-parameter nonlinear lexical design, then remove one additional relative-bias coefficient from the more heavily pruned second attention head.
mechanism: Eighth-longest tolerant-head relative-bias fixing
evidence_used: The 680-parameter design achieved 99.94% after sequentially pruning the second head through its seventh-longest bias, while the corresponding first-head pruning previously reduced accuracy to 92.07%; this motivates the narrowest adjacent one-parameter reduction in the tolerant head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3969, "parameters": 678, "training_steps": 4999}

RECENT RESULT
hypothesis: A correctly constructed 679-parameter transformer will retain at least 99% accuracy because it fixes only the second head’s eighth-longest relative-position bias, whereas the failed 678-parameter trial removed two parameters and therefore did not test this intermediate design.
change: Remove exactly one additional learned relative-bias coefficient from the second attention head and reconstruct it as zero.
mechanism: Corrected single-coefficient tolerant-head bias pruning
evidence_used: The 680-parameter design achieved 99.94% accuracy; the attempted next design reported 678 rather than the intended 679 parameters and collapsed to 39.69%, motivating an exact one-parameter intermediate test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 679, "training_steps": 4999}



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
