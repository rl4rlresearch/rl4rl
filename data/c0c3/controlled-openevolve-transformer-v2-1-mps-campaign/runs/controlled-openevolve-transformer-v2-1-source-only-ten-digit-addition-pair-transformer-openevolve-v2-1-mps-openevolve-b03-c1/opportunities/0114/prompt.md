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
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1041, "training_steps": 4999}
prior_hypothesis: Applying the same score-preserving query–key rotation to the second head will reduce the verified model from 1,042 to 1,041 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing one redundant shared-value scale coordinate will reduce the verified model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because both independent head gains remain learned and can absorb any common rescaling of the shared value map.
change: Fix the largest initialized shared-value coefficient as a buffer, learn the other 27 coefficients, and reconstruct the full value map during attention while preserving both head-specific gains.
mechanism: Common value-scale gauge fixing
evidence_used: Disjoint routing with two learned gains achieved 99.96% at 1,046 parameters, whereas tying those gains reduced accuracy to 95.25%; this patch preserves their independence and removes only the exact common scaling redundancy between the shared value map and both gains.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1045, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one initialized key coefficient in the first attention head will reduce the model from 1,045 to 1,044 parameters while retaining at least 99% accuracy, because reciprocal scaling of that head’s query map and key map leaves every attention score unchanged.
change: Store the largest initialized first-head key coefficient as a buffer, learn all remaining key coefficients, and reconstruct the complete key map during attention.
mechanism: Single-head reciprocal query–key scale gauge fixing
evidence_used: Fixing the analogous common value-map scaling redundancy while retaining independent head gains achieved 99.95% accuracy at 1,045 parameters; this tests another exact continuous attention gauge one coordinate at a time.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1044, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one initialized key coefficient in the second attention head will reduce the verified model from 1,044 to 1,043 parameters while retaining at least 99% accuracy, because each head has an independent reciprocal query–key scaling symmetry and fixing the first head’s analogous coefficient achieved 99.99% accuracy.
change: Fix the largest initialized key coefficient in each attention head as a buffer, learn all remaining key coefficients, and reconstruct the complete key map during attention.
mechanism: Second-head reciprocal query–key scale gauge fixing
evidence_used: The current 1,044-parameter design fixed one first-head key coefficient and achieved 99.99% accuracy, directly validating this exact gauge removal; applying the same independent symmetry to the second head is the smallest evidence-backed next change.
result: the trained implementation did not satisfy the learned-model requirement
reported_values: {"training_steps": 4999.0}

RECENT RESULT
hypothesis: Sharing the remaining twenty-third-farthest attention bias across both heads will reduce the verified model from 1,044 to 1,043 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance, add one learned scalar shared across heads for that distance, and insert it immediately before the existing twenty-second-farthest bias.
mechanism: Cross-head twenty-third-farthest relative-bias tying
evidence_used: Nine consecutive adjacent cross-head distance ties through the twenty-second-farthest distance retained at least 99% accuracy, most recently reaching 99.91%; earlier attempts at this next tie provided no adverse accuracy evidence because they reproduced the unchanged implementation or were unverifiable.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Fixing one initialized query coefficient in the second attention head will reduce the model from 1,044 to 1,043 parameters while retaining at least 99% accuracy, because reciprocal query–key scaling is an exact independent symmetry for each head.
change: Store the largest initialized second-head query coefficient as a buffer, learn all remaining query coefficients, and reconstruct the complete query map during attention while preserving the existing first-head key gauge fix.
mechanism: Complementary second-head query–key gauge fixing
evidence_used: Fixing the first head’s analogous key-scale gauge achieved 99.99% accuracy at 1,044 parameters. The prior second-head key attempt failed the learned-model requirement rather than accuracy, so applying the independent symmetry through the complementary query map tests whether that failure was implementation-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1043, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one query coefficient through a score-preserving joint rotation of the first head’s query and key feature spaces will reduce the model from 1,043 to 1,042 parameters while retaining at least 99% accuracy.
change: Apply an initialized orthogonal rotation that zeros one first-head query coefficient, omit that fixed zero from the packed trainable QKV parameter, and reconstruct it during attention.
mechanism: First-head orthogonal query–key gauge fixing
evidence_used: Independent reciprocal query–key gauge fixes reduced the model to 1,043 parameters with 99.98% accuracy; this uses another exact query–key symmetry while preserving both head routes and every attention score.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1042, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the same score-preserving query–key rotation to the second head will reduce the verified model from 1,042 to 1,041 parameters while retaining at least 99% accuracy.
change: Rotate two query/key features in the second head to zero one query coefficient, omit that fixed zero from the packed QKV parameter, and reconstruct it during attention.
mechanism: Second-head orthogonal query–key gauge fixing
evidence_used: The identical first-head orthogonal gauge removal reduced the model to 1,042 parameters with 99.93% accuracy; the second head has the same independent dot-product-preserving symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1041, "training_steps": 4999}

RECENT RESULT
hypothesis: Eliminating a second first-head query coefficient through a score-preserving query–key rotation will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy.
change: Rotate the second and third features of the first attention head to zero one additional query coefficient, exclude that fixed zero from the packed QKV parameter, and reconstruct it during attention.
mechanism: Additional first-head orthogonal query–key gauge fixing
evidence_used: The first- and second-head orthogonal gauge removals preserved all attention scores and achieved 99.93% at 1,042 parameters and 99.90% at 1,041 parameters respectively, directly supporting another independent orthogonal gauge removal.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing content-dependent query–key scores with two learned causal relative-distance kernels will reduce the model from 1,041 to 925 parameters while retaining at least 99% accuracy, because fixed-format addition requires distinct positional routes but not necessarily token-dependent routing.
change: Remove all learned query and key maps and their bias after initialization; retain learned per-head relative biases, the shared learned value map, and distinct independently scaled residual slots.
mechanism: Learned relative-kernel causal attention
evidence_used: Disjoint head-slot routing achieved 99.96%, while merging routes collapsed accuracy to 0.01%, identifying route identity as load-bearing. The shared value map succeeded, and extensive cross-head relative-bias sharing also retained accuracy, making learned relative-distance kernels a plausible compact source of the two required routes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest attention bias across both heads will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy.
change: Shorten the head-specific relative-bias core by one distance and insert one learned scalar shared across both heads immediately before the existing twenty-second-farthest bias.
mechanism: Cross-head twenty-third-farthest relative-bias tying
evidence_used: Nine consecutive adjacent cross-head distance ties through the twenty-second-farthest distance retained at least 99% accuracy; the current 1,041-parameter design achieved 99.90%, while previous attempts at this next tie were unverifiable rather than accuracy failures.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Eliminating another second-head query coefficient through a score-preserving orthogonal query–key rotation will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy.
change: Rotate the second and third features of the second attention head, omit the resulting fixed-zero query coefficient from the packed QKV parameter, and reconstruct it during attention.
mechanism: Residual second-head orthogonal query–key gauge fixing
evidence_used: The first two orthogonal gauge removals preserved attention scores and achieved 99.93% and 99.90% accuracy. The attempted additional first-head removal timed out without adverse accuracy evidence, so the symmetric second-head test is the most direct implementation-specific follow-up.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing one second-head key coefficient in a feature distinct from the existing second-head query pivot will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy, because reciprocal scaling of that feature’s query row and key row preserves every attention score.
change: Select the largest initialized eligible second-head key coefficient as a fixed buffer, exclude it from the packed QKV parameter, and reconstruct both key pivots during attention.
mechanism: Independent feature-wise query–key scale gauge fixing
evidence_used: Reciprocal query–key gauge fixes reached 1,043 parameters with 99.98% accuracy, and subsequent score-preserving orthogonal fixes reached 1,041 parameters with 99.90%. This uses another exact query–key symmetry while avoiding the additional rotation logic associated with the recent timeouts.
result: training did not finish within the verification time limit



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
