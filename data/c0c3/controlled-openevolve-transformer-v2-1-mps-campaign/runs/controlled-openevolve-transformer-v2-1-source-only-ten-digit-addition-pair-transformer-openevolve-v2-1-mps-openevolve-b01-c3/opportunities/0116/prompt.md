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
verified_results: {"accuracy": 0.9979, "parameters": 1080, "training_steps": 4999}
prior_hypothesis: A 1,080-parameter transformer will maintain at least 99% accuracy because the value map receives non-affine LayerNorm outputs, whose all-ones input component is always zero, so removing that unidentifiable input direction preserves its effective learned function class.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1058, "training_steps": 4999}
prior_hypothesis: The 1,058-parameter transformer will maintain at least 99% accuracy because the verified 1,059-parameter design achieved 99.79%, and sharing its scalar MLP output bias with the attention triplet scalar preserves both learned biases and the load-bearing triplet/quartet attention boundary while adding only one cross-layer equality constraint.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9979, "parameters": 1059, "training_steps": 4999}
prior_hypothesis: A 1,059-parameter transformer will maintain at least 99% accuracy because the verified 1,060-parameter `[a,b,b,c,c,c,c]` attention-bias design achieved 99.95%, and merging its first singleton with the adjacent pair preserves the demonstrated quartet boundary.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995, "parameters": 1060, "training_steps": 4999}
prior_hypothesis: A 1,060-parameter transformer will maintain at least 99% accuracy because the verified 1,061-parameter model achieved 99.98%; unlike the failed quintet tie, this preserves the successful final quartet and its boundary while tying only two remaining singleton bias coordinates.

## Recent verification evidence

RECENT RESULT
hypothesis: The resulting 1,066-parameter transformer will maintain at least 99% accuracy because the verified 1,067-parameter model reached 99.82%, and two successive MLP output-bias pair ties already preserved 99%+ accuracy.
change: Adopt the verified 1,067-parameter mean-zero attention and MLP factorization, then remove one additional parameter by tying a third pair of MLP output-bias coordinates.
mechanism: Triple-pair-tied MLP output bias
evidence_used: The 1,067-parameter mean-zero MLP-input design achieved 99.82%; independently, the second MLP output-bias pair tie produced a 1,079-parameter model with 99.95%, motivating one more isolated tie while preserving all learned matrices, normalization, and attention routing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1066, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 1,065-parameter transformer will maintain at least 99% accuracy because the verified 1,066-parameter model achieved 99.95%, and this adds only one equality constraint to its three-times-pair-tied MLP output bias while preserving all learned matrices, attention routing, normalization, and residual paths.
change: Replace the seven-coordinate MLP output bias pattern `[a,b,b,c,c,d,d]` with `[a,a,a,b,b,c,c]`, reducing the learned bias from four scalars to three.
mechanism: Triplet-and-pair-tied MLP output bias
evidence_used: The current 1,066-parameter design reached 99.95% after a third successive MLP output-bias tie; the next isolated reduction extends that demonstrated bias-sharing mechanism by one parameter without altering any weight matrix or previously sensitive component.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1065, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,064-parameter transformer will maintain at least 99% accuracy because the verified 1,065-parameter model reached 99.96%, and this adds only one equality constraint to its three-scalar MLP output bias while preserving all learned matrices, attention routing, normalization, and residual paths.
change: Adopt the verified triplet-and-pair bias sharing, then tie its two bias-pair scalars to form the seven-coordinate pattern `[a,a,a,b,b,b,b]`, reducing the model by one parameter.
mechanism: Triplet-and-quartet-tied MLP output bias
evidence_used: The 1,065-parameter design achieved 99.96% after four successive MLP output-bias sharing reductions retained at least 99% accuracy, motivating one more isolated bias tie without altering previously sensitive components.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1064, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,063-parameter transformer will maintain at least 99% accuracy because the verified 1,064-parameter model achieved 100%, and this adds only one equality constraint to its triplet-and-quartet MLP output bias while preserving all learned matrices, attention routing, normalization, and residual paths.
change: Adopt the verified mean-zero MLP input factorization and replace the two-scalar triplet-and-quartet output bias with one scalar shared across all seven basis coordinates.
mechanism: Scalar-shared MLP output bias
evidence_used: The 1,064-parameter triplet-and-quartet bias design achieved 100% after successive MLP output-bias sharing reductions all retained at least 99% accuracy, motivating one final isolated bias-sharing reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1063, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 1,062-parameter transformer will maintain at least 99% accuracy because the verified 1,063-parameter model achieved 99.64%, and this removes only its final scalar-shared MLP output bias while preserving every learned matrix, attention route, normalization, and residual path.
change: Restrict the MLP input projection to the verified seven-dimensional mean-zero basis and make its mean-zero output projection bias-free.
mechanism: Bias-free mean-zero MLP projections
evidence_used: The 1,063-parameter scalar-shared-bias design reached 99.64%; successive reductions of the same MLP output bias retained at least 99% accuracy, making removal of its final scalar the most isolated next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9871, "parameters": 1062, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 1,062-parameter transformer will maintain at least 99% accuracy because it preserves the scalar MLP output bias whose removal reduced accuracy to 98.71%, while adding only one equality constraint to the already pair-tied attention output bias of the 99.64%-accurate 1,063-parameter model.
change: Reduce the learned attention output bias from six scalars to five by changing its seven-coordinate expansion from a final pair to a final triplet.
mechanism: Triplet-tied attention output bias
evidence_used: The current 1,063-parameter model achieved 99.64%, whereas deleting its final MLP output-bias scalar failed at 98.71%; successive coordinate sharing elsewhere preserved accuracy, so extending the existing attention-bias pair tie is the most isolated alternative one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1062, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,056-parameter transformer will achieve at least 99% accuracy because positional heads can specialize through their learned attention distributions while sharing one feature representation of digit content, with a single learned scalar retaining relative head weighting.
change: Adopt the verified 1,062-parameter scalar MLP bias and triplet-tied attention bias, then replace the second head’s seven-coordinate value adapter with one learned head-level gain. This challenges the prior assumption that each routing head needs feature-wise value specialization.
mechanism: Scalar-mixed shared value codebook
evidence_used: Reference Design 1 achieved 99.94% with 1,062 parameters, and the 1,087-parameter shared-value design achieved 99.98%; since addition is operand-symmetric and the heads retain independent learned relative-position biases, routing—not coordinate-wise value scaling—is plausibly the load-bearing specialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9711, "parameters": 1056, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,061-parameter transformer will maintain at least 99% accuracy because the verified 1,062-parameter model achieved 99.94%, and this adds only one equality constraint to its triplet-tied attention output bias while preserving the load-bearing scalar MLP bias, feature-wise head adapter, learned matrices, and positional routing.
change: Adopt the verified scalar-shared MLP output bias and reduce the attention output bias from six learned scalars to four by expanding its final scalar across four of the seven mean-zero basis coordinates.
mechanism: Quartet-tied attention output bias
evidence_used: The 1,062-parameter design retained 99.94% after extending the attention output-bias pair to a triplet, whereas removing the scalar MLP bias produced only 98.71% and replacing the feature-wise head adapter with a scalar produced 97.11%; therefore one further attention-bias tie is the most isolated supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1061, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,060-parameter transformer will maintain at least 99% accuracy because the verified 1,061-parameter model achieved 99.98%, and this adds only one equality constraint to its quartet-tied attention output bias while preserving the load-bearing scalar MLP bias, feature-wise head adapter, learned matrices, and positional routing.
change: Adopt the verified mean-zero MLP input projection and scalar MLP output bias, then reduce the learned attention output bias from four scalars to three by expanding its final scalar across five of the seven mean-zero coordinates.
mechanism: Quintet-tied attention output bias
evidence_used: The 1,062-parameter triplet-tied attention-bias model reached 99.94%, and extending that tie to a quartet produced 99.98% at 1,061 parameters; in contrast, removing the scalar MLP bias or collapsing the feature-wise head adapter failed, making one further attention-bias tie the most supported isolated reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9797, "parameters": 1060, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,060-parameter transformer will maintain at least 99% accuracy because the verified 1,061-parameter model achieved 99.98%; unlike the failed quintet tie, this preserves the successful final quartet and its boundary while tying only two remaining singleton bias coordinates.
change: Reduce the attention output bias from four scalars to three using the pattern `[a,b,b,c,c,c,c]`, preserving the first coordinate and the verified final quartet.
mechanism: Pair-tied singleton attention output bias
evidence_used: Extending the final quartet to a quintet reduced accuracy to 97.97%, indicating that the coordinate adjoining the quartet needs a distinct value from it; the 1,061-parameter quartet design achieved 99.98%, motivating an alternative isolated tie among its non-quartet coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1060, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,059-parameter transformer will maintain at least 99% accuracy because the verified 1,060-parameter `[a,b,b,c,c,c,c]` attention-bias design achieved 99.95%, and merging its first singleton with the adjacent pair preserves the demonstrated quartet boundary.
change: Adopt the verified final quartet and tie the three remaining attention output-bias coordinates, producing `[a,a,a,b,b,b,b]` and reducing the current model by three parameters.
mechanism: Triplet-and-quartet-tied attention output bias
evidence_used: The 1,060-parameter pair-tied-singleton design reached 99.95%, while the failed quintet design showed that the coordinate adjoining the final quartet must remain distinct; this patch preserves that boundary and applies the triplet-and-quartet sharing pattern that previously achieved 100% for the MLP bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1059, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,058-parameter transformer will maintain at least 99% accuracy because the verified 1,059-parameter design achieved 99.79%, and sharing its scalar MLP output bias with the attention triplet scalar preserves both learned biases and the load-bearing triplet/quartet attention boundary while adding only one cross-layer equality constraint.
change: Adopt the verified scalar MLP bias and triplet/quartet attention bias, then reuse the attention triplet parameter as the MLP scalar bias, removing one deduplicated learned parameter.
mechanism: Cross-layer scalar bias sharing
evidence_used: The 1,059-parameter triplet/quartet attention design reached 99.79%; the failed quintet showed that the attention group boundary matters, while the 1,063-parameter scalar MLP bias independently reached 99.64%, motivating parameter sharing without deleting either mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1058, "training_steps": 4999}



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
