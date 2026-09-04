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
verified_results: {"accuracy": 0.9995, "parameters": 1060, "training_steps": 4999}
prior_hypothesis: A 1,060-parameter transformer will maintain at least 99% accuracy because the verified 1,061-parameter model achieved 99.98%; unlike the failed quintet tie, this preserves the successful final quartet and its boundary while tying only two remaining singleton bias coordinates.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1055, "training_steps": 4999}
prior_hypothesis: A 1,055-parameter transformer will maintain at least 99% accuracy because the verified 1,056-parameter `[a,b,c,d,d,e,e]` gain design achieved 100%, and tying the next adjacent pair retains four independently learned feature gains.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9923000000000001, "parameters": 1054, "training_steps": 4999}
prior_hypothesis: A 1,054-parameter transformer will maintain at least 99% accuracy because the verified 1,055-parameter `[a,b,b,c,c,d,d]` gain design achieved 99.88%, and tying its remaining singleton into the adjacent pair retains three independently learned feature gains.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1056, "training_steps": 4999}
prior_hypothesis: A 1,056-parameter transformer will maintain at least 99% accuracy because the verified 1,057-parameter model achieved 99.88%, and tying one additional adjacent pair preserves five independently learned secondary-head gains rather than collapsing all feature specialization.

## Recent verification evidence

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

RECENT RESULT
hypothesis: The resulting 1,057-parameter transformer will maintain at least 99% accuracy because the verified 1,058-parameter model achieved 99.96%, and tying only the final pair of its seven secondary-head feature gains preserves six learned gain values instead of the single gain that failed at 97.11%.
change: Adopt the verified 1,058-parameter mean-zero projections, triplet/quartet attention bias, and cross-layer scalar bias sharing, then remove one parameter by tying the final two coordinates of the secondary-head value adapter.
mechanism: Pair-tied secondary-head value gain
evidence_used: The 1,058-parameter reference reached 99.96%, while collapsing all seven secondary-head gains to one scalar reached only 97.11%; an isolated pair tie directly tests whether most feature-wise specialization can be retained with one fewer parameter.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1057, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,056-parameter transformer will maintain at least 99% accuracy because the verified 1,057-parameter model achieved 99.88%, and tying one additional adjacent pair preserves five independently learned secondary-head gains rather than collapsing all feature specialization.
change: Preserve the verified final gain pair and tie the two preceding secondary-head gain coordinates, changing the expansion from `[a,b,c,d,e,f,f]` to `[a,b,c,d,d,e,e]`.
mechanism: Two-pair secondary-head value gain
evidence_used: The current isolated final-pair tie retained 99.88%, while collapsing all seven gains to one scalar reached only 97.11%; additionally, the attention-bias experiments showed that preserving a successful group boundary while tying other coordinates can succeed when extending that group fails.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1056, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,055-parameter transformer will maintain at least 99% accuracy because the verified 1,056-parameter `[a,b,c,d,d,e,e]` gain design achieved 100%, and tying the next adjacent pair retains four independently learned feature gains.
change: Change the secondary-head value-gain expansion to `[a,b,b,c,c,d,d]`, removing one learned parameter while preserving all attention routing, biases, projections, and residual paths.
mechanism: Three-pair secondary-head value-gain sharing
evidence_used: The 1,056-parameter two-pair gain design achieved 100%, while collapsing all seven gains to one scalar failed at 97.11%; this motivates another isolated adjacent-pair tie rather than broad gain sharing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1055, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,054-parameter transformer will maintain at least 99% accuracy because the verified 1,055-parameter `[a,b,b,c,c,d,d]` gain design achieved 99.88%, and tying its remaining singleton into the adjacent pair retains three independently learned feature gains.
change: Adopt the verified cross-layer scalar-bias sharing and compress the secondary-head gain expansion to `[a,a,a,b,b,c,c]`, removing five deduplicated parameters from the current design.
mechanism: Triplet-and-two-pair secondary-head value-gain sharing
evidence_used: The 1,055-parameter three-pair gain design reached 99.88%, while collapsing all seven gains to one scalar reached only 97.11%; the successful sequence of isolated adjacent ties motivates one further tie while preserving three feature groups.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9923000000000001, "parameters": 1054, "training_steps": 4999}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
