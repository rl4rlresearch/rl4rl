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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 861, "training_steps": 4999}
prior_hypothesis: The resulting 861-parameter transformer will maintain at least 99% accuracy because the current rank-six, final-pair-tied model achieved 99.99%, while the same `[a,b,c,d,d,e,e]` gain pattern achieved 99.97% with the six-unit MLP before vocabulary compression.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1004, "training_steps": 4999}
prior_hypothesis: A 1,004-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions to eleven, ten, and nine units achieved 100%, 99.89%, and 99.96%, respectively; reducing the verified nine-unit design by one more unit tests the same isolated 14-parameter reduction.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9993000000000001, "parameters": 973, "training_steps": 4999}
prior_hypothesis: The resulting 973-parameter transformer will maintain at least 99% accuracy because the current 974-parameter model achieved 99.97%, while the same `[a,b,b,c,c,d,d]` gain pattern achieved 99.88% in the larger-MLP design.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 976, "training_steps": 4999}
prior_hypothesis: A 976-parameter transformer will maintain at least 99% accuracy because five successive MLP-width reductions through seven units retained at least 99%, and the verified seven-unit 990-parameter model achieved 100%; reducing the width by one tests the same isolated 14-parameter reduction.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,018-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions from twelve to eleven and eleven to ten units retained 100% and 99.89% accuracy; reducing the verified 1,032-parameter design by one more unit tests the same isolated 14-parameter reduction.
change: Adopt the verified 1,032-parameter architecture and reduce its MLP hidden width from ten to nine units, preserving its feature-wise secondary-head gains, three-group attention bias, independent scalar MLP bias, routing, embeddings, and residual width.
mechanism: Nine-unit nonlinear carry bottleneck
evidence_used: The eleven-unit model achieved 100% at 1,046 parameters, and the otherwise corresponding ten-unit model achieved 99.89% at 1,032 parameters, showing two successful fourteen-parameter bottleneck reductions with substantial remaining accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1018, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,004-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions to eleven, ten, and nine units achieved 100%, 99.89%, and 99.96%, respectively; reducing the verified nine-unit design by one more unit tests the same isolated 14-parameter reduction.
change: Adopt the verified 1,018-parameter architecture and reduce its MLP hidden width from nine to eight units, preserving its full secondary-head gains, three-group attention bias, independent scalar MLP bias, routing, embeddings, and residual width.
mechanism: Eight-unit nonlinear carry bottleneck
evidence_used: The otherwise corresponding eleven-, ten-, and nine-unit models all exceeded 99% accuracy, with the nine-unit 1,018-parameter model reaching 99.96%; this repeated success makes another identical width reduction the clearest informative test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1004, "training_steps": 4999}

RECENT RESULT
hypothesis: A 990-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions to eleven, ten, nine, and eight units all exceeded 99%, with the verified eight-unit 1,004-parameter model reaching 99.99%; reducing it by one further unit tests the same isolated 14-parameter reduction.
change: Adopt the verified 1,004-parameter architecture and reduce its MLP hidden width from eight to seven units, preserving full secondary-head gains, the three-group attention bias, independent scalar MLP bias, attention routing, embeddings, and residual width.
mechanism: Seven-unit nonlinear carry bottleneck
evidence_used: The otherwise corresponding eleven-, ten-, nine-, and eight-unit models achieved 100%, 99.89%, 99.96%, and 99.99%; the eight-unit result has the strongest observed margin and directly motivates another identical bottleneck reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 990, "training_steps": 4999}

RECENT RESULT
hypothesis: A 976-parameter transformer will maintain at least 99% accuracy because five successive MLP-width reductions through seven units retained at least 99%, and the verified seven-unit 990-parameter model achieved 100%; reducing the width by one tests the same isolated 14-parameter reduction.
change: Reduce the MLP hidden width from seven to six units while preserving the verified attention, embeddings, projections, biases, residual width, training procedure, and decoding interface.
mechanism: Six-unit nonlinear carry bottleneck
evidence_used: The otherwise corresponding eight-unit and seven-unit models achieved 99.99% and 100% accuracy at 1,004 and 990 parameters, respectively, directly motivating one further bottleneck-width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 976, "training_steps": 4999}

RECENT RESULT
hypothesis: A 962-parameter transformer will maintain at least 99% accuracy because the otherwise identical six-unit, 976-parameter model achieved 100%; reducing the MLP by one further unit tests the same repeatedly successful 14-parameter compression.
change: Adopt the verified six-unit architecture and reduce its learned MLP hidden width to five units while preserving all attention, embedding, projection, normalization, training, checkpointing, and decoding behavior.
mechanism: Five-unit nonlinear carry bottleneck
evidence_used: Successive MLP-width reductions through six units all retained at least 99% accuracy, and the six-unit reference achieved 100%, providing the strongest direct evidence for testing one additional width reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9648, "parameters": 962, "training_steps": 4999}

RECENT RESULT
hypothesis: A 975-parameter transformer will maintain at least 99% accuracy because the six-unit, 976-parameter design achieved 100%, while the same isolated final-pair gain tie previously retained 99.88%; preserving six MLP units avoids the five-unit bottleneck that fell to 96.48%.
change: Adopt the verified six-unit MLP and parameterize the seven secondary-head gains as `[a,b,c,d,e,f,f]`, removing one additional learned parameter.
mechanism: Six-unit MLP with final-pair secondary-head gain sharing
evidence_used: The six-unit reference reached 100% at 976 parameters, whereas five units failed at 96.48%; independently, tying only the final secondary-head gain pair met the requirement at 99.88%, motivating that compression while retaining the proven MLP width.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 975, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 974-parameter transformer will maintain at least 99% accuracy because the verified six-unit, final-pair-tied 975-parameter model achieved 99.98%, while extending gain sharing from `[a,b,c,d,e,f,f]` to `[a,b,c,d,d,e,e]` independently achieved 100% in the larger-MLP design.
change: Reduce the MLP width from ten to six units and represent the seven secondary-head gains with five learned values expanded as `[a,b,c,d,d,e,e]`.
mechanism: Six-unit MLP with two-pair secondary-head gain sharing
evidence_used: The six-unit MLP is the narrowest verified bottleneck at 100%, and its 975-parameter final-pair-tied variant reached 99.98%; the independently verified two-pair gain pattern reached 100%, motivating one additional isolated gain tie without crossing the failed five-unit bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 974, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 973-parameter transformer will maintain at least 99% accuracy because the current 974-parameter model achieved 99.97%, while the same `[a,b,b,c,c,d,d]` gain pattern achieved 99.88% in the larger-MLP design.
change: Replace the five learned secondary-head gains expanded as `[a,b,c,d,d,e,e]` with four gains expanded as `[a,b,b,c,c,d,d]`, removing one learned parameter while retaining the verified six-unit MLP.
mechanism: Six-unit MLP with three-pair secondary-head gain sharing
evidence_used: The current six-unit, two-pair-gain model reached 99.97%, and the independently verified three-pair pattern reached 99.88%; this supports testing one isolated additional adjacent gain tie without crossing the failed five-unit MLP bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 973, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 864-parameter transformer will maintain at least 99% accuracy because the six-unit MLP reached 100% while the 973-parameter gain-tied model reached 99.93%; retaining that nonlinear width but constraining the previously untested token/logit codebook to six centered features should preserve the learned computation.
change: Replace the full seven-coordinate centered vocabulary embedding with a learned six-coordinate embedding projected through a fixed orthonormal mean-zero basis, removing 111 parameters while preserving the eight-dimensional residual stream, six-unit MLP, causal attention, and tied learned logits.
mechanism: Rank-six tied vocabulary bottleneck
evidence_used: The five-unit MLP failed at 96.48%, identifying nonlinear width as load-bearing, whereas six units repeatedly exceeded 99%. The old designs implicitly assumed all seven learned vocabulary-feature coordinates were necessary; this patch instead tests a lower-rank learned input/output representation without revisiting the failed MLP bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 862, "training_steps": 4999}

RECENT RESULT
hypothesis: A 749-parameter transformer will maintain at least 99% accuracy because the verified rank-six, 862-parameter design achieved 99.99%; reducing only its learned vocabulary feature rank by one tests whether that high-margin representation retains another 113-parameter compression.
change: Adopt the verified final-pair gain sharing and reduce the centered tied vocabulary embedding from six to five learned feature coordinates, preserving the six-unit MLP, eight-dimensional residual stream, causal attention, training procedure, and protected decoding interface.
mechanism: Rank-five tied vocabulary bottleneck
evidence_used: The rank-six vocabulary bottleneck reached 99.99% at 862 parameters, while the five-unit MLP failed at 96.48%; this motivates compressing the successful vocabulary codebook rather than the demonstrated load-bearing nonlinear width.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.985, "parameters": 749, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 860-parameter transformer will maintain at least 99% accuracy because the rank-six 862-parameter design achieved 99.99%, while the `[a,b,b,c,c,d,d]` gain pattern independently achieved 99.93% with the same six-unit MLP.
change: Adopt the verified rank-six tied vocabulary embedding and six-unit MLP, then represent seven secondary-head gains with four learned values expanded as `[a,b,b,c,c,d,d]`.
mechanism: Rank-six vocabulary bottleneck with three-pair secondary-head gain sharing
evidence_used: Rank five failed at 98.50%, so this preserves the successful rank-six embedding; the rank-six final-pair-tied model reached 99.99%, and three-pair gain sharing already exceeded the accuracy threshold in the 973-parameter design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7854000000000001, "parameters": 860, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 861-parameter transformer will maintain at least 99% accuracy because the current rank-six, final-pair-tied model achieved 99.99%, while the same `[a,b,c,d,d,e,e]` gain pattern achieved 99.97% with the six-unit MLP before vocabulary compression.
change: Replace six learned secondary-head gains expanded as `[a,b,c,d,e,f,f]` with five gains expanded as `[a,b,c,d,d,e,e]`, removing one parameter while preserving the successful rank-six embedding and six-unit MLP.
mechanism: Rank-six vocabulary bottleneck with two-pair secondary-head gain sharing
evidence_used: The 862-parameter rank-six model reached 99.99%; although jumping directly to four gains failed at 78.54%, the five-gain pattern independently retained 99.97%, making this isolated intermediate tie the most informative next compression.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 861, "training_steps": 4999}



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
