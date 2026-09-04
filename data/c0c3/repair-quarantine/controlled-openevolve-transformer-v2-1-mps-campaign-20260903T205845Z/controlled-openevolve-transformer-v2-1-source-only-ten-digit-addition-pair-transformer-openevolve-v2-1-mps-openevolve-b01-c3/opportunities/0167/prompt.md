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
verified_results: {"accuracy": 0.9987999999999999, "parameters": 748, "training_steps": 4999}
prior_hypothesis: Restricting the verified 861-parameter design’s tied token/logit embedding from rank six to rank five will retain at least 99% accuracy while reducing the model to 748 parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9989, "parameters": 747, "training_steps": 4999}
prior_hypothesis: Combining the verified rank-five embedding with the independently verified `[a,b,b,c,c,d,d]` gain pattern will achieve at least 99% accuracy with 747 parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9993000000000001, "parameters": 973, "training_steps": 4999}
prior_hypothesis: The resulting 973-parameter transformer will maintain at least 99% accuracy because the current 974-parameter model achieved 99.97%, while the same `[a,b,b,c,c,d,d]` gain pattern achieved 99.88% in the larger-MLP design.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.998, "parameters": 746, "training_steps": 4999}
prior_hypothesis: The verified 747-parameter design will retain at least 99% accuracy when two adjacent final LayerNorm scales are tied, reducing the model to 746 parameters without further constraining attention.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Restricting the verified 861-parameter design’s tied token/logit embedding from rank six to rank five will retain at least 99% accuracy while reducing the model to 748 parameters.
change: Adopt the verified six-unit MLP and paired secondary-head gain, then remove one additional learned vocabulary coordinate from both input representation and output classification.
mechanism: Five-coordinate tied vocabulary bottleneck
evidence_used: Reference Design 1 achieved 99.99% accuracy with 861 parameters using rank-six embeddings, a six-unit MLP, and `[a,b,c,d,d,e,e]` secondary gains; this isolates the new test to whether its token representation has one redundant rank.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 748, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the current six-unit MLP to five units will retain at least 99% accuracy while lowering the learned parameter count from 748 to 734.
change: Change the default feed-forward width from six to five, isolating a 14-parameter reduction without altering attention, embeddings, training, or decoding.
mechanism: Five-unit nonlinear bottleneck
evidence_used: The current 748-parameter rank-five model achieved 99.88% accuracy with a six-unit MLP, while the references show repeated MLP-width reductions retained at least 99%; this makes the next one-unit width reduction the most direct test of remaining nonlinear-capacity redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.055999999999999994, "parameters": 734, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified rank-five embedding with the independently verified `[a,b,b,c,c,d,d]` gain pattern will achieve at least 99% accuracy with 747 parameters.
change: Reduce the tied embedding from rank six to rank five and tie one additional secondary-head gain while retaining the six-unit MLP.
mechanism: Rank-five vocabulary bottleneck with four-way secondary-head gain tying
evidence_used: The 748-parameter rank-five design achieved 99.88% accuracy, whereas reducing its MLP width failed; Reference Design 2’s four-parameter secondary-gain pattern achieved 99.93%, motivating this orthogonal one-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 747, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the successful secondary-head gain compression from `[a,b,b,c,c,d,d]` to `[a,b,b,c,c,c,c]` will reduce the verified 747-parameter rank-five model to 746 parameters while retaining at least 99% accuracy.
change: Adopt the verified rank-five tied embedding and replace the secondary head’s four learned value gains with three gains expanded in groups of one, two, and four coordinates.
mechanism: Rank-five vocabulary bottleneck with three-way dyadic secondary-head gain tying
evidence_used: Reference Design 2 achieved 99.89% accuracy with 747 parameters, and its one-parameter gain reduction slightly exceeded the 748-parameter Reference Design 1; this motivates testing one further isolated gain tie while preserving the six-unit MLP whose reduction failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9067000000000001, "parameters": 746, "training_steps": 4999}

RECENT RESULT
hypothesis: The verified 747-parameter design will retain at least 99% accuracy when two adjacent final LayerNorm scales are tied, reducing the model to 746 parameters without further constraining attention.
change: Adopt the verified rank-five embedding and four-way secondary-head gain pattern, then replace seven independent final-normalization scales with six parameters expanded as `[a,b,c,d,e,f,f,1]`.
mechanism: Rank-five vocabulary bottleneck with paired final-normalization scales
evidence_used: Reference Design 2 achieved 99.89% accuracy with 747 parameters; the failed 746-parameter experiment further compressed attention gains and fell to 90.67%, motivating an orthogonal one-parameter reduction in final normalization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 746, "training_steps": 4999}

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
