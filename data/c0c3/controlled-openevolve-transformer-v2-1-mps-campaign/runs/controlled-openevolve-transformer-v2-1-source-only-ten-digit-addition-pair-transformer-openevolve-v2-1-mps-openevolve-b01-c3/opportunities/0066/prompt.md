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
verified_results: {"accuracy": 1.0, "parameters": 1277, "training_steps": 4999}
prior_hypothesis: A 1,277-parameter model will retain at least 99% accuracy because the verified 1,278-parameter model achieved 100%, while sharing the ninth-farthest bias affects only nine query-key pairs at full context and preserves head-specific biases at every more frequent distance.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1279, "training_steps": 4999}
prior_hypothesis: A 1,279-parameter model will retain at least 99% accuracy because the verified 1,280-parameter model achieved 100%, while sharing the seventh-farthest bias affects only seven query-key pairs at full context and preserves head-specific biases at every more frequent distance.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1278, "training_steps": 4999}
prior_hypothesis: A 1,278-parameter model will retain at least 99% accuracy because the verified 1,279-parameter model achieved 99.99%, while additionally sharing the eighth-farthest bias affects only eight query-key pairs at full context and preserves head-specific biases at every more frequent distance.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1275, "training_steps": 4999}
prior_hypothesis: A 1,275-parameter model will retain at least 99% accuracy because the verified 1,276-parameter model achieved 99.96%, while sharing the eleventh-farthest bias affects only eleven query-key pairs at full context and preserves head-specific biases at every more frequent distance.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,283-parameter model will retain at least 99% accuracy because the verified 1,284-parameter model achieved 99.94%, while sharing the third-farthest bias affects only three query-key pairs at full context and leaves all more frequently used routing bins head-specific.
change: Share the three maximum-distance attention biases across heads while retaining independent head-specific biases at every shorter causal distance.
mechanism: Cross-head sharing of the three sparsest relative-distance biases
evidence_used: The 1,284-parameter design successfully shared the two sparsest relative-distance biases with 99.94% accuracy; extending the same mechanism to the next-sparsest bin is the smallest incremental and most directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1283, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,282-parameter model will retain at least 99% accuracy because the verified 1,283-parameter model achieved 100%, while additionally sharing the fourth-farthest bias preserves an adaptive value and affects only four query-key pairs at full context.
change: Anchor distance-zero biases at zero and share the four maximum-distance attention biases across heads, while retaining independent head-specific biases at every shorter causal distance.
mechanism: Cross-head sharing of the four sparsest relative-distance biases
evidence_used: The 1,283-parameter design achieved 100% accuracy after successfully extending cross-head sharing from the two sparsest distance bins to three; sharing the next-sparsest bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1282, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,281-parameter model will retain at least 99% accuracy because the verified 1,282-parameter model achieved 99.94%, while additionally sharing the fifth-farthest bias affects only five query-key pairs at full context and preserves independent head-specific biases at every more frequent distance.
change: Replace absolute position embeddings with gauge-anchored relative-attention biases, sharing the five maximum-distance biases across heads while retaining the current learned projections, MLP bias quartets, and attention-output bias tie.
mechanism: Cross-head sharing of the five sparsest relative-distance biases
evidence_used: The 1,282-parameter design met the requirement after progressively extending successful cross-head sharing from one through four sparsest distance bins; sharing the next-sparsest bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1281, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,280-parameter model will retain at least 99% accuracy because the verified 1,281-parameter model achieved 99.98%, while sharing the sixth-farthest bias affects only six query-key pairs at full context and preserves head-specific biases at every more frequent distance.
change: Extend the current cross-head sharing from the five to the six maximum-distance attention biases, reducing the model by one learned parameter.
mechanism: Cross-head sharing of the six sparsest relative-distance biases
evidence_used: Progressive sharing of the one through five sparsest distance bins produced verified 1,285- through 1,281-parameter models that all exceeded 99% accuracy; the current five-bin design reached 99.98%, directly motivating the next single-bin reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1280, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,279-parameter model will retain at least 99% accuracy because the verified 1,280-parameter model achieved 100%, while sharing the seventh-farthest bias affects only seven query-key pairs at full context and preserves head-specific biases at every more frequent distance.
change: Extend cross-head sharing from the six to the seven maximum-distance attention biases, reducing the model by one learned parameter beyond the verified 1,280-parameter design.
mechanism: Cross-head sharing of the seven sparsest relative-distance biases
evidence_used: Progressive sharing of one through six sparsest distance bins consistently exceeded 99% accuracy, and the six-bin 1,280-parameter design achieved 100%; extending the same mechanism by one bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1279, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,278-parameter model will retain at least 99% accuracy because the verified 1,279-parameter model achieved 99.99%, while additionally sharing the eighth-farthest bias affects only eight query-key pairs at full context and preserves head-specific biases at every more frequent distance.
change: Extend cross-head sharing from the three to the eight maximum-distance attention biases, reducing the current model by five learned parameters and the best verified design by one.
mechanism: Cross-head sharing of the eight sparsest relative-distance biases
evidence_used: Progressive sharing of one through seven sparsest distance bins consistently exceeded 99% accuracy, with the seven-bin 1,279-parameter design reaching 99.99%; sharing the next-sparsest bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1278, "training_steps": 4999}

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
hypothesis: A 1,277-parameter model will retain at least 99% accuracy because the verified 1,278-parameter model achieved 100%, while sharing the ninth-farthest bias affects only nine query-key pairs at full context and preserves head-specific biases at every more frequent distance.
change: Extend cross-head sharing from the current six to the nine maximum-distance attention biases, reducing the current model by three learned parameters and the best verified design by one.
mechanism: Cross-head sharing of the nine sparsest relative-distance biases
evidence_used: Progressive sharing of one through eight sparsest distance bins consistently exceeded 99% accuracy, and the eight-bin 1,278-parameter design achieved 100%; sharing the next-sparsest bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1277, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,276-parameter model will retain at least 99% accuracy because the verified 1,277-parameter model achieved 100%, while sharing the tenth-farthest bias affects only ten query-key pairs at full context and preserves head-specific biases at every more frequent distance.
change: Extend cross-head sharing from the current two to the ten maximum-distance attention biases, reducing the current model by eight learned parameters and the best verified design by one.
mechanism: Cross-head sharing of the ten sparsest relative-distance biases
evidence_used: Progressive sharing through nine sparsest distance bins consistently exceeded 99% accuracy, and the nine-bin 1,277-parameter design achieved 100%; sharing the next-sparsest bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1276, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,275-parameter model will retain at least 99% accuracy because the verified 1,276-parameter model achieved 99.96%, while sharing the eleventh-farthest bias affects only eleven query-key pairs at full context and preserves head-specific biases at every more frequent distance.
change: Extend cross-head sharing from the ten to the eleven maximum-distance attention biases, reducing the model by one learned parameter.
mechanism: Cross-head sharing of the eleven sparsest relative-distance biases
evidence_used: The current 1,276-parameter design met the requirement after progressive sharing through ten sparsest distance bins; extending the same mechanism by one bin is the smallest directly supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1275, "training_steps": 4999}



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
